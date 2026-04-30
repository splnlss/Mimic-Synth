"""
Bucket 2 V1.2: OB-Xf capture rig.
Renders (parameter vector, note) -> WAV via DawDreamer.
Runs on macOS, Windows, Linux.

V1.2 changes vs V1.1:
- Settle loop runs with the CORRECT patch context at all times:
  * At vector boundaries: settles BEFORE the patch change, draining the
    OLD patch's release tail (v1.1 fought the new patch's characteristics).
  * Between notes within a vector: settles with the current patch loaded
    (correct, since params haven't changed).
- Measures patch self-noise baseline (no MIDI) after loading each new
  patch. Stored in parquet as `self_noise`. Patches that self-oscillate
  or have LFO artifacts are expected to have energy in the pre-roll
  window — the bleed detector would false-positive on these without
  the baseline.
- hard_reset (graph reload) as fallback when settle times out.
- Uses Sobol random_base2(m) instead of random(n) to avoid silent
  truncation of non-power-of-2 sample counts.

M=10 by default (2^10 = 1,024 vectors). Use M=14 for production.
"""
import sys
from pathlib import Path
import hashlib
import math
import platform
import numpy as np
import pandas as pd
import yaml
import soundfile as sf
from scipy.stats.qmc import Sobol
from tqdm import tqdm

# Allow running as `python s02_capture/capture_v1_2.py` from the repo root
# or as `python capture_v1_2.py` from inside s02_capture/
sys.path.insert(0, str(Path(__file__).parent.parent))
from defaults import PROFILE_PATH, S02_DIR, S02_WAV_DIR, S02_PARQUET, SAMPLE_RATE, BUFFER_SIZE

M = 14                       # Sobol exponent: 2^M vectors (14 → 16,384 vectors)
OUT_DIR      = S02_DIR
WAV_DIR      = S02_WAV_DIR
PARQUET_PATH = S02_PARQUET
CHECKPOINT_EVERY = 50        # flush parquet every N vectors

# Settle pass — renders silent chunks to drain release tails between captures.
SETTLE_CHUNK_SEC = 0.05      # 50ms chunks (faster polling than v1.1's 100ms)
SETTLE_MAX_SEC = 10.0        # long OB-Xf releases with high resonance can ring for 8s+
SETTLE_THRESHOLD = 1e-4      # match the silence detector — no point settling below noise floor

WAV_DIR.mkdir(parents=True, exist_ok=True)


def load_profile(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_plugin_path(profile):
    """Pick the right plugin path for the current OS."""
    sys = platform.system()
    key = {
        "Darwin":  "plugin_path_macos",
        "Windows": "plugin_path_windows",
        "Linux":   "plugin_path_linux",
    }[sys]
    path = Path(profile["synth"][key]).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"OB-Xf not found at {path}. Install from "
            "https://github.com/surge-synthesizer/OB-Xf/releases"
        )
    return str(path)


def build_name_index(synth):
    """Return a {param_name: index} dict for the loaded plugin."""
    n = synth.get_plugin_parameter_size()
    return {synth.get_parameter_name(i): i for i in range(n)}


def apply_params(synth, params_dict, profile, name_idx):
    """Write normalised [0,1] param vector to the VST.
    Mutates params_dict in-place so stored values match what the synth received."""
    for name in list(params_dict):
        value = params_dict[name]
        spec = profile["parameters"][name]
        if not spec["continuous"]:
            n_cats = len(spec.get("categories", [])) or 2
            value = round(value * (n_cats - 1)) / (n_cats - 1)
            params_dict[name] = value
        synth.set_parameter(name_idx[name], float(value))


def reset(synth, profile, name_idx):
    for name, canonical in profile["reset"].items():
        if name in name_idx:
            synth.set_parameter(name_idx[name], float(canonical))


def measure_self_noise(engine, synth, duration=0.2):
    """Render a silent chunk (no MIDI) and return the peak amplitude.
    Call after loading a new patch to measure its self-noise floor —
    self-oscillating filters, LFO artifacts, cross-mod feedback, etc.
    Duration matches pre_roll_sec so the measurement covers the same
    window the bleed detector inspects."""
    synth.clear_midi()
    engine.render(duration)
    return float(np.max(np.abs(engine.get_audio())))


def settle(engine, synth, played_notes,
           chunk=SETTLE_CHUNK_SEC,
           max_sec=SETTLE_MAX_SEC,
           threshold=SETTLE_THRESHOLD):
    """Drain the current patch's release tail by rendering silent chunks.
    Clears MIDI buffer, then polls until output peak < threshold.
    Returns elapsed seconds; if >= max_sec the synth never settled."""
    synth.clear_midi()
    elapsed = 0.0
    while elapsed < max_sec:
        synth.clear_midi()
        engine.render(chunk)
        peak = float(np.max(np.abs(engine.get_audio())))
        if peak < threshold:
            return elapsed
        elapsed += chunk
    return elapsed


def hard_reset(engine, synth, settle_after=True, threshold=SETTLE_THRESHOLD):
    """Reload the processing graph to nuke all internal VST state, then
    run a second settle pass. The graph reload partially clears state,
    making residual energy easier to drain on the second pass."""
    engine.load_graph([(synth, [])])
    if settle_after:
        settle(engine, synth, [], threshold=threshold)


def render_one(engine, synth, note, profile):
    """Render a single (note, param-state) capture. Returns mono float32.
    No settle pass — caller is responsible for settling before the patch change."""
    synth.clear_midi()
    vel = profile["probe"]["velocity"]
    pre_roll = float(profile["probe"].get("pre_roll_sec", 0.0))
    synth.add_midi_note(note, vel, pre_roll, profile["probe"]["hold_sec"])
    engine.set_bpm(120)
    engine.render(profile["probe"]["render_sec"])
    audio = engine.get_audio()               # (channels, samples)
    return audio.mean(axis=0).astype(np.float32)


def sample_vectors(m, modulated_params, seed=0):
    """Sobol-sample the modulated parameter subspace in [0,1]^d.
    Uses random_base2(m) to guarantee 2^m balanced points."""
    d = len(modulated_params)
    sobol = Sobol(d=d, scramble=True, seed=seed)
    return sobol.random_base2(m=m)


def capture_vector(engine, synth, vec, notes, profile, name_idx,
                   modulated, played_notes_prev=None):
    """Capture one parameter vector across all notes.

    Handles the full v1.2 lifecycle: settle previous patch, load new patch,
    measure self-noise, settle between notes, hard_reset fallback.

    Args:
        played_notes_prev: notes played by the previous vector (for settle).
            Pass None or [] for the first vector.

    Returns:
        list of dicts (one per successfully rendered note), each containing:
            hash, note, audio (np.ndarray), self_noise, params_dict
        Plus an int count of unsettled events.
    """
    unsettled = 0

    # Settle previous patch's tail
    if played_notes_prev:
        settled_sec = settle(engine, synth, played_notes_prev)
        if settled_sec >= SETTLE_MAX_SEC:
            unsettled += 1
            hard_reset(engine, synth)

    # Load new patch
    reset(synth, profile, name_idx)
    params_dict = dict(zip(modulated, vec))
    apply_params(synth, params_dict, profile, name_idx)

    # Measure self-noise
    self_noise = measure_self_noise(engine, synth)
    effective_threshold = max(SETTLE_THRESHOLD, self_noise * 2.0)

    results = []
    last_note = None
    for note in notes:
        # Settle between notes within this vector
        if last_note is not None:
            settled_sec = settle(engine, synth, [last_note],
                                 threshold=effective_threshold)
            if settled_sec >= SETTLE_MAX_SEC:
                unsettled += 1
                hard_reset(engine, synth, threshold=effective_threshold)
                reset(synth, profile, name_idx)
                apply_params(synth, params_dict, profile, name_idx)

        audio = render_one(engine, synth, note, profile)
        last_note = note

        h = hashlib.md5(vec.tobytes() + bytes([note])).hexdigest()[:12]
        results.append({
            "hash": h,
            "note": note,
            "audio": audio,
            "self_noise": self_noise,
            "params_dict": params_dict,
        })

    return results, unsettled


def _prompt_resume_or_overwrite(n_rows: int) -> str:
    # Auto-resume for non-interactive runs (piped stdin, nohup, background jobs)
    import sys
    if not sys.stdin.isatty():
        return "resume"
    while True:
        ans = input(
            f"\nExisting dataset found at {PARQUET_PATH} with {n_rows} rows.\n"
            "  [c]ontinue (skip already-captured vectors)\n"
            "  [o]verwrite (delete and start fresh)\n"
            "  [a]bort\n"
            "Choice [c/o/a]: "
        ).strip().lower()
        if ans in ("c", "continue"): return "resume"
        if ans in ("o", "overwrite"): return "overwrite"
        if ans in ("a", "abort", ""): return "abort"
        print("  (please answer c, o, or a)")


def _load_existing_rows() -> tuple[list[dict], set[str]]:
    if not PARQUET_PATH.exists():
        return [], set()
    df = pd.read_parquet(PARQUET_PATH)
    return df.to_dict("records"), set(df["hash"].astype(str).tolist())


def _flush(rows: list[dict]) -> None:
    tmp = PARQUET_PATH.with_suffix(".parquet.tmp")
    pd.DataFrame(rows).to_parquet(tmp)
    tmp.replace(PARQUET_PATH)


def main():
    profile = load_profile(PROFILE_PATH)
    modulated = [
        name for name, spec in profile["parameters"].items()
        if spec.get("importance", 0) > 0
    ]
    notes = profile["probe"]["notes"]

    plugin_path = resolve_plugin_path(profile)
    import dawdreamer as daw  # noqa: PLC0415 — lazy import; pure helpers must not require DawDreamer
    engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
    synth = engine.make_plugin_processor("obxf", plugin_path)
    name_idx = build_name_index(synth)

    missing = [n for n in modulated if n not in name_idx]
    if missing:
        raise RuntimeError(
            f"Profile references parameters not exposed by this OB-Xf build: {missing}. "
            "Run enumerate_params.py and reconcile s01_profiles/obxf.yaml."
        )

    rows, done_hashes = _load_existing_rows()
    if rows:
        choice = _prompt_resume_or_overwrite(len(rows))
        if choice == "abort":
            print("Aborted.")
            return
        if choice == "overwrite":
            PARQUET_PATH.unlink(missing_ok=True)
            for wav in WAV_DIR.glob("*.wav"):
                wav.unlink()
            rows, done_hashes = [], set()
            print("Cleared existing dataset.")
        else:
            print(f"Resuming — {len(done_hashes)} captures already present.")

    n_samples = 2 ** M
    vectors = sample_vectors(M, modulated)
    engine.load_graph([(synth, [])])

    expected = [
        [hashlib.md5(v.tobytes() + bytes([n])).hexdigest()[:12] for n in notes]
        for v in vectors
    ]
    fully_done = sum(1 for hs in expected if all(h in done_hashes for h in hs))
    if fully_done:
        print(f"  {fully_done}/{n_samples} vectors already complete — skipping.")

    skipped_resume = 0
    silent_count = 0
    clip_count = 0
    stuck_count = 0
    unsettled_count = 0
    wav_bytes_written = 0
    total_captures = n_samples * len(notes)
    played_notes_this_vector: list[int] = []

    def _postfix():
        cur_mb = wav_bytes_written / 1e6
        if len(done_hashes) > 0:
            eta_mb = wav_bytes_written / len(done_hashes) * total_captures / 1e6
            mb_str = f"{cur_mb:.0f}/{eta_mb:.0f}"
        else:
            mb_str = f"{cur_mb:.0f}"
        d = {"sr": f"{SAMPLE_RATE//1000}kHz", "captures": len(done_hashes), "mb": mb_str}
        if silent_count:    d["silent"] = silent_count
        if clip_count:      d["clip"] = clip_count
        if stuck_count:     d["stuck"] = stuck_count
        if unsettled_count: d["unsettled"] = unsettled_count
        return d

    pbar = tqdm(total=n_samples, initial=fully_done, desc="capturing")
    pbar.set_postfix(**_postfix())

    for i, (vec, expected_hashes) in enumerate(zip(vectors, expected)):
        if all(h in done_hashes for h in expected_hashes):
            skipped_resume += 1
            continue

        # ── SETTLE: drain the PREVIOUS patch's tail before loading new params ──
        if played_notes_this_vector:
            settled_sec = settle(engine, synth, played_notes_this_vector)
            if settled_sec >= SETTLE_MAX_SEC:
                unsettled_count += 1
                hard_reset(engine, synth)
        played_notes_this_vector = []

        # ── Load new patch ──
        reset(synth, profile, name_idx)
        params_dict = dict(zip(modulated, vec))
        apply_params(synth, params_dict, profile, name_idx)

        # ── Measure patch self-noise (no MIDI) ──
        self_noise = measure_self_noise(engine, synth)

        # ── Render each note ──
        # Use self_noise as settle threshold floor — patches that self-oscillate
        # can never settle below their noise floor, so don't waste time trying.
        effective_threshold = max(SETTLE_THRESHOLD, self_noise * 2.0)

        last_note = None
        for note, h in zip(notes, expected_hashes):
            if h in done_hashes:
                continue
            # Settle between notes within a vector — same patch, just drain
            # the previous note's release tail.
            if last_note is not None:
                settled_sec = settle(engine, synth, [last_note],
                                     threshold=effective_threshold)
                if settled_sec >= SETTLE_MAX_SEC:
                    unsettled_count += 1
                    hard_reset(engine, synth, threshold=effective_threshold)
                    # Re-apply current patch after hard reset
                    reset(synth, profile, name_idx)
                    apply_params(synth, params_dict, profile, name_idx)
            audio = render_one(engine, synth, note, profile)
            last_note = note
            played_notes_this_vector.append(note)

            peak = float(np.max(np.abs(audio)))
            if peak < 1e-4:
                silent_count += 1
                pbar.write(f"  warning: silent render (vec={i}, note={note}) — skipping")
                pbar.set_postfix(**_postfix())
                continue
            if peak > 0.99 and int(np.sum(np.abs(audio) > 0.99)) >= 5:
                clip_count += 1
                pbar.write(f"  warning: clipped render (vec={i}, note={note})")
            wav_path = WAV_DIR / f"{h}_n{note}.wav"
            sf.write(wav_path, audio, SAMPLE_RATE)
            wav_bytes_written += audio.nbytes
            rows.append({
                "hash": h,
                "note": note,
                "wav": str(wav_path),
                "self_noise": self_noise,
                **{f"p_{k}": v for k, v in params_dict.items()},
            })
            done_hashes.add(h)

        pbar.update(1)
        pbar.set_postfix(**_postfix())

        if (i + 1) % CHECKPOINT_EVERY == 0 and rows:
            _flush(rows)

    pbar.close()
    if rows:
        _flush(rows)
    if skipped_resume:
        print(f"Skipped {skipped_resume} vectors already captured.")
    print(f"Saved {len(rows)} captures to {PARQUET_PATH}")


if __name__ == "__main__":
    main()

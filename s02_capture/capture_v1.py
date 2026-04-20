"""
Bucket 2 V1: OB-Xf capture rig.
Renders (parameter vector, note) -> WAV via DawDreamer.
Runs on macOS, Windows, Linux.

N_SAMPLES=50 by default for fast validation. Scale to 10_000+ for real dataset runs.
"""
from pathlib import Path
import hashlib
import platform
import numpy as np
import pandas as pd
import yaml
import soundfile as sf
from scipy.stats.qmc import Sobol
import dawdreamer as daw
from tqdm import tqdm

SAMPLE_RATE = 48000
BUFFER_SIZE = 512
N_SAMPLES = 10_000       # set to 100_000 for full production runs
PROFILE_PATH = "s01_profiles/obxf.yaml"
OUT_DIR = Path("data")
WAV_DIR = OUT_DIR / "wav"
PARQUET_PATH = OUT_DIR / "samples.parquet"
CHECKPOINT_EVERY = 50    # flush parquet every N vectors so an interrupt loses ≤N vectors
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
            params_dict[name] = value  # keep stored value in sync with synth state
        synth.set_parameter(name_idx[name], float(value))


def reset(synth, profile, name_idx):
    for name, canonical in profile["reset"].items():
        if name in name_idx:
            synth.set_parameter(name_idx[name], float(canonical))


def render_one(engine, synth, note, profile):
    """Render a single (note, param-state) capture. Returns mono float32."""
    synth.clear_midi()
    vel = profile["probe"]["velocity"]
    synth.add_midi_note(note, vel, 0.0, profile["probe"]["hold_sec"])
    engine.set_bpm(120)
    engine.render(profile["probe"]["render_sec"])
    audio = engine.get_audio()               # (channels, samples)
    return audio.mean(axis=0).astype(np.float32)


def sample_vectors(n, modulated_params, seed=0):
    """Sobol-sample the modulated parameter subspace in [0,1]^d."""
    d = len(modulated_params)
    sobol = Sobol(d=d, scramble=True, seed=seed)
    return sobol.random(n)


def _prompt_resume_or_overwrite(n_rows: int) -> str:
    """Ask the user what to do about an existing dataset. Returns 'resume', 'overwrite', or 'abort'."""
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
    """Return (rows, set_of_hashes) from an existing parquet, or ([], set()) if none."""
    if not PARQUET_PATH.exists():
        return [], set()
    df = pd.read_parquet(PARQUET_PATH)
    return df.to_dict("records"), set(df["hash"].astype(str).tolist())


def _flush(rows: list[dict]) -> None:
    """Write current rows to parquet atomically (tmp + rename)."""
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

    vectors = sample_vectors(N_SAMPLES, modulated)
    engine.load_graph([(synth, [])])

    # Precompute per-vector expected hashes so we can report resume progress up front.
    expected = [
        [hashlib.md5(v.tobytes() + bytes([n])).hexdigest()[:12] for n in notes]
        for v in vectors
    ]
    fully_done = sum(1 for hs in expected if all(h in done_hashes for h in hs))
    if fully_done:
        print(f"  {fully_done}/{N_SAMPLES} vectors already complete — skipping.")

    skipped_resume = 0
    silent_count = 0
    clip_count = 0
    stuck_count = 0
    wav_bytes_written = 0
    total_captures = N_SAMPLES * len(notes)

    def _postfix():
        cur_mb = wav_bytes_written / 1e6
        if len(done_hashes) > 0:
            eta_mb = wav_bytes_written / len(done_hashes) * total_captures / 1e6
            mb_str = f"{cur_mb:.0f}/{eta_mb:.0f}"
        else:
            mb_str = f"{cur_mb:.0f}"
        d = {"sr": f"{SAMPLE_RATE//1000}kHz", "captures": len(done_hashes), "mb": mb_str}
        if silent_count: d["silent"] = silent_count
        if clip_count:   d["clip"] = clip_count
        if stuck_count:  d["stuck"] = stuck_count
        return d

    pbar = tqdm(total=N_SAMPLES, initial=fully_done, desc="capturing")
    pbar.set_postfix(**_postfix())

    for i, (vec, expected_hashes) in enumerate(zip(vectors, expected)):
        if all(h in done_hashes for h in expected_hashes):
            skipped_resume += 1
            continue
        pbar.update(1)

        reset(synth, profile, name_idx)
        params_dict = dict(zip(modulated, vec))
        apply_params(synth, params_dict, profile, name_idx)

        for note, h in zip(notes, expected_hashes):
            if h in done_hashes:
                continue
            audio = render_one(engine, synth, note, profile)
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
                **{f"p_{k}": v for k, v in params_dict.items()},
            })
            done_hashes.add(h)

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

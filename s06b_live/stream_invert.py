#!/usr/bin/env python3
"""
stream_invert.py — Streaming audio-to-parameter inversion for OB-Xf.

v4 changes:
  - Fine-grained note segmentation: energy + pitch-discontinuity detection at 10 ms
    resolution correctly splits legato multi-note targets (no silence between notes).
  - Per-region MIDI note: each detected note region gets its own MIDI note derived
    from the region's median pitch. Two separate note values per region:
      midi_note      — exact MIDI integer sent to DawDreamer for pitch accuracy
      surrogate_note — snapped to nearest profile training note so the surrogate
                       stays in-distribution (prevents Osc 1 Pitch hallucination)
  - Warm-start reset at region boundaries so the surrogate search restarts fresh
    when the surrogate_note changes.
  - Render emits separate note-on/off per region with the exact midi_note and
    precise sample-accurate timings (not coarse frame boundaries).
"""
# ── Make the project root importable regardless of how this script is invoked.
# Needed when running as a plain script (python stream_invert.py) or via
# `conda run` without PYTHONPATH set.  `python -m s06b_live.stream_invert`
# does not need this because -m puts the package root on sys.path automatically,
# but direct invocation does not.
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

import defaults as _defs
from s05_surrogate.model import Surrogate
from s06_invert.invert import _load_surrogate


# ── Pitch detection ──────────────────────────────────────────────────────────

def detect_pitch_autocorr(audio, sr):
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio * np.hanning(len(audio))
    autocorr = signal.correlate(audio, audio, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    # Search within valid lag range: period of max pitch → period of min pitch
    min_lag = max(1, int(sr / 5000))   # ~10 samples for 5000 Hz ceiling
    max_lag = int(sr / 50)             # 960 samples for 50 Hz floor
    search = autocorr[min_lag:max_lag]
    peaks, _ = signal.find_peaks(search, distance=min_lag)
    if len(peaks) == 0:
        return _detect_pitch_fft(audio, sr)
    # Adjust indices back to full autocorr space
    peaks = peaks + min_lag
    peak_heights = autocorr[peaks]
    best_idx = float(peaks[np.argmax(peak_heights)])
    # Parabolic sub-sample refinement: delta = 0.5*(left-right)/(left-2*center+right)
    i = int(best_idx)
    if 0 < i < len(autocorr) - 1:
        alpha = autocorr[i - 1]
        beta = autocorr[i]
        gamma = autocorr[i + 1]
        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > 1e-10:
            best_idx = i + 0.5 * (alpha - gamma) / denom
    pitch_hz = sr / best_idx
    if 50 <= pitch_hz <= 5000:
        return pitch_hz
    return None


def _detect_pitch_fft(audio, sr):
    freqs, psd = signal.welch(audio, sr, nperseg=min(len(audio), 2048))
    pitch_hz = freqs[np.argmax(psd)]
    if 50 <= pitch_hz <= 5000:
        return pitch_hz
    return None


# ── Note region detection (energy + pitch-change aware) ──────────────────────

def _hz_to_midi_note(hz, fallback=60):
    """Convert Hz to the nearest MIDI note (0–127)."""
    if hz is None or hz <= 0:
        return fallback
    midi = 12.0 * np.log2(hz / 440.0) + 69.0
    return int(np.clip(round(midi), 0, 127))


def _snap_to_profile_note(midi_note: int, profile_notes: list[int]) -> int:
    """Return the profile training note closest to midi_note.

    Keeps the surrogate in-distribution. Out-of-distribution note values
    cause the surrogate to hallucinate Osc 1 Pitch offsets. The surrogate
    was only trained on profile_notes, so we must pass one of those as the
    note context and let Osc 1 Pitch compensate for any remaining pitch delta.
    """
    if not profile_notes:
        return midi_note
    return min(profile_notes, key=lambda n: abs(n - midi_note))


def detect_note_regions(
    audio,
    sr,
    profile_notes: list[int] | None = None,
    energy_threshold_factor: float = 0.05,
    pitch_change_threshold_st: float = 3.0,
    min_note_ms: float = 50.0,
    analysis_win_ms: float = 20.0,
    analysis_hop_ms: float = 10.0,
):
    """Fine-grained note segmentation using energy + pitch discontinuity.

    Runs at analysis_hop_ms resolution (default 10 ms) independent of the
    coarser inversion window. Pitch jumps ≥ pitch_change_threshold_st semitones
    trigger a region boundary even when energy stays high (legato playing).

    midi_note: nearest integer MIDI note to the region's median pitch.
    surrogate_note: snapped to the nearest profile training note so the
      surrogate stays in-distribution; Osc 1 Pitch compensates for any delta.

    Returns a list of dicts:
        onset_sec      : float  — region start time (seconds)
        offset_sec     : float  — region end time (seconds)
        median_hz      : float  — median detected pitch within region
        midi_note      : int    — nearest MIDI note to median_hz (0–127)
        surrogate_note : int    — nearest profile note (used as surrogate context)
    """
    win_n = int(analysis_win_ms / 1000.0 * sr)
    hop_n = int(analysis_hop_ms / 1000.0 * sr)
    min_frames = max(1, int(min_note_ms / analysis_hop_ms))

    # Fine-grained frame analysis
    frames = []
    for start in range(0, len(audio) - win_n + 1, hop_n):
        win = audio[start: start + win_n]
        rms = float(np.sqrt(np.mean(win ** 2)))
        hz = detect_pitch_autocorr(win, sr)
        frames.append({"t_sec": start / sr, "rms": rms, "hz": hz})

    if not frames:
        return []

    rms_arr = np.array([f["rms"] for f in frames])
    threshold = rms_arr.max() * energy_threshold_factor

    regions = []
    in_note = False
    onset = 0
    # Reference pitch for jump detection — updated instantaneously within a region.
    # Short octave-error frames trigger spurious 1-frame regions that get discarded
    # by min_frames, so we don't need IIR smoothing here.
    prev_valid_hz = None

    for i, f in enumerate(frames):
        is_active = f["rms"] > threshold
        hz = f["hz"]

        pitch_jump = False
        if hz and prev_valid_hz and in_note:
            delta_st = abs(12.0 * np.log2(hz / prev_valid_hz))
            if delta_st > pitch_change_threshold_st:
                pitch_jump = True

        if is_active and not in_note:
            onset = i
            in_note = True
            prev_valid_hz = hz

        elif in_note and (not is_active or pitch_jump):
            if i - onset >= min_frames:
                region_pitches = [frames[j]["hz"] for j in range(onset, i) if frames[j]["hz"]]
                med_hz = float(np.median(region_pitches)) if region_pitches else None
                midi = _hz_to_midi_note(med_hz)
                regions.append({
                    "onset_sec": frames[onset]["t_sec"],
                    "offset_sec": frames[i - 1]["t_sec"],
                    "median_hz": med_hz,
                    "midi_note": midi,
                    "surrogate_note": _snap_to_profile_note(midi, profile_notes or []),
                })
            in_note = False
            if is_active and pitch_jump:
                # Immediately open the next region (legato transition)
                onset = i
                in_note = True
                prev_valid_hz = hz

        # Update reference pitch — only when stable within a region
        if hz and not pitch_jump:
            prev_valid_hz = hz

    if in_note and len(frames) - onset >= min_frames:
        region_pitches = [frames[j]["hz"] for j in range(onset, len(frames)) if frames[j]["hz"]]
        med_hz = float(np.median(region_pitches)) if region_pitches else None
        midi = _hz_to_midi_note(med_hz)
        regions.append({
            "onset_sec": frames[onset]["t_sec"],
            "offset_sec": frames[-1]["t_sec"],
            "median_hz": med_hz,
            "midi_note": midi,
            "surrogate_note": _snap_to_profile_note(midi, profile_notes or []),
        })

    return regions


def _region_for_frame(t_sec, note_regions):
    """Return the note region dict containing t_sec, or None if silent."""
    for r in note_regions:
        if r["onset_sec"] <= t_sec <= r["offset_sec"]:
            return r
    return None


# ── Pinned params ────────────────────────────────────────────────────────────
# These params are pinned during surrogate inversion to prevent the surrogate
# from finding degenerate solutions that produce the right embedding but the
# wrong audio. Mapping: param name (without `p_` prefix) → fixed [0,1] value.
#
# Why each is pinned:
#   "Osc 1 Pitch" → 0.5 (centre, no transposition)
#       OB-Xf's Osc 1 Pitch is a ±24-semitone transpose. The surrogate
#       happily pushes this to 1.0 (+24 st) to match the embedding, which
#       shifts a MIDI 85 (1109 Hz) target up to ~4.5 kHz. We send the exact
#       MIDI note to DawDreamer (see surrogate_note vs midi_note logic),
#       so Osc 1 Pitch must stay at centre.
#
#   "Amp Env Release" → 0.2 (short release)
#       Free during inversion the surrogate finds long-release solutions to
#       match the target's tail energy. This causes notes to ring past the
#       region boundary and through the end of the sample. 0.2 ≈ ~150 ms
#       release: enough for natural-sounding decay, not enough to bleed
#       across regions.
#
#   "LFO 1 to Osc 1 Pitch" → 0.0 (no LFO modulation on pitch)
#       Prevents pitch wobble that would corrupt the per-region pitch.
PINNED_PARAMS: dict[str, float] = {
    "Osc 1 Pitch": 0.5,
    "Amp Env Release": 0.2,
    "LFO 1 to Osc 1 Pitch": 0.0,
}


def _pinned_indices(param_cols: list[str]) -> dict[int, float]:
    """Map column index → pinned value for params present in `param_cols`."""
    out: dict[int, float] = {}
    for name, val in PINNED_PARAMS.items():
        col = f"p_{name}"
        if col in param_cols:
            out[param_cols.index(col)] = val
    return out


# ── Pitch → MIDI pitch bend + Osc 1 Pitch ────────────────────────────────────

OSC1_PITCH_RANGE_SEMITONES = 12.0
MIDI_PITCH_BEND_RANGE_SEMITONES = 2.0  # ±2 semitones via MIDI pitch bend


def pitch_hz_to_midi_bend(pitch_hz, base_note):
    """Convert pitch_hz to (midi_note, pitch_bend_value).

    pitch_bend_value is in [0, 1] where 0.5 = center (no bend).
    MIDI pitch bend range is ±2 semitones.
    """
    if pitch_hz is None or pitch_hz <= 0:
        return base_note, 0.5
    midi = 12.0 * np.log2(pitch_hz / 440.0) + 69.0
    rounded_note = int(round(midi))
    bend_semitones = midi - rounded_note
    bend_value = 0.5 + bend_semitones / (2.0 * MIDI_PITCH_BEND_RANGE_SEMITONES)
    return rounded_note, float(np.clip(bend_value, 0.0, 1.0))


def pitch_hz_to_osc1(pitch_hz, base_note):
    if pitch_hz is None or pitch_hz <= 0:
        return 0.5
    midi = 12.0 * np.log2(pitch_hz / 440.0) + 69.0
    offset = midi - base_note
    value = 0.5 + offset / (2.0 * OSC1_PITCH_RANGE_SEMITONES)
    return float(np.clip(value, 0.0, 1.0))


# ── Smoothing ────────────────────────────────────────────────────────────────

def smooth_trajectory(values, window_size=3):
    values = np.array(values, dtype=np.float64)
    if len(values) <= window_size:
        return values
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(values, kernel, mode='same')
    edges = window_size // 2
    if edges > 0:
        smoothed[:edges] = values[:edges]
        smoothed[-edges:] = values[-edges:]
    return smoothed


# ── Gradient inversion (all params free, including Osc 2) ────────────────────

def grad_invert(
    surrogate, target_emb, note, d_params,
    n_starts=4, steps=50, lr=5e-2, device="cuda", init_params=None,
    pin_indices: dict[int, float] | None = None,
):
    """Multi-start gradient inversion through the surrogate.

    pin_indices: maps {param_index: fixed_value}. Each step the pinned indices
        are forced to their fixed value before the surrogate forward, and their
        gradients are zeroed before the optimizer step so they never drift.
        See PINNED_PARAMS for the rationale.
    """
    surrogate.eval()
    target = target_emb.to(device).unsqueeze(0)
    note_t = torch.full((1,), note / 127.0, device=device)
    pin_indices = pin_indices or {}

    best_score = float("inf")
    best_params = None

    for i in range(n_starts):
        if init_params is not None and i == 0:
            if isinstance(init_params, torch.Tensor):
                params = init_params.clone().unsqueeze(0).to(device).requires_grad_(True)
            else:
                params = torch.from_numpy(init_params).clone().unsqueeze(0).to(device).requires_grad_(True)
        else:
            params = torch.rand(1, d_params, device=device, requires_grad=True)

        # Initialise pinned params to their fixed value (random init may have
        # placed them anywhere; we don't want a pre-pin drift).
        if pin_indices:
            with torch.no_grad():
                for idx, val in pin_indices.items():
                    params[0, idx] = val

        opt = torch.optim.Adam([params], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            clamped = params.clamp(0.0, 1.0)
            if pin_indices:
                # Replace pinned positions in the forward path so the
                # surrogate sees the fixed value but gradients still flow
                # to the other params.
                pin_mask = torch.ones_like(clamped)
                pin_fixed = torch.zeros_like(clamped)
                for idx, val in pin_indices.items():
                    pin_mask[0, idx] = 0.0
                    pin_fixed[0, idx] = val
                clamped = clamped * pin_mask + pin_fixed
            pred = surrogate(clamped, note_t)
            loss = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
            loss.backward()
            if pin_indices and params.grad is not None:
                for idx in pin_indices:
                    params.grad[0, idx] = 0.0
            opt.step()

        final = params.detach().clamp(0.0, 1.0)
        # Force pinned values in the returned params (they won't have moved
        # if the masking + zero-grad worked, but be defensive).
        if pin_indices:
            for idx, val in pin_indices.items():
                final[0, idx] = val
        with torch.no_grad():
            pred = surrogate(final, note_t)
            score = (1.0 - F.cosine_similarity(pred, target, dim=-1)).item()

        if score < best_score:
            best_score = score
            best_params = final.squeeze(0).cpu()

    assert best_params is not None
    return best_score, best_params


# ── Main inversion ───────────────────────────────────────────────────────────

def stream_invert(
    target_wav: Path,
    surrogate_checkpoint: Path,
    profile_path: Path,
    out_dir: Path,
    device: str = "cuda",
    win_sec: float = 0.1,
    hop_sec: float = 0.05,
    n_starts: int = 4,
    grad_steps: int = 50,
    smooth_window: int = 3,
    skip_render: bool = False,
    refine_iterations: int = 3,
    refine_threshold: float = 0.01,
) -> dict:
    audio, sr = sf.read(str(target_wav), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio_duration = len(audio) / sr  # preserved for render length

    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    surrogate, manifest = _load_surrogate(Path(surrogate_checkpoint), device)
    surrogate = surrogate.to(device)
    surrogate.eval()
    param_cols = manifest["param_cols"]
    d_params = len(param_cols)

    # Indices to pin during inversion (keeps pitch stable + prevents long ring).
    pin_idx = _pinned_indices(param_cols)
    if pin_idx:
        print("Pinned params during inversion:")
        for idx, val in pin_idx.items():
            print(f"  {param_cols[idx]} = {val}")

    from s04_embed.embed import Embedder
    embedder = Embedder(device=device)

    profile_notes = profile.get("probe", {}).get("notes", [])

    # ── Fine-grained note region detection ───────────────────────────────
    note_regions = detect_note_regions(audio, sr, profile_notes=profile_notes)

    print(f"Detected {len(note_regions)} note region(s):")
    for i, r in enumerate(note_regions):
        print(f"  Region {i}: {r['onset_sec']:.3f}s–{r['offset_sec']:.3f}s  "
              f"pitch={r['median_hz']:.0f}Hz  MIDI={r['midi_note']}  "
              f"surrogate_note={r['surrogate_note']}")

    if not note_regions:
        # Fallback: treat entire audio as one region at the nearest profile note
        freqs, psd = signal.welch(audio, sr, nperseg=min(len(audio), 4096))
        global_hz = float(freqs[np.argmax(psd)])
        midi = _hz_to_midi_note(global_hz if 50 <= global_hz <= 5000 else None)
        snapped = _snap_to_profile_note(midi, profile_notes)
        note_regions = [{"onset_sec": 0.0, "offset_sec": len(audio)/sr,
                         "median_hz": global_hz, "midi_note": midi,
                         "surrogate_note": snapped}]
        print(f"  (fallback) {note_regions[0]}")

    # ── Coarse frame analysis (embedding + per-frame pitch) ───────────────
    win_samples = int(win_sec * sr)
    hop_samples = int(hop_sec * sr)
    n_frames = max(0, (len(audio) - win_samples) // hop_samples + 1)

    pitch_hz_list = []
    emb_list = []

    for start in tqdm(range(0, len(audio) - win_samples + 1, hop_samples), desc="Analyzing"):
        window = audio[start: start + win_samples]
        pitch_hz_list.append(detect_pitch_autocorr(window, sr))
        emb_list.append(embedder.encodec_embed(window, sr, pool="mean"))

    # ── Per-frame region lookup & pitch trajectory ───────────────────────
    # Each coarse frame maps to a note region (or None if silent).
    frame_regions = []
    frame_notes = []        # exact MIDI note for pitch bend + render
    frame_surrogate_notes = []  # snapped-to-profile note for surrogate context
    for i in range(n_frames):
        t_sec = i * hop_sec + win_sec / 2.0   # centre of the window
        r = _region_for_frame(t_sec, note_regions)
        frame_regions.append(r)
        frame_notes.append(r["midi_note"] if r else note_regions[0]["midi_note"])
        frame_surrogate_notes.append(
            r["surrogate_note"] if r else note_regions[0]["surrogate_note"]
        )

    pitch_bends = []
    osc1_pitch_values = []
    for i, p in enumerate(pitch_hz_list):
        # Pitch bend relative to the exact region MIDI note
        region_note = frame_notes[i]
        _, bend = pitch_hz_to_midi_bend(p, region_note)
        pitch_bends.append(bend)
        osc1_pitch_values.append(pitch_hz_to_osc1(p, region_note))

    pitch_bends_smooth = smooth_trajectory(pitch_bends, window_size=3)
    osc1_pitch_smooth = smooth_trajectory(osc1_pitch_values, window_size=3)

    # ── Gradient inversion per frame ─────────────────────────────────────
    results = []
    prev_params = None
    prev_surrogate_note = None

    for i in tqdm(range(n_frames), desc="Inverting"):
        timestamp = i * hop_sec
        emb_torch = torch.tensor(emb_list[i], dtype=torch.float32).to(device)
        frame_note = frame_notes[i]
        surrogate_note = frame_surrogate_notes[i]

        # Reset warm-start when the surrogate note context changes
        if surrogate_note != prev_surrogate_note:
            prev_params = None
        prev_surrogate_note = surrogate_note

        if prev_params is not None:
            score, params = grad_invert(
                surrogate, emb_torch, surrogate_note, d_params,
                n_starts=1, steps=grad_steps, device=device,
                init_params=prev_params, pin_indices=pin_idx,
            )
        else:
            score, params = grad_invert(
                surrogate, emb_torch, surrogate_note, d_params,
                n_starts=n_starts, steps=grad_steps, device=device,
                pin_indices=pin_idx,
            )

        res = {
            "timestamp": timestamp,
            "pitch_hz": float(pitch_hz_list[i]) if pitch_hz_list[i] else np.nan,
            "midi_note": frame_note,
            "surrogate_note": surrogate_note,
            "pitch_bend": pitch_bends_smooth[i],
            "osc1_pitch": osc1_pitch_smooth[i],
            "score": float(score),
            "active": frame_regions[i] is not None,
        }
        for col, val in zip(param_cols, params):
            res[col] = float(val)

        results.append(res)
        prev_params = params

    # ── Post-processing ──────────────────────────────────────────────────
    df = pd.DataFrame(results)

    for col in param_cols:
        if col in df.columns:
            df[col] = smooth_trajectory(df[col].values, smooth_window)

    best_row = df.loc[df["score"].idxmin()]

    # ── Save outputs ─────────────────────────────────────────────────────
    run_dir = out_dir / target_wav.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(run_dir / "stream_params.parquet")

    best_patch = {
        "target": str(target_wav),
        "note_regions": [
            {"onset_sec": r["onset_sec"], "offset_sec": r["offset_sec"],
             "median_hz": r["median_hz"], "midi_note": r["midi_note"],
             "surrogate_note": r["surrogate_note"]}
            for r in note_regions
        ],
        "pitch_hz": float(best_row["pitch_hz"]) if pd.notnull(best_row["pitch_hz"]) else None,
        "score": float(best_row["score"]),
        "params": {col: float(best_row[col]) for col in param_cols},
    }
    with open(run_dir / "best_patch.yaml", "w") as f:
        yaml.dump(best_patch, f, default_flow_style=False)

    pitch_traj = {
        "target": str(target_wav),
        "sr": sr,
        "audio_duration": audio_duration,
        "note_regions": [
            {"onset_sec": r["onset_sec"], "offset_sec": r["offset_sec"],
             "median_hz": r["median_hz"], "midi_note": r["midi_note"],
             "surrogate_note": r["surrogate_note"]}
            for r in note_regions
        ],
        "osc1_pitch_range_semitones": OSC1_PITCH_RANGE_SEMITONES,
        "midi_pitch_bend_range_semitones": MIDI_PITCH_BEND_RANGE_SEMITONES,
        "frames": [],
    }
    for _, row in df.iterrows():
        pitch_traj["frames"].append({
            "timestamp": float(row["timestamp"]),
            "pitch_hz": float(row["pitch_hz"]) if pd.notnull(row["pitch_hz"]) else None,
            "midi_note": int(row["midi_note"]),
            "pitch_bend": float(row["pitch_bend"]),
            "osc1_pitch": float(row["osc1_pitch"]),
            "score": float(row["score"]),
            "active": bool(row["active"]),
        })
    with open(run_dir / "pitch_trajectory.yaml", "w") as f:
        yaml.dump(pitch_traj, f, default_flow_style=False)

    print(f"✓ Streaming tracking complete: {len(df)} frames")
    print(f"✓ Best score: {best_row['score']:.4f}")
    print(f"✓ Saved to {run_dir / 'stream_params.parquet'}")

    # ── Render ───────────────────────────────────────────────────────────
    if not skip_render:
        _render_stream(df, pitch_traj, profile_path, run_dir, note_regions, audio_duration)

        if refine_iterations > 0:
            _refine_loop(
                target_wav=target_wav,
                profile_path=profile_path,
                run_dir=run_dir,
                note_regions=note_regions,
                param_cols=param_cols,
                surrogate=surrogate,
                embedder=embedder,
                device=device,
                audio_duration=audio_duration,
                max_iterations=refine_iterations,
                threshold=refine_threshold,
            )

    return {"df": df, "best": best_row, "run_dir": run_dir}


# ── Render with per-region note on/off + pitch bend ──────────────────────────

def _render_stream(
    df: pd.DataFrame,
    pitch_traj: dict,
    profile_path: Path,
    out_dir: Path,
    note_regions: list,
    audio_duration: float = None,
):
    """Render with per-region note on/off using precise onset/offset timings."""
    import dawdreamer as daw

    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    sr = profile.get("probe", {}).get("sample_rate", 48000)
    timestamps = df["timestamp"].values
    hop_sec = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.05
    # Render the full original audio duration, not just up to the last inversion frame
    total_sec = audio_duration if audio_duration else timestamps[-1] + hop_sec

    vst_path = Path(profile["synth"]["plugin_path_linux"])
    engine = daw.RenderEngine(sr, 512)
    plugin = engine.make_plugin_processor("synth", str(vst_path))
    engine.load_graph([(plugin, [])])

    num_params = plugin.get_plugin_parameter_size()
    param_name_to_index = {plugin.get_parameter_name(i): i for i in range(num_params)}

    param_cols = [c for c in df.columns if c.startswith("p_")]

    # Per-region note on/off.
    # Note-off for region N is sent at the onset of region N+1 so the release
    # phase starts at the correct transition point, not at the fine-grained
    # energy-fade frame (which can be 20 ms before the next note-on, leaving
    # insufficient time for even a short release to complete).
    for i, r in enumerate(note_regions):
        note_on = r["onset_sec"]
        if i < len(note_regions) - 1:
            note_off = note_regions[i + 1]["onset_sec"]
        else:
            note_off = r["offset_sec"]
        note_dur = max(0.0, note_off - note_on)
        if note_dur > 0:
            plugin.add_midi_note(r["midi_note"], 100, note_on, note_dur)

    # Parameter automation
    for col in param_cols:
        p_name = col.removeprefix("p_")
        if p_name in param_name_to_index:
            p_idx = param_name_to_index[p_name]
            data = np.column_stack((timestamps, df[col].values))
            plugin.set_automation(p_idx, data)

    # Pitch bend (if exposed as a named VST parameter)
    pitch_bends = [f["pitch_bend"] for f in pitch_traj["frames"]]
    pb_timestamps = [f["timestamp"] for f in pitch_traj["frames"]]
    if "Pitch Bend" in param_name_to_index:
        pb_data = np.column_stack((pb_timestamps, pitch_bends))
        plugin.set_automation(param_name_to_index["Pitch Bend"], pb_data)

    n_regions = len(note_regions)
    notes_str = ", ".join(f"MIDI {r['midi_note']} ({r['median_hz']:.0f}Hz)" for r in note_regions)
    print(f"Rendering stream: {total_sec:.2f}s  {n_regions} region(s): {notes_str}")
    engine.render(total_sec)
    audio_out = plugin.get_audio()

    out_path = out_dir / "rendered.wav"
    sf.write(str(out_path), audio_out.transpose(), sr)

    max_val = np.max(np.abs(audio_out))
    if max_val > 0:
        norm = audio_out / max_val * 0.5
        sf.write(str(out_path).replace(".wav", "_normalized.wav"), norm.transpose(), sr)

    print(f"✓ Rendered to {out_path}")


# ── Self-learning refinement loop (full-result driven) ───────────────────────

def _refine_loop(
    target_wav: Path,
    profile_path: Path,
    run_dir: Path,
    note_regions: list,
    param_cols: list,
    surrogate,
    embedder,
    device: str,
    audio_duration: float = None,
    max_iterations: int = 3,
    threshold: float = 0.01,
):
    """Render → embed → compare FULL result → adjust → re-render.

    Primary driver: full rendered audio vs full target audio comparison.
    Secondary: per-frame surrogate gradient to suggest direction.
    Only applies adjustments that improve the full result.
    """
    import dawdreamer as daw

    target_audio, sr = sf.read(str(target_wav), dtype="float32")
    if target_audio.ndim == 2:
        target_audio = target_audio.mean(axis=1)

    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    sr_profile = profile.get("probe", {}).get("sample_rate", 48000)
    vst_path = Path(profile["synth"]["plugin_path_linux"])

    df = pd.read_parquet(run_dir / "stream_params.parquet")
    timestamps = df["timestamp"].values
    hop_sec = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.05
    total_sec = audio_duration if audio_duration else timestamps[-1] + hop_sec

    # Same pin set as the initial inversion — refinement must not push these
    # parameters either (Osc 1 Pitch, Amp Env Release, LFO 1 to Osc 1 Pitch).
    pin_idx = _pinned_indices(param_cols)
    pinned_cols = {param_cols[i] for i in pin_idx}

    # Force the dataframe values to the pinned values so the FIRST render
    # actually reflects the pin (defensive — initial inversion already
    # produced these values, but a stale parquet from an older v4 run could
    # still have un-pinned values).
    for idx, val in pin_idx.items():
        df[param_cols[idx]] = val

    best_score = float("inf")
    best_df = df.copy()

    target_emb = embedder.encodec_embed(target_audio, sr, pool="mean")
    target_emb_torch = torch.tensor(target_emb, dtype=torch.float32).to(device)

    def _render_and_score(current_df):
        engine = daw.RenderEngine(sr_profile, 512)
        plugin = engine.make_plugin_processor("synth", str(vst_path))
        engine.load_graph([(plugin, [])])

        num_params = plugin.get_plugin_parameter_size()
        pn2i = {plugin.get_parameter_name(i): i for i in range(num_params)}

        for i, r in enumerate(note_regions):
            note_on = r["onset_sec"]
            note_off = note_regions[i + 1]["onset_sec"] if i < len(note_regions) - 1 else r["offset_sec"]
            dur = max(0.0, note_off - note_on)
            if dur > 0:
                plugin.add_midi_note(r["midi_note"], 100, note_on, dur)

        for col in param_cols:
            p_name = col.removeprefix("p_")
            if p_name in pn2i:
                data = np.column_stack((timestamps, current_df[col].values))
                plugin.set_automation(pn2i[p_name], data)

        engine.render(total_sec)
        rendered = plugin.get_audio().transpose()
        if rendered.ndim == 2:
            rendered = rendered.mean(axis=1)

        emb = embedder.encodec_embed(rendered, sr_profile, pool="mean")
        emb_t = torch.tensor(emb, dtype=torch.float32).to(device)
        score = (1.0 - F.cosine_similarity(emb_t.unsqueeze(0), target_emb_torch.unsqueeze(0))).item()
        return score, rendered

    for iteration in range(max_iterations):
        print(f"\n--- Refinement iteration {iteration + 1}/{max_iterations} ---")

        score, _ = _render_and_score(df)
        print(f"  Current full-result score: {score:.4f}")

        if score < best_score:
            best_score = score
            best_df = df.copy()

        if score < threshold:
            print(f"  Score below threshold — refinement complete.")
            break

        # Surrogate gradient per frame — use surrogate_note (in-distribution profile note)
        surrogate.eval()

        frame_grads = []
        for i in range(len(df)):
            frame_note = int(df["surrogate_note"].iloc[i]) if "surrogate_note" in df.columns else int(df["midi_note"].iloc[i])
            note_t = torch.full((1,), frame_note / 127.0, device=device)

            param_tensor = torch.zeros(1, len(param_cols), device=device)
            for j, col in enumerate(param_cols):
                param_tensor[0, j] = df[col].iloc[i]
            param_tensor.requires_grad_(True)

            opt = torch.optim.Adam([param_tensor], lr=5e-3)
            for _ in range(50):
                opt.zero_grad()
                clamped = param_tensor.clamp(0.0, 1.0)
                if pin_idx:
                    mask = torch.ones_like(clamped)
                    fixed = torch.zeros_like(clamped)
                    for idx, val in pin_idx.items():
                        mask[0, idx] = 0.0
                        fixed[0, idx] = val
                    clamped = clamped * mask + fixed
                pred = surrogate(clamped, note_t)
                loss = (1.0 - F.cosine_similarity(pred, target_emb_torch.unsqueeze(0), dim=-1)).mean()
                loss.backward()
                if param_tensor.grad is not None:
                    for idx in pin_idx:
                        param_tensor.grad[0, idx] = 0.0
                opt.step()

            new_vals = param_tensor.detach().clamp(0.0, 1.0).squeeze(0).cpu().numpy()
            for idx, val in pin_idx.items():
                new_vals[idx] = val
            frame_grads.append(new_vals)

        best_local_score = score
        best_local_df = df.copy()
        best_alpha = 0.0

        for alpha in [0.05, 0.1, 0.15, 0.2, 0.3]:
            trial_df = df.copy()
            for i in range(len(df)):
                for j, col in enumerate(param_cols):
                    if col in pinned_cols:
                        continue
                    old_val = trial_df[col].iloc[i]
                    new_val = frame_grads[i][j]
                    trial_df.at[i, col] = np.clip(old_val + (new_val - old_val) * alpha, 0.0, 1.0)

            trial_score, _ = _render_and_score(trial_df)
            print(f"    α={alpha:.2f} → full score: {trial_score:.4f}")

            if trial_score < best_local_score:
                best_local_score = trial_score
                best_local_df = trial_df.copy()
                best_alpha = alpha

        if best_alpha > 0:
            df = best_local_df
            print(f"  ✓ Best α={best_alpha:.2f}, new score: {best_local_score:.4f}")
        else:
            print(f"  No improvement found — stopping refinement.")
            break

    if best_score < float("inf"):
        best_df.to_parquet(run_dir / "stream_params.parquet")
        best_row = best_df.loc[best_df["score"].idxmin()]
        best_patch = {
            "target": str(target_wav),
            "note_regions": [
                {"onset_sec": r["onset_sec"], "offset_sec": r["offset_sec"],
                 "median_hz": r["median_hz"], "midi_note": r["midi_note"],
                 "surrogate_note": r["surrogate_note"]}
                for r in note_regions
            ],
            "pitch_hz": float(best_row["pitch_hz"]) if pd.notnull(best_row["pitch_hz"]) else None,
            "score": float(best_row["score"]),
            "refined_score": float(best_score),
            "params": {col: float(best_row[col]) for col in param_cols},
        }
        with open(run_dir / "best_patch.yaml", "w") as f:
            yaml.dump(best_patch, f, default_flow_style=False)

        print(f"\nRe-rendering best result (score={best_score:.4f})...")
        pitch_traj_refined = {
            "frames": [
                {"timestamp": float(row["timestamp"]), "pitch_bend": float(row.get("pitch_bend", 0.5))}
                for _, row in best_df.iterrows()
            ]
        }
        _render_stream(best_df, pitch_traj_refined, profile_path, run_dir, note_regions, audio_duration)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="Target audio file (.wav)")
    ap.add_argument("--surrogate", default=None, help="Surrogate model checkpoint")
    ap.add_argument("--profile", default=None, help="Profile YAML (auto-detect if None)")
    ap.add_argument("--out", default=str(_defs.S06_PATCHES_DIR), help="Output directory")
    ap.add_argument("--win-sec", type=float, default=0.1)
    ap.add_argument("--hop-sec", type=float, default=0.05)
    ap.add_argument("--n-starts", type=int, default=4)
    ap.add_argument("--grad-steps", type=int, default=50)
    ap.add_argument("--smooth-window", type=int, default=3)
    ap.add_argument("--device", default="cuda", help="torch device")
    ap.add_argument("--no-render", action="store_true", help="Skip DawDreamer render step")
    ap.add_argument("--refine-iterations", type=int, default=3)
    ap.add_argument("--refine-threshold", type=float, default=0.01)

    args = ap.parse_args()

    if args.surrogate is None:
        runs = sorted(_defs.S05_RUNS_DIR.glob("run_*")) if _defs.S05_RUNS_DIR.exists() else []
        if not runs:
            ap.error("No surrogate runs found; pass --surrogate explicitly.")
        args.surrogate = str(runs[-1] / "state_dict.pt")

    if args.profile is None:
        args.profile = str(Path(__file__).resolve().parent.parent / "s01_profiles" / "obxf.yaml")

    stream_invert(
        target_wav=Path(args.target),
        surrogate_checkpoint=Path(args.surrogate),
        profile_path=Path(args.profile),
        out_dir=Path(args.out),
        device=args.device,
        win_sec=args.win_sec,
        hop_sec=args.hop_sec,
        n_starts=args.n_starts,
        grad_steps=args.grad_steps,
        smooth_window=args.smooth_window,
        skip_render=args.no_render,
        refine_iterations=args.refine_iterations,
        refine_threshold=args.refine_threshold,
    )

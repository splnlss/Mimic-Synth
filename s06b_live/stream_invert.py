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
from datetime import datetime
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


def _detect_pitch_pyworld(
    audio: np.ndarray,
    sr: int,
    hop_ms: float = 5.0,
    f0_floor: float = 200.0,    # 200Hz minimum: crane scream range is 750-1110Hz
    f0_ceil: float = 2000.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract continuous F0 pitch trajectory using WORLD vocoder (pyworld).

    Returns (f0_hz, timestamps) arrays of shape (n_frames,) at hop_ms
    resolution, or None if pyworld is not available.

    pyworld advantages over autocorrelation / CREPE:
    - Detects short bursts (single 5ms frames at 1018Hz at 85ms, 1012Hz at 135ms)
    - Continuous voiced/unvoiced tracking — no confidence threshold needed
    - F0 = 0 for unvoiced frames (no NaN management needed)
    - Works at 48kHz natively — no resampling required
    - Fast (deterministic, no GPU)
    - Stonemask refinement significantly reduces octave errors

    f0_floor should be set above the lowest harmonic you want to capture.
    For bird calls in the 750-1110Hz range, 200Hz avoids sub-harmonic errors.
    """
    try:
        import pyworld as pw
    except ImportError:
        return None

    f0_raw, t = pw.dio(
        audio.astype(np.float64), int(sr),
        f0_floor=f0_floor, f0_ceil=f0_ceil,
        frame_period=hop_ms,
    )
    f0 = pw.stonemask(audio.astype(np.float64), f0_raw, t, int(sr))
    return f0.astype(np.float32), t.astype(np.float32)


def _make_pitch_fn(
    f0_hz: np.ndarray,
    t_array: np.ndarray,
) -> "Callable[[float], float | None]":
    """Build a t_sec → hz lookup from a pyworld F0 array.

    Returns None for unvoiced frames (F0 = 0).
    Uses nearest-neighbour lookup on the time array.
    """
    def pitch_fn(t_sec: float) -> float | None:
        idx = int(np.searchsorted(t_array, t_sec))
        idx = max(0, min(len(f0_hz) - 1, idx))
        hz = f0_hz[idx]
        return None if hz <= 0 else float(hz)

    return pitch_fn


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
    pitch_change_threshold_st: float = 1.0,
    min_note_ms: float = 20.0,
    analysis_win_ms: float = 20.0,
    analysis_hop_ms: float = 10.0,
    pitch_fn=None,   # Optional: callable(t_sec) → hz or None (from CREPE)
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
        t_sec = start / sr
        hz = pitch_fn(t_sec) if pitch_fn is not None else detect_pitch_autocorr(win, sr)
        frames.append({"t_sec": t_sec, "rms": rms, "hz": hz})

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


def _write_pitch_bend_midi(
    note_regions: list[dict],
    fine_pitch_frames: list[dict],
    output_path,
    pb_range_st: float = 3.0,
) -> float:
    """Write a MIDI file with note-on/off events and fine pitch bend automation.

    Pitch bend affects ALL oscillators simultaneously (unlike Osc 1 Pitch which
    only moves Osc 1). Both Osc 1 and Osc 2 track the source pitch together.

    Calibration (measured against OB-Xf):
        Pitch Bend Up/Down param = 0.042 → ±2.06 semitones at full bend (±8192).
        pb_range_param = 0.042 * (pb_range_st / 2.06)

    Args:
        note_regions: from detect_note_regions.
        fine_pitch_frames: from _compute_fine_pitch_trajectory (pyworld at 5ms).
        output_path: where to write the .mid file.
        pb_range_st: total pitch bend range in semitones. ±3st covers the crane
            scream's within-region drop from 1109Hz to ~950Hz (−2.66st).

    Returns:
        pb_range_param — the VST value to set for Pitch Bend Up and Pitch Bend Down.
    """
    try:
        import mido
    except ImportError:
        return None

    # Calibrated from OB-Xf measurement: 0.042 → 2.06 st at full ±8192 deflection.
    PB_CALIB = 2.06 / 0.042       # semitones per param unit
    pb_range_param = pb_range_st / PB_CALIB

    ticks_per_beat = 960
    tempo = 500000                 # 120 BPM

    def sec_to_tick(t_sec: float) -> int:
        return int(round(t_sec * ticks_per_beat * 1_000_000 / tempo))

    # Build a flat event list, then sort by time for MIDI delta encoding
    events: list[tuple] = []      # (time_sec, type, *args)

    for i, r in enumerate(note_regions):
        note_on_t = r["onset_sec"]
        if i < len(note_regions) - 1:
            note_off_t = note_regions[i + 1]["onset_sec"]
        else:
            note_off_t = r["offset_sec"]
        dur = max(0.001, note_off_t - note_on_t)
        events.append((note_on_t,  "note_on",  r["midi_note"], 100))
        events.append((note_on_t + dur, "note_off", r["midi_note"], 0))

    # Build a fast lookup: timestamp → base MIDI note frequency
    note_hz_by_region: dict[tuple, float] = {}
    for r in note_regions:
        base_hz = 440.0 * (2 ** ((r["midi_note"] - 69) / 12.0))
        note_hz_by_region[(r["onset_sec"], r["offset_sec"])] = base_hz

    def base_hz_at(t: float) -> float | None:
        for r in note_regions:
            if r["onset_sec"] <= t <= r["offset_sec"]:
                return 440.0 * (2 ** ((r["midi_note"] - 69) / 12.0))
        return None

    # Emit pitch bend for every frame that has a detected pitch (pyworld or
    # autocorrelation fallback). Frames without pitch detection are simply
    # skipped — the synth holds the last bend value automatically.
    # No extrapolation needed since the hybrid tracker extends into the tail.
    for frame in fine_pitch_frames:
        t = frame["timestamp"]
        pitch_hz = frame.get("pitch_hz")
        if pitch_hz is None:
            continue

        base = base_hz_at(t)
        if base is None:
            continue

        offset_st = 12.0 * np.log2(pitch_hz / base)
        pb_value = int(offset_st / pb_range_st * 8192)
        events.append((t, "pitchbend", max(-8192, min(8191, pb_value))))

    events.sort(key=lambda e: e[0])

    mid = mido.MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    current_tick = 0
    for ev in events:
        t_sec = ev[0]
        tick = sec_to_tick(t_sec)
        dt = max(0, tick - current_tick)
        if ev[1] == "note_on":
            track.append(mido.Message("note_on", channel=0, note=ev[2], velocity=ev[3], time=dt))
            current_tick = tick
        elif ev[1] == "note_off":
            track.append(mido.Message("note_off", channel=0, note=ev[2], velocity=ev[3], time=dt))
            current_tick = tick
        elif ev[1] == "pitchbend":
            track.append(mido.Message("pitchwheel", channel=0, pitch=ev[2], time=dt))
            current_tick = tick

    mid.save(str(output_path))
    return pb_range_param


def _compute_fine_pitch_trajectory(
    audio: np.ndarray,
    sr: int,
    note_regions: list[dict],
    win_ms: float = 20.0,
    hop_ms: float = 10.0,
    pitch_fn=None,   # Optional: precomputed CREPE pitch callable
) -> list[dict]:
    """High-resolution pitch tracking at hop_ms intervals.

    Runs autocorrelation pitch detection at much finer resolution than the
    surrogate inversion (which runs at 50ms hop). Returns per-frame pitch
    and the corresponding Osc 1 Pitch automation value.

    The returned osc1_pitch values are applied as VST automation in the render
    to reproduce fine pitch glides (within-region bends, release drops, etc.)
    that coarser tracking and fixed MIDI notes cannot capture.

    Pitch range: OB-Xf Osc 1 Pitch is ±24 semitones (0.5=center). A glide
    from 1109Hz to 950Hz on MIDI note 85 corresponds to:
        offset = -2.66 semitones → osc1_pitch ≈ 0.5 - 2.66/48 ≈ 0.444
    """
    win_n = int(win_ms / 1000 * sr)
    hop_n = int(hop_ms / 1000 * sr)
    frames: list[dict] = []

    for start in range(0, len(audio) - win_n + 1, hop_n):
        t = (start + win_n // 2) / sr
        r = _region_for_frame(t, note_regions)

        # Primary: pyworld F0 (high accuracy for voiced portions)
        if pitch_fn is not None:
            pitch_hz = pitch_fn(t)
        else:
            pitch_hz = None

        # Fallback: autocorrelation for unvoiced frames WITHIN a note region.
        # pyworld loses tracking when the note amplitude fades, but autocorrelation
        # still detects the pitch (e.g., 823Hz at t=0.58s in the crane scream tail).
        # Only apply within regions to avoid tracking noise in silent gaps.
        if pitch_hz is None and r is not None:
            win = audio[start : start + win_n]
            pitch_hz = detect_pitch_autocorr(win, sr)
            # Reject sub-harmonic or harmonic overtone errors: the true pitch
            # should stay within ±1 octave of the region's MIDI note.
            if pitch_hz is not None:
                expected_hz = 440.0 * (2 ** ((r["midi_note"] - 69) / 12.0))
                if not (expected_hz * 0.50 < pitch_hz < expected_hz * 2.0):
                    pitch_hz = None

        base_note = r["midi_note"] if r else (note_regions[0]["midi_note"] if note_regions else 60)
        osc1 = pitch_hz_to_osc1(pitch_hz, base_note) if pitch_hz else 0.5
        frames.append({
            "timestamp": float(t),
            "pitch_hz": float(pitch_hz) if pitch_hz else None,
            "osc1_pitch": float(osc1),
        })

    return frames


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

# OB-Xf Osc 1 Pitch is a ±24-semitone transpose: 0.0 = -24st, 0.5 = center, 1.0 = +24st.
# (Confirmed: surrogate pushing to 1.0 shifts MIDI 85 / 1109Hz to ~4.4kHz = +24 semitones.)
OSC1_PITCH_RANGE_SEMITONES = 24.0
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
    hill_iterations: int = 2,
    hill_offsets: tuple[float, ...] = (-0.15, -0.05, 0.05, 0.15),
    run_cmaes: bool = False,
    cmaes_mode: str = "hybrid",     # "global" | "per-region" | "hybrid"
    cmaes_popsize: int = 16,
    cmaes_maxiter: int = 20,
    cmaes_sigma0: float = 0.08,
    # Fine pitch tracking — applied as Osc 1 Pitch automation in render
    pitch_win_ms: float = 20.0,     # analysis window for fine pitch (ms)
    pitch_hop_ms: float = 10.0,     # analysis hop for fine pitch (ms)
    # Note detection sensitivity
    min_note_ms: float = 20.0,      # minimum note duration for region detection
    pitch_threshold_st: float = 1.0, # semitone jump triggering a new note region
) -> dict:
    # ── Mono enforcement ─────────────────────────────────────────────────────
    # All target files must be mono before analysis. See s07_refine/mono_utils.py.
    from s07_refine.mono_utils import ensure_mono
    _, _, target_wav = ensure_mono(target_wav)

    audio, sr = sf.read(str(target_wav), dtype="float32")
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

    # ── pyworld F0 pitch extraction (full audio, deterministic) ──────────
    # pyworld at 5ms resolution detects short bursts (single-frame voiced
    # events at e.g. 85ms/135ms), continuous pitch glides, and voiced/unvoiced
    # transitions without confidence thresholds or octave ambiguity.
    # Falls back to per-frame autocorrelation if pyworld is unavailable.
    pyworld_result = _detect_pitch_pyworld(audio, sr, hop_ms=pitch_hop_ms)
    if pyworld_result is not None:
        pw_f0, pw_t = pyworld_result
        pitch_fn = _make_pitch_fn(pw_f0, pw_t)
        n_voiced = int(np.sum(pw_f0 > 0))
        print(f"pyworld F0: {len(pw_f0)} frames @ {pitch_hop_ms:.0f}ms, "
              f"{n_voiced} voiced ({100*n_voiced//max(1,len(pw_f0))}%)")
    else:
        pitch_fn = None
        print("pyworld not available — using autocorrelation pitch detection")

    # ── Fine-grained note region detection ───────────────────────────────
    note_regions = detect_note_regions(
        audio, sr,
        profile_notes=profile_notes,
        pitch_change_threshold_st=pitch_threshold_st,
        min_note_ms=min_note_ms,
        pitch_fn=pitch_fn,
    )

    print(f"Detected {len(note_regions)} note region(s) "
          f"(threshold={pitch_threshold_st:.1f}st, min={min_note_ms:.0f}ms):")
    for i, r in enumerate(note_regions):
        hz_str = f"{r['median_hz']:.0f}Hz" if r['median_hz'] else "?Hz"
        print(f"  Region {i}: {r['onset_sec']:.3f}s–{r['offset_sec']:.3f}s  "
              f"pitch={hz_str}  MIDI={r['midi_note']}  "
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

    # ── Fine pitch tracking (high-resolution, applied to Osc 1 Pitch in render) ──
    # Runs at pitch_hop_ms resolution (default 10ms) to capture sub-50ms pitch
    # glides, bends, and burst transients that the 50ms surrogate analysis misses.
    print(f"Computing fine pitch trajectory "
          f"({pitch_win_ms:.0f}ms win / {pitch_hop_ms:.0f}ms hop"
          f"{', pyworld' if pitch_fn is not None else ', autocorr'})...")
    fine_pitch_frames = _compute_fine_pitch_trajectory(
        audio, sr, note_regions, win_ms=pitch_win_ms, hop_ms=pitch_hop_ms,
        pitch_fn=pitch_fn,
    )
    print(f"  → {len(fine_pitch_frames)} frames "
          f"(vs {(len(audio)//int(hop_sec*sr)+1)} surrogate frames)")

    # ── Coarse frame analysis (embedding + per-frame pitch) ───────────────
    win_samples = int(win_sec * sr)
    hop_samples = int(hop_sec * sr)
    n_frames = max(0, (len(audio) - win_samples) // hop_samples + 1)

    pitch_hz_list = []
    emb_list = []

    for start in tqdm(range(0, len(audio) - win_samples + 1, hop_samples), desc="Analyzing"):
        t = (start + win_samples // 2) / sr
        window = audio[start: start + win_samples]
        if pitch_fn is not None:
            pitch_hz_list.append(pitch_fn(t))
        else:
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
    run_dir = out_dir / target_wav.stem / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(run_dir / "stream_params.parquet")

    # Save fine pitch trajectory — loaded by _render_stream for Osc 1 Pitch automation.
    with open(run_dir / "fine_pitch_trajectory.yaml", "w") as f:
        yaml.dump({
            "win_ms": pitch_win_ms,
            "hop_ms": pitch_hop_ms,
            "osc1_pitch_range_semitones": OSC1_PITCH_RANGE_SEMITONES,
            "frames": fine_pitch_frames,
        }, f, default_flow_style=False)

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

        if hill_iterations > 0:
            _hill_climb_step(
                target_wav=target_wav,
                profile_path=profile_path,
                run_dir=run_dir,
                note_regions=note_regions,
                param_cols=param_cols,
                pinned_cols={param_cols[i] for i in pin_idx},
                embedder=embedder,
                device=device,
                audio_duration=audio_duration,
                n_passes=hill_iterations,
                offsets=hill_offsets,
            )

        if run_cmaes:
            _cmaes_step(
                target_wav=target_wav,
                profile_path=profile_path,
                run_dir=run_dir,
                note_regions=note_regions,
                param_cols=param_cols,
                pinned_cols={param_cols[i] for i in pin_idx},
                embedder=embedder,
                device=device,
                audio_duration=audio_duration,
                mode=cmaes_mode,
                sigma0=cmaes_sigma0,
                popsize=cmaes_popsize,
                maxiter=cmaes_maxiter,
                audio=audio,
                sr=sr,
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

    # ── MIDI note events + pitch bend via MIDI file ───────────────────────
    # If a fine pitch trajectory exists, generate a MIDI file containing
    # both note-on/off events and fine pitch bend messages (5ms resolution).
    # This gives true pitch bend affecting ALL oscillators simultaneously,
    # unlike the Osc 1 Pitch VST automation which only moves Osc 1.
    #
    # Pitch bend range calibration (measured against OB-Xf):
    #   Pitch Bend Up/Down = 0.042 → ±2.06 semitones at full bend (±8192 MIDI).
    fine_traj_path = out_dir / "fine_pitch_trajectory.yaml"
    midi_generated = False

    if fine_traj_path.exists():
        with open(fine_traj_path) as _fh:
            fine_data = yaml.safe_load(_fh)
        fine_frames = fine_data.get("frames", [])

        if fine_frames:
            midi_path = out_dir / "pitch_bend.mid"
            pb_range_st = 6.0     # ±6 semitones: crane scream drops ~5st
            pb_range_param = _write_pitch_bend_midi(
                note_regions, fine_frames, midi_path, pb_range_st=pb_range_st,
            )

            if pb_range_param is not None:
                # Set the VST pitch bend range before loading the MIDI file
                if "Pitch Bend Up" in param_name_to_index:
                    plugin.set_parameter(param_name_to_index["Pitch Bend Up"], pb_range_param)
                if "Pitch Bend Down" in param_name_to_index:
                    plugin.set_parameter(param_name_to_index["Pitch Bend Down"], pb_range_param)

                plugin.clear_midi()
                plugin.load_midi(str(midi_path))
                midi_generated = True

                voiced = sum(1 for f in fine_frames if f.get("pitch_hz"))
                print(f"  MIDI pitch bend: {len(fine_frames)} frames, "
                      f"{voiced} voiced, ±{pb_range_st:.1f}st range "
                      f"(pb_param={pb_range_param:.3f})")

    if not midi_generated:
        # Fallback: add_midi_note without pitch bend
        for i, r in enumerate(note_regions):
            note_on = r["onset_sec"]
            if i < len(note_regions) - 1:
                note_off = note_regions[i + 1]["onset_sec"]
            else:
                note_off = r["offset_sec"]
            note_dur = max(0.0, note_off - note_on)
            if note_dur > 0:
                plugin.add_midi_note(r["midi_note"], 100, note_on, note_dur)

    # Parameter automation (timbre params — Osc 1 Pitch stays at 0.5 since
    # pitch is now handled by MIDI pitch bend above)
    for col in param_cols:
        p_name = col.removeprefix("p_")
        if p_name in param_name_to_index:
            p_idx = param_name_to_index[p_name]
            data = np.column_stack((timestamps, df[col].values))
            plugin.set_automation(p_idx, data)

    n_regions = len(note_regions)
    notes_str = ", ".join(
        f"MIDI {r['midi_note']} ({r['median_hz']:.0f}Hz)" if r['median_hz'] else f"MIDI {r['midi_note']}"
        for r in note_regions
    )
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

    from s07_refine.audio_compare import compute_mrstft_features, compute_ap_features, score_audio_composite
    target_mrstft = compute_mrstft_features(target_audio)
    target_ap = compute_ap_features(target_audio, sr)

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

        score = score_audio_composite(
            rendered, sr_profile, target_emb_torch, embedder, device,
            target_mrstft, target_ap,
        )
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
                {"timestamp": float(row["timestamp"]),
                 "pitch_bend": float(row.get("pitch_bend", 0.5)),
                 "osc1_pitch": float(row.get("osc1_pitch", 0.5))}
                for _, row in best_df.iterrows()
            ]
        }
        _render_stream(best_df, pitch_traj_refined, profile_path, run_dir, note_regions, audio_duration)


# ── Hill-climbing refinement on real VST renders (s07 strategy 1) ───────────

def _hill_climb_step(
    target_wav: Path,
    profile_path: Path,
    run_dir: Path,
    note_regions: list,
    param_cols: list[str],
    pinned_cols: set[str],
    embedder,
    device: str,
    audio_duration: float,
    n_passes: int,
    offsets: tuple[float, ...],
):
    """Run s07 hill-climbing on the post-α-refinement parquet, then re-render.

    Loads `stream_params.parquet`, embeds the target, hands off to
    `s07_refine.vst_hill_climb.hill_climb`, persists the improved trajectory,
    and re-renders. Pinned params are left untouched (the hill-climber skips
    them via `pinned_cols`).
    """
    from s07_refine.vst_hill_climb import hill_climb
    from s07_refine.audio_compare import compute_mrstft_features, compute_ap_features

    target_audio, sr = sf.read(str(target_wav), dtype="float32")
    if target_audio.ndim == 2:
        target_audio = target_audio.mean(axis=1)

    target_emb = embedder.encodec_embed(target_audio, sr, pool="mean")
    target_emb_t = torch.tensor(target_emb, dtype=torch.float32, device=device)
    target_mrstft = compute_mrstft_features(target_audio)
    target_ap = compute_ap_features(target_audio, sr)

    active = "EnCodec+MRSTFT" + ("+AP" if target_ap is not None else "")
    df = pd.read_parquet(run_dir / "stream_params.parquet")
    timestamps = df["timestamp"].values
    hop_sec = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.05
    total_sec = audio_duration if audio_duration else timestamps[-1] + hop_sec

    print(f"\n=== Hill-climb refinement ({n_passes} pass(es), offsets={list(offsets)}) ===")
    print(f"    Scoring: {active} composite (renormalised weights)")

    refined_df, final_score, change_log = hill_climb(
        df=df,
        note_regions=note_regions,
        param_cols=param_cols,
        pinned_cols=pinned_cols,
        profile_path=profile_path,
        total_sec=total_sec,
        target_emb_t=target_emb_t,
        embedder=embedder,
        device=device,
        offsets=offsets,
        n_passes=n_passes,
        target_mrstft=target_mrstft,
        target_ap=target_ap,
    )

    # Persist updated trajectory + best patch + change log.
    refined_df.to_parquet(run_dir / "stream_params.parquet")

    best_row = refined_df.loc[refined_df["score"].idxmin()]
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
        "hill_climb_score": float(final_score),
        "params": {col: float(best_row[col]) for col in param_cols},
    }
    with open(run_dir / "best_patch.yaml", "w") as f:
        yaml.dump(best_patch, f, default_flow_style=False)

    with open(run_dir / "hill_climb_log.yaml", "w") as f:
        yaml.dump(
            {"final_score": float(final_score), "moves": change_log},
            f, default_flow_style=False,
        )

    pitch_traj = {
        "frames": [
            {"timestamp": float(row["timestamp"]),
             "pitch_bend": float(row.get("pitch_bend", 0.5)),
             "osc1_pitch": float(row.get("osc1_pitch", 0.5))}
            for _, row in refined_df.iterrows()
        ]
    }
    print(f"\nRe-rendering hill-climbed result (score={final_score:.4f})...")
    _render_stream(refined_df, pitch_traj, profile_path, run_dir, note_regions, audio_duration)


# ── CMA-ES refinement on real VST renders (s07 strategy 2) ──────────────────

def _cmaes_step(
    target_wav: Path,
    profile_path: Path,
    run_dir: Path,
    note_regions: list,
    param_cols: list[str],
    pinned_cols: set[str],
    embedder,
    device: str,
    audio_duration: float,
    mode: str = "hybrid",
    sigma0: float = 0.08,
    popsize: int = 16,
    maxiter: int = 20,
    audio: np.ndarray | None = None,
    sr: int = 48000,
):
    """Run s07 CMA-ES on the post-hill-climb parquet, then re-render.

    Loads `stream_params.parquet`, runs target analysis to build a smart x0
    (blended with the current hill-climb result), then calls
    `s07_refine.vst_cmaes.cmaes_refine`. Saves updated parquet/yaml/wav and a
    `cmaes_log.yaml` for diagnostics.
    """
    from s07_refine.vst_cmaes import cmaes_refine
    from s07_refine.target_analysis import analyze_target, suggest_x0, print_analysis

    # Load target audio for analysis (may already be loaded by caller)
    if audio is None:
        target_audio, sr = sf.read(str(target_wav), dtype="float32")
        if target_audio.ndim == 2:
            target_audio = target_audio.mean(axis=1)
    else:
        target_audio = audio

    # Target analysis: answer the 5 design questions
    print("\n=== Target Analysis (s07 CMA-ES warm-start) ===")
    analysis = analyze_target(target_audio, sr, note_regions)
    print_analysis(analysis)

    df = pd.read_parquet(run_dir / "stream_params.parquet")

    # Apply target-analysis warm-start to the 15 surrogate params only.
    # The 7 extra params are initialised inside cmaes_refine from profile reset values.
    current_x0 = np.array([float(df[c].median()) for c in param_cols])
    x0_smart = suggest_x0(analysis, param_cols, pinned_cols, current_x0)
    for i, col in enumerate(param_cols):
        if col not in pinned_cols:
            df[col] = np.clip(df[col] + (x0_smart[i] - current_x0[i]) * 0.5, 0.0, 1.0)

    timestamps = df["timestamp"].values
    hop_sec = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.05
    total_sec = audio_duration if audio_duration else timestamps[-1] + hop_sec

    print(f"\n=== CMA-ES refinement (mode={mode}, popsize={popsize}, maxiter={maxiter}) ===")

    result = cmaes_refine(
        df=df,
        note_regions=note_regions,
        param_cols=param_cols,
        pinned_cols=pinned_cols,
        profile_path=profile_path,
        total_sec=total_sec,
        target_wav=target_wav,
        embedder=embedder,
        device=device,
        mode=mode,
        sigma0=sigma0,
        popsize=popsize,
        maxiter=maxiter,
    )

    # All param columns = surrogate 15 + extra 7 (now in result.best_df)
    all_param_cols = [c for c in result.best_df.columns if c.startswith("p_")]

    # Persist
    result.best_df.to_parquet(run_dir / "stream_params.parquet")
    best_row = result.best_df.loc[result.best_df["score"].idxmin()]
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
        "cmaes_global_score": float(result.global_score),
        "cmaes_region_scores": [float(s) for s in result.region_scores],
        "osc_config": result.osc_config,
        "extra_params": result.extra_param_names,
        "params": {col: float(best_row[col]) for col in all_param_cols
                   if col in best_row.index},
    }
    with open(run_dir / "best_patch.yaml", "w") as f:
        yaml.dump(best_patch, f, default_flow_style=False)

    cmaes_log = {
        "mode": result.mode,
        "global_score": float(result.global_score),
        "region_scores": [float(s) for s in result.region_scores],
        "osc_config": result.osc_config,
        "extra_params_added": result.extra_param_names,
        "n_renders": result.n_renders,
        "restarts_used": result.restarts_used,
        "param_deltas": {k: float(v) for k, v in result.param_deltas.items()},
        "iterations": result.iteration_log,
    }
    with open(run_dir / "cmaes_log.yaml", "w") as f:
        yaml.dump(cmaes_log, f, default_flow_style=False)

    pitch_traj = {
        "frames": [
            {"timestamp": float(row["timestamp"]),
             "pitch_bend": float(row.get("pitch_bend", 0.5)),
             "osc1_pitch": float(row.get("osc1_pitch", 0.5))}
            for _, row in result.best_df.iterrows()
        ]
    }
    print(f"\nRe-rendering CMA-ES result (global={result.global_score:.4f}, "
          f"regions={[f'{s:.4f}' for s in result.region_scores]}, "
          f"config='{result.osc_config}')...")
    _render_stream(result.best_df, pitch_traj, profile_path, run_dir,
                   note_regions, audio_duration)


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
    ap.add_argument("--refine-iterations", type=int, default=3,
                    help="α-search refinement iterations (surrogate-gradient driven)")
    ap.add_argument("--refine-threshold", type=float, default=0.01)
    ap.add_argument("--hill-iterations", type=int, default=2,
                    help="s07 hill-climb passes on real VST renders (0 to disable)")
    ap.add_argument("--hill-offsets", type=str, default="-0.15,-0.05,0.05,0.15",
                    help="comma-separated list of per-param offsets to try each pass")
    ap.add_argument("--cmaes", action="store_true",
                    help="run s07 CMA-ES after hill-climb (requires: pip install cma)")
    ap.add_argument("--cmaes-mode", default="hybrid",
                    choices=["global", "per-region", "hybrid"],
                    help="global=fast bypass (single patch, ~0.029); "
                         "per-region=independent per note region; "
                         "hybrid=global first then conditional per-region (default)")
    ap.add_argument("--cmaes-popsize", type=int, default=16)
    ap.add_argument("--cmaes-maxiter", type=int, default=20)
    ap.add_argument("--cmaes-sigma0", type=float, default=0.08,
                    help="CMA-ES step size (0.05=refine, 0.10-0.15=explore)")
    ap.add_argument("--pitch-win-ms", type=float, default=20.0,
                    help="Window size for fine pitch tracking (ms, default 20)")
    ap.add_argument("--pitch-hop-ms", type=float, default=10.0,
                    help="Hop size for fine pitch tracking (ms, default 10)")
    ap.add_argument("--min-note-ms", type=float, default=20.0,
                    help="Minimum note duration for region detection (ms, default 20)")
    ap.add_argument("--pitch-threshold-st", type=float, default=1.0,
                    help="Semitone jump triggering a new note region (default 1.0)")

    args = ap.parse_args()

    hill_offsets = tuple(float(x) for x in args.hill_offsets.split(",") if x.strip())

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
        hill_iterations=args.hill_iterations,
        hill_offsets=hill_offsets,
        run_cmaes=args.cmaes,
        cmaes_mode=args.cmaes_mode,
        cmaes_popsize=args.cmaes_popsize,
        cmaes_maxiter=args.cmaes_maxiter,
        cmaes_sigma0=args.cmaes_sigma0,
        pitch_win_ms=args.pitch_win_ms,
        pitch_hop_ms=args.pitch_hop_ms,
        min_note_ms=args.min_note_ms,
        pitch_threshold_st=args.pitch_threshold_st,
    )

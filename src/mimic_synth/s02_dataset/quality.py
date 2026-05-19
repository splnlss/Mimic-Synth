"""
Per-capture quality gates for Bucket 3 dataset generation.

Each check takes a mono float32 audio array + metadata and returns a
boolean (for pass/fail) or a small dict of stats. Keep these fast — they
run on every single capture.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# Defaults tuned for 48 kHz mono renders.
SILENCE_PEAK_THRESHOLD = 1e-4
CLIP_PEAK_THRESHOLD = 0.99
CLIP_MIN_SAMPLES = 5
STUCK_TAIL_MS = 50          # how much of the end of the capture to inspect
STUCK_RMS_RATIO_DB = 6.0    # tail RMS within 6 dB of sustain RMS → stuck
BLEED_WINDOW_MS = 20        # leading window that must be near silent
BLEED_PEAK_THRESHOLD = 0.01


@dataclass
class CaptureStats:
    rms: float
    peak: float
    silent: bool
    clipped: bool
    stuck: bool
    prev_bleed: bool

    def is_valid(self) -> bool:
        return not (self.silent or self.clipped or self.stuck or self.prev_bleed)


def is_silent(audio: np.ndarray, threshold: float = SILENCE_PEAK_THRESHOLD) -> bool:
    return float(np.max(np.abs(audio))) < threshold


def is_clipped(
    audio: np.ndarray,
    threshold: float = CLIP_PEAK_THRESHOLD,
    min_samples: int = CLIP_MIN_SAMPLES,
) -> bool:
    return int(np.sum(np.abs(audio) > threshold)) >= min_samples


def _rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def is_stuck_note(
    audio: np.ndarray,
    sample_rate: int,
    hold_sec: float,
    release_sec: float,
    pre_roll_sec: float = 0.0,
    tail_ms: float = STUCK_TAIL_MS,
    rms_ratio_db: float = STUCK_RMS_RATIO_DB,
) -> bool:
    """After pre_roll+hold+release has elapsed, the last `tail_ms` of the capture
    should be near the noise floor. If its RMS is within `rms_ratio_db` of the
    sustain-window RMS, the note never released."""
    total_samples = len(audio)
    expected_end = int((pre_roll_sec + hold_sec + release_sec) * sample_rate)
    tail_samples = int((tail_ms / 1000.0) * sample_rate)

    if total_samples <= expected_end or tail_samples <= 0:
        # Capture is too short to evaluate; don't flag.
        return False

    tail = audio[-tail_samples:]
    # Sustain window: the last ~100 ms before the note-off.
    note_off = int((pre_roll_sec + hold_sec) * sample_rate)
    sustain_end = note_off
    sustain_start = max(0, sustain_end - int(0.1 * sample_rate))
    sustain = audio[sustain_start:sustain_end]

    tail_rms = _rms(tail)
    sustain_rms = _rms(sustain)
    if sustain_rms <= 1e-9:
        return False                 # nothing playing; not a stuck-note case

    # tail_rms / sustain_rms in dB; if close, something's still ringing.
    ratio_db = 20.0 * np.log10(tail_rms / sustain_rms + 1e-12)
    return ratio_db > -rms_ratio_db


def has_prev_note_bleed(
    audio: np.ndarray,
    sample_rate: int,
    window_ms: float = BLEED_WINDOW_MS,
    threshold: float = BLEED_PEAK_THRESHOLD,
    self_noise: float = 0.0,
) -> bool:
    """First `window_ms` of the capture — before the probe note's own attack
    can physically ramp in — should be near silent.

    If `self_noise` is provided (from a silent render of the patch with no
    MIDI), it's used as the effective threshold floor. Patches that
    self-oscillate or have LFO artifacts will produce energy in the
    pre-roll window that isn't bleed from a previous note."""
    window_samples = int((window_ms / 1000.0) * sample_rate)
    if window_samples <= 0 or len(audio) <= window_samples:
        return False
    leading = audio[:window_samples]
    effective_threshold = max(threshold, self_noise * 2.0)
    return float(np.max(np.abs(leading))) >= effective_threshold


def analyse(
    audio: np.ndarray,
    sample_rate: int,
    hold_sec: float,
    release_sec: float,
    pre_roll_sec: float = 0.0,
    self_noise: float = 0.0,
) -> CaptureStats:
    """Run every check and return a CaptureStats."""
    return CaptureStats(
        rms=_rms(audio),
        peak=float(np.max(np.abs(audio))),
        silent=is_silent(audio),
        clipped=is_clipped(audio),
        stuck=is_stuck_note(audio, sample_rate, hold_sec, release_sec, pre_roll_sec),
        prev_bleed=has_prev_note_bleed(audio, sample_rate, self_noise=self_noise),
    )

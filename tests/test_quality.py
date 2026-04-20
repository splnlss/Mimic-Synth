"""Unit tests for s03_dataset.quality — synthetic signals, no DawDreamer."""
import numpy as np
import pytest

from s03_dataset.quality import (
    is_silent, is_clipped, is_stuck_note, has_prev_note_bleed, analyse,
)

SR = 44100


def _tone(freq, dur_sec, amp=0.3, sr=SR):
    t = np.arange(int(dur_sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _env_tone(hold_sec, release_sec, amp=0.3, tail_sec=0.1, tail_amp=0.0, sr=SR):
    """Concatenate: hold tone + linear release ramp + tail at `tail_amp`."""
    hold = _tone(220, hold_sec, amp=amp, sr=sr)
    rel_len = int(release_sec * sr)
    env = np.linspace(1.0, 0.0, rel_len, dtype=np.float32)
    rel = _tone(220, release_sec, amp=amp, sr=sr)[:rel_len] * env
    tail = _tone(220, tail_sec, amp=tail_amp, sr=sr)
    return np.concatenate([hold, rel, tail])


class TestSilence:
    def test_silent_zeros(self):
        assert is_silent(np.zeros(1000, dtype=np.float32))

    def test_not_silent(self):
        assert not is_silent(_tone(440, 0.1))

    def test_borderline(self):
        x = np.full(1000, 5e-5, dtype=np.float32)
        assert is_silent(x)


class TestClipping:
    def test_not_clipped(self):
        assert not is_clipped(_tone(440, 0.1, amp=0.5))

    def test_clipped(self):
        x = _tone(440, 0.1, amp=1.0).copy()
        x[:100] = 1.0
        assert is_clipped(x)

    def test_below_min_samples(self):
        x = _tone(440, 0.1, amp=0.5).copy()
        x[:3] = 1.0
        assert not is_clipped(x, min_samples=5)


class TestStuckNote:
    def test_clean_release_not_stuck(self):
        x = _env_tone(hold_sec=0.5, release_sec=0.2, tail_sec=0.1, tail_amp=0.0)
        assert not is_stuck_note(x, SR, hold_sec=0.5, release_sec=0.2)

    def test_sustained_tail_is_stuck(self):
        x = _env_tone(hold_sec=0.5, release_sec=0.2, tail_sec=0.1, tail_amp=0.3)
        assert is_stuck_note(x, SR, hold_sec=0.5, release_sec=0.2)

    def test_silent_sustain_not_flagged(self):
        x = np.zeros(int(SR * 0.9), dtype=np.float32)
        assert not is_stuck_note(x, SR, hold_sec=0.5, release_sec=0.2)

    def test_too_short_not_flagged(self):
        x = _tone(220, 0.1)
        assert not is_stuck_note(x, SR, hold_sec=0.5, release_sec=0.2)


class TestPrevBleed:
    def test_silent_leading_ok(self):
        leading = np.zeros(int(SR * 0.05), dtype=np.float32)
        body = _tone(440, 0.5)
        assert not has_prev_note_bleed(np.concatenate([leading, body]), SR)

    def test_loud_leading_flagged(self):
        leading = _tone(440, 0.05, amp=0.5)
        body = _tone(440, 0.5)
        assert has_prev_note_bleed(np.concatenate([leading, body]), SR)

    def test_short_audio_not_flagged(self):
        assert not has_prev_note_bleed(np.zeros(10, dtype=np.float32), SR)


class TestAnalyse:
    def test_clean_capture_valid(self):
        x = _env_tone(hold_sec=0.5, release_sec=0.2, tail_amp=0.0)
        x = np.concatenate([np.zeros(int(SR * 0.03), dtype=np.float32), x])
        stats = analyse(x, SR, hold_sec=0.5, release_sec=0.2)
        assert stats.is_valid()
        assert stats.peak > 0

    def test_silent_capture_invalid(self):
        stats = analyse(np.zeros(SR, dtype=np.float32), SR, 0.5, 0.2)
        assert stats.silent and not stats.is_valid()

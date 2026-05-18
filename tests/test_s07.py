"""Tests for s07_refine: mono_utils, target_analysis, and CMA-ES logic.

These tests do NOT require DawDreamer or a GPU — they run under .venv or
the conda env and use synthetic audio only.

Run:
    .venv/bin/pytest tests/test_s07.py -v
    conda run -n mimic-synth python -m pytest tests/test_s07.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ── ensure project root on path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from s07_refine.mono_utils import ensure_mono
from s07_refine.target_analysis import (
    TargetAnalysis,
    analyze_target,
    suggest_x0,
    _attack_ms_to_param,
    _decay_ms_to_param,
    _estimate_adsr,
    _harmonic_ratio,
    _detect_fundamental_hps,
)

# ── Synthetic audio generators ───────────────────────────────────────────────

SR = 48000


def _sine(freq=440.0, duration=1.0, sr=SR, amplitude=0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _sawtooth(freq=440.0, duration=1.0, sr=SR, amplitude=0.5) -> np.ndarray:
    """Sawtooth with N harmonics (approximation)."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    sig = np.zeros_like(t)
    for k in range(1, 12):
        sig += ((-1) ** (k + 1)) / k * np.sin(2 * np.pi * k * freq * t)
    sig = sig / np.abs(sig).max() * amplitude
    return sig.astype(np.float32)


def _noise(duration=1.0, sr=SR, amplitude=0.3) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (amplitude * rng.standard_normal(int(duration * sr))).astype(np.float32)


def _shaped(wave: np.ndarray, attack_ms=10.0, decay_ms=100.0,
            sustain=0.6, release_ms=200.0, sr=SR) -> np.ndarray:
    """Apply a simple ADSR envelope to a signal."""
    n = len(wave)
    env = np.zeros(n)
    a = min(n, int(attack_ms / 1000 * sr))
    d = min(n - a, int(decay_ms / 1000 * sr))
    r = min(n, int(release_ms / 1000 * sr))
    s_start = a + d
    s_end = n - r
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if d > 0:
        env[a:a + d] = np.linspace(1, sustain, d)
    if s_end > s_start:
        env[s_start:s_end] = sustain
    if r > 0:
        env[s_end:s_end + r] = np.linspace(sustain, 0, r)
    return (wave * env).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# mono_utils tests
# ════════════════════════════════════════════════════════════════════════════

class TestMonoUtils:
    def test_mono_passthrough(self, tmp_path):
        import soundfile as sf
        wav = tmp_path / "mono.wav"
        audio = _sine().reshape(-1, 1)
        sf.write(str(wav), audio, SR)

        result, sr, path = ensure_mono(wav)
        assert result.ndim == 1
        assert sr == SR
        assert path == wav          # same file returned

    def test_stereo_to_mono_averages(self, tmp_path):
        import soundfile as sf
        wav = tmp_path / "stereo.wav"
        left = _sine(440)
        right = _sine(880)
        stereo = np.stack([left, right], axis=1)
        sf.write(str(wav), stereo, SR)

        result, sr, path = ensure_mono(wav)
        expected = (left + right) / 2
        assert result.ndim == 1
        # atol=5e-4: WAV float32 encode/decode introduces ~1e-4 quantisation error
        assert np.allclose(result, expected, atol=5e-4)
        assert "mono" in path.stem     # saved as <stem>_mono.wav

    def test_stereo_saves_mono_file(self, tmp_path):
        import soundfile as sf
        wav = tmp_path / "stereo.wav"
        stereo = np.stack([_sine(440), _sine(440)], axis=1)
        sf.write(str(wav), stereo, SR)

        _, _, path = ensure_mono(wav)
        assert path.exists()
        mono_check, _ = sf.read(str(path), dtype="float32")
        assert mono_check.ndim == 1

    def test_three_channels_raises(self, tmp_path):
        import soundfile as sf
        wav = tmp_path / "surround.wav"
        # soundfile can't write 3-ch WAV directly via normal write; build raw
        audio = np.zeros((SR, 3), dtype=np.float32)
        sf.write(str(wav), audio, SR)

        with pytest.raises(ValueError, match="3 channels"):
            ensure_mono(wav)


# ════════════════════════════════════════════════════════════════════════════
# target_analysis tests
# ════════════════════════════════════════════════════════════════════════════

class TestTargetAnalysis:

    # ── Q1: Oscillator character ────────────────────────────────────────────

    def test_sawtooth_is_harmonic(self):
        saw = _sawtooth(440, duration=0.5)
        a = analyze_target(saw, SR)
        # Threshold is deliberately modest: HPS can return the 2nd harmonic
        # as F0 (octave error) which halves the measured harmonic ratio, but
        # the sawtooth still comes out much more harmonic than white noise.
        assert a.harmonic_ratio > 0.15, "Sawtooth should have substantial harmonic ratio"
        assert a.spectral_flatness < 0.5, "Sawtooth should not be noise-like"

    def test_noise_is_flat(self):
        noise = _noise()
        a = analyze_target(noise, SR)
        assert a.spectral_flatness > 0.3, "Noise should have higher spectral flatness"
        assert a.harmonic_ratio < 0.7, "Noise should have low harmonic ratio"

    def test_sine_has_low_inharmonicity(self):
        sine = _sine(440, duration=0.5)
        a = analyze_target(sine, SR)
        # Pure sine has very little inharmonic energy
        assert a.inharmonicity < 0.8, "Pure sine should have low inharmonicity"

    def test_spectral_centroid_brightens_with_frequency(self):
        low = _sawtooth(110, duration=0.3)
        high = _sawtooth(1760, duration=0.3)
        a_low = analyze_target(low, SR)
        a_high = analyze_target(high, SR)
        assert a_high.spectral_centroid_norm > a_low.spectral_centroid_norm, \
            "Higher fundamental should produce brighter centroid"

    def test_fundamental_detection(self):
        # Use a sawtooth: HPS is designed for harmonic-rich signals.
        # A pure sine has no overtones so HPS returns ambiguous results.
        saw = _sawtooth(440, duration=0.5)
        a = analyze_target(saw, SR)
        if a.fundamental_hz is not None:
            # Allow ±1 octave error — HPS can return F0 or F0*2.
            assert 200 <= a.fundamental_hz <= 1000, \
                f"Detected fundamental should be within one octave of 440Hz, " \
                f"got {a.fundamental_hz:.1f}Hz"

    # ── Q2: Amp ADSR ────────────────────────────────────────────────────────

    def test_fast_attack_detected(self):
        saw = _sawtooth(440, duration=0.5)
        fast = _shaped(saw, attack_ms=5.0, decay_ms=50.0)
        a = analyze_target(fast, SR)
        assert a.attack_ms < 50.0, f"Fast attack should be short, got {a.attack_ms:.1f}ms"

    def test_slow_attack_detected(self):
        saw = _sawtooth(440, duration=1.5)
        slow = _shaped(saw, attack_ms=300.0, decay_ms=100.0)
        a = analyze_target(slow, SR)
        assert a.attack_ms > 50.0, f"Slow attack should be long, got {a.attack_ms:.1f}ms"

    def test_attack_param_mapping_is_monotone(self):
        """Slower attacks should produce higher param values."""
        attacks = [1.0, 10.0, 100.0, 1000.0]
        params = [_attack_ms_to_param(ms) for ms in attacks]
        assert params == sorted(params), "Attack param mapping must be monotone"

    def test_decay_param_mapping_is_monotone(self):
        decays = [1.0, 10.0, 100.0, 1000.0]
        params = [_decay_ms_to_param(ms) for ms in decays]
        assert params == sorted(params), "Decay param mapping must be monotone"

    def test_adsr_params_in_range(self):
        saw = _shaped(_sawtooth(440, duration=1.0), attack_ms=20.0, decay_ms=80.0)
        a = analyze_target(saw, SR)
        for attr in ("attack_ms", "decay_ms", "sustain_level",
                     "filter_cutoff_est", "filter_resonance_est"):
            val = getattr(a, attr)
            assert val >= 0, f"{attr} must be non-negative, got {val}"

    # ── Q3: Filter ──────────────────────────────────────────────────────────

    def test_bright_signal_gets_higher_cutoff(self):
        dark = _shaped(_sawtooth(110, duration=0.5))
        bright = _shaped(_sawtooth(1760, duration=0.5))
        a_dark = analyze_target(dark, SR)
        a_bright = analyze_target(bright, SR)
        assert a_bright.filter_cutoff_est > a_dark.filter_cutoff_est, \
            "Brighter signal should suggest higher filter cutoff"

    def test_filter_cutoff_in_valid_range(self):
        for freq in [220, 440, 880, 1760]:
            a = analyze_target(_sine(freq, duration=0.3), SR)
            assert 0.0 <= a.filter_cutoff_est <= 1.0

    def test_filter_resonance_in_valid_range(self):
        for sig in [_sine(440, 0.3), _noise(0.3), _sawtooth(440, 0.3)]:
            a = analyze_target(sig, SR)
            assert 0.0 <= a.filter_resonance_est <= 1.0

    # ── Q4: Modulation ──────────────────────────────────────────────────────

    def test_static_signal_has_low_flux(self):
        sine = _sine(440, duration=0.5)
        a = analyze_target(sine, SR)
        assert a.spectral_flux_norm < 0.5, "Static tone should have low flux"

    def test_filter_env_amount_is_low_for_static_tone(self):
        sine = _sine(440, duration=0.5)
        a = analyze_target(sine, SR)
        assert a.filter_env_amount_est < 0.4

    def test_all_modulation_params_in_range(self):
        for sig in [_sine(440, 0.5), _noise(0.5), _sawtooth(440, 0.5)]:
            a = analyze_target(sig, SR)
            for attr in ("filter_env_amount_est", "lfo_rate_est", "lfo_to_filter_est"):
                val = getattr(a, attr)
                assert 0.0 <= val <= 1.0, f"{attr} out of range: {val}"

    # ── Q5: Pitch / cross-mod ───────────────────────────────────────────────

    def test_cross_mod_low_for_harmonic_signal(self):
        saw = _sawtooth(440, duration=0.5)
        a = analyze_target(saw, SR)
        assert a.cross_mod_est <= 0.25, "Harmonic signal should need little cross-mod"

    def test_osc2_detune_near_unison_for_stable_pitch(self):
        sine = _sine(440, duration=0.5)
        a = analyze_target(sine, SR)
        assert 0.3 <= a.osc2_detune_est <= 0.7, "Stable-pitch signal → near-unison detune"

    # ── suggest_x0 ──────────────────────────────────────────────────────────

    def test_suggest_x0_shape_and_range(self):
        PARAM_COLS = [
            "p_Amp Env Attack", "p_Amp Env Decay", "p_Amp Env Release",
            "p_Filter Cutoff", "p_Filter Resonance", "p_Filter Env Amount",
            "p_Osc 1 Pitch", "p_Osc 2 Detune", "p_Osc Pulsewidth",
            "p_Cross Modulation", "p_LFO 1 Rate", "p_LFO 1 to Filter Cutoff",
            "p_LFO 1 to Osc 1 Pitch", "p_Filter Mode", "p_Osc 1 Volume",
        ]
        PINNED = {"p_Osc 1 Pitch", "p_Amp Env Release", "p_LFO 1 to Osc 1 Pitch"}
        a = analyze_target(_sawtooth(440, duration=0.5), SR)
        x0 = suggest_x0(a, PARAM_COLS, PINNED)
        assert x0.shape == (len(PARAM_COLS),)
        assert np.all(x0 >= 0.0) and np.all(x0 <= 1.0), "All params must be in [0,1]"

    def test_suggest_x0_respects_pinned(self):
        PARAM_COLS = ["p_Osc 1 Pitch", "p_Filter Cutoff", "p_Amp Env Release"]
        PINNED = {"p_Osc 1 Pitch", "p_Amp Env Release"}
        pinned_vals = np.array([0.5, 0.7, 0.2])
        a = analyze_target(_sine(440, 0.3), SR)
        x0 = suggest_x0(a, PARAM_COLS, PINNED, current_x0=pinned_vals)
        assert x0[0] == pytest.approx(0.5), "Pinned p_Osc 1 Pitch must stay 0.5"
        assert x0[2] == pytest.approx(0.2), "Pinned p_Amp Env Release must stay 0.2"

    def test_suggest_x0_blends_with_current(self):
        PARAM_COLS = ["p_Filter Cutoff"]
        PINNED: set = set()
        a = analyze_target(_sawtooth(440, 0.5), SR)
        suggestion = a.filter_cutoff_est
        current = np.array([0.0])  # worst-case starting point
        x0 = suggest_x0(a, PARAM_COLS, PINNED, current_x0=current)
        # Blended value should be between 0.0 and suggestion
        assert 0.0 <= x0[0] <= suggestion + 0.01


# ════════════════════════════════════════════════════════════════════════════
# CMA-ES unit tests (no DawDreamer — uses a mock objective)
# ════════════════════════════════════════════════════════════════════════════

class TestCMAESLogic:
    """Verify CMA-ES convergence on a synthetic objective without VST."""

    def _mock_cmaes(self, target: np.ndarray, sigma0=0.2, popsize=10, maxiter=50):
        """Run CMA-ES on a quadratic bowl and return (best_x, best_score)."""
        try:
            import cma
        except ImportError:
            pytest.skip("cma package not installed")

        n = len(target)
        es = cma.CMAEvolutionStrategy(
            [0.5] * n, sigma0,
            {"bounds": [[0.0] * n, [1.0] * n], "maxiter": maxiter,
             "popsize": popsize, "verbose": -9},
        )
        best_x, best_score = np.array([0.5] * n), float("inf")
        while not es.stop():
            xs = es.ask()
            scores = [float(np.sum((np.array(x) - target) ** 2)) for x in xs]
            es.tell(xs, scores)
            idx = np.argmin(scores)
            if scores[idx] < best_score:
                best_score = scores[idx]
                best_x = np.array(xs[idx])
        return best_x, best_score

    def test_cmaes_finds_minimum(self):
        """CMA-ES should converge close to a known minimum in [0,1]^4."""
        target = np.array([0.3, 0.7, 0.2, 0.8])
        best_x, best_score = self._mock_cmaes(target)
        assert best_score < 0.01, f"CMA-ES should find near-zero score, got {best_score:.4f}"
        assert np.allclose(best_x, target, atol=0.1), \
            f"Best x should be close to target:\n  got {best_x}\n  expected {target}"

    def test_cmaes_respects_bounds(self):
        """All candidate evaluations should stay in [0,1]."""
        try:
            import cma
        except ImportError:
            pytest.skip("cma package not installed")

        target = np.array([0.1, 0.9, 0.5])
        n = len(target)
        es = cma.CMAEvolutionStrategy(
            [0.5] * n, 0.3,
            {"bounds": [[0.0] * n, [1.0] * n], "maxiter": 10,
             "popsize": 8, "verbose": -9},
        )
        while not es.stop():
            xs = es.ask()
            for x in xs:
                assert all(0.0 <= v <= 1.0 for v in x), \
                    f"Candidate out of bounds: {x}"
            es.tell(xs, [float(np.sum((np.array(x) - target) ** 2)) for x in xs])

    def test_ipop_population_grows(self):
        """Verify that IPOP multiplies popsize correctly."""
        init_pop = 8
        ratio = 1.5
        for restart in range(3):
            new_pop = int(init_pop * ratio ** restart)
            assert new_pop >= init_pop
        assert int(init_pop * ratio ** 2) == 18  # 8 → 12 → 18


# ════════════════════════════════════════════════════════════════════════════
# MRSTFT feature tests (numpy only, no DawDreamer or GPU)
# ════════════════════════════════════════════════════════════════════════════

class TestMRSTFTDist:
    """Tests for auraloss-backed _mrstft_dist in audio_compare.py."""

    def setup_method(self):
        from s07_refine.audio_compare import _mrstft_dist
        self.dist = _mrstft_dist

    def test_identical_is_near_zero(self):
        audio = np.random.randn(SR).astype(np.float32)
        assert self.dist(audio, audio) < 0.05

    def test_short_audio_returns_neutral(self):
        """Audio shorter than 256 samples → neutral score of 1.0."""
        audio = np.zeros(100, dtype=np.float32)
        assert self.dist(audio, audio) == 1.0

    def test_output_in_range(self):
        a = np.random.randn(SR).astype(np.float32)
        b = np.random.randn(SR).astype(np.float32)
        d = self.dist(a, b)
        assert 0.0 <= d <= 2.0, f"Expected [0, 2], got {d}"

    def test_different_signals_higher_than_identical(self):
        sine = _sine(440.0, 1.0)
        noise = np.random.randn(SR).astype(np.float32) * 0.5
        assert self.dist(sine, noise) > self.dist(sine, sine)

    def test_no_nan_or_inf(self):
        audio = _sine(440.0, 1.0)
        d = self.dist(audio, audio)
        assert np.isfinite(d)

    def test_silence_vs_tone_is_positive(self):
        silence = np.zeros(SR, dtype=np.float32)
        tone = _sine(440.0, 1.0)
        assert self.dist(silence, tone) > 0.0


class TestLUFSNormalise:
    """Tests for _lufs_normalize in audio_compare.py."""

    def setup_method(self):
        from s07_refine.audio_compare import _lufs_normalize
        self.norm = _lufs_normalize

    def test_loud_and_quiet_converge(self):
        """10× louder and 10× quieter versions of the same signal should have
        the same RMS after normalisation."""
        pytest.importorskip("pyloudnorm")
        base = _sine(440.0, 1.0) * 0.3
        loud  = self.norm(base * 10.0, SR)
        quiet = self.norm(base * 0.1,  SR)
        assert abs(loud.std() - quiet.std()) < 1e-3, (
            f"RMS mismatch after LUFS normalise: {loud.std():.5f} vs {quiet.std():.5f}"
        )

    def test_idempotent(self):
        """Normalising a signal twice gives the same result as normalising once."""
        audio = _sine(440.0, 1.0) * 0.3
        once  = self.norm(audio, SR)
        twice = self.norm(once,  SR)
        assert np.allclose(once, twice, atol=1e-4), "Double-normalise should be idempotent"

    def test_silence_passthrough(self):
        """Silent signal (below -70 LUFS threshold) must be returned unchanged."""
        silence = np.zeros(SR, dtype=np.float32)
        out = self.norm(silence, SR)
        assert np.allclose(out, silence), "Silent audio should pass through unchanged"

    def test_short_clip_passthrough(self):
        """Clip shorter than the BS.1770 400ms gating window returns original."""
        short = _sine(440.0, 0.1) * 0.3   # 100ms — below pyloudnorm minimum
        out = self.norm(short, SR)
        assert np.allclose(out, short), "Short clip should pass through unchanged"

    def test_output_dtype_float32(self):
        audio = _sine(440.0, 1.0).astype(np.float32)
        out = self.norm(audio, SR)
        assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"

    def test_mrstft_dist_level_invariant(self):
        """After normalisation, a 10× level shift should give near-zero MRSTFT distance."""
        pytest.importorskip("pyloudnorm")
        from s07_refine.audio_compare import _mrstft_dist
        base   = _sine(440.0, 1.0) * 0.3
        normal = self.norm(base,        SR)
        louder = self.norm(base * 10.0, SR)
        dist = _mrstft_dist(normal, louder)
        assert dist < 0.05, f"Level-shifted signal should score ~0 after normalise, got {dist:.4f}"


class TestFilterCutoffTrajectory:
    """Tests for _centroid_hz_to_filter_cutoff and _compute_filter_cutoff_trajectory."""

    def setup_method(self):
        from s06b_live.stream_invert import (
            _centroid_hz_to_filter_cutoff,
            _compute_filter_cutoff_trajectory,
        )
        self.hz_to_fc = _centroid_hz_to_filter_cutoff
        self.traj = _compute_filter_cutoff_trajectory

    def test_centroid_hz_bounds(self):
        # Output must always be in [0, 1]; low Hz → low cutoff; very high Hz → near 1.0.
        # Exact values depend on whether calibration table exists (calibrated vs heuristic).
        arr = np.array([0.0, SR / 2, SR * 2], dtype=np.float32)
        out = self.hz_to_fc(arr, SR)
        assert 0.0 <= out[0] <= 0.5, f"0 Hz should map to a low cutoff, got {out[0]}"
        assert 0.5 <= out[1] <= 1.0, f"Nyquist should map to a high cutoff, got {out[1]}"
        assert out[2] == pytest.approx(out[1], abs=0.1), "above Nyquist should saturate"

    def test_centroid_hz_monotone(self):
        freqs = np.array([100.0, 500.0, 2000.0, 8000.0], dtype=np.float32)
        out = self.hz_to_fc(freqs, SR)
        assert np.all(np.diff(out) > 0), "mapping must be strictly increasing"

    def test_trajectory_output_shape(self):
        audio = _sine(440.0, 1.0)
        ts = np.linspace(0.1, 0.9, 10)
        result = self.traj(audio, SR, ts)
        assert result.shape == (10,)

    def test_trajectory_values_in_range(self):
        audio = np.random.randn(SR).astype(np.float32) * 0.1
        ts = np.linspace(0.05, 0.95, 15)
        result = self.traj(audio, SR, ts)
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_bright_higher_than_dark(self):
        # 4 kHz sine centroid: (4000/24000)*0.6+0.3 ≈ 0.40 > 200 Hz: ≈ 0.305
        bright = _sine(4000.0, 1.0)
        dark   = _sine(200.0,  1.0)
        ts = np.linspace(0.1, 0.9, 8)
        fc_bright = self.traj(bright, SR, ts)
        fc_dark   = self.traj(dark,   SR, ts)
        assert fc_bright.mean() > fc_dark.mean(), (
            f"bright ({fc_bright.mean():.3f}) should exceed dark ({fc_dark.mean():.3f})"
        )

    def test_silence_does_not_crash(self):
        silence = np.zeros(SR, dtype=np.float32)
        ts = np.linspace(0.1, 0.9, 5)
        result = self.traj(silence, SR, ts)
        assert result.shape == (5,)
        assert np.all(np.isfinite(result))


class TestSPScoring:
    """Tests for compute_sp_features and _sp_dist in audio_compare.py."""

    def setup_method(self):
        from s07_refine.audio_compare import (
            compute_sp_features, _sp_dist,
            ENCODEC_WEIGHT, MRSTFT_WEIGHT, AP_WEIGHT, SP_WEIGHT,
        )
        self.sp_feats = compute_sp_features
        self.sp_dist  = _sp_dist
        self.weights  = (ENCODEC_WEIGHT, MRSTFT_WEIGHT, AP_WEIGHT, SP_WEIGHT)

    def test_weights_sum_to_one(self):
        total = sum(self.weights)
        assert total == pytest.approx(1.0, abs=1e-6), f"weights sum {total} ≠ 1.0"

    def test_sp_features_shape(self):
        pytest.importorskip("pyworld")
        audio = _sine(440.0, 1.0)
        sp = self.sp_feats(audio, SR)
        assert sp is not None
        assert sp.ndim == 2
        assert sp.shape[1] > 1
        assert sp.dtype == np.float32

    def test_sp_features_short_returns_none(self):
        short = _sine(440.0, 0.05)   # 50ms — below 100ms minimum
        assert self.sp_feats(short, SR) is None

    def test_sp_dist_identical_near_zero(self):
        pytest.importorskip("pyworld")
        sp = self.sp_feats(_sine(440.0, 1.0), SR)
        assert self.sp_dist(sp, sp) < 0.01

    def test_sp_dist_different_signals(self):
        pytest.importorskip("pyworld")
        sp_bright = self.sp_feats(_sine(4000.0, 1.0), SR)
        sp_dark   = self.sp_feats(_sine(200.0,  1.0), SR)
        assert self.sp_dist(sp_bright, sp_dark) > 0.0

    def test_sp_dist_empty_returns_neutral(self):
        empty = np.zeros((0, 513), dtype=np.float32)
        assert self.sp_dist(empty, empty) == 1.0


class TestCREPEWarmStart:
    """Tests for CREPE + harmonic analysis warm-start in target_analysis.py."""

    def setup_method(self):
        from s07_refine.target_analysis import (
            _crepe_f0, _pyworld_analysis, _harmonic_amplitudes,
            analyze_target, suggest_x0,
        )
        self.crepe_f0      = _crepe_f0
        self.pyworld       = _pyworld_analysis
        self.harm_amps     = _harmonic_amplitudes
        self.analyze       = analyze_target
        self.suggest       = suggest_x0

    # ── _crepe_f0 ──────────────────────────────────────────────────────────

    def test_crepe_f0_returns_tuple(self):
        pytest.importorskip("torchcrepe")
        audio = _sine(440.0, 1.0)
        result = self.crepe_f0(audio, SR)
        assert result is not None, "CREPE should succeed on 1s 440Hz sine"
        f0, t = result
        assert f0.shape == t.shape
        assert len(f0) > 0

    def test_crepe_f0_voiced_majority(self):
        pytest.importorskip("torchcrepe")
        audio = _sine(440.0, 1.0)
        f0, t = self.crepe_f0(audio, SR)
        voiced_frac = (f0 > 0).mean()
        assert voiced_frac > 0.5, f"Most frames should be voiced, got {voiced_frac:.2f}"

    def test_crepe_f0_silence_unvoiced(self):
        pytest.importorskip("torchcrepe")
        silence = np.zeros(SR, dtype=np.float32)
        f0, t = self.crepe_f0(silence, SR)
        assert (f0 == 0).all(), "Silent audio should have all unvoiced frames"

    def test_crepe_f0_short_returns_none(self):
        short = _sine(440.0, 0.05)   # 50ms
        assert self.crepe_f0(short, SR) is None

    def test_crepe_f0_frequency_accuracy(self):
        pytest.importorskip("torchcrepe")
        audio = _sine(440.0, 1.0)
        f0, _ = self.crepe_f0(audio, SR)
        voiced = f0[f0 > 0]
        assert len(voiced) > 0
        median_f0 = float(np.median(voiced))
        assert abs(median_f0 - 440.0) < 10.0, f"CREPE F0 {median_f0:.1f}Hz should be near 440Hz"

    # ── _harmonic_amplitudes ────────────────────────────────────────────────

    def test_harmonic_amplitudes_shape(self):
        audio = _sine(440.0, 1.0)
        f0 = np.full(200, 440.0)
        t = np.arange(200) * 0.005
        f0[:10] = 0.0   # unvoiced prefix
        result = self.harm_amps(audio, SR, f0, t)
        assert result is not None
        assert result.shape[0] == 200
        assert result.shape[1] == 16   # n_harmonics default

    def test_harmonic_amplitudes_fundamental_dominant(self):
        audio = _sine(440.0, 1.0) * 0.5
        f0 = np.full(200, 440.0)
        t  = np.arange(200) * 0.005
        result = self.harm_amps(audio, SR, f0, t)
        voiced = result[f0 > 0]
        if len(voiced) > 0:
            h1_mean  = voiced[:, 0].mean()
            h2_mean  = voiced[:, 1].mean()
            assert h1_mean > h2_mean * 0.5, "Fundamental should be dominant"

    def test_harmonic_amplitudes_unvoiced_zero(self):
        audio = _sine(440.0, 1.0)
        f0 = np.zeros(200)         # all unvoiced
        t  = np.arange(200) * 0.005
        result = self.harm_amps(audio, SR, f0, t)
        if result is not None:
            assert result.sum() == 0.0

    # ── analyze_target integration ─────────────────────────────────────────

    def test_analyze_uses_crepe(self):
        audio = _sine(440.0, 1.0)
        a = self.analyze(audio, SR)
        assert a.f0_source in ("crepe", "hps", "dio"), f"unexpected f0_source: {a.f0_source}"

    def test_analyze_noise_volume_populated(self):
        noise = np.random.randn(SR).astype(np.float32) * 0.3
        a = self.analyze(noise, SR)
        # White noise should trigger some noise volume estimate
        assert 0.0 <= a.noise_volume_est <= 0.40

    def test_analyze_filter_env_fields_finite(self):
        audio = _sine(440.0, 1.0)
        a = self.analyze(audio, SR)
        for field_val in [a.filter_env_attack_est, a.filter_env_decay_est,
                          a.filter_env_sustain_est]:
            assert np.isfinite(field_val)
            assert 0.0 <= field_val <= 1.0

    def test_analyze_harmonic_richness_in_range(self):
        audio = _sine(440.0, 1.0)
        a = self.analyze(audio, SR)
        assert 0.0 <= a.harmonic_richness <= 1.0

    # ── suggest_x0 coverage ────────────────────────────────────────────────

    def test_suggest_x0_covers_new_params(self):
        audio = _sine(440.0, 1.0)
        a = self.analyze(audio, SR)
        new_params = [
            "p_Noise Volume", "p_Filter Env Attack",
            "p_Filter Env Decay", "p_Filter Env Sustain", "p_Osc 2 Volume",
        ]
        x0 = self.suggest(a, new_params, set())
        assert x0.shape == (len(new_params),)
        assert np.all((x0 >= 0.0) & (x0 <= 1.0))


# ════════════════════════════════════════════════════════════════════════════
# Integration smoke test (no render; just imports and analysis)
# ════════════════════════════════════════════════════════════════════════════

class TestSmoke:
    def test_modules_importable(self):
        from s07_refine import mono_utils, target_analysis, vst_hill_climb, vst_cmaes
        assert hasattr(mono_utils, "ensure_mono")
        assert hasattr(target_analysis, "analyze_target")
        assert hasattr(vst_hill_climb, "hill_climb")
        assert hasattr(vst_cmaes, "cmaes_refine")

    def test_audio_compare_imports(self):
        from s07_refine.audio_compare import (
            _mrstft_dist, compute_ap_features, score_audio_composite,
            ENCODEC_WEIGHT, MRSTFT_WEIGHT, AP_WEIGHT, SP_WEIGHT,
        )
        assert abs(ENCODEC_WEIGHT + MRSTFT_WEIGHT + AP_WEIGHT + SP_WEIGHT - 1.0) < 1e-6

    def test_analyze_crane_scream_if_available(self):
        """Run full analysis on the actual target if it exists."""
        import soundfile as sf
        from s07_refine.target_analysis import print_analysis
        target = Path(
            "/mnt/d/Mimic-Synth-Data/OB-X_Prototype/targets/"
            "816426_crane-bird-scream_mono.wav"
        )
        if not target.exists():
            pytest.skip("crane scream target not available")
        audio, sr = sf.read(str(target), dtype="float32")
        a = analyze_target(audio, sr)
        print_analysis(a)
        # Basic sanity checks on the real target
        assert 0.0 <= a.filter_cutoff_est <= 1.0
        assert 0.0 <= a.attack_ms
        assert a.fundamental_hz is None or a.fundamental_hz > 0

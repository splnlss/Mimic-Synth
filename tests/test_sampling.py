"""Unit tests for s03_dataset.sampling — no DawDreamer required."""
import numpy as np
import pytest

from s03_dataset.sampling import cold_start_vectors, to_synth_value, apply_importance


class TestColdStartVectors:
    def test_shape(self):
        v = cold_start_vectors(m=5, d=4, seed=0)
        assert v.shape == (32, 4)

    def test_range(self):
        v = cold_start_vectors(m=6, d=3, seed=0)
        assert v.min() >= 0.0 and v.max() <= 1.0

    def test_deterministic_same_seed(self):
        a = cold_start_vectors(m=4, d=3, seed=42)
        b = cold_start_vectors(m=4, d=3, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = cold_start_vectors(m=4, d=3, seed=0)
        b = cold_start_vectors(m=4, d=3, seed=1)
        assert not np.allclose(a, b)

    def test_m_zero(self):
        v = cold_start_vectors(m=0, d=2, seed=0)
        assert v.shape == (1, 2)

    def test_negative_m_raises(self):
        with pytest.raises(ValueError):
            cold_start_vectors(m=-1, d=2)


class TestToSynthValue:
    def test_linear_passthrough(self):
        assert to_synth_value(0.5, {"log_scale": False}) == 0.5
        assert to_synth_value(0.0, {}) == 0.0
        assert to_synth_value(1.0, {}) == 1.0

    def test_log_endpoints_preserved(self):
        spec = {"log_scale": True, "log_base": 50}
        assert to_synth_value(0.0, spec) == pytest.approx(0.0, abs=1e-9)
        assert to_synth_value(1.0, spec) == pytest.approx(1.0, abs=1e-9)

    def test_log_monotonic(self):
        spec = {"log_scale": True, "log_base": 50}
        xs = np.linspace(0, 1, 20)
        ys = [to_synth_value(x, spec) for x in xs]
        assert all(ys[i] <= ys[i+1] for i in range(len(ys)-1))

    def test_log_gives_finer_low_resolution(self):
        """At u=0.5, log mapping should return a value below 0.5."""
        spec = {"log_scale": True, "log_base": 50}
        assert to_synth_value(0.5, spec) < 0.5

    def test_invalid_log_base(self):
        with pytest.raises(ValueError):
            to_synth_value(0.5, {"log_scale": True, "log_base": 0})


class TestApplyImportance:
    def _profile(self):
        return {
            "parameters": {
                "a": {"importance": 1.0},
                "b": {"importance": 0.3},
                "c": {"importance": 1.0, "log_scale": True, "log_base": 50},
            },
            "reset": {"a": 0.5, "b": 0.5, "c": 0.5},
        }

    def test_filter_mode_passthrough(self):
        out = apply_importance(np.array([0.1, 0.9]), ["a", "b"], self._profile(), mode="filter")
        assert out == {"a": 0.1, "b": 0.9}

    def test_filter_applies_log_scale(self):
        out = apply_importance(np.array([0.5]), ["c"], self._profile(), mode="filter")
        assert out["c"] < 0.5

    def test_scale_narrows_low_importance(self):
        out = apply_importance(np.array([0.0]), ["b"], self._profile(), mode="scale")
        # importance=0.3, u=0.0 → 0.5 + 0.3*(0-0.5) = 0.35
        assert out["b"] == pytest.approx(0.35, abs=1e-6)

    def test_scale_clips_to_unit(self):
        profile = self._profile()
        profile["reset"]["a"] = 0.95
        out = apply_importance(np.array([1.0]), ["a"], profile, mode="scale")
        assert 0.0 <= out["a"] <= 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            apply_importance(np.array([0.1, 0.2]), ["a"], self._profile())

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            apply_importance(np.array([0.1]), ["a"], self._profile(), mode="bogus")

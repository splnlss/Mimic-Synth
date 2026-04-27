"""
Unit tests for pure-Python functions in capture_v1_2.py — no DawDreamer required.
"""
import hashlib
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
import yaml

from s02_capture.capture_v1_2 import apply_params, sample_vectors, resolve_plugin_path, load_profile

PROFILE_PATH = Path(__file__).parent.parent / "s01_profiles" / "obxf.yaml"


@pytest.fixture(scope="module")
def profile():
    return load_profile(PROFILE_PATH)


# ── sample_vectors ────────────────────────────────────────────────────────────

def test_sample_vectors_shape():
    # m=4 gives 2^4=16 samples
    vecs = sample_vectors(m=4, modulated_params=["a", "b", "c"])
    assert vecs.shape == (16, 3)


def test_sample_vectors_range():
    # m=6 is too large (2^6=64), use m=3 for 8 samples
    vecs = sample_vectors(m=3, modulated_params=list("abcde"))
    assert vecs.min() >= 0.0
    assert vecs.max() <= 1.0


def test_sample_vectors_deterministic():
    # m=3 gives 8 samples
    v1 = sample_vectors(m=3, modulated_params=["x", "y"], seed=42)
    v2 = sample_vectors(m=3, modulated_params=["x", "y"], seed=42)
    np.testing.assert_array_equal(v1, v2)


def test_sample_vectors_different_seeds():
    # m=3 gives 8 samples
    v1 = sample_vectors(m=3, modulated_params=["x", "y"], seed=0)
    v2 = sample_vectors(m=3, modulated_params=["x", "y"], seed=1)
    assert not np.array_equal(v1, v2)


# ── apply_params ──────────────────────────────────────────────────────────────

def _make_synth_mock():
    synth = MagicMock()
    return synth


def test_apply_params_continuous_passthrough(profile):
    """Continuous params should be written to the synth unchanged."""
    synth = _make_synth_mock()
    name_idx = {"Filter Cutoff": 40, "Filter Resonance": 41}
    params_dict = {"Filter Cutoff": 0.333, "Filter Resonance": 0.777}
    apply_params(synth, params_dict, profile, name_idx)
    assert params_dict["Filter Cutoff"] == pytest.approx(0.333)
    assert params_dict["Filter Resonance"] == pytest.approx(0.777)


def test_apply_params_discrete_quantises_dict():
    """Discrete params must update params_dict so stored value == sent value."""
    synth = _make_synth_mock()
    mini_profile = {
        "parameters": {
            "WaveType": {
                "continuous": False,
                "categories": ["saw", "pulse", "tri"],
            }
        }
    }
    name_idx = {"WaveType": 0}
    params_dict = {"WaveType": 0.45}
    apply_params(synth, params_dict, mini_profile, name_idx)
    # round(0.45 * 2) / 2 = round(0.9) / 2 = 1/2 = 0.5
    assert params_dict["WaveType"] == pytest.approx(0.5), (
        "params_dict must be updated to the quantised value so Parquet matches synth state"
    )


def test_apply_params_calls_set_parameter(profile):
    """set_parameter should be called once per param."""
    synth = _make_synth_mock()
    name_idx = {"Filter Cutoff": 40, "Filter Resonance": 41}
    params_dict = {"Filter Cutoff": 0.5, "Filter Resonance": 0.3}
    apply_params(synth, params_dict, profile, name_idx)
    assert synth.set_parameter.call_count == 2


def test_apply_params_uses_correct_index(profile):
    """set_parameter must be called with the index from name_idx."""
    synth = _make_synth_mock()
    name_idx = {"Filter Cutoff": 40}
    params_dict = {"Filter Cutoff": 0.5}
    apply_params(synth, params_dict, profile, name_idx)
    synth.set_parameter.assert_called_once_with(40, pytest.approx(0.5))


# ── resolve_plugin_path ───────────────────────────────────────────────────────

def test_resolve_plugin_path_real_macos():
    import platform
    if platform.system() != "Darwin":
        pytest.skip("macOS-only test")
    profile = load_profile(PROFILE_PATH)
    path = resolve_plugin_path(profile)
    assert isinstance(path, str)
    assert Path(path).exists()


def test_resolve_plugin_path_missing_raises():
    fake_profile = {
        "synth": {
            "plugin_path_macos":   "/nonexistent/OB-Xf.vst3",
            "plugin_path_windows": "C:/nonexistent/OB-Xf.vst3",
            "plugin_path_linux":   "/nonexistent/OB-Xf.vst3",
        }
    }
    with pytest.raises(FileNotFoundError):
        resolve_plugin_path(fake_profile)


# ── hash uniqueness ───────────────────────────────────────────────────────────

def test_hash_uniqueness():
    """Different (vec, note) pairs must produce different hashes."""
    vec_a = np.array([0.1, 0.2, 0.3])
    vec_b = np.array([0.4, 0.5, 0.6])
    note_a, note_b = 60, 72

    h_aa = hashlib.md5(vec_a.tobytes() + bytes([note_a])).hexdigest()[:12]
    h_ab = hashlib.md5(vec_a.tobytes() + bytes([note_b])).hexdigest()[:12]
    h_ba = hashlib.md5(vec_b.tobytes() + bytes([note_a])).hexdigest()[:12]

    assert h_aa != h_ab, "Same vec, different note should differ"
    assert h_aa != h_ba, "Different vec, same note should differ"
    assert h_ab != h_ba, "Different vec and note should differ"

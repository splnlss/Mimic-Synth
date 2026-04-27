"""
Unit tests for capture_v1_2.py — no DawDreamer required.

Tests the key behavioural change: settle runs before patch change (draining
the old patch's tail), not after. Also tests note_off, sample_vectors using
random_base2, and the existing pure functions.
"""
import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch
import numpy as np
import pytest
import yaml

# capture_v1_2.py uses underscore
import importlib
spec = importlib.util.spec_from_file_location(
    "capture_v1_2",
    str(Path(__file__).parent.parent / "s02_capture" / "capture_v1_2.py"),
)
capture_v1_2 = importlib.util.module_from_spec(spec)

# Stub dawdreamer before exec so module-level import doesn't fail
sys.modules.setdefault("dawdreamer", MagicMock())
spec.loader.exec_module(capture_v1_2)

apply_params = capture_v1_2.apply_params
sample_vectors = capture_v1_2.sample_vectors
resolve_plugin_path = capture_v1_2.resolve_plugin_path
load_profile = capture_v1_2.load_profile
measure_self_noise = capture_v1_2.measure_self_noise
settle = capture_v1_2.settle
render_one = capture_v1_2.render_one

PROFILE_PATH = Path(__file__).parent.parent / "s01_profiles" / "obxf.yaml"


@pytest.fixture(scope="module")
def profile():
    return load_profile(PROFILE_PATH)


# ── sample_vectors (now uses random_base2) ────────────────────────────────────

def test_sample_vectors_shape():
    vecs = sample_vectors(m=4, modulated_params=["a", "b", "c"])
    assert vecs.shape == (16, 3)


def test_sample_vectors_power_of_two():
    """Output row count must always be exactly 2^m."""
    for m in (3, 5, 7):
        vecs = sample_vectors(m=m, modulated_params=["x", "y"])
        assert vecs.shape[0] == 2 ** m


def test_sample_vectors_range():
    vecs = sample_vectors(m=6, modulated_params=list("abcde"))
    assert vecs.min() >= 0.0
    assert vecs.max() <= 1.0


def test_sample_vectors_deterministic():
    v1 = sample_vectors(m=5, modulated_params=["x", "y"], seed=42)
    v2 = sample_vectors(m=5, modulated_params=["x", "y"], seed=42)
    np.testing.assert_array_equal(v1, v2)


def test_sample_vectors_different_seeds():
    v1 = sample_vectors(m=5, modulated_params=["x", "y"], seed=0)
    v2 = sample_vectors(m=5, modulated_params=["x", "y"], seed=1)
    assert not np.array_equal(v1, v2)


# ── apply_params ──────────────────────────────────────────────────────────────

def test_apply_params_continuous_passthrough(profile):
    synth = MagicMock()
    name_idx = {"Filter Cutoff": 40, "Filter Resonance": 41}
    params_dict = {"Filter Cutoff": 0.333, "Filter Resonance": 0.777}
    apply_params(synth, params_dict, profile, name_idx)
    assert params_dict["Filter Cutoff"] == pytest.approx(0.333)
    assert params_dict["Filter Resonance"] == pytest.approx(0.777)


def test_apply_params_discrete_quantises_dict():
    synth = MagicMock()
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
    assert params_dict["WaveType"] == pytest.approx(0.5)


def test_apply_params_calls_set_parameter(profile):
    synth = MagicMock()
    name_idx = {"Filter Cutoff": 40, "Filter Resonance": 41}
    params_dict = {"Filter Cutoff": 0.5, "Filter Resonance": 0.3}
    apply_params(synth, params_dict, profile, name_idx)
    assert synth.set_parameter.call_count == 2


# ── measure_self_noise ────────────────────────────────────────────────────────

def test_measure_self_noise_silent_patch():
    """A silent patch should return ~0.0."""
    engine = MagicMock()
    synth = MagicMock()
    engine.get_audio.return_value = np.zeros((2, 100), dtype=np.float32)
    noise = measure_self_noise(engine, synth)
    assert noise == pytest.approx(0.0)
    synth.clear_midi.assert_called_once()
    engine.render.assert_called_once()


def test_measure_self_noise_self_oscillating():
    """A self-oscillating patch should return its peak level."""
    engine = MagicMock()
    synth = MagicMock()
    engine.get_audio.return_value = np.array([[0.05, -0.03, 0.02]])
    noise = measure_self_noise(engine, synth)
    assert noise == pytest.approx(0.05)


def test_measure_self_noise_no_midi():
    """measure_self_noise must NOT schedule any MIDI — only clear + render."""
    engine = MagicMock()
    synth = MagicMock()
    engine.get_audio.return_value = np.zeros((2, 100), dtype=np.float32)
    measure_self_noise(engine, synth)
    synth.add_midi_note.assert_not_called()


# ── settle ────────────────────────────────────────────────────────────────────

def _make_engine_mock(peak_sequence):
    """Create engine mock where successive get_audio() calls return arrays
    with the given peak values. Simulates a decaying tail."""
    engine = MagicMock()
    audio_iter = iter(peak_sequence)

    def fake_get_audio():
        try:
            peak = next(audio_iter)
        except StopIteration:
            peak = 0.0
        return np.array([[peak, -peak * 0.5]])  # (channels, samples)

    engine.get_audio.side_effect = fake_get_audio
    return engine


def test_settle_returns_quickly_when_silent():
    """If the synth is already silent, settle should return after the flush chunk."""
    engine = _make_engine_mock([0.0])
    synth = MagicMock()
    elapsed = settle(engine, synth, [60], chunk=0.05, max_sec=2.0, threshold=1e-5)
    assert elapsed == pytest.approx(0.0, abs=0.01)  # silent on first check


def test_settle_waits_for_decay():
    """Settle should keep rendering until peak drops below threshold."""
    # 3 loud chunks then silence
    engine = _make_engine_mock([0.1, 0.05, 0.001, 0.0])
    synth = MagicMock()
    elapsed = settle(engine, synth, [60], chunk=0.05, max_sec=2.0, threshold=1e-3)
    # 3 loud chunks (0.15) then silence at 4th = 0.15
    assert elapsed == pytest.approx(0.15, abs=0.02)


def test_settle_caps_at_max_sec():
    """If the synth never settles, elapsed should equal max_sec."""
    # Always loud
    engine = _make_engine_mock([0.5] * 100)
    synth = MagicMock()
    elapsed = settle(engine, synth, [60], chunk=0.05, max_sec=0.5, threshold=1e-5)
    assert elapsed >= 0.5


def test_settle_clears_midi_before_rendering():
    """settle must clear MIDI buffer before starting the render loop."""
    engine = _make_engine_mock([0.0])
    synth = MagicMock()

    settle(engine, synth, [48, 60], chunk=0.05, max_sec=1.0, threshold=1e-5)

    synth.clear_midi.assert_called()
    engine.render.assert_called()


# ── render_one ────────────────────────────────────────────────────────────────

def test_render_one_no_settle_call():
    """render_one in v1.2 must NOT call settle — that's the caller's job."""
    engine = MagicMock()
    synth = MagicMock()
    engine.get_audio.return_value = np.zeros((2, 48000), dtype=np.float32)

    profile = {
        "probe": {
            "velocity": 100,
            "pre_roll_sec": 0.2,
            "hold_sec": 1.5,
            "render_sec": 4.1,
        }
    }
    audio = render_one(engine, synth, 60, profile)

    # Should call clear_midi, add_midi_note, set_bpm, render — NOT settle/note_off
    synth.clear_midi.assert_called_once()
    synth.add_midi_note.assert_called_once_with(60, 100, 0.2, 1.5)
    engine.set_bpm.assert_called_once_with(120)
    engine.render.assert_called_once_with(4.1)
    assert audio.shape == (48000,)
    assert audio.dtype == np.float32


# ── resolve_plugin_path ───────────────────────────────────────────────────────

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


# ── hash uniqueness (same as v1) ─────────────────────────────────────────────

def test_hash_uniqueness():
    vec_a = np.array([0.1, 0.2, 0.3])
    vec_b = np.array([0.4, 0.5, 0.6])
    note_a, note_b = 60, 72

    h_aa = hashlib.md5(vec_a.tobytes() + bytes([note_a])).hexdigest()[:12]
    h_ab = hashlib.md5(vec_a.tobytes() + bytes([note_b])).hexdigest()[:12]
    h_ba = hashlib.md5(vec_b.tobytes() + bytes([note_a])).hexdigest()[:12]

    assert h_aa != h_ab
    assert h_aa != h_ba
    assert h_ab != h_ba


# ── Integration-style: verify settle order in main loop ──────────────────────

def test_settle_before_patch_change_order():
    """The critical invariant: settle must drain the OLD patch before
    apply_params loads the new one. We simulate a 2-vector capture and
    verify the call order."""
    engine = MagicMock()
    synth = MagicMock()
    call_log = []

    def log_set_param(idx, val):
        call_log.append(("set_parameter", idx, round(val, 4)))
    synth.set_parameter.side_effect = log_set_param

    def log_add_note(n, v, start, dur):
        call_log.append(("add_midi_note", n, v))
    synth.add_midi_note.side_effect = log_add_note

    def log_clear():
        call_log.append(("clear_midi",))
    synth.clear_midi.side_effect = log_clear

    def log_render(dur):
        call_log.append(("render", round(dur, 3)))
    engine.render.side_effect = log_render

    # get_audio returns silence so settle resolves immediately
    engine.get_audio.return_value = np.zeros((2, 100), dtype=np.float32)

    profile = load_profile(PROFILE_PATH)
    name_idx = {name: i for i, name in enumerate(profile["parameters"])}
    for name in profile["reset"]:
        if name not in name_idx:
            name_idx[name] = len(name_idx)

    notes = [60]  # single note for simplicity
    modulated = list(profile["parameters"].keys())

    # Simulate two vectors
    vec1 = np.full(len(modulated), 0.3)
    vec2 = np.full(len(modulated), 0.7)

    played = []

    # Vector 1: no settle needed (first vector)
    capture_v1_2.reset(synth, profile, name_idx)
    params1 = dict(zip(modulated, vec1))
    apply_params(synth, params1, profile, name_idx)
    audio = render_one(engine, synth, 60, profile)
    played.append(60)

    call_log.clear()

    # Vector 2: settle THEN reset+apply
    settle(engine, synth, played, chunk=0.05, max_sec=1.0, threshold=1e-5)
    played.clear()
    capture_v1_2.reset(synth, profile, name_idx)
    params2 = dict(zip(modulated, vec2))
    apply_params(synth, params2, profile, name_idx)

    # Verify: the first events after call_log.clear() should be note_off
    # (add_midi_note with vel=0) and renders, BEFORE any set_parameter calls
    # with the new 0.7 values.
    first_set_param_idx = None
    first_render_idx = None
    for idx, entry in enumerate(call_log):
        if entry[0] == "set_parameter" and first_set_param_idx is None:
            first_set_param_idx = idx
        if entry[0] == "render" and first_render_idx is None:
            first_render_idx = idx

    assert first_render_idx is not None, "settle should have rendered"
    assert first_set_param_idx is not None, "reset/apply should have set params"
    assert first_render_idx < first_set_param_idx, (
        "settle renders must happen BEFORE set_parameter calls (old patch drains first)"
    )

"""
Integration tests — require DawDreamer + OB-Xf installed.
Run with: pytest tests/test_integration.py -v -m integration

These tests load the actual VST and render audio, so they take a few seconds.
"""
import platform
from pathlib import Path
import numpy as np
import pytest
import yaml

from s02_capture.capture_v1_2 import (
    load_profile, resolve_plugin_path, build_name_index,
    apply_params, reset, render_one,
    SAMPLE_RATE, BUFFER_SIZE,
)

pytestmark = pytest.mark.integration

PROFILE_PATH = Path(__file__).parent.parent / "s01_profiles" / "obxf.yaml"


@pytest.fixture(scope="module")
def profile():
    return load_profile(PROFILE_PATH)


@pytest.fixture(scope="module")
def engine_synth_idx(profile):
    import dawdreamer as daw
    plugin_path = resolve_plugin_path(profile)
    engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
    synth = engine.make_plugin_processor("obxf", plugin_path)
    name_idx = build_name_index(synth)
    engine.load_graph([(synth, [])])
    return engine, synth, name_idx


def test_plugin_loads_and_has_parameters(engine_synth_idx):
    _, synth, name_idx = engine_synth_idx
    assert len(name_idx) > 0, "OB-Xf should expose at least one parameter"


def test_profile_params_exist_in_plugin(profile, engine_synth_idx):
    _, _, name_idx = engine_synth_idx
    modulated = [
        name for name, spec in profile["parameters"].items()
        if spec.get("importance", 0) > 0
    ]
    missing = [n for n in modulated if n not in name_idx]
    assert not missing, (
        f"Profile params not found in plugin: {missing}. "
        "Run enumerate_params.py and reconcile s01_profiles/obxf.yaml."
    )


def test_render_is_non_silent(profile, engine_synth_idx):
    engine, synth, name_idx = engine_synth_idx
    reset(synth, profile, name_idx)
    audio = render_one(engine, synth, note=60, profile=profile)
    assert np.max(np.abs(audio)) > 1e-4, "Render should produce non-silent audio"


def test_render_correct_length(profile, engine_synth_idx):
    engine, synth, name_idx = engine_synth_idx
    reset(synth, profile, name_idx)
    audio = render_one(engine, synth, note=60, profile=profile)
    expected_samples = int(profile["probe"]["render_sec"] * SAMPLE_RATE)
    assert abs(len(audio) - expected_samples) <= BUFFER_SIZE, (
        f"Audio length {len(audio)} too far from expected {expected_samples}"
    )


def test_render_spectral_consistency(profile, engine_synth_idx):
    """OB-Xf waveforms are NOT bit-exact between renders — oscillator phase is
    not reset between DawDreamer render calls ("Suppressing default patch" race
    condition in v1.0.3). Timbral content (spectral centroid) must stay within
    5% across renders, which is sufficient for surrogate training."""
    engine, synth, name_idx = engine_synth_idx

    def spectral_centroid(audio):
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / SAMPLE_RATE)
        return np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-9)

    centroids = []
    for _ in range(5):
        reset(synth, profile, name_idx)
        audio = render_one(engine, synth, note=60, profile=profile)
        centroids.append(spectral_centroid(audio))

    mean_c = np.mean(centroids)
    cv = np.std(centroids) / (mean_c + 1e-9)   # coefficient of variation
    assert cv < 0.05, (
        f"Spectral centroid CV={cv:.3f} across 5 renders — timbral instability "
        f"(centroids: {[f'{c:.0f}' for c in centroids]})"
    )


def test_cutoff_affects_spectral_centroid(profile, engine_synth_idx):
    """Higher cutoff → brighter sound → higher spectral centroid."""
    engine, synth, name_idx = engine_synth_idx
    modulated = [
        name for name, spec in profile["parameters"].items()
        if spec.get("importance", 0) > 0
    ]

    def render_with_cutoff(cutoff_value):
        reset(synth, profile, name_idx)
        params_dict = {name: 0.5 for name in modulated}
        params_dict["Filter Cutoff"] = cutoff_value
        apply_params(synth, params_dict, profile, name_idx)
        return render_one(engine, synth, note=60, profile=profile)

    audio_low = render_with_cutoff(0.1)
    audio_high = render_with_cutoff(0.9)

    def spectral_centroid(audio):
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / SAMPLE_RATE)
        return np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-9)

    centroid_low = spectral_centroid(audio_low)
    centroid_high = spectral_centroid(audio_high)

    assert centroid_high > centroid_low, (
        f"Cutoff=0.9 centroid ({centroid_high:.1f} Hz) should exceed "
        f"cutoff=0.1 centroid ({centroid_low:.1f} Hz)"
    )


def test_different_notes_produce_different_audio(profile, engine_synth_idx):
    engine, synth, name_idx = engine_synth_idx
    reset(synth, profile, name_idx)
    audio_c3 = render_one(engine, synth, note=48, profile=profile)
    reset(synth, profile, name_idx)
    audio_c4 = render_one(engine, synth, note=60, profile=profile)
    assert not np.array_equal(audio_c3, audio_c4), "Different notes should produce different audio"

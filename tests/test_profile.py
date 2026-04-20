"""
Unit tests for s01_profiles/obxf.yaml — no DawDreamer required.
"""
from pathlib import Path
import pytest
import yaml

PROFILE_PATH = Path(__file__).parent.parent / "s01_profiles" / "obxf.yaml"


@pytest.fixture(scope="module")
def profile():
    with open(PROFILE_PATH) as f:
        return yaml.safe_load(f)


def test_yaml_loads(profile):
    assert profile is not None


def test_top_level_keys(profile):
    for key in ("synth", "parameters", "probe", "reset"):
        assert key in profile, f"Missing top-level key: {key}"


def test_synth_plugin_paths(profile):
    synth = profile["synth"]
    for key in ("plugin_path_macos", "plugin_path_windows", "plugin_path_linux"):
        assert key in synth, f"Missing plugin path: {key}"
        assert synth[key], f"Empty plugin path: {key}"


def test_parameter_fields(profile):
    required_fields = {"continuous", "encoding", "range", "importance"}
    for name, spec in profile["parameters"].items():
        missing = required_fields - set(spec.keys())
        assert not missing, f"Parameter '{name}' missing fields: {missing}"


def test_parameter_ranges(profile):
    for name, spec in profile["parameters"].items():
        lo, hi = spec["range"]
        assert lo == 0.0, f"{name}: range lower bound should be 0.0"
        assert hi == 1.0, f"{name}: range upper bound should be 1.0"


def test_render_sec_equals_hold_plus_release(profile):
    probe = profile["probe"]
    expected = probe["hold_sec"] + probe["release_sec"]
    assert probe["render_sec"] == pytest.approx(expected), (
        f"render_sec ({probe['render_sec']}) != hold_sec + release_sec ({expected})"
    )


def test_probe_notes_ascending(profile):
    notes = profile["probe"]["notes"]
    assert len(notes) == 5
    assert notes == sorted(notes), "probe notes should be in ascending order"
    assert all(0 <= n <= 127 for n in notes), "all notes must be valid MIDI (0-127)"


def test_modulated_params_have_positive_importance(profile):
    modulated = [
        name for name, spec in profile["parameters"].items()
        if spec.get("importance", 0) > 0
    ]
    assert len(modulated) > 0, "No modulated parameters found"


def test_reset_values_in_range(profile):
    for name, value in profile["reset"].items():
        assert 0.0 <= float(value) <= 1.0, (
            f"Reset value for '{name}' ({value}) is outside [0, 1]"
        )

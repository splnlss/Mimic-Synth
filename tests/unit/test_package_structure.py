"""Structural tests for the mimic_synth package layout.

Verify that all public modules import correctly, key symbols are reachable,
and the config module exposes the expected path attributes.
These tests do not require DawDreamer, torch, or GPU.
"""
import importlib
import sys
from pathlib import Path

import pytest

# ── Package import smoke-tests ────────────────────────────────────────────────

MODULES = [
    "mimic_synth",
    "mimic_synth.config",
    "mimic_synth.s02_dataset.sampling",
    "mimic_synth.s02_dataset.quality",
    "mimic_synth.s02_dataset.manifest",
    "mimic_synth.s02_dataset.sequences",
    "mimic_synth.s02_dataset.build_dataset",
    "mimic_synth.s02_dataset.verify_dataset",
    "mimic_synth.s06_refine.mono_utils",
]

@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    """Every listed module must be importable without DawDreamer or torch."""
    mod = importlib.import_module(module)
    assert mod is not None


# ── config.py attribute coverage ─────────────────────────────────────────────

def test_config_path_attributes():
    from mimic_synth import config
    required = [
        "SAMPLE_RATE", "BUFFER_SIZE",
        "DATA_ROOT", "PROJECT_NAME", "PROJECT_DIR",
        "S01_DIR", "PROFILE_PATH", "CAL_PATH",
        "S02_DIR", "S02_WAV_DIR", "S02_PARQUET",
        "S03_DIR", "S03_PARQUET",
        "S04_DIR", "S04_EMBEDDINGS",
        "S05_DIR", "S05_RUNS_DIR",
        "S06_DIR", "S06_PATCHES_DIR",
        "TARGETS_DIR",
    ]
    for attr in required:
        assert hasattr(config, attr), f"config.{attr} missing"


def test_config_path_types():
    from mimic_synth import config
    path_attrs = [
        "DATA_ROOT", "PROJECT_DIR", "S01_DIR", "PROFILE_PATH",
        "S02_DIR", "S03_DIR", "S04_DIR", "S05_DIR",
    ]
    for attr in path_attrs:
        val = getattr(config, attr)
        assert isinstance(val, Path), f"config.{attr} should be Path, got {type(val)}"


def test_config_env_override(monkeypatch):
    """MIMIC_DATA_ROOT env var must be respected."""
    monkeypatch.setenv("MIMIC_DATA_ROOT", "/tmp/test_root")
    monkeypatch.setenv("MIMIC_PROJECT", "TestProject")
    # Flush both the submodule and the parent package so re-import is clean
    for key in list(sys.modules):
        if key == "mimic_synth" or key.startswith("mimic_synth."):
            del sys.modules[key]
    import mimic_synth.config as cfg
    assert cfg.DATA_ROOT == Path("/tmp/test_root")
    assert cfg.PROJECT_NAME == "TestProject"
    assert cfg.PROJECT_DIR == Path("/tmp/test_root/TestProject")
    # Flush again so later tests see the real config
    for key in list(sys.modules):
        if key == "mimic_synth" or key.startswith("mimic_synth."):
            del sys.modules[key]


def test_config_sample_rate():
    from mimic_synth import config
    assert config.SAMPLE_RATE == 48000
    assert config.BUFFER_SIZE == 512


# ── Package directory structure ───────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent.parent

def test_src_layout_exists():
    assert (REPO_ROOT / "src" / "mimic_synth").is_dir()
    assert (REPO_ROOT / "src" / "mimic_synth" / "__init__.py").exists()


@pytest.mark.parametrize("pkg", [
    "s01_capture", "s02_dataset", "s03_embed",
    "s04_surrogate", "s05_invert", "s06_refine",
])
def test_stage_package_exists(pkg):
    pkg_dir = REPO_ROOT / "src" / "mimic_synth" / pkg
    assert pkg_dir.is_dir(), f"Package dir missing: {pkg_dir}"
    assert (pkg_dir / "__init__.py").exists(), f"__init__.py missing in {pkg}"


def test_pyproject_toml_exists():
    assert (REPO_ROOT / "pyproject.toml").exists()


def test_makefile_exists():
    assert (REPO_ROOT / "Makefile").exists()


def test_tests_unit_dir():
    assert (REPO_ROOT / "tests" / "unit").is_dir()
    assert (REPO_ROOT / "tests" / "integration").is_dir()


def test_docs_dir():
    assert (REPO_ROOT / "docs").is_dir()


def test_scripts_dir():
    assert (REPO_ROOT / "scripts").is_dir()
    assert (REPO_ROOT / "scripts" / "calibrate_synth.py").exists()
    assert (REPO_ROOT / "scripts" / "enumerate_params.py").exists()


def test_old_stage_dirs_removed():
    """Old flat stage dirs must be gone after refactor.

    s06b_live is excluded: it may still be present if a background inversion
    job was running at refactor time and hasn't been cleaned up yet.
    """
    old_dirs = [
        "s02_capture", "s03_dataset", "s04_embed",
        "s05_surrogate", "s06_invert", "s07_refine",
    ]
    for d in old_dirs:
        assert not (REPO_ROOT / d).exists(), f"Old dir still present: {d}"


def test_defaults_py_removed():
    assert not (REPO_ROOT / "defaults.py").exists(), "defaults.py should be gone"


# ── Key active-code files exist ───────────────────────────────────────────────

@pytest.mark.parametrize("rel_path", [
    "src/mimic_synth/config.py",
    "src/mimic_synth/s01_capture/capture_v1_2.py",
    "src/mimic_synth/s02_dataset/build_dataset.py",
    "src/mimic_synth/s02_dataset/sampling.py",
    "src/mimic_synth/s03_embed/embed.py",
    "src/mimic_synth/s04_surrogate/model.py",
    "src/mimic_synth/s04_surrogate/train.py",
    "src/mimic_synth/s05_invert/stream_invert.py",
    "src/mimic_synth/s06_refine/audio_compare.py",
    "src/mimic_synth/s06_refine/vst_cmaes.py",
    "src/mimic_synth/s06_refine/vst_hill_climb.py",
    "src/mimic_synth/s06_refine/target_analysis.py",
])
def test_key_file_exists(rel_path):
    assert (REPO_ROOT / rel_path).exists(), f"Missing: {rel_path}"


# ── No synth-specific strings in library code ─────────────────────────────────

def test_no_hardcoded_synth_names_in_library():
    """Library code must not hardcode synth names (obxf, OB-Xf, crane, surge)."""
    forbidden = ["surge-synthesizer", "OB-Xf releases", "crane scream", "bird-call-funny"]
    lib_root = REPO_ROOT / "src" / "mimic_synth"
    violations = []
    for py in lib_root.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        # Skip archive files
        if py.name in ("capture_v1-1.py", "capture_v1.py"):
            continue
        text = py.read_text(errors="replace")
        for term in forbidden:
            if term in text:
                violations.append(f"{py.relative_to(REPO_ROOT)}: '{term}'")
    assert not violations, "Synth-specific strings found:\n" + "\n".join(violations)


# ── Sampling module correctness (fast, no GPU) ───────────────────────────────

def test_sampling_sobol_shape():
    from mimic_synth.s02_dataset.sampling import cold_start_vectors
    vecs = cold_start_vectors(m=4, d=5, seed=0)
    assert vecs.shape == (16, 5)
    assert vecs.min() >= 0.0
    assert vecs.max() <= 1.0


def test_quality_imports():
    from mimic_synth.s02_dataset.quality import analyse
    assert callable(analyse)


def test_manifest_imports():
    from mimic_synth.s02_dataset.manifest import new_manifest, write_manifest
    assert callable(new_manifest)
    assert callable(write_manifest)

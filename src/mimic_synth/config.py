"""Central path configuration for MimicSynth.

Single responsibility: know WHICH project is active and WHERE it lives.
Everything else is derived from the project's own s01_project-profile/project.yaml.

To switch projects:
    export MIMIC_DATA_ROOT=/mnt/d/Mimic-Synth-Data
    export MIMIC_PROJECT=Minimoog_Model_D
or edit the two fallback values below.

Stage paths are derived from PROJECT_DIR using the stage names defined in
project.yaml. If project.yaml is absent (e.g. during initial setup) they
fall back to the conventional names so the pipeline still runs.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Project pointer (the only two values to change when switching projects) ──
DATA_ROOT    = Path(os.environ.get("MIMIC_DATA_ROOT", "/mnt/d/Mimic-Synth-Data"))
PROJECT_NAME = os.environ.get("MIMIC_PROJECT",   "OB-X_Prototype")
PROJECT_DIR  = DATA_ROOT / PROJECT_NAME

# ── S01: project profile (all project-specific config lives here) ────────────
S01_DIR      = PROJECT_DIR / "s01_project-profile"
PROFILE_PATH = S01_DIR / "profile.yaml"
CAL_PATH     = S01_DIR / "calibration.npz"

# ── Stage paths — loaded from project.yaml if available, else conventional ──

def _load_project_yaml() -> dict:
    p = S01_DIR / "project.yaml"
    if not p.exists():
        return {}
    try:
        import yaml
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _stage(key: str, fallback: str) -> Path:
    """Return PROJECT_DIR / stages[key] from project.yaml, or PROJECT_DIR / fallback."""
    cfg = _load_project_yaml()
    rel = cfg.get("stages", {}).get(key, fallback)
    return PROJECT_DIR / rel


# Physical constants (never project-specific)
SAMPLE_RATE = 48000
BUFFER_SIZE = 512

# ── Stage directories ────────────────────────────────────────────────────────
S02_DIR     = _stage("s02_capture",   "s02_capture")
S02_WAV_DIR = S02_DIR / "wav"
S02_PARQUET = S02_DIR / "samples.parquet"

S03_DIR     = _stage("s03_dataset",   "s03_dataset")
S03_PARQUET = S03_DIR / "samples.parquet"

S04_DIR        = _stage("s04_embed",      "s04_embed")
S04_EMBEDDINGS = S04_DIR / "encodec_embeddings.npy"

S05_DIR      = _stage("s05_surrogate",  "s05_surrogate")
S05_RUNS_DIR = S05_DIR / "runs"

S06_DIR         = _stage("s06_invert",    "s06_invert")
S06_PATCHES_DIR = S06_DIR / "patches"

TARGETS_DIR = _stage("targets",          "targets")

_REPO_ROOT = Path(__file__).parent.parent.parent

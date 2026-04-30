"""
Project-wide defaults for MimicSynth.

Production code should prefer reading from the profile YAML (e.g.
profile["probe"]["sample_rate"]) when a profile is available. These
constants exist for scripts that run before a profile is loaded
(enumerate_params.py) and for unit tests that don't use a real profile.
"""
from pathlib import Path

SAMPLE_RATE = 48000
BUFFER_SIZE = 512

# ── External data storage ─────────────────────────────────────────────────────
# All pipeline outputs (WAVs, parquet, embeddings, model runs, patches) live
# under PROJECT_DIR on the external drive. Only source code lives in this repo.
DATA_ROOT    = Path("/mnt/d/Mimic-Synth-Data")
PROJECT_NAME = "OB-X_Prototype"
PROJECT_DIR  = DATA_ROOT / PROJECT_NAME

# s02 — raw capture outputs
S02_DIR     = PROJECT_DIR / "s02_capture"
S02_WAV_DIR = S02_DIR / "wav"
S02_PARQUET = S02_DIR / "samples.parquet"

# s03 — quality-gated dataset
S03_DIR     = PROJECT_DIR / "s03_dataset"
S03_PARQUET = S03_DIR / "samples.parquet"

# s04 — EnCodec embeddings
S04_DIR        = PROJECT_DIR / "s04_embed"
S04_EMBEDDINGS = S04_DIR / "encodec_embeddings.npy"

# s05 — surrogate model runs
S05_DIR      = PROJECT_DIR / "s05_surrogate"
S05_RUNS_DIR = S05_DIR / "runs"

# s06 — inversion outputs
S06_DIR         = PROJECT_DIR / "s06_invert"
S06_PATCHES_DIR = S06_DIR / "patches"

# target audio files used as inversion inputs (not pipeline outputs)
TARGETS_DIR = PROJECT_DIR / "targets"

# ── Repo-side paths (stay with the code, never move) ─────────────────────────
_REPO_ROOT   = Path(__file__).parent
PROFILE_PATH = _REPO_ROOT / "s01_profiles" / "obxf.yaml"

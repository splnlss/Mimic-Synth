# MimicSynth Refactoring Plan

Two-step migration to a `src` layout with consistent stage numbering across repo and project folder.

**Guiding principles:**
- Stage numbers (`s01_`, `s02_`, …) are kept — they communicate pipeline order
- Numbers reset from 01 in the repo (profile becomes a config file, not a stage)
- Repo and project folder use identical stage names
- `inputs/` and `outputs/` replace the ad-hoc `targets/` and patches locations
- No functionality is changed during this refactor — rename and rewire only

---

## Stage mapping

| Old repo dir | New repo package | Old data dir | New data dir |
|---|---|---|---|
| `s01_profiles/` | _(moves to project as `profile.yaml`)_ | — | — |
| `s02_capture/` | `src/mimic_synth/s01_capture/` | `s02_capture/` | `s01_capture/` |
| `s03_dataset/` | `src/mimic_synth/s02_dataset/` | `s03_dataset/` | `s02_dataset/` |
| `s04_embed/` | `src/mimic_synth/s03_embed/` | `s04_embed/` | `s03_embed/` |
| `s05_surrogate/` | `src/mimic_synth/s04_surrogate/` | `s05_surrogate/` | `s04_surrogate/` |
| `s06_invert/` + `s06b_live/` | `src/mimic_synth/s05_invert/` | _(output only)_ | `outputs/` |
| `s07_refine/` | `src/mimic_synth/s06_refine/` | _(output only)_ | `outputs/` |
| `defaults.py` | `src/mimic_synth/config.py` | — | `project.yaml` |
| `build_instructions/` | `docs/` | — | — |
| `calibrate_synth.py` etc. | `scripts/` | — | — |
| `targets/` | — | `targets/` | `inputs/` |

---

## Step 1 — Repository restructuring

**Goal:** Move all source code into `src/mimic_synth/` with new stage names, update all imports, add `pyproject.toml` and `Makefile`. Data folder is untouched in this step.

### 1.1 Create the new directory tree

```
src/
  mimic_synth/
    __init__.py              (empty)
    config.py                (was defaults.py)
    s01_capture/             (was s02_capture/)
      __init__.py
      capture_v1_2.py        (current version)
      capture_v1.py          (archive)
      capture_v1-1.py        (archive)
    s02_dataset/             (was s03_dataset/)
      __init__.py
      build_dataset.py
      manifest.py
      quality.py
      sampling.py
      sequences.py
      verify_dataset.py
    s03_embed/               (was s04_embed/)
      __init__.py
      embed.py
      index_dataset.py
      verify_embeddings.py
    s04_surrogate/           (was s05_surrogate/)
      __init__.py
      model.py
      train.py
      verify_surrogate.py
    s05_invert/              (merge of s06_invert/ + s06b_live/)
      __init__.py
      grad_search.py         (from s06_invert/)
      cmaes_search.py        (from s06_invert/)
      invert.py              (from s06_invert/)
      validate.py            (from s06_invert/)
      render_stream.py       (from s06b_live/ — active version)
      stream_invert.py       (from s06b_live/ — v4, active version)
      stream_invert_offline.py  (from s06_invert/stream_invert.py — legacy)
      render_stream_legacy.py   (from s06_invert/render_stream.py — legacy)
      pitch_mapping.py       (from s06b_live/)
      profile_constraints.py (from s06b_live/)
      build_manifest.py      (from s06b_live/)
    s06_refine/              (was s07_refine/)
      __init__.py
      audio_compare.py
      mono_utils.py
      target_analysis.py
      vst_cmaes.py
      vst_hill_climb.py
tests/
  unit/                      (move non-integration tests here)
  integration/               (move @pytest.mark.integration tests here)
docs/                        (was build_instructions/)
scripts/                     (root-level utility scripts)
  calibrate_synth.py
  enumerate_params.py
  analyze_audio.py
  check_audio.py
  compare_crane.py
  convert_to_mono.py
  estimate_pitch.py
  invert_and_render.py
  render_crane.py
  test_render.py
  verify_crane.py
  verify_render.py
  run_s02_s03_s04.sh
notebooks/                   (empty, for future exploratory work)
pyproject.toml               (new)
Makefile                     (new)
README.md
CLAUDE.md
LICENSE
SECURITY.md
.gitignore                   (update to exclude src/__pycache__ etc.)
```

**Remove from root after migration:**
- `defaults.py`
- `s01_profiles/`
- `s02_capture/`
- `s03_dataset/`
- `s04_embed/`
- `s05_surrogate/`
- `s06_invert/`
- `s06b_live/`
- `s07_refine/`
- `build_instructions/`
- All root-level `*.py` utility scripts
- Stale root-level logs: `s03_build_output.log`, `s03_rebuild.log`, `s03_run.log`, `s04_embed_resume.log`, `s05_surrogate/train.log`
- `run_s02_s03_s04.sh`

### 1.2 Write `src/mimic_synth/config.py`

Identical API to current `defaults.py` — same exported names, same derived paths. Downstream code that does `import defaults as _defs` changes to `from mimic_synth import config as _defs`. No logic changes.

```python
# src/mimic_synth/config.py
import os
from pathlib import Path

SAMPLE_RATE = 48000
BUFFER_SIZE = 512

DATA_ROOT    = Path(os.environ.get("MIMIC_DATA_ROOT", "/mnt/d/Mimic-Synth-Data"))
PROJECT_NAME = os.environ.get("MIMIC_PROJECT", "OB-X_Prototype")
PROJECT_DIR  = DATA_ROOT / PROJECT_NAME

S01_DIR     = PROJECT_DIR / "s01_capture"
S01_WAV_DIR = S01_DIR / "wav"
S01_PARQUET = S01_DIR / "samples.parquet"

S02_DIR     = PROJECT_DIR / "s02_dataset"
S02_PARQUET = S02_DIR / "samples.parquet"

S03_DIR        = PROJECT_DIR / "s03_embed"
S03_EMBEDDINGS = S03_DIR / "encodec_embeddings.npy"

S04_DIR      = PROJECT_DIR / "s04_surrogate"
S04_RUNS_DIR = S04_DIR / "runs"

INPUTS_DIR  = PROJECT_DIR / "inputs"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

_REPO_ROOT   = Path(__file__).parent.parent.parent  # src/mimic_synth/config.py → repo root
PROFILE_PATH = PROJECT_DIR / "profile.yaml"         # profile lives in project folder
```

Note: `S02_DIR` etc. now refer to the *new* stage numbers matching the data folder rename in Step 2. `PROFILE_PATH` now points into the project folder, not the repo.

### 1.3 Update all intra-package imports

**Pattern:** `import defaults as _defs` → `from mimic_synth import config as _defs`  
**Pattern:** `from defaults import X` → `from mimic_synth.config import X`

**Cross-stage absolute imports to rewrite:**

| File | Old import | New import |
|---|---|---|
| `s05_invert/invert.py` | `from s05_surrogate.model import Surrogate` | `from mimic_synth.s04_surrogate.model import Surrogate` |
| `s05_invert/invert.py` | `from s06_invert.grad_search import grad_invert` | `from .grad_search import grad_invert` |
| `s05_invert/invert.py` | `from s06_invert.cmaes_search import cmaes_invert` | `from .cmaes_search import cmaes_invert` |
| `s05_invert/stream_invert_offline.py` | `from s05_surrogate.model import Surrogate` | `from mimic_synth.s04_surrogate.model import Surrogate` |
| `s05_invert/stream_invert_offline.py` | `from s06_invert.invert import _load_surrogate, _estimate_best_note` | `from .invert import _load_surrogate, _estimate_best_note` |
| `s05_invert/validate.py` | `from s05_surrogate.model import Surrogate` | `from mimic_synth.s04_surrogate.model import Surrogate` |
| `s05_invert/validate.py` | `from s06_invert.grad_search import grad_invert` | `from .grad_search import grad_invert` |
| `s05_invert/validate.py` | `from s06_invert.cmaes_search import cmaes_invert` | `from .cmaes_search import cmaes_invert` |
| `s05_invert/stream_invert.py` | `from s05_surrogate.model import Surrogate` | `from mimic_synth.s04_surrogate.model import Surrogate` |
| `s05_invert/stream_invert.py` | `from s06_invert.invert import _load_surrogate` | `from .invert import _load_surrogate` |
| `s05_invert/stream_invert_offline.py` | `from s06_invert.grad_search import grad_invert` _(deferred, inside function)_ | `from .grad_search import grad_invert` |
| `s05_invert/stream_invert.py` | `from s07_refine.mono_utils import ensure_mono` _(deferred)_ | `from mimic_synth.s06_refine.mono_utils import ensure_mono` |
| `s05_invert/stream_invert.py` | `from s07_refine.audio_compare import compute_ap_features, compute_sp_features, score_audio_composite, _lufs_normalize` _(deferred, ×2)_ | `from mimic_synth.s06_refine.audio_compare import ...` |
| `s05_invert/stream_invert.py` | `from s07_refine.vst_hill_climb import hill_climb` _(deferred)_ | `from mimic_synth.s06_refine.vst_hill_climb import hill_climb` |
| `s05_invert/stream_invert.py` | `from s07_refine.vst_cmaes import cmaes_refine` _(deferred)_ | `from mimic_synth.s06_refine.vst_cmaes import cmaes_refine` |
| `s05_invert/stream_invert.py` | `from s07_refine.target_analysis import analyze_target, suggest_x0, print_analysis` _(deferred)_ | `from mimic_synth.s06_refine.target_analysis import ...` |
| `s05_invert/stream_invert.py` | `from s07_refine.audio_compare import render_trajectory` _(deferred)_ | `from mimic_synth.s06_refine.audio_compare import render_trajectory` |
| `s06_refine/vst_cmaes.py` | `from s07_refine.audio_compare import ...` | `from .audio_compare import ...` |
| `s06_refine/vst_hill_climb.py` | `from s07_refine.audio_compare import render_and_score` | `from .audio_compare import render_and_score` |
| `s02_dataset/build_dataset.py` | `from s02_capture.capture_v1_2 import ...` _(deferred)_ | `from mimic_synth.s01_capture.capture_v1_2 import ...` |
| `s02_dataset/sequences.py` | `from s02_capture.capture_v1_2 import ...` _(deferred)_ | `from mimic_synth.s01_capture.capture_v1_2 import ...` |

**Tests — all imports to rewrite:**

| Test file | Old import | New import |
|---|---|---|
| `test_capture_unit.py` | `from s02_capture.capture_v1_2 import ...` | `from mimic_synth.s01_capture.capture_v1_2 import ...` |
| `test_capture_v1_2.py` | `from s02_capture...` | `from mimic_synth.s01_capture...` |
| `test_embed.py` | `from s04_embed.embed import Embedder, _stft` | `from mimic_synth.s03_embed.embed import Embedder, _stft` |
| `test_embed_index.py` | `from s04_embed.index_dataset import ...` | `from mimic_synth.s03_embed.index_dataset import ...` |
| `test_embed_index.py` | `from s04_embed.verify_embeddings import ...` | `from mimic_synth.s03_embed.verify_embeddings import ...` |
| `test_integration.py` | `from s02_capture.capture_v1_2 import ...` | `from mimic_synth.s01_capture.capture_v1_2 import ...` |
| `test_manifest.py` | `from s03_dataset.manifest import ...` | `from mimic_synth.s02_dataset.manifest import ...` |
| `test_quality.py` | `from s03_dataset.quality import ...` | `from mimic_synth.s02_dataset.quality import ...` |
| `test_s05_surrogate.py` | `from s05_surrogate.model import Surrogate` | `from mimic_synth.s04_surrogate.model import Surrogate` |
| `test_s07.py` | `from s07_refine.mono_utils import ...` | `from mimic_synth.s06_refine.mono_utils import ...` |
| `test_s07.py` | `from s07_refine.target_analysis import ...` | `from mimic_synth.s06_refine.target_analysis import ...` |
| `test_sampling.py` | `from s03_dataset.sampling import ...` | `from mimic_synth.s02_dataset.sampling import ...` |
| `test_sequences.py` | `from s03_dataset.sequences import ...` | `from mimic_synth.s02_dataset.sequences import ...` |
| `test_verify_dataset.py` | `from s03_dataset.verify_dataset import ...` | `from mimic_synth.s02_dataset.verify_dataset import ...` |

### 1.4 Write `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "mimic-synth"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[project.scripts]
mimic-capture  = "mimic_synth.s01_capture.capture_v1_2:main"
mimic-build    = "mimic_synth.s02_dataset.build_dataset:main"
mimic-embed    = "mimic_synth.s03_embed.index_dataset:main"
mimic-train    = "mimic_synth.s04_surrogate.train:main"
mimic-invert   = "mimic_synth.s05_invert.stream_invert:main"
mimic-refine   = "mimic_synth.s06_refine.vst_cmaes:main"
mimic-verify-dataset    = "mimic_synth.s02_dataset.verify_dataset:main"
mimic-verify-embeddings = "mimic_synth.s03_embed.verify_embeddings:main"
mimic-verify-surrogate  = "mimic_synth.s04_surrogate.verify_surrogate:main"

[tool.pytest.ini_options]
pythonpath = ["src"]
markers = ["integration: tests that require DawDreamer and OB-Xf installed"]
testpaths = ["tests"]

[tool.setuptools.package-dir]
"" = "src"
```

Note: `capture_v1_2.py` needs a `main()` entry point added (trivially wraps existing `__main__` block).

### 1.5 Write `Makefile`

```makefile
CONDA = conda run -n mimic-synth

.PHONY: install capture build embed train invert refine \
        verify-dataset verify-embeddings verify-surrogate \
        test test-unit test-integration lint

install:
	pip install -e ".[dev]"

capture:
	$(CONDA) mimic-capture

build:
	$(CONDA) mimic-build

embed:
	$(CONDA) mimic-embed --pool mean --batch-size 64

train:
	$(CONDA) mimic-train

invert:
	$(CONDA) mimic-invert --target $(TARGET)

refine:
	$(CONDA) mimic-refine --target $(TARGET) --cmaes

verify-dataset:
	$(CONDA) mimic-verify-dataset

verify-embeddings:
	$(CONDA) mimic-verify-embeddings

verify-surrogate:
	$(CONDA) mimic-verify-surrogate

test:
	.venv/bin/pytest tests/unit -m "not integration"

test-integration:
	$(CONDA) python -m pytest tests/integration -m integration

lint:
	.venv/bin/ruff check src/ tests/
```

### 1.6 Reorganise `tests/`

Move tests to match the unit/integration split:
- `tests/unit/` — all current tests except `test_integration.py`
- `tests/integration/` — `test_integration.py`

Add `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`.

### 1.7 Update root-level scripts before moving

These scripts import from stage packages and must be updated before moving to `scripts/`:

| Script | Imports to update |
|---|---|
| `verify_render.py` | `from s04_embed.embed` → `from mimic_synth.s03_embed.embed` |
| `verify_crane.py` | `from s04_embed.embed` → `from mimic_synth.s03_embed.embed` |
| `invert_and_render.py` | `from s04_embed.embed`, `from s05_surrogate.model`, `from s06_invert.*` → new paths |

### 1.8 Move docs and scripts

- `build_instructions/*.md` → `docs/*.md` (keep filenames)
- Root-level utility `.py` files → `scripts/`
- `run_s02_s03_s04.sh` → `scripts/`
- `s07_refine/obxf_calibration.npz` → leave in place for now; Step 2 moves it to the project folder

### 1.8 Update `.gitignore`

Add:
```
src/__pycache__/
src/mimic_synth/__pycache__/
*.log
logs/
```

### 1.9 Install and verify

```bash
# In conda env (for pipeline code)
conda activate mimic-synth
pip install -e .

# Run unit tests with .venv
.venv/bin/pip install -e .
.venv/bin/pytest tests/unit -m "not integration" -v

# Smoke-test imports
conda run -n mimic-synth python -c "
from mimic_synth.s02_dataset.build_dataset import main
from mimic_synth.s03_embed.index_dataset import main
from mimic_synth.s04_surrogate.model import Surrogate
from mimic_synth.s05_invert.stream_invert import main
from mimic_synth.s06_refine.vst_cmaes import main
print('All imports OK')
"
```

---

## Step 2 — Project data folder restructuring

**Goal:** Rename data stage directories to match the new numbering, rename `targets/` → `inputs/`, create `outputs/`, `reports/`, `logs/`, move profile and calibration file into the project, and introduce `project.yaml`.

> ⚠️ **Critical dependency:** The `s02_dataset/samples.parquet` (old `s03_dataset`) stores **absolute paths** into `s01_capture/wav/` (old `s02_capture/wav/`). Renaming the capture directory invalidates those paths. The dataset must be rebuilt from the renamed capture directory after the rename (step 2.3).

### 2.1 Rename stage directories

Run from `/mnt/d/Mimic-Synth-Data/OB-X_Prototype/`:

```bash
mv s02_capture  s01_capture
mv s03_dataset  s02_dataset
mv s04_embed    s03_embed
mv s05_surrogate s04_surrogate
```

### 2.2 Rename workspace folders

```bash
mv targets inputs
mkdir -p outputs reports logs
```

### 2.3 Rebuild s02_dataset (parquet paths are now stale)

The parquet stored absolute paths to `s02_capture/wav/` — those paths no longer exist. Rebuild from the renamed capture:

```bash
conda activate mimic-synth
mimic-build   # reads S01_DIR from config.py, writes to S02_DIR
mimic-verify-dataset
```

### 2.4 Re-run s03_embed (embeddings are indexed against the old parquet)

```bash
mimic-embed --pool mean --batch-size 64
mimic-verify-embeddings
```

### 2.5 Move project-specific files out of repo

```bash
# Profile — copy to project folder, keep template in docs/
cp /path/to/repo/s01_profiles/obxf.yaml \
   /mnt/d/Mimic-Synth-Data/OB-X_Prototype/profile.yaml

# Calibration file — project-specific, not repo material
mv /path/to/repo/s07_refine/obxf_calibration.npz \
   /mnt/d/Mimic-Synth-Data/OB-X_Prototype/s04_surrogate/obxf_calibration.npz
```

Update `config.py` calibration path:
```python
CALIBRATION_PATH = S04_DIR / "obxf_calibration.npz"
```

Update any hardcoded references to `s07_refine/obxf_calibration.npz` in `s06_refine/` to use `_defs.CALIBRATION_PATH`.

### 2.6 Write `project.yaml`

```yaml
# /mnt/d/Mimic-Synth-Data/OB-X_Prototype/project.yaml
synth: OB-Xf
version: 1.0.3
profile: profile.yaml        # relative to this file

capture:
  m: 14                      # Sobol exponent → 16,384 vectors
  sample_rate: 48000

stages:
  s01_capture:   s01_capture/
  s02_dataset:   s02_dataset/
  s03_embed:     s03_embed/
  s04_surrogate: s04_surrogate/
  inputs:        inputs/
  outputs:       outputs/
  reports:       reports/
  logs:          logs/
```

This file is informational for now — `config.py` still derives paths from env vars. Future work: make `config.py` load this file when `MIMIC_PROJECT` is set to a directory.

### 2.7 Final project folder layout

```
OB-X_Prototype/
  project.yaml
  profile.yaml
  PROJECT_STATUS.md
  s01_capture/
    samples.parquet
    wav/
  s02_dataset/
    samples.parquet
    manifest.yaml
  s03_embed/
    encodec_embeddings.npy
    encodec_embeddings_done.npy
  s04_surrogate/
    runs/
      run_20260429_145056/   (stale — retrain after s03 rebuild)
    obxf_calibration.npz
  inputs/                    (was targets/)
    crane-scream.wav
    bird-call-funny.wav
    ...
  outputs/                   (was patches in s06b_live/)
    crane-scream/
      20260514_143022/
        best_patch.yaml
        rendered.wav
        trajectory.yaml
        cmaes_log.yaml
        hill_climb_log.yaml
  reports/
  logs/
```

### 2.8 Update `PROJECT_STATUS.md`

Update all stage references and paths to reflect new numbering.

### 2.9 Move existing output sessions

```bash
# Move any existing session output from old locations to outputs/
# s06b_live typically wrote to /mnt/d/Mimic-Synth-Data/OB-X_Prototype/ directly
# or to a timestamped subfolder — locate and move:
find /mnt/d/Mimic-Synth-Data/OB-X_Prototype -name "best_patch.yaml" \
  ! -path "*/outputs/*" -exec echo {} \;
# Manually move confirmed session folders to outputs/<target-stem>/<timestamp>/
```

### 2.10 Verify end-to-end

```bash
conda activate mimic-synth

# Verify dataset and embeddings reflect new paths
mimic-verify-dataset
mimic-verify-embeddings

# Smoke-test inversion on a known target
mimic-invert --target /mnt/d/Mimic-Synth-Data/OB-X_Prototype/inputs/crane-scream.wav
```

---

## Checklist

### Step 1 — Repo

- [ ] `src/mimic_synth/` tree created
- [ ] `config.py` written with env-var support
- [ ] All stage packages moved and renamed
- [ ] `s05_invert/` merged from `s06_invert/` + `s06b_live/`
- [ ] All cross-stage absolute imports rewritten (see table in 1.3)
- [ ] All test imports rewritten (see table in 1.3)
- [ ] `pyproject.toml` written
- [ ] `Makefile` written
- [ ] `pip install -e .` succeeds in both envs
- [ ] Unit tests pass: `.venv/bin/pytest tests/unit -m "not integration"`
- [ ] Smoke-test imports pass
- [ ] `docs/` populated from `build_instructions/`
- [ ] `scripts/` populated from root-level utilities
- [ ] Old top-level dirs removed
- [ ] `.gitignore` updated
- [ ] `CLAUDE.md` and `README.md` updated with new commands

### Step 2 — Data folder

- [ ] Stage dirs renamed (`s01_capture` … `s04_surrogate`)
- [ ] `targets/` → `inputs/`
- [ ] `outputs/`, `reports/`, `logs/` created
- [ ] `s02_dataset` rebuilt (`mimic-build`) — parquet paths now point to `s01_capture`
- [ ] `s03_embed` rebuilt (`mimic-embed`)
- [ ] `profile.yaml` copied to project folder
- [ ] `obxf_calibration.npz` moved to `s04_surrogate/`
- [ ] `config.py` `CALIBRATION_PATH` updated
- [ ] `project.yaml` written
- [ ] Existing session outputs moved to `outputs/`
- [ ] `PROJECT_STATUS.md` updated
- [ ] End-to-end inversion smoke test passes

---

## Step 3 — README rewrite

**Goal:** Replace the current README with one that follows best practices from RealPython, Shaun Fulton (Medium), and Markdown Visualizer: clear elevator pitch up front, badges, table of contents, features, installation, CLI usage, tech stack, project layout, and contributing. All commands updated to the new `mimic-*` CLI. Deep technical content (pipeline internals, algorithm diagrams) moved to `docs/`.

**Guiding principle:** The README is a landing page, not a manual. A first-time visitor should understand what the project does and how to run it within 60 seconds. Algorithm depth lives in `docs/`.

---

### 3.1 What's wrong with the current README

| Problem | Location in current file |
|---|---|
| No elevator pitch — first section is a stage table | Line 3 |
| No badges | Missing entirely |
| No table of contents | Missing (file is 504 lines) |
| No Features section — unique capabilities buried in prose | Lines 177–248 |
| No Tech Stack section | Missing |
| No Contributing section | Missing |
| `defaults.py` referenced throughout — stale after Step 1 | Lines 43–48 |
| All `python -m s0N_xxx` commands stale after Step 1 | Lines 109, 115, 122, 129–135, 148–149, etc. |
| `conda run -n mimic-synth python -m s06b_live...` stale | Lines 195, 204, 213 |
| Stage folder paths (`s02_capture/`, `s06b_live/`) stale after Steps 1–2 | Lines 9–17, 268–334 |
| Algorithm diagrams (S06b, S07 ASCII flowcharts) — excellent but too early | Lines 370–451 |
| Quality roadmap at the bottom — useful but deeply nested | Lines 462–497 |
| Installation split across three non-adjacent sections | Lines 22–33, 66–93 |
| Data storage config references `defaults.py` by name | Lines 43–48 |

---

### 3.2 New README structure

```
# MimicSynth
[badges]
[one-sentence description]
[2–3 sentence description]

## Table of Contents
## Features
## Tech Stack
## Installation
## Configuration
## Quick Start
## Pipeline
## Usage — Inverting a Sound
## Project Structure
## Extending to a New Synth
## Running Tests
## Roadmap
## License
```

---

### 3.3 Section-by-section content plan

#### Header

```markdown
# MimicSynth

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20WSL2-lightgrey)]()

A Python pipeline that listens to a synthesizer sound and finds the knob settings that reproduce it.
```

Follow with a 2–3 sentence description covering: what it does (parameter-to-timbre mapping via ML), what it targets (VST synths via DawDreamer, hardware synths via MIDI), and the key output (a `best_patch.yaml` loadable directly onto the synth).

Optionally: a before/after spectrogram image or waveform comparison showing target vs rendered audio (add to `docs/assets/` and embed here). If not available at time of writing, leave a placeholder `![demo](docs/assets/demo.png)`.

---

#### Table of Contents

Manual, anchored links to all H2 sections. Required given length.

---

#### Features

Bullet list of what makes MimicSynth stand out — written for someone who found it on GitHub, not for someone already deep in the code:

- **Full inversion pipeline** — from raw audio to synth patch in under 30 seconds (fast mode) or ~15 minutes (full CMA-ES quality)
- **Synth-agnostic design** — profile YAML describes any VST or hardware synth; only the profile changes between instruments
- **Surrogate neural network** — differentiable proxy for the synth trained on Sobol-sampled patches; enables gradient-based search without rendering through the VST
- **Real-VST refinement** — CMA-ES closes the surrogate-to-real gap by evaluating every candidate through the actual plugin
- **Oscillator config and interval scouting** — discrete outer search over waveform types and Osc 2 pitch intervals before continuous CMA-ES
- **Composite scoring** — EnCodec 40% + auraloss MRSTFT 25% + aperiodicity 20% + spectral envelope 15%, all LUFS-normalised
- **Automatic pitch tracking** — pyworld F0 at 5ms resolution with autocorrelation fallback; writes MIDI pitch bend file
- **Calibrated filter cutoff** — measured OB-Xf sweep (420–4134 Hz) replaces heuristic formulas
- **Checkpoint/resume** — every long-running stage checkpoints and resumes automatically when interrupted
- **Best result so far: 0.029 cosine distance** on a crane scream (saw+pulse config, CMA-ES global mode)

---

#### Tech Stack

Table covering every major dependency with a one-line reason for use:

| Library | Purpose |
|---|---|
| [DawDreamer](https://github.com/DBraun/DawDreamer) | Headless VST host — renders audio from plugin + MIDI + automation |
| [EnCodec](https://github.com/facebookresearch/encodec) | 128-d audio embeddings (48 kHz pre-quantiser latents) as perceptual distance metric |
| [auraloss](https://github.com/csteinmetz1/auraloss) | Multi-resolution STFT loss for composite scoring |
| [pyworld](https://github.com/JeremyCCHsu/Python-WORLD) | F0/aperiodicity/spectral envelope analysis (WORLD vocoder) |
| [cma](https://github.com/CMA-ES/pycma) | CMA-ES optimiser for real-VST parameter search |
| [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | LUFS normalisation before all scoring comparisons |
| [torchcrepe](https://github.com/maxrmorrison/torchcrepe) | Neural pitch detection (fallback / cross-check) |
| PyTorch | Surrogate MLP training and gradient-based inversion |
| pandas / pyarrow | Parquet storage for captured samples and per-frame parameters |
| scipy (Sobol) | Quasi-random parameter sampling for dataset generation |

Python: 3.11 (conda env for pipeline), 3.14 (venv for tests).  
Platform: Linux / WSL2. GPU: CUDA recommended (RTX 3070+) for embedding and surrogate training.

---

#### Installation

**Prerequisites** (new sub-section, currently scattered):

```markdown
### Prerequisites

- Conda (Miniconda or Anaconda)
- OB-Xf VST3 installed: download from the [v1.0.3 release](https://github.com/surge-synthesizer/OB-Xf/releases/tag/v1.0.3), install to `~/.vst3/`
- CUDA-capable GPU recommended for embedding and training (CPU works but is slow)
- External drive or sufficient disk space (~50 GB for a full production capture)
```

**Install steps** — currently split across lines 30–33 and 66–93, needs consolidation:

```markdown
### Clone and install

git clone https://github.com/splnlss/mimic-synth
cd mimic-synth

# Pipeline env (DawDreamer + PyTorch + CUDA)
conda env create -f environment.yml
conda activate mimic-synth
pip install -e .

# Test env (CPU-only, no DawDreamer)
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Note: `environment.yml` needs to be created as part of this step (currently missing from repo — `requirements.txt` exists but is pip-only and incomplete for DawDreamer).

---

#### Configuration

New section — replaces the current "Data storage" section (lines 37–62) but without referencing `defaults.py` by name:

```markdown
### Configure project location

Set the project directory via environment variable (or accept the default):

export MIMIC_DATA_ROOT=/mnt/d/Mimic-Synth-Data   # default
export MIMIC_PROJECT=OB-X_Prototype               # default

# Or point directly at a project.yaml:
export MIMIC_PROJECT=/path/to/OB-X_Prototype/project.yaml
```

Show the project folder layout (`inputs/`, `outputs/`, `reports/`, `logs/`, stage dirs) — the new structure from Step 2, not the old `s02_capture/` paths.

---

#### Quick Start

3-command block — the hook for someone who just wants to try it:

```markdown
## Quick Start

# 1. Verify the plugin loads
mimic-capture --list-params

# 2. Invert a sound (fast mode, ~30s)
mimic-invert --target inputs/my-sound.wav --hill-iterations 0

# 3. Find the output
ls outputs/my-sound/
# best_patch.yaml  rendered.wav  trajectory.yaml
```

---

#### Pipeline

Keep the stage table (currently lines 6–18) but update folder paths to new stage numbers and merge S06/S06b into S05, S07 into S06:

| Stage | Package | CLI | Purpose |
|---|---|---|---|
| S01 | — | — | Synth profile (`profile.yaml` in project folder) |
| S02 | `mimic_synth.s01_capture` | `mimic-capture` | Render (params × notes) to WAV via DawDreamer |
| S03 | `mimic_synth.s02_dataset` | `mimic-build` | Quality-gate captures, write manifest |
| S04 | `mimic_synth.s03_embed` | `mimic-embed` | Pre-compute 128-d EnCodec embeddings |
| S05 | `mimic_synth.s04_surrogate` | `mimic-train` | Train surrogate MLP f(params, note) → embedding |
| S06 | `mimic_synth.s05_invert` | `mimic-invert` | Invert target audio to patch (surrogate + α-refinement) |
| S07 | `mimic_synth.s06_refine` | `mimic-refine` | Real-VST refinement: hill-climb + CMA-ES |

Under this, one sub-section **"Running the full pipeline"** with a Makefile-style command sequence:

```bash
make capture          # S01 capture (long — days for production M=14)
make build            # S02 dataset
make embed            # S03 embeddings
make train            # S04 surrogate
make verify-dataset
make verify-embeddings
make verify-surrogate
```

---

#### Usage — Inverting a Sound

This replaces the current "6b + 7" section (lines 166–248). Keep the mode table and quality ladder. Update all commands from `conda run -n mimic-synth python -m s06b_live.stream_invert` to `mimic-invert`:

```bash
# Fast (~30s): surrogate + α-refinement only
mimic-invert --target inputs/my-sound.wav --hill-iterations 0

# Standard (~5 min): + hill-climb
mimic-invert --target inputs/my-sound.wav

# Full quality (~15 min): + CMA-ES
mimic-invert --target inputs/my-sound.wav --cmaes

# Full quality, more iterations (~20 min)
mimic-invert --target inputs/my-sound.wav --cmaes --cmaes-maxiter 30
```

Keep the quality ladder table and CMA-ES tuning flags table — they are genuinely useful.

Move the ASCII algorithm flowcharts (S06b data flow, S07 hill-climb/CMA-ES pseudocode, currently lines 370–451) to `docs/algorithm.md` and link from here with one line: _"For a detailed walkthrough of the inversion and refinement algorithms, see [docs/algorithm.md](docs/algorithm.md)."_

---

#### Project Structure

Replace the current flat file listing (lines 266–335) with the new post-refactor layout. Two sub-sections:

**Repo:**
```
mimic-synth/
  src/mimic_synth/
    s01_capture/      Capture rig — DawDreamer + Sobol sampling
    s02_dataset/      Quality gates, manifest, dataset builder
    s03_embed/        EnCodec embedding (128-d mean pooled)
    s04_surrogate/    Surrogate MLP — train, verify
    s05_invert/       Inversion — grad search, CMA-ES, stream invert
    s06_refine/       Real-VST refinement — hill-climb, CMA-ES
    config.py         Project path resolver (reads MIMIC_PROJECT env var)
  tests/
    unit/
    integration/
  docs/               Algorithm docs, per-stage design notes
  scripts/            One-off utilities (calibrate_synth.py, enumerate_params.py…)
  Makefile
  pyproject.toml
```

**Project data folder:**
```
OB-X_Prototype/
  project.yaml        Project config
  profile.yaml        Synth profile (OB-Xf parameters, probe notes, etc.)
  s01_capture/        Raw WAVs + samples.parquet
  s02_dataset/        Quality-gated dataset + manifest.yaml
  s03_embed/          encodec_embeddings.npy
  s04_surrogate/      Model runs + obxf_calibration.npz
  inputs/             Target audio files for inversion
  outputs/            Inversion results (best_patch.yaml, rendered.wav, …)
  reports/            Verification reports
  logs/               Pipeline logs
```

---

#### Extending to a New Synth

Short new section — one of the project's key design goals is synth-agnosticism. Currently buried in CLAUDE.md architecture notes. 3–4 steps:

1. Copy `docs/profiles/obxf.yaml` as a template
2. Define parameters, ranges, importance weights, probe notes
3. Set `MIMIC_PROJECT` to a new project folder with the new profile
4. Run the pipeline — all stages adapt automatically (input_dim is read from the profile)

Link to `docs/01-profile.md` for the full profile format reference.

---

#### Running Tests

Update from current (lines 252–263):

```bash
# Unit tests (no DawDreamer or GPU needed)
make test

# Integration tests (requires DawDreamer + OB-Xf installed)
make test-integration

# Specific stage tests
.venv/bin/pytest tests/unit/test_sampling.py -v
conda run -n mimic-synth python -m pytest tests/unit/test_s05_surrogate.py -v
```

---

#### Roadmap

Move the quality roadmap (currently lines 462–497) here but condense it to a checklist format — the current version is written as an internal spec, not a public roadmap. Group into three themes: Richer synthesis, Better analysis, Better optimization. Keep the "Done" strikethroughs.

---

#### License

Keep as-is (MIT, one line).

---

### 3.4 Content that moves out of README

| Current content | Moves to |
|---|---|
| S06b algorithm flowchart (lines 370–408) | `docs/algorithm.md` |
| S07 hill-climb/CMA-ES pseudocode (lines 410–451) | `docs/algorithm.md` |
| PINNED_PARAMS design notes (lines 241–246) | `docs/algorithm.md` |
| Profile YAML format section (lines 339–358) | `docs/01-profile.md` |
| Detailed `surrogate_note` explanation (lines 244–245) | `docs/algorithm.md` |
| Quality roadmap detailed bullets (lines 462–497) | Condensed roadmap stays; full details move to `docs/roadmap.md` |

---

### 3.5 New content to write from scratch

| Section | Notes |
|---|---|
| Elevator pitch (2–3 sentences) | Emphasise: audio → knobs, any synth, gradient + CMA-ES, open-source |
| Badges row | Python version, license, tests, platform |
| Table of Contents | Manual anchors to all H2 headings |
| Features bullet list | 8–10 bullets, written for a GitHub visitor not a developer |
| Tech Stack table | Every major dep with one-line rationale |
| Quick Start (3 commands) | Uses `mimic-invert`, reads from `inputs/`, writes to `outputs/` |
| Configuration section | `MIMIC_PROJECT` env var, project.yaml location |
| "Extending to a new synth" | 4-step numbered guide |
| `environment.yml` | Needs to be created; referenced in Installation |

---

### 3.6 Commands reference — old → new

Every command in the README must be updated. Full mapping:

| Old command | New command |
|---|---|
| `python enumerate_params.py` | `mimic-capture --list-params` |
| `python s02_capture/capture_v1_2.py` | `mimic-capture` or `make capture` |
| `python -m s03_dataset.build_dataset` | `mimic-build` or `make build` |
| `python -m s03_dataset.build_dataset --m 14` | `mimic-build --m 14` |
| `python -m s03_dataset.verify_dataset` | `mimic-verify-dataset` or `make verify-dataset` |
| `python -m s04_embed.index_dataset --pool mean --batch-size 64` | `mimic-embed --pool mean --batch-size 64` or `make embed` |
| `python -m s04_embed.verify_embeddings` | `mimic-verify-embeddings` or `make verify-embeddings` |
| `python -m s05_surrogate.train` | `mimic-train` or `make train` |
| `python -m s05_surrogate.verify_surrogate` | `mimic-verify-surrogate` or `make verify-surrogate` |
| `conda run -n mimic-synth python -m s06b_live.stream_invert --target X --hill-iterations 0` | `mimic-invert --target X --hill-iterations 0` |
| `conda run -n mimic-synth python -m s06b_live.stream_invert --target X` | `mimic-invert --target X` |
| `conda run -n mimic-synth python -m s06b_live.stream_invert --target X --cmaes` | `mimic-invert --target X --cmaes` |
| `conda run -n mimic-synth python calibrate_synth.py` | `python scripts/calibrate_synth.py` |
| `.venv/bin/pytest -m "not integration"` | `make test` |
| `conda run -n mimic-synth python -m pytest tests/test_s05_surrogate.py ...` | `make test-integration` or direct pytest |

---

### 3.7 Step 3 checklist

- [ ] `environment.yml` created (referenced in Installation)
- [ ] `docs/algorithm.md` written (receives flowcharts from README)
- [ ] `docs/01-profile.md` written (receives profile format section)
- [ ] `docs/roadmap.md` written (receives detailed quality roadmap)
- [ ] `docs/assets/` created for images (placeholder or actual demo asset)
- [ ] README header rewritten: title, badges, elevator pitch
- [ ] Table of Contents added
- [ ] Features section written (8–10 bullets, newcomer-friendly)
- [ ] Tech Stack table written
- [ ] Installation consolidated into Prerequisites + Clone + Install steps
- [ ] Configuration section written (`MIMIC_PROJECT` env var)
- [ ] Quick Start section written (3 commands, `mimic-invert`)
- [ ] Pipeline section updated (new stage numbers + CLI commands)
- [ ] Usage section updated (all `mimic-invert` commands, keep quality ladder)
- [ ] Project Structure updated (post-refactor repo + data folder layout)
- [ ] "Extending to a new synth" section written
- [ ] Running Tests updated (Makefile targets)
- [ ] Roadmap condensed (checklist format)
- [ ] All 17 old commands replaced with new CLI equivalents (see §3.6)
- [ ] `defaults.py` references removed
- [ ] Old stage folder paths (`s02_capture/`, `s06b_live/`) removed
- [ ] Removed content verified as present in `docs/`

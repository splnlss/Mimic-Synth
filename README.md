# MimicSynth

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20WSL2-lightgrey)]()

A Python pipeline that listens to a sound and finds the synthesizer knob settings that reproduce it. Give it a bird call, a crane scream, or any timbral target — it returns a `best_patch.yaml` you can load directly onto a compatible VST.

MimicSynth works by training a differentiable surrogate model of a synth's parameter-to-timbre mapping, then inverting that model with gradient descent and CMA-ES. The surrogate runs in milliseconds; the final refinement stage drives the real plugin for highest fidelity.

![demo](docs/assets/demo.png)

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Pipeline](#pipeline)
- [Usage — Inverting a Sound](#usage--inverting-a-sound)
- [Project Structure](#project-structure)
- [Extending to a New Synth](#extending-to-a-new-synth)
- [Running Tests](#running-tests)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features

- **Full inversion pipeline** — audio to synth patch in under 30 seconds (fast mode) or ~15 minutes (full CMA-ES quality)
- **Synth-agnostic design** — a profile YAML describes any VST or hardware synth; only the profile changes between instruments
- **Surrogate neural network** — FiLM+ResBlocks MLP with per-layer note conditioning; differentiable proxy that enables gradient-based search without rendering through the VST
- **Real-VST refinement** — CMA-ES closes the surrogate-to-real gap by evaluating every candidate through the actual plugin
- **Oscillator config and interval scouting** — discrete outer search over waveform types and Osc 2 pitch intervals before continuous CMA-ES
- **Composite audio scoring** — EnCodec 33% + auraloss MRSTFT 22% + aperiodicity 17% + spectral envelope 13% + envelope shape 15%, all LUFS-normalised
- **Automatic pitch tracking** — pyworld F0 at 5 ms resolution with autocorrelation fallback; writes MIDI pitch-bend output
- **Checkpoint / resume** — every long-running stage checkpoints and resumes automatically on restart
- **Best result so far: 0.029 cosine distance** on a crane-scream target (saw+pulse config, CMA-ES global mode)

---

## Tech Stack

| Library | Purpose |
|---|---|
| [DawDreamer](https://github.com/DBraun/DawDreamer) | Headless VST host — renders audio from plugin + MIDI + automation |
| [EnCodec](https://github.com/facebookresearch/encodec) | 128-d audio embeddings (48 kHz pre-quantiser latents) as perceptual distance metric |
| [LAION-CLAP](https://github.com/LAION-AI/CLAP) | Optional 512-d audio embeddings for surrogate training (`--embed-model clap`) |
| [auraloss](https://github.com/csteinmetz1/auraloss) | Multi-resolution STFT loss — composite scoring and auxiliary surrogate training loss |
| [pyworld](https://github.com/JeremyCCHsu/Python-WORLD) | F0 / aperiodicity / spectral-envelope analysis (WORLD vocoder) |
| [cma](https://github.com/CMA-ES/pycma) | CMA-ES optimiser for real-VST parameter search |
| [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | LUFS normalisation before all scoring comparisons |
| [torchcrepe](https://github.com/maxrmorrison/torchcrepe) | Neural pitch detection (fallback / cross-check) |
| PyTorch | Surrogate MLP training and gradient-based inversion |
| pandas / pyarrow | Parquet storage for captured samples and per-frame parameters |
| scipy (Sobol) | Quasi-random parameter sampling for dataset generation |

Python 3.11 (conda env for pipeline), 3.14 (venv for tests). Platform: Linux / WSL2. GPU: CUDA recommended (RTX 3070+) for embedding and surrogate training.

---

## Installation

### Prerequisites

- Conda (Miniconda or Anaconda)
- A supported VST3 synthesizer installed at the system default path (e.g. `~/.vst3/`)
- CUDA-capable GPU recommended for embedding and training (CPU works but is slow)
- ~50 GB disk space for a full production capture (M=14, 16 notes)

### Clone and install

```bash
git clone https://github.com/splnlss/mimic-synth
cd mimic-synth

# Pipeline env (DawDreamer + PyTorch + CUDA)
conda env create -f environment.yml
conda activate mimic-synth
pip install -e .

# Test env (CPU-only, no DawDreamer or GPU needed)
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

---

## Configuration

All data paths are resolved from two environment variables:

```bash
export MIMIC_DATA_ROOT=/mnt/d/Mimic-Synth-Data   # default
export MIMIC_PROJECT=OB-X_Prototype               # default project name
```

The project folder at `$MIMIC_DATA_ROOT/$MIMIC_PROJECT/` must contain a `profile.yaml` describing the synth. All stage outputs are written under this folder. See [docs/01-profile.md](docs/01-profile.md) for the profile format.

Project data folder layout:

```
OB-X_Prototype/
  profile.yaml        Synth profile (parameters, probe config, behavior notes)
  PROJECT_STATUS.md   Source of truth for stage completion and quality results
  s01_capture/        Raw WAVs + samples.parquet
  s02_dataset/        Quality-gated dataset + manifest.yaml
  s03_embed/          encodec_embeddings.npy (+ optional clap, mrstft)
  s04_surrogate/      Model runs/ + calibration.npz
  inputs/             Target audio files for inversion
  outputs/            Inversion results per target (best_patch.yaml, rendered.wav, …)
  reports/            Verification reports
  logs/               Pipeline logs
```

---

## Quick Start

```bash
# 1. Capture a small dev dataset (~30 min for M=10, 16 notes)
conda activate mimic-synth && mimic-capture --m 10

# 2. Build, embed, and train the surrogate (~1 hour on GPU)
mimic-build && mimic-embed && mimic-train

# 3. Invert a target sound (fast mode, ~30s)
mimic-invert --target inputs/my-sound.wav --hill-iterations 0

# 4. Find the output
ls outputs/my-sound/
# best_patch.yaml  rendered.wav  trajectory.yaml  settings.md
```

---

## Pipeline

| Stage | Package | CLI | Purpose |
|---|---|---|---|
| S01 | — | — | Synth profile (`profile.yaml` in project folder) |
| S02 | `mimic_synth.s01_capture` | `mimic-capture` | Render (params × notes) to WAV via DawDreamer |
| S03 | `mimic_synth.s02_dataset` | `mimic-build` | Quality-gate captures, write dataset + manifest |
| S04 | `mimic_synth.s03_embed` | `mimic-embed` | Pre-compute 128-d EnCodec embeddings |
| S05 | `mimic_synth.s04_surrogate` | `mimic-train` | Train surrogate MLP f(params, note) → embedding |
| S06 | `mimic_synth.s05_invert` | `mimic-invert` | Invert target audio (surrogate + α-refinement) |
| S07 | `mimic_synth.s06_refine` | `mimic-refine` | Real-VST refinement: hill-climb + CMA-ES |

### Running the full pipeline

```bash
conda activate mimic-synth
make capture          # S01 capture (days for production M=14; M=10 for dev)
make build            # S02 dataset
make embed            # S03 embeddings
make train            # S05 surrogate
make verify-dataset
make verify-embeddings
make verify-surrogate
```

Data flow:

```
profile.yaml → sampling (Sobol) → capture (DawDreamer WAVs) → quality gates
                                                                     ↓
                                                         samples.parquet + WAVs
                                                                     ↓
                                                          EnCodec embedding → .npy
                                                                     ↓
                                               surrogate MLP training (params, note → latent)
                                                                     ↓
                                      target.wav → gradient inversion → CMA-ES refinement
                                                                     ↓
                                                         best_patch.yaml + rendered.wav
```

---

## Usage — Inverting a Sound

MimicSynth offers three quality modes. All write output to `outputs/<target-stem>/<timestamp>/`.

```bash
# Fast (~30s): surrogate inversion + α-refinement only
# Good for iteration and preview
mimic-invert --target inputs/my-sound.wav --hill-iterations 0

# Standard (~5 min): + hill-climb (per-param coordinate descent on real VST)
# Good for daily use on most targets
mimic-invert --target inputs/my-sound.wav

# Full quality (~15 min): + CMA-ES (osc config + interval scouting, IPOP restart)
# Best for final-quality renders
mimic-invert --target inputs/my-sound.wav --cmaes

# Full quality, more iterations (~20 min)
mimic-invert --target inputs/my-sound.wav --cmaes --cmaes-maxiter 30

# Maximum quality (~30 min)
mimic-invert --target inputs/my-sound.wav --cmaes \
    --cmaes-sigma0 0.12 --cmaes-popsize 24 --cmaes-maxiter 30
```

### Quality ladder (crane-scream target)

| Mode | Real-synth cosine distance | Time |
|---|---|---|
| Surrogate only | ~0.21 | ~30s |
| + α-refinement | ~0.10 | ~30s |
| + Hill-climb | ~0.094 | ~5 min |
| + CMA-ES | **~0.029–0.042** | ~15 min |

Stereo targets are automatically converted to mono (L+R average).

For a detailed walkthrough of the inversion and refinement algorithms, see [docs/algorithm.md](docs/algorithm.md).

---

## Project Structure

### Repository

```
mimic-synth/
  src/mimic_synth/
    config.py             Project path resolver (reads MIMIC_PROJECT env var)
    s01_capture/          Capture rig — DawDreamer + Sobol sampling
    s02_dataset/          Quality gates, manifest, dataset builder
    s03_embed/            EnCodec embedding (128-d mean-pooled)
    s04_surrogate/        Surrogate MLP — train, verify
    s05_invert/           Inversion — grad search, stream invert, α-refinement
    s06_refine/           Real-VST refinement — hill-climb, CMA-ES
  tests/
    unit/                 Fast tests (no DawDreamer or GPU)
    integration/          Tests that require DawDreamer + VST installed
  docs/                   Algorithm docs, per-stage design notes, profiles
    profiles/             Synth profile templates
  scripts/                One-off utilities (calibrate_synth.py, enumerate_params.py…)
  Makefile
  pyproject.toml
  README.md
  CLAUDE.md
  LICENSE
```

### Project data folder (outside repo)

```
OB-X_Prototype/
  project.yaml            Project config (synth name, Sobol M, stage paths)
  profile.yaml            Synth profile (parameters, probe notes, behavior)
  PROJECT_STATUS.md       Stage completion + quality metrics
  s01_capture/            Raw WAVs + samples.parquet (~50 GB for M=14)
  s02_dataset/            Quality-gated dataset + manifest.yaml
  s03_embed/              encodec_embeddings.npy (+ optional clap, mrstft)
  s04_surrogate/
    runs/                 Model checkpoints (state_dict.pt, surrogate.onnx)
    calibration.npz       Measured filter cutoff / envelope curves
  inputs/                 Target audio files for inversion
  outputs/                Per-target inversion results
    my-sound/
      20260518_143022/
        best_patch.yaml
        rendered.wav
        trajectory.yaml
        cmaes_log.yaml
        settings.md
  reports/                Verification CSV reports
  logs/                   Pipeline run logs
```

---

## Extending to a New Synth

MimicSynth is designed to work with any VST or hardware synthesizer. Only the profile changes between instruments:

1. Copy `docs/profiles/template.yaml` to your project folder as `profile.yaml`
2. Define parameters, value ranges, importance weights, and probe notes for your synth
3. Set `MIMIC_PROJECT` to a new project folder (or `MIMIC_DATA_ROOT` if on a different drive)
4. Run the pipeline — all stages adapt automatically (input dimensionality is derived from the profile)

```bash
export MIMIC_PROJECT=MyNewSynth
mimic-capture --m 10   # generates profile-appropriate Sobol vectors
```

See [docs/01-profile.md](docs/01-profile.md) for the full profile format reference, including parameter importance weighting, log-scale transforms, discrete categories, and reset values.

---

## Running Tests

```bash
# Unit tests (no DawDreamer or GPU needed) — use .venv
make test

# All tests with coverage
.venv/bin/pytest tests/unit -m "not integration" -v

# Specific stage tests
.venv/bin/pytest tests/unit/test_sampling.py -v
.venv/bin/pytest tests/unit/test_quality.py -v

# Integration tests (requires DawDreamer + VST installed) — use conda
make test-integration

# Surrogate / invert tests (require torch) — use conda
conda run -n mimic-synth python -m pytest tests/unit/test_s05_surrogate.py -v
```

Tests are organised as:

| Test file | What it covers |
|---|---|
| `test_sampling.py` | Sobol sampling shape, importance weighting, log-scale |
| `test_quality.py` | Silence / clipping / stuck / bleed detectors |
| `test_manifest.py` | YAML manifest round-trip |
| `test_sequences.py` | Temporal parameter trajectory generation |
| `test_embed.py` | EnCodec shape and dtype (synthetic audio, GPU auto-skip) |
| `test_embed_index.py` | index_dataset checkpoint/resume logic |
| `test_s05_surrogate.py` | Surrogate forward pass, gradient flow (torch) |
| `test_s07.py` | audio_compare scoring, mono_utils, target_analysis |
| `test_package_structure.py` | Module imports, config attributes, directory layout |
| `test_integration.py` | Full render via DawDreamer + OB-Xf (requires VST) |

---

## Roadmap

### Richer synthesis
- [ ] Hardware synth support via MIDI + audio interface loopback (profile-driven, no DawDreamer)
- [ ] Multi-timbral profiles (split / layer modes)
- [ ] Parameter macro groups (e.g. "brightness", "warmth") in profile

### Better analysis
- [x] pyworld F0 / aperiodicity / spectral envelope
- [x] Calibrated filter cutoff from measured sweep
- [x] Amplitude envelope scoring (Pearson correlation)
- [ ] Onset / transient feature extraction for percussive targets
- [ ] Longer-context temporal modelling (LFO shapes, filter sweeps)

### Better optimisation
- [x] CMA-ES with IPOP restart
- [x] Oscillator config scouting (saw / pulse / saw+pulse)
- [x] Osc 2 interval scouting (7 discrete musical intervals)
- [x] Calibrated filter-cutoff warm-start
- [ ] Population seeding from nearest-neighbour in embedding space
- [ ] Multi-objective Pareto front (timbral match vs. playability)

### Deployment
- [ ] ONNX export for inference without PyTorch
- [ ] nn~ / Max/MSP integration (S08)
- [ ] Web demo

---

## License

MIT. See [LICENSE](LICENSE).

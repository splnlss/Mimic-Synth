# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MimicSynth** (`MimicSynth/`) is an ML audio dataset pipeline that captures synthesizer parameter-timbre mappings. It renders parameter vectors through a VST plugin (OB-Xf) via DawDreamer and produces labeled WAV datasets for training inverse models. The pipeline is stage-based, with directories prefixed `s01_` through `s04_`.

## Environments

Two Python environments are in use — they are not interchangeable:

| Env | Activation | Used for |
|-----|-----------|----------|
| `.venv` (Python 3.14) | `source .venv/bin/activate` | CPU stages: tests, capture, dataset build/verify |
| `mimic-synth` conda (Python 3.11) | `conda activate mimic-synth` | GPU stages: s04_embed (torch 2.11+cu130) |

`torch` is **only** in the conda env. Running embed commands with `.venv/bin/python` will fail with `ModuleNotFoundError`.

## Common Commands

CPU stages — run from `MimicSynth/` with the local venv:

```bash
source .venv/bin/activate

# Run all tests
.venv/bin/pytest

# Skip integration tests (no DawDreamer/OB-Xf needed)
.venv/bin/pytest -m "not integration"

# Run a single test file / test
.venv/bin/pytest tests/test_sampling.py
.venv/bin/pytest tests/test_sampling.py::test_cold_start_vectors_shape

# Capture audio (from s02_capture/)
cd s02_capture && ../.venv/bin/python capture_v1_2.py

# Build dataset
.venv/bin/python -m s03_dataset.build_dataset --profile s01_profiles/obxf.yaml --m 10 --out s03_dataset/data/

# Verify dataset
.venv/bin/python -m s03_dataset.verify_dataset --dataset s02_capture/data/ --profile s01_profiles/obxf.yaml

# List VST parameters
.venv/bin/python enumerate_params.py
```

GPU stages — activate the conda env first:

```bash
conda activate mimic-synth

# Embed dataset (pre-compute EnCodec embeddings, supports resume)
# --batch-size controls clips per GPU forward pass; 32 is conservative, try 64-128 on 4090
python -m s04_embed.index_dataset --dataset s02_capture/data/ --out s04_embed/data/ --pool mean --batch-size 32

# Force a specific GPU (e.g. use 3070 instead of 4090)
python -m s04_embed.index_dataset --dataset s02_capture/data/ --out s04_embed/data/ --pool mean --device cuda:1

# Verify embeddings
python -m s04_embed.verify_embeddings --embeddings s04_embed/data/encodec_embeddings.npy --dataset s02_capture/data/
```

## Architecture

### Pipeline stages

- **s01_profiles/** — YAML synth profiles defining parameters, ranges, importance weights, probe config (note, velocity, timing), reset values, and platform-specific plugin paths. `obxf.yaml` is the only profile.
- **s02_capture/** — Capture rig scripts that load a profile, instantiate DawDreamer with the VST, generate Sobol-sampled parameter vectors, and render each (vector × note) combination to WAV. `capture_v1_2.py` is the current version (settle-before-patch-change, per-note settle with adaptive threshold, self-noise measurement, hard_reset fallback).
- **s03_dataset/** — Python package with modules run via `python -m`:
  - `sampling.py` — Sobol quasi-random sampling with scrambling, importance weighting (filter/scale modes), and log-scale transforms
  - `quality.py` — Per-capture quality gates: silence, clipping, stuck notes, prev_bleed detection
  - `manifest.py` — YAML manifest for reproducibility tracking
  - `sequences.py` — Temporal sequence generation (interpolated parameter trajectories for frame-to-frame dynamics training)
  - `build_dataset.py` — CLI orchestrator
  - `verify_dataset.py` — Post-hoc dataset auditor (exits non-zero if any failure rate >1%)
- **s04_embed/** — Audio embedding via EnCodec 48 kHz encoder (continuous pre-quantiser latents):
  - `embed.py` — `Embedder` class: `encodec_embed()` for pooled vectors (128-d mean / 256-d meanstd), `encodec_sequence()` for frame-wise [128, T], `mrstft_feats()` for auxiliary multi-res STFT
  - `index_dataset.py` — CLI to pre-compute `encodec_embeddings.npy` aligned 1-to-1 with `samples.parquet`. Supports checkpoint/resume/overwrite (same pattern as capture_v1_2.py)
  - `verify_embeddings.py` — Post-hoc embedding auditor: shape alignment, completeness, NaN/Inf, latent stats, nearest/farthest neighbor spot-check

### GPU / hardware

- **GPUs**: RTX 4090 24 GB (cuda:0, primary), RTX 3070 8 GB (cuda:1). `embed.py` selects `cuda` (= cuda:0) by default.
- **torch.compile**: `Embedder.__init__` compiles `self.enc.encoder` with `mode="reduce-overhead"` on CUDA. First forward pass incurs a ~30s JIT warmup; all subsequent calls in the same process are faster. Does not compile on CPU (`Embedder(device="cpu")` is eager, used by tests).
- **fp16 autocast**: encoder forward passes run in `torch.float16` on CUDA via `torch.amp.autocast`. EnCodec pre-quantiser latents are safe in fp16; outputs are cast back to float32 before returning numpy arrays.
- **Batch size**: `index_dataset.py` accumulates WAVs and embeds in batches (`--batch-size`, default 32). EnCodec 48 kHz is ~15M params (~60 MB). The 4090 has headroom for batches of 64–128. Increase if GPU utilization is low.
- **Device override**: pass `--device cuda:1` to `index_dataset.py` to route to the 3070 (e.g., if the 4090 is occupied).

### Key patterns

- **DawDreamer API**: Parameters are set by index, not name. Use `build_name_index(synth)` to get `{name: index}` mapping, then `synth.set_parameter(idx, value)`. MIDI is managed via `synth.clear_midi()` / `synth.add_midi_note()`. There is no `add_midi_message` API in dawdreamer 0.8.x.
- **Settle loop** (v1.2): Renders silent chunks between captures until synth output peak < threshold (max 10s). Runs *before* patch change (draining old patch's tail) and between notes within a vector. Uses adaptive threshold `max(1e-4, self_noise * 2)` so self-oscillating patches don't stall. Falls back to `hard_reset` (graph reload + second settle pass) when settle times out.
- **Sobol sampling**: Uses `scipy.stats.qmc.Sobol` with `random_base2(m=m)` — always pass the exponent `m`, not a sample count, to avoid silent truncation.
- **Quality thresholds**: `prev_bleed` checks first 20ms for peak > `max(0.01, self_noise * 2)`; `silence` is peak < 1e-4; `clipping` is 5+ samples > 0.99; `stuck` is tail RMS within 6dB of sustain.
- **Self-noise baseline**: `measure_self_noise()` renders 200ms with no MIDI after each patch load. Stored in parquet as `self_noise`. Used by both the settle loop (adaptive threshold) and the bleed detector (threshold floor) to handle patches that self-oscillate or have LFO artifacts.
- **Profile importance field**: `importance: 0` excludes a parameter from sampling entirely. Values 0–1 control sampling density.
- **Sample rate**: Standardized at 48 kHz.
- **Checkpoint/resume pattern**: Long-running scripts (`capture_v1_2.py`, `index_dataset.py`) checkpoint periodically and prompt `[c]ontinue / [o]verwrite / [a]bort` on restart when prior output exists. New scripts that process the full dataset should follow this pattern.
- **EnCodec embedding**: 48 kHz model produces 128-d latents at 150 Hz frame rate (not 75 Hz — that's the 24 kHz model). Pre-quantiser continuous latents via `model.encoder(x)`, not quantised codes. Latents are unbounded (not L2-normalised) — do not normalise to unit sphere.

### Data flow

```
obxf.yaml → sampling.py (Sobol vectors) → capture_v1_2.py (DawDreamer render) → WAV + samples.parquet
                                                                                        ↓
                                                                              quality.py (per-capture gates)
                                                                                        ↓
                                                                              verify_dataset.py (aggregate audit)
                                                                                        ↓
                                                                              embed.py (EnCodec) → encodec_embeddings.npy
```

## Testing

- Unit tests in `tests/` cover sampling, quality, manifest, sequences, capture logic, embeddings, and the embedding verifier
- Integration tests (`test_integration.py`) require DawDreamer + OB-Xf VST installed; marked with `@pytest.mark.integration`
- Embed tests (`test_embed.py`, `test_embed_index.py`) use synthetic tones — no capture data or GPU required
- Quality gate thresholds in `quality.py` are tuned for specific audio characteristics — don't change without understanding the capture pipeline

## External References

- **Build instructions**: `/Users/splnlss/Obsidian/Splnlss_Vault/Soundz/Build Instructions/` — Obsidian vault with per-stage design docs for the full pipeline (01 Profile through 08 Package). Consult when implementing new stages.

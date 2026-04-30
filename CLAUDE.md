# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MimicSynth** (`MimicSynth/`) is an ML audio dataset pipeline that captures synthesizer parameter-timbre mappings. It renders parameter vectors through a VST plugin (OB-Xf) via DawDreamer and produces labeled WAV datasets for training inverse models. The pipeline is stage-based, with directories prefixed `s01_` through `s08_`.

## Stage Status

| Stage | Name | Status | Notes |
|-------|------|--------|-------|
| s01 | Profiles | ✅ Complete | `obxf.yaml` only |
| s02 | Capture | 🔄 Active | M=14 production run in progress — 15 notes, ~10,648/16,384 vectors (~65%), ETA ~6h |
| s03 | Dataset | ✅ Complete | Dev-scale (5,120 rows, M=10, 5 notes); prod rebuild pending s02 completion |
| s04 | Embed | ✅ Complete | `encodec_embeddings.npy` (16,995 rows, from s02 partial); rebuild pending s03 prod |
| s05 | Surrogate | ✅ Complete | `run_20260429_145056`: val_loss 0.0061; cos-sim 0.9988 |
| s06 | Invert | ⚠️ WIP | Search + render both working. `invert.py` tested on bird call (score 0.0011, note 84). Full test-split validation not yet run. |
| s07 | Refine | 🔲 Not started | Closes gap between surrogate score and real-synth score |
| s08 | Package | 🔲 Not started | ONNX + nn~ / Max integration |

## Environments

Two Python environments are in use — they are not interchangeable:

| Env | Activation | Used for |
|-----|-----------|----------|
| `.venv` (Python 3.14) | `source .venv/bin/activate` | Unit tests and CPU-only utilities (no DawDreamer, no torch) |
| `mimic-synth` conda (Python 3.11) | `conda activate mimic-synth` | All pipeline stages: capture (DawDreamer), embed, surrogate, invert (torch 2.11+cu130) |

`torch` and `dawdreamer` are **only** in the conda env. Running pipeline commands with `.venv/bin/python` will fail with `ModuleNotFoundError`.

## Common Commands

Tests — use `.venv` (no DawDreamer or torch needed for non-integration tests):

```bash
# Run all non-integration tests
.venv/bin/pytest -m "not integration"

# Run a single test file
.venv/bin/pytest tests/test_sampling.py

# s05/s06 tests require torch — run under conda
conda run -n mimic-synth python -m pytest tests/test_s05_surrogate.py tests/test_invert.py -v
```

Pipeline stages — all require conda:

```bash
conda activate mimic-synth

# Capture audio (s02)
cd s02_capture && python capture_v1_2.py

# Build dataset — post-hoc from existing capture (no DawDreamer render)
python -m s03_dataset.build_dataset \
    --profile s01_profiles/obxf.yaml \
    --from-capture s02_capture/data/ \
    --out s03_dataset/data/

# Build dataset — live capture (M=10 dev, M=14 production)
python -m s03_dataset.build_dataset \
    --profile s01_profiles/obxf.yaml --m 10 --out s03_dataset/data/

# Verify dataset
python -m s03_dataset.verify_dataset \
    --dataset s03_dataset/data/ --profile s01_profiles/obxf.yaml

# Embed dataset (s04) — --batch-size 64-128 on 4090
# Current embeddings built from s02_capture/data/ (16,995 rows)
python -m s04_embed.index_dataset \
    --dataset s02_capture/data/ --out s04_embed/data/ --pool mean --batch-size 64

# Verify embeddings
python -m s04_embed.verify_embeddings \
    --embeddings s04_embed/data/encodec_embeddings.npy --dataset s03_dataset/data/

# Train surrogate (s05)
python -m s05_surrogate.train \
    --dataset s03_dataset/data/samples.parquet \
    --embeddings s04_embed/data/encodec_embeddings.npy \
    --out s05_surrogate/runs/

# Verify surrogate — full round-trip check on held-out test split
python -m s05_surrogate.verify_surrogate \
    --checkpoint s05_surrogate/runs/run_20260429_145056/state_dict.pt \
    --dataset s03_dataset/data/samples.parquet \
    --embeddings s04_embed/data/encodec_embeddings.npy \
    --profile s01_profiles/obxf.yaml

# Invert a target audio clip (s06) — pitch-detects dominant note, then grad+CMA-ES search
python -m s06_invert.invert \
    --target path/to/target.wav \
    --surrogate s05_surrogate/runs/run_20260429_145056/state_dict.pt \
    --profile s01_profiles/obxf.yaml \
    --out patches/

# Validate inversion on held-out test split (~85 min for 512 samples)
python -m s06_invert.validate \
    --surrogate s05_surrogate/runs/run_20260429_145056/state_dict.pt \
    --dataset s03_dataset/data/samples.parquet \
    --embeddings s04_embed/data/encodec_embeddings.npy \
    --profile s01_profiles/obxf.yaml --stability

# List VST parameters (requires conda for DawDreamer)
python enumerate_params.py
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
  - `build_dataset.py` — CLI orchestrator; `--from-capture` mode (post-hoc, no DawDreamer) and `--m` mode (live capture)
  - `verify_dataset.py` — Post-hoc dataset auditor (exits non-zero if any failure rate >1%)
- **s04_embed/** — Audio embedding via EnCodec 48 kHz encoder (continuous pre-quantiser latents):
  - `embed.py` — `Embedder` class: `encodec_embed()` for pooled vectors (128-d mean / 256-d meanstd), `encodec_sequence()` for frame-wise [128, T], `mrstft_feats()` for auxiliary multi-res STFT
  - `index_dataset.py` — CLI to pre-compute `encodec_embeddings.npy` aligned 1-to-1 with `samples.parquet`. Supports checkpoint/resume/overwrite
  - `verify_embeddings.py` — Post-hoc embedding auditor: shape alignment, completeness, NaN/Inf, latent stats, nearest/farthest neighbor spot-check
- **s05_surrogate/** ✅ — Forward model: differentiable neural approximation of `f(params, note) → EnCodec latent`. Best run: `runs/run_20260429_145056/` (val_loss 0.0061, test-split cos-sim mean 0.9988, p10 0.9971).
  - `model.py` — `Surrogate` MLP (v1: 4-layer GELU, hidden=512, ~1M params) and `SurrogateDataset`
  - `train.py` — CLI training loop: AdamW + AMP, MSE + 0.3×cosine loss, random_split(seed=42) 80/10/10, checkpoint/resume, ONNX export
  - `verify_surrogate.py` — Three checks: (1) round-trip on held-out test split (cos-sim ≥ 0.9), (2) per-parameter sweep from reset values, (3) gradient non-degeneracy. All pass on current checkpoint.
- **s06_invert/** ⚠️ WIP — Offline patch search: target audio → best synth parameter vector.
  - `grad_search.py` — Multi-start Adam through the frozen surrogate, clamped to `[0,1]` every step. Tested: score 0.0002 on held-out capture.
  - `cmaes_search.py` — CMA-ES refinement seeded from grad result. Requires `pip install cma`.
  - `invert.py` — End-to-end CLI: embeds target, brute-forces over profile notes, writes `patches/<stem>/best_patch.yaml + candidates.parquet + target_embedding.npy`. Tested on `613846_bird-call-funny.wav` (score 0.0011, note 84).
  - `render_stream.py` — Renders a `best_patch.yaml` through OB-Xf via DawDreamer. Uses correct API (`get_plugin_parameter_size`, `get_parameter_name`). Verified: produces `rendered.wav` + `rendered_normalized.wav` (cos-sim 0.0002 on bird call target).
  - `stream_invert.py` — Sliding-window inversion for longer or real-time targets; segments audio and inverts each window independently.
  - `validate.py` — Batch validation on held-out test split; reproduces train.py split (random_split seed=42). **Not yet run on full test split.**

### Known issues / WIP items

- **s06 render step**: ✅ Working. `render_stream.py` loads `best_patch.yaml`, applies params via DawDreamer (`get_plugin_parameter_size` / `get_parameter_name` / `set_parameter`), renders WAV. Key API note: no `get_num_parameters` in DawDreamer 0.8.x — use `get_plugin_parameter_size()`.
- **Automation Monitoring**: A cron job `mimic-synth-monitor` polls s02 capture progress every 10m.
- **s05 train/val/test split**: code uses `random_split(seed=42)` (not hash-modulo as build doc specifies). Equivalent for a fixed dataset; would drift if new captures are added.
- **s05 no cosine LR decay or early stopping**: build doc calls for both. Current checkpoint converged well anyway (val_loss 0.0061).
- **s06 validate not yet run**: full 512-sample test-split validation (~85 min) not yet executed.
- **Dataset is dev-scale**: current data is M=10 (5,120 samples). Production target is M=14 (81,920 samples).

### GPU / hardware

- **GPUs**: RTX 4090 24 GB (cuda:0, primary), RTX 3070 8 GB (cuda:1). `embed.py` selects `cuda` (= cuda:0) by default.
- **torch.compile**: `Embedder.__init__` compiles `self.enc.encoder` with `mode="reduce-overhead"` on CUDA. First forward pass incurs a ~30s JIT warmup; all subsequent calls in the same process are faster. Does not compile on CPU (`Embedder(device="cpu")` is eager, used by tests).
- **fp16 autocast**: encoder forward passes run in `torch.float16` on CUDA via `torch.amp.autocast`. EnCodec pre-quantiser latents are safe in fp16; outputs are cast back to float32 before returning numpy arrays.
- **Batch size**: `index_dataset.py` accumulates WAVs and embeds in batches (`--batch-size`, default 32). EnCodec 48 kHz is ~15M params (~60 MB). The 4090 has headroom for batches of 64–128. Increase if GPU utilization is low.
- **Device override**: pass `--device cuda:1` to route to the 3070 (e.g., if the 4090 is occupied).

### Key patterns

- **DawDreamer API**: Parameters are set by index, not name. Use `synth.get_plugin_parameter_size()` to get the param count, then `synth.get_parameter_name(i)` to build a `{name: index}` dict. Set values with `synth.set_parameter(idx, value)`. MIDI is managed via `synth.clear_midi()` / `synth.add_midi_note()`. There is no `add_midi_message` or `get_num_parameters` API in DawDreamer 0.8.x.
- **Settle loop** (v1.2): Renders silent chunks between captures until synth output peak < threshold (max 10s). Runs *before* patch change (draining old patch's tail) and between notes within a vector. Uses adaptive threshold `max(1e-4, self_noise * 2)` so self-oscillating patches don't stall. Falls back to `hard_reset` (graph reload + second settle pass) when settle times out.
- **Sobol sampling**: Uses `scipy.stats.qmc.Sobol` with `random_base2(m=m)` — always pass the exponent `m`, not a sample count, to avoid silent truncation.
- **Quality thresholds**: `prev_bleed` checks first 20ms for peak > `max(0.01, self_noise * 2)`; `silence` is peak < 1e-4; `clipping` is 5+ samples > 0.99; `stuck` is tail RMS within 6dB of sustain.
- **Self-noise baseline**: `measure_self_noise()` renders 200ms with no MIDI after each patch load. Stored in parquet as `self_noise`. Used by both the settle loop (adaptive threshold) and the bleed detector (threshold floor) to handle patches that self-oscillate or have LFO artifacts.
- **Profile importance field**: `importance: 0` excludes a parameter from sampling entirely. Values 0–1 control sampling density.
- **Sample rate**: Standardized at 48 kHz.
- **Checkpoint/resume pattern**: Long-running scripts (`capture_v1_2.py`, `index_dataset.py`) checkpoint periodically and prompt `[c]ontinue / [o]verwrite / [a]bort` on restart when prior output exists. New scripts that process the full dataset should follow this pattern.
- **EnCodec embedding**: 48 kHz model produces 128-d latents at 150 Hz frame rate (not 75 Hz — that's the 24 kHz model). Pre-quantiser continuous latents via `model.encoder(x)`, not quantised codes. Latents are unbounded (not L2-normalised) — do not normalise to unit sphere.
- **Surrogate loss**: `MSE + 0.3 * cosine_distance`. MSE dominates (EnCodec latents carry magnitude); cosine is a directional regulariser. Do **not** L2-normalise the surrogate output head.
- **Inversion multi-start**: gradient descent through the surrogate is non-convex — always use multiple random starts (default 16). CMA-ES seeds from the best grad result. Params clamped to `[0,1]` before every surrogate forward pass.

### Data flow

```
obxf.yaml → sampling.py (Sobol vectors) → capture_v1_2.py (DawDreamer render) → WAV + samples.parquet
                                                                                        ↓
                                                                              quality.py (per-capture gates)
                                                                                        ↓
                                                                              verify_dataset.py (aggregate audit)
                                                                                        ↓
                                                                              embed.py (EnCodec) → encodec_embeddings.npy
                                                                                        ↓
                                              samples.parquet + encodec_embeddings.npy → s05_surrogate/train.py
                                                                                        ↓
                                                                              state_dict.pt + surrogate.onnx
                                                                                        ↓
                                                                     target.wav → s06_invert (grad descent + CMA-ES)
                                                                                        ↓
                                                                     patches/<target>/best_patch.yaml + rendered.wav
                                                                                        ↓
                                                                              s07_refine (real-synth gap closure)
```

## Testing

- Unit tests in `tests/` cover sampling, quality, manifest, sequences, capture logic, embeddings, surrogate shapes/gradients, and inversion search
- Run non-integration tests under `.venv`; s05/s06 tests require torch and must run under conda
- Integration tests (`test_integration.py`) require DawDreamer + OB-Xf VST installed; marked with `@pytest.mark.integration`
- Embed tests (`test_embed.py`, `test_embed_index.py`) and invert tests (`test_invert.py`) use synthetic data — no capture data or GPU required for shape/gradient checks; GPU tests auto-skip if CUDA unavailable
- Quality gate thresholds in `quality.py` are tuned for specific audio characteristics — don't change without understanding the capture pipeline

## External References

- **Build instructions**: `build_instructions/` — per-stage design docs for the full pipeline (01 Profile through 08 Package). Consult when implementing new stages. Covers: 01 Profile (OB-Xf, Juno-106, Novation Peak), 02 Capture (VST + hardware rigs), 03 Dataset, 04 Embed, 05 Surrogate, 06 Invert, 06b Live, 07 Refine, 08 Package.

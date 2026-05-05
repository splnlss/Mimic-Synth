# MimicSynth

A pipeline for building audio datasets from synthesizers and training models that learn the parameter-to-timbre mapping of a synth. Currently targets the [OB-Xf](https://github.com/surge-synthesizer/OB-Xf) (free, open-source OB-Xa emulation) via DawDreamer, with architecture designed to extend to hardware/analog synths via MIDI + audio interface.

## Pipeline stages

| Stage | Folder | Status | Purpose |
|-------|--------|--------|---------|
| S01 | `s01_profiles/` | ✅ Complete | Synth profile — parameter definitions, importance weights, probe config |
| S02 | `s02_capture/` | 🔄 Active | Capture rig — renders (param vector, note) to WAV via DawDreamer |
| S03 | `s03_dataset/` | ⏳ Pending | Dataset builder — Sobol sampling, quality gates, manifest, verifier |
| S04 | `s04_embed/` | ⏳ Pending | Audio embedding — EnCodec 48 kHz pre-quantiser latents (128-d) |
| S05 | `s05_surrogate/` | ✅ Complete | Forward model — MLP mapping (params, note) → EnCodec latent |
| S06 | `s06_invert/` | ✅ Complete | Patch search — grad descent + CMA-ES inversion of target audio |
| **S06b** | `s06b_live/` | ✅ Working | Streaming inversion v4 — per-region note segmentation, snapped surrogate notes, pinned params, refinement loop |
| S07 | `s07_refine/` | 🔲 Planned | VST-loop refinement — close the surrogate-to-real cosine-distance gap (see `build_instructions/07 Refine VST Loop.md`) |
| S08 | `s08_package/` | 🔲 Planned | Package — ONNX export + nn~ / Max/Pd integration |

---

## Requirements

- [OB-Xf VST3](https://github.com/surge-synthesizer/OB-Xf/releases) installed at the system default path
- Two Python environments (see below)

### Environments

| Env | Setup | Used for |
|-----|-------|---------|
| `.venv` (Python 3.14) | `python -m venv .venv && pip install -r requirements.txt` | Unit tests, CPU-only utilities |
| `mimic-synth` conda (Python 3.11) | `conda env create -f environment.yml` | All pipeline stages (DawDreamer + torch/CUDA) |

`torch` and `dawdreamer` are only in the conda env. Running pipeline commands with `.venv/bin/python` will fail.

---

## Data storage

All pipeline outputs (WAVs, parquet, embeddings, model checkpoints, patches) live on an external drive — **not** in this repo. Source code stays here.

Data root is configured in `defaults.py`:

```python
DATA_ROOT    = Path("/mnt/d/Mimic-Synth-Data")   # D:\Mimic-Synth-Data on Windows
PROJECT_NAME = "OB-X_Prototype"
```

Directory layout under `DATA_ROOT / PROJECT_NAME`:

```
OB-X_Prototype/
  s02_capture/          raw WAVs + samples.parquet
  s03_dataset/          quality-gated dataset + manifest.yaml
  s04_embed/            encodec_embeddings.npy
  s05_surrogate/runs/   model checkpoints (state_dict.pt, surrogate.onnx)
  s06_invert/patches/   inversion results (best_patch.yaml, rendered.wav, …)
  targets/              target audio files used as inversion inputs
```

Change `DATA_ROOT` and `PROJECT_NAME` in `defaults.py` to relocate or switch between projects.

---

## Quickstart

All pipeline commands require the conda env:

```bash
conda activate mimic-synth
```

### 1. Verify the plugin loads

```bash
python enumerate_params.py
```

Lists all parameters exposed by the OB-Xf VST.

### 2. Capture raw audio (S02)

Renders 2^M Sobol-sampled parameter vectors × 16 notes into the data directory. Production uses M=14 (16,384 vectors × 16 notes = 262,144 captures). All output paths come from `defaults.py` — no flags needed.

```bash
python s02_capture/capture_v1_2.py
```

Checkpoints every 50 vectors. If interrupted, re-run — it resumes automatically (interactive: choose **[c]ontinue**; non-interactive/background: resumes without prompting).

Key features (v1.2):
- **Settle-before-patch-change**: drains the old patch's release tail before loading new parameters
- **Per-note settle**: drains between notes within the same parameter vector
- **Adaptive settle threshold**: uses measured self-noise floor so self-oscillating patches don't stall
- **Self-noise baseline**: 200ms silent render after each patch load, stored as `self_noise` in parquet
- **Hard reset fallback**: reloads the graph when the settle loop times out

### 3. Build the dataset (S03)

Post-hoc mode reads an existing capture, applies quality gates, and writes a manifest. No re-rendering needed. All paths default via `defaults.py`.

```bash
python -m s03_dataset.build_dataset \
    --from-capture /mnt/d/Mimic-Synth-Data/OB-X_Prototype/s02_capture
```

Or live capture from scratch (slower, requires DawDreamer):

```bash
python -m s03_dataset.build_dataset --m 14
```

Verify the result:

```bash
python -m s03_dataset.verify_dataset \
    --dataset /mnt/d/Mimic-Synth-Data/OB-X_Prototype/s03_dataset
```

### 4. Embed the dataset (S04)

Pre-computes 128-d EnCodec embeddings aligned 1-to-1 with `samples.parquet`. All paths default via `defaults.py`.

```bash
python -m s04_embed.index_dataset --pool mean --batch-size 64
```

Checkpoints every 500 rows. Verify with:

```bash
python -m s04_embed.verify_embeddings
```

### 5. Train the surrogate (S05)

Trains a 4-layer MLP to approximate `f(params, note) → EnCodec latent`. All paths default via `defaults.py`.

```bash
python -m s05_surrogate.train
```

Verify with a full round-trip check on the held-out test split (spec criterion: cos-sim ≥ 0.9):

```bash
python -m s05_surrogate.verify_surrogate
```

Auto-selects the latest run in `S05_RUNS_DIR`. Current best: `run_20260429_145056` — val loss 0.0061, test-split cos-sim 0.9988 (dev-scale data; production retrain pending).

### 6. Invert a target sound (S06) ⚠️ WIP

Given a target WAV, finds the OB-Xf parameter vector whose surrogate-predicted embedding is closest to the target. Uses pitch detection to select the best MIDI note, then runs multi-start gradient descent followed by CMA-ES refinement.

Profile, output directory, and surrogate checkpoint all default via `defaults.py`:

```bash
python -m s06_invert.invert --target path/to/target.wav
```

Output: `s06_invert/patches/<target_stem>/best_patch.yaml`, `candidates.parquet`, `target_embedding.npy`.

### 6b. Streaming inversion for long / multi-pitch targets (S06b) ✅ Working

For targets with pitch changes or sustained notes, use the sliding-window streaming inverter. It segments the target into note regions (energy + pitch-discontinuity at 10 ms resolution), tracks pitch per region, runs warm-start gradient descent through the surrogate, and refines against real VST renders.

```bash
# Defaults: 100ms window, 50ms hop, 50 grad steps, 4 warm-starts, 3 refine iterations
conda run -n mimic-synth python -m s06b_live.stream_invert \
    --target path/to/target.wav

# With custom parameters
conda run -n mimic-synth python -m s06b_live.stream_invert \
    --target path/to/target.wav \
    --win-sec 0.1 --hop-sec 0.05 \
    --grad-steps 50 --n-starts 4 \
    --refine-iterations 3

# Direct script invocation also works (sys.path is auto-fixed):
conda run -n mimic-synth python /home/sanss/Mimic-Synth/s06b_live/stream_invert.py \
    --target path/to/target.wav

# Skip render + refinement (analysis only)
conda run -n mimic-synth python -m s06b_live.stream_invert \
    --target path/to/target.wav --no-render
```

Output: `s06_invert/patches/<target_stem>/stream_params.parquet`, `pitch_trajectory.yaml`, `best_patch.yaml`, `rendered.wav`.

Key v4 design decisions (all required to produce audible output that matches the target):

- **Per-region dual MIDI notes** — `midi_note` is the exact integer (sent to DawDreamer for correct pitch playback); `surrogate_note` is snapped to the nearest profile training note (passed to the surrogate so it stays in-distribution and doesn't extrapolate to garbage params).
- **Pinned parameters during inversion** — `PINNED_PARAMS` in `s06b_live/stream_invert.py` fixes `Osc 1 Pitch=0.5`, `Amp Env Release=0.2`, and `LFO 1 to Osc 1 Pitch=0.0`. Without these pins the surrogate finds degenerate solutions that match the target *embedding* but produce wrong audio (pitch shifted +24 semitones, notes ringing through the entire sample). See the `PINNED_PARAMS` block in the script for the rationale per param.
- **Sample-accurate render duration** — the render uses `audio_duration = len(target) / sr` so the output WAV matches the target length exactly.

---

## Run tests

Non-integration tests run under `.venv` (no DawDreamer or GPU needed):

```bash
.venv/bin/pytest -m "not integration"
```

S05/S06 tests require torch — run under conda:

```bash
conda run -n mimic-synth python -m pytest tests/test_s05_surrogate.py tests/test_invert.py -v
```

---

## Project structure

```
defaults.py                  # Data root, project name, all stage paths
enumerate_params.py          # Utility: list all VST parameters
build_instructions/          # Per-stage design docs (01–08)

s01_profiles/
  obxf.yaml                  # OB-Xf parameter profile

s02_capture/
  capture_v1_2.py            # Current capture rig (M=14 production)

s03_dataset/
  build_dataset.py           # Dataset builder CLI (--from-capture or --m)
  sampling.py                # Sobol sampling + importance weighting
  quality.py                 # Per-capture quality gates
  manifest.py                # Reproducibility manifest
  sequences.py               # Temporal sequence dataset builder
  verify_dataset.py          # Post-hoc auditor

s04_embed/
  embed.py                   # Embedder class (EnCodec 48kHz)
  index_dataset.py           # Pre-compute embeddings CLI
  verify_embeddings.py       # Post-hoc embedding auditor

s05_surrogate/
  model.py                   # Surrogate MLP + SurrogateDataset
  train.py                   # Training loop CLI
  verify_surrogate.py        # Round-trip + sweep + gradient checks

s06_invert/
  grad_search.py             # Multi-start gradient descent
  cmaes_search.py            # CMA-ES refinement
  invert.py                  # End-to-end inversion CLI
  render_stream.py           # Render best_patch.yaml via DawDreamer
  stream_invert.py           # Sliding-window inversion for long targets
  validate.py                # Batch validation on held-out test split

s06b_live/
  stream_invert.py           # v4 streaming inverter — per-region notes,
                             #   pinned params, real-VST refinement loop

tests/
  test_sampling.py
  test_quality.py
  test_manifest.py
  test_sequences.py
  test_verify_dataset.py
  test_capture_unit.py
  test_capture_v1_2.py
  test_embed.py
  test_embed_index.py
  test_s05_surrogate.py
  test_invert.py
  test_integration.py        # Requires DawDreamer + OB-Xf

# Data lives on external drive — not in repo
# /mnt/d/Mimic-Synth-Data/OB-X_Prototype/   (see defaults.py)
```

---

## Profile format

```yaml
parameters:
  "Filter Cutoff":
    encoding: vst
    range: [0.0, 1.0]
    continuous: true
    log_scale: true     # perceptual log mapping for frequency-type params
    importance: 1.0     # 0 = fixed at reset, 1 = full range sampled

probe:
  notes: [34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 72, 84]
  velocity: 100
  hold_sec: 1.5
  release_sec: 4.5
  sample_rate: 48000
```

`importance: 0` excludes a parameter from sampling. `log_scale: true` gives finer resolution at low values.

---

## How inversion works

The surrogate learns `f(params, note) → EnCodec latent` from captured data. Inversion asks the reverse question: given a target audio embedding, find the params that minimise cosine distance through the frozen surrogate.

**S06 — single-shot inversion (offline)**

```
target.wav → EnCodec → target_embedding [128-d]
                              ↓
         search: find params p s.t. surrogate(p, note) ≈ target_embedding
              ├── multi-start gradient descent (Adam, 32 starts × 500 steps)
              └── CMA-ES refinement (seeded from best grad result)
                              ↓
                    best_patch.yaml  →  DawDreamer  →  rendered.wav
```

**S06b — streaming, per-region inversion**

```
target.wav  ─►  energy + pitch-jump segmentation @ 10 ms  ─►  note regions
                                       ↓
       per region:  midi_note (exact)  +  surrogate_note (snapped to profile)
                                       ↓
   per coarse frame (100 ms win, 50 ms hop):
       grad_invert(surrogate, target_emb_frame, surrogate_note,
                   pin_indices = {Osc 1 Pitch=0.5,
                                  Amp Env Release=0.2,
                                  LFO 1 to Osc 1 Pitch=0.0})
                                       ↓
            stream_params.parquet  +  pitch_trajectory.yaml
                                       ↓
   render via DawDreamer (per-region note-on/off, sample-accurate timing)
                                       ↓
   refinement loop: render → embed → compare full result → α-search
                    on per-frame surrogate gradients (pins still applied)
                                       ↓
                                rendered.wav
```

The pinned params are critical: without them the surrogate finds solutions where Osc 1 Pitch is maxed (+24 semitones) and Amp Env Release is held high — producing audio that matches the target embedding but doesn't actually sound like the target.

Output is synth knob positions (0–1 per parameter), not MIDI transcription. Load `best_patch.yaml` onto the synth and play the recovered MIDI note to hear the result.

---

## License

MIT

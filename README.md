# MimicSynth

A pipeline for building audio datasets from synthesizers and training models that learn the parameter-to-timbre mapping of a synth. Currently targets the [OB-Xf](https://github.com/surge-synthesizer/OB-Xf) (free, open-source OB-Xa emulation) via DawDreamer, with architecture designed to extend to hardware/analog synths via MIDI + audio interface.

## Pipeline stages

| Stage | Folder | Status | Purpose |
|-------|--------|--------|---------|
| S01 | `s01_profiles/` | ✅ Complete | Synth profile — parameter definitions, importance weights, probe config |
| S02 | `s02_capture/` | ✅ Complete | Capture rig — renders (param vector, note) to WAV via DawDreamer |
| S03 | `s03_dataset/` | ✅ Complete | Dataset builder — Sobol sampling, quality gates, manifest, verifier |
| S04 | `s04_embed/` | ✅ Complete | Audio embedding — EnCodec 48 kHz pre-quantiser latents (128-d) |
| S05 | `s05_surrogate/` | ✅ Complete | Forward model — MLP mapping (params, note) → EnCodec latent |
| S06 | `s06_invert/` | ⚠️ WIP | Patch search — grad descent + CMA-ES inversion of target audio |
| S07 | `s07_refine/` | 🔲 Planned | Refine — close gap between surrogate score and real-synth output |
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

Renders 2^M Sobol-sampled parameter vectors × N notes into `s02_capture/data/`. Production uses M=14 (16,384 vectors × 15 notes = 245,760 captures).

```bash
cd s02_capture && python capture_v1_2.py
```

Checkpoints every 50 vectors — if interrupted, re-run and choose **[c]ontinue**.

Key features (v1.2):
- **Settle-before-patch-change**: drains the old patch's release tail before loading new parameters
- **Per-note settle**: drains between notes within the same parameter vector
- **Adaptive settle threshold**: uses measured self-noise floor so self-oscillating patches don't stall
- **Self-noise baseline**: 200ms silent render after each patch load, stored as `self_noise` in parquet
- **Hard reset fallback**: reloads the graph when the settle loop times out

### 3. Build the dataset (S03)

Post-hoc mode reads an existing capture, applies quality gates, and writes a manifest. No re-rendering needed.

```bash
python -m s03_dataset.build_dataset \
    --profile s01_profiles/obxf.yaml \
    --from-capture s02_capture/data/ \
    --out s03_dataset/data/
```

Or live capture from scratch (slower, requires DawDreamer):

```bash
python -m s03_dataset.build_dataset \
    --profile s01_profiles/obxf.yaml --m 14 --out s03_dataset/data/
```

Verify the result:

```bash
python -m s03_dataset.verify_dataset \
    --dataset s03_dataset/data/ --profile s01_profiles/obxf.yaml
```

### 4. Embed the dataset (S04)

Pre-computes 128-d EnCodec embeddings aligned 1-to-1 with `samples.parquet`.

```bash
python -m s04_embed.index_dataset \
    --dataset s03_dataset/data/ --out s04_embed/data/ \
    --pool mean --batch-size 64
```

Checkpoints every 500 rows. Verify with:

```bash
python -m s04_embed.verify_embeddings \
    --embeddings s04_embed/data/encodec_embeddings.npy \
    --dataset s03_dataset/data/
```

### 5. Train the surrogate (S05)

Trains a 4-layer MLP to approximate `f(params, note) → EnCodec latent`.

```bash
python -m s05_surrogate.train \
    --dataset s03_dataset/data/samples.parquet \
    --embeddings s04_embed/data/encodec_embeddings.npy \
    --out s05_surrogate/runs/
```

Verify with a full round-trip check on the held-out test split (spec criterion: cos-sim ≥ 0.9):

```bash
python -m s05_surrogate.verify_surrogate \
    --checkpoint s05_surrogate/runs/<run_id>/state_dict.pt \
    --dataset s03_dataset/data/samples.parquet \
    --embeddings s04_embed/data/encodec_embeddings.npy \
    --profile s01_profiles/obxf.yaml
```

Current best checkpoint: `runs/run_20260429_145056/` — val loss 0.0061, test-split cos-sim 0.9988.

### 6. Invert a target sound (S06) ⚠️ WIP

Given a target WAV, finds the OB-Xf parameter vector whose surrogate-predicted embedding is closest to the target. Uses pitch detection to select the best MIDI note, then runs multi-start gradient descent followed by CMA-ES refinement.

```bash
python -m s06_invert.invert \
    --target path/to/target.wav \
    --surrogate s05_surrogate/runs/run_20260429_145056/state_dict.pt \
    --profile s01_profiles/obxf.yaml \
    --out patches/
```

Output: `patches/<target_stem>/best_patch.yaml`, `candidates.parquet`, `target_embedding.npy`.

Render the recovered patch through OB-Xf:

```bash
python s06_invert/render_stream.py \
    --patch patches/<target_stem>/best_patch.yaml \
    --profile s01_profiles/obxf.yaml \
    --out patches/<target_stem>/rendered.wav
```

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
defaults.py                  # Shared constants (SAMPLE_RATE, BUFFER_SIZE)
enumerate_params.py          # Utility: list all VST parameters
build_instructions/          # Per-stage design docs (01–08)

s01_profiles/
  obxf.yaml                  # OB-Xf parameter profile

s02_capture/
  capture_v1_2.py            # Current capture rig
  data/                      # samples.parquet + wav/ (gitignored)

s03_dataset/
  build_dataset.py           # Dataset builder CLI (--from-capture or --m)
  sampling.py                # Sobol sampling + importance weighting
  quality.py                 # Per-capture quality gates
  manifest.py                # Reproducibility manifest
  sequences.py               # Temporal sequence dataset builder
  verify_dataset.py          # Post-hoc auditor
  data/                      # samples.parquet + wav/ (gitignored)

s04_embed/
  embed.py                   # Embedder class (EnCodec 48kHz)
  index_dataset.py           # Pre-compute embeddings CLI
  verify_embeddings.py       # Post-hoc embedding auditor
  data/                      # encodec_embeddings.npy (gitignored)

s05_surrogate/
  model.py                   # Surrogate MLP + SurrogateDataset
  train.py                   # Training loop CLI
  verify_surrogate.py        # Round-trip + sweep + gradient checks
  runs/                      # Checkpoints (gitignored)

s06_invert/
  grad_search.py             # Multi-start gradient descent
  cmaes_search.py            # CMA-ES refinement
  invert.py                  # End-to-end inversion CLI
  render_stream.py           # Render best_patch.yaml via DawDreamer
  stream_invert.py           # Sliding-window inversion for long targets
  validate.py                # Batch validation on held-out test split

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
  notes: [36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 72, 84]
  velocity: 100
  hold_sec: 1.5
  release_sec: 4.5
  sample_rate: 48000
```

`importance: 0` excludes a parameter from sampling. `log_scale: true` gives finer resolution at low values.

---

## How inversion works

The surrogate learns `f(params, note) → EnCodec latent` from captured data. Inversion asks the reverse question: given a target audio embedding, find the params that minimise cosine distance through the frozen surrogate.

```
target.wav → EnCodec → target_embedding [128-d]
                              ↓
         search: find params p s.t. surrogate(p, note) ≈ target_embedding
              ├── multi-start gradient descent (Adam, 16 starts × 300 steps)
              └── CMA-ES refinement (seeded from best grad result)
                              ↓
                    best_patch.yaml  →  DawDreamer  →  rendered.wav
```

The output is synth knob positions (0–1 per parameter), not MIDI transcription. Load `best_patch.yaml` onto the synth and play the suggested note to hear the result.

---

## License

MIT

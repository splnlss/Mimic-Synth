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
| **S06b** | `s06b_live/` | ✅ Working | **Inversion** — surrogate-driven: note segmentation, per-region MIDI, PINNED_PARAMS, α-refinement. Fast (~30s). |
| **S07** | `s07_refine/` | ✅ Working | **Refinement** — real-VST-driven: hill-climb + CMA-ES with osc config scouting. No surrogate. Best quality (~15 min). |
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

### 6b + 7. Inversion (S06b) and Refinement (S07)

S06b and S07 are two distinct stages that run in sequence. Understanding the difference helps you pick the right mode:

| | S06b (Inversion) | S07 (Refinement) |
|---|---|---|
| **Driven by** | Surrogate neural network | Real OB-Xf renders |
| **Speed** | ~30 seconds | 5–15 minutes |
| **Output quality** | cosine dist ~0.10 | cosine dist ~0.03–0.09 |
| **Use when** | Iterating, previewing | Final render |

**S06b** uses gradient descent through the frozen surrogate — fast because it avoids the VST entirely during search. S06b also includes an α-refinement loop that scales the surrogate's suggested direction by testing a handful of real renders. This alone drops from ~0.21 to ~0.10.

**S07** abandons the surrogate and evaluates every candidate by rendering through OB-Xf. S07 Strategy 1 (hill-climb) does per-param coordinate descent. S07 Strategy 2 (CMA-ES) additionally scouts oscillator waveform configurations (saw, pulse, saw+pulse) and runs a population-based search that learns parameter correlations.

**All targets are automatically converted to mono** — if you pass a stereo file, the pipeline saves `<stem>_mono.wav` alongside it and uses that. For 3+ channels you will be prompted for instructions.

**Each run writes to a timestamped subfolder**: `patches/<target_stem>/YYYYMMDD_HHMMSS/`

#### Mode 1 — Fast (~30s): surrogate inversion + α-refinement only

```bash
conda run -n mimic-synth python -m s06b_live.stream_invert \
    --target path/to/target.wav --hill-iterations 0
```

Good for iteration and checking pitch/timing before committing to a long run.

#### Mode 2 — Standard (~5 min): + hill-climb (S07 Strategy 1)

```bash
# Default — hill-climb is on (--hill-iterations 2)
conda run -n mimic-synth python -m s06b_live.stream_invert \
    --target path/to/target.wav
```

For each unpinned parameter, tries offsets ±0.05 and ±0.15 globally across all frames; keeps the offset that lowers the real-render cosine distance. Saves `hill_climb_log.yaml` showing which params moved and by how much (useful diagnostic).

#### Mode 3 — Full quality (~15 min): + CMA-ES (S07 Strategy 2)

```bash
conda run -n mimic-synth python -m s06b_live.stream_invert \
    --target path/to/target.wav --cmaes
```

Runs target analysis (spectral centroid, ADSR estimation, harmonic ratio, LFO detection) to answer 5 design questions and warm-start the CMA-ES. Scouts 3 oscillator configs, runs CMA-ES on the best, restarts with larger population if it stagnates. Achieved **0.029 cosine distance** on the crane scream (saw+pulse config, vs 0.21 surrogate-only).

CMA-ES tuning flags:

| Flag | Default | Effect |
|---|---|---|
| `--cmaes-sigma0` | 0.08 | Search radius: 0.05=refine, 0.12=explore |
| `--cmaes-popsize` | 16 | Population per iteration |
| `--cmaes-maxiter` | 20 | Max iterations before restart |

Output files per run: `stream_params.parquet`, `pitch_trajectory.yaml`, `best_patch.yaml`, `rendered.wav`, `rendered_normalized.wav`, `hill_climb_log.yaml` (if hill-climb ran), `cmaes_log.yaml` (if CMA-ES ran).

Key design constraints (all required for correct audio):

- **`PINNED_PARAMS`** — `Osc 1 Pitch=0.5` (no transpose; pitch comes from MIDI note), `Amp Env Release=0.2` (prevents notes ringing through), `LFO 1 to Osc 1 Pitch=0.0` (no pitch wobble). Without these the surrogate AND the real-synth search both find degenerate solutions.
- **`surrogate_note`** — detected MIDI note snapped to nearest profile training note for the surrogate context; exact MIDI note used for DawDreamer render. Keeps the surrogate in-distribution.
- **Reset values applied before every render** — `audio_compare.py` applies all profile reset values first so OB-Xf starts from a known state regardless of previous plugin state.

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
  stream_invert.py           # v4 — inversion entry point; all S07 modes
                             #   wired in via --hill-iterations / --cmaes

s07_refine/
  mono_utils.py              # ensure_mono(): stereo→mono, 3+ch raises
  target_analysis.py         # TargetAnalysis: answers 5 design questions
                             #   (osc type, ADSR, filter, modulation, pitch)
  audio_compare.py           # render_trajectory() + score_audio(); applies
                             #   profile reset values before every render
  vst_hill_climb.py          # Strategy 1: per-param coordinate descent
  vst_cmaes.py               # Strategy 2: osc config scouting + CMA-ES
                             #   with IPOP restart

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

**S06b — Inversion (surrogate-driven, ~30s)**

```
target.wav  ──► mono conversion (auto)
                      ↓
        energy + pitch-jump segmentation @ 10 ms → note regions
                      ↓
        per region:  midi_note (exact, → DawDreamer)
                     surrogate_note (snapped to profile, → surrogate)
                      ↓
        per coarse frame (100ms window, 50ms hop):
            grad_invert(surrogate, frame_emb, surrogate_note,
                        PINNED_PARAMS = {Osc 1 Pitch=0.5,
                                         Amp Env Release=0.2,
                                         LFO 1 to Osc 1 Pitch=0.0})
                      ↓
        stream_params.parquet  +  pitch_trajectory.yaml
                      ↓
        render (DawDreamer, per-region note-on/off, reset values applied)
                      ↓
        α-refinement: surrogate gradient → direction, real renders → scaling
                      ↓
        rendered.wav  [cosine dist ≈ 0.10]
```

**S07 — Refinement (real-VST-driven, ~5–15 min)**

```
stream_params.parquet (from S06b)
                      ↓
    ┌─── Strategy 1: Hill-climb (default, ~5 min) ───────────────────────┐
    │  for each unpinned param p:                                         │
    │      for offset in [-0.15, -0.05, +0.05, +0.15]:                  │
    │          trial = clip(all_frames[p] + offset, 0, 1)                │
    │          score = real_render_and_embed(trial)                       │
    │      keep offset that lowers score                                  │
    │  repeat until no param improves → hill_climb_log.yaml              │
    └────────────────────────────────────────────────────────────────────┘
                      ↓  [cosine dist ≈ 0.09]
    ┌─── Strategy 2: CMA-ES (--cmaes flag, ~15 min) ─────────────────────┐
    │  target_analysis: spectral centroid → filter cutoff                 │
    │                   ADSR shape → Amp Env Attack/Decay                 │
    │                   harmonic ratio → resonance/cross-mod              │
    │                   spectral flux → Filter Env Amount / LFO rate      │
    │                   → smart x0 blended with hill-climb result         │
    │                                                                      │
    │  osc config scouting: saw / pulse / saw+pulse × short CMA-ES        │
    │  → best config (saw+pulse won on crane scream)                       │
    │                                                                      │
    │  full CMA-ES on winning config (popsize=16, maxiter=20)             │
    │  IPOP restart: 1.5× population if stagnating                        │
    │  → cmaes_log.yaml (param deltas, osc config, n_renders)             │
    └────────────────────────────────────────────────────────────────────┘
                      ↓  [cosine dist ≈ 0.03–0.04]
                  rendered.wav  (timestamped subfolder)
```

`PINNED_PARAMS` is enforced at every stage. Without them the optimizer shifts pitch +24 semitones and holds notes indefinitely — both produce lower embedding distance but completely wrong audio.

Output is synth knob positions (0–1 per parameter). Load `best_patch.yaml` onto the synth and play the recovered MIDI note to hear the result.

---

## License

MIT

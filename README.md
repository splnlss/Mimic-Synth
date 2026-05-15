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
| S06 | `s06_invert/` | ✅ Complete | Patch search — grad descent + CMA-ES inversion of target audio |
| **S06b** | `s06b_live/` | ✅ Working | **Inversion** — surrogate-driven: note segmentation, pyworld pitch tracking, MIDI pitch bend, PINNED_PARAMS, α-refinement. Fast (~30s). |
| **S07** | `s07_refine/` | ✅ Working | **Refinement** — real-VST-driven: hill-climb + CMA-ES (global/per-region/hybrid) with expanded 22-param space and composite scoring. No surrogate. Best quality (~15 min). |
| S08 | `s08_package/` | 🔲 Planned | Package — ONNX export for deployment |

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

Each project folder contains a `PROJECT_STATUS.md` that tracks stage completion, quality results, the active surrogate checkpoint, and next steps. That file is the source of truth for project state; this README describes the software only.

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

Renders 2^M Sobol-sampled parameter vectors × N notes into the data directory (N and M are defined in the profile). All output paths come from `defaults.py` — no flags needed.

> **Important:** always run capture via `conda activate` + direct python, not `conda run`. `conda run` buffers stdout and the tqdm progress bar won't appear.

```bash
conda activate mimic-synth
python s02_capture/capture_v1_2.py
```

Checkpoints every 50 vectors. If interrupted, re-run — it resumes automatically (interactive: choose **[c]ontinue**; non-interactive/background: resumes without prompting).

> **Fresh start:** if the profile changes (new parameters or notes), the old parquet must be deleted before restarting or old rows with NaN param values will contaminate the dataset. Delete `s02_capture/samples.parquet` and `s02_capture/wav/` before re-running.

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

Auto-selects the latest run in `S05_RUNS_DIR`. See the project's `PROJECT_STATUS.md` for the current checkpoint and quality metrics.

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

**S06b** uses gradient descent through the frozen surrogate — fast because it avoids the VST entirely during search. Pitch is tracked using pyworld F0 (5ms resolution, full-audio, deterministic) with an autocorrelation fallback. Fine pitch glides are written as MIDI pitch bend automation (affecting all oscillators simultaneously), not Osc 1 Pitch VST automation. S06b also runs an α-refinement loop that scales the surrogate's suggested direction by testing a handful of real renders, dropping from ~0.21 to ~0.10.

**S07** abandons the surrogate and evaluates every candidate by rendering through the VST. S07 Strategy 1 (hill-climb) does per-param coordinate descent. S07 Strategy 2 (CMA-ES) scouts oscillator waveform configurations **and Osc 2 Pitch intervals** (7 discrete musical intervals: octave down through octave up), then runs a population-based search across the full surrogate param space plus any extra params defined in `cmaes_extra_params`, scoring each candidate with a composite distance: **EnCodec 40% + auraloss MRSTFT 25% + aperiodicity 20% + spectral envelope 15%**, all LUFS-normalised to −23 LUFS. Attack transient detection (librosa onset) sets a per-target dynamic upper bound on attack time.

Per-frame **Filter Cutoff** is driven by a measured calibration curve (OB-Xf sweep: 420 Hz at cutoff=0 → 4134 Hz at cutoff=1), replacing the previous heuristic Nyquist-normalised formula. Ring Mod and Cross Modulation are smoothed with a 350ms window (7 frames) after CMA-ES to prevent high-frequency aliasing at region boundaries. The CMA-ES checkpoint writes `rendered.wav` at each IPOP restart improvement so progress is audible during the search.

CMA-ES operates in three modes:
- **`global`** (fastest): single CMA-ES pass applying global per-param offsets. Achieved **0.029** on crane scream.
- **`per-region`** (experimental): independent CMA-ES per note region, each scored against that region's own target embedding. Prone to discontinuities on short regions.
- **`hybrid`** (default): global first, then per-region fine-tune where it beats the global result by >5%. Linear crossfade at region boundaries.

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

For each unpinned parameter, tries offsets ±0.05 and ±0.15 globally across all frames; keeps the offset that lowers the composite scoring distance. Saves `hill_climb_log.yaml` showing which params moved and by how much.

#### Mode 3 — Full quality (~15 min): + CMA-ES (S07 Strategy 2)

```bash
conda run -n mimic-synth python -m s06b_live.stream_invert \
    --target path/to/target.wav --cmaes
```

Runs target analysis (spectral centroid, ADSR estimation, harmonic ratio, LFO detection) to warm-start the CMA-ES. Scouts 3 oscillator configs, then runs the selected config in hybrid mode (global + per-region fine-tune). IPOP restart with 1.5× population if stagnating. Achieved **0.029 cosine distance** on the crane scream (saw+pulse config, vs 0.21 surrogate-only).

CMA-ES tuning flags:

| Flag | Default | Purpose | Trade-off |
|---|---|---|---|
| `--cmaes-mode` | `hybrid` | `global` (fastest, best score), `per-region` (experimental), `hybrid` (global then per-region fine-tune) | `global` achieved 0.029; `hybrid` is safer for complex multi-region targets |
| `--cmaes-sigma0` | 0.08 | Search radius around x0 | 0.05 = refine near warm-start; 0.12 = explore widely; too large wastes budget on bad regions |
| `--cmaes-popsize` | 16 | Candidates evaluated per iteration | Larger = better coverage but proportionally more renders; 24 gives ~50% more renders per iteration |
| `--cmaes-maxiter` | 20 | Max CMA-ES iterations before IPOP restart | 30 recommended when combined with interval scouting; adds ~5 min but aids convergence |

**Recommended quality ladder** (crane scream baseline):

| Command | Time | Score |
|---|---|---|
| `--hill-iterations 0` | ~30s | ~0.17 |
| *(default)* | ~5 min | ~0.15 |
| `--cmaes` | ~15 min | ~0.09–0.13 |
| `--cmaes --cmaes-maxiter 30` | ~20 min | ~0.05–0.09 |
| `--cmaes --cmaes-popsize 24 --cmaes-maxiter 30` | ~30 min | best quality |

Output files per run: `stream_params.parquet`, `pitch_trajectory.yaml`, `fine_pitch_trajectory.yaml`, `pitch_bend.mid`, `best_patch.yaml`, `rendered.wav`, `rendered_normalized.wav`, `hill_climb_log.yaml` (if hill-climb ran), `cmaes_log.yaml` (if CMA-ES ran).

Key design constraints (all required for correct audio):

- **`PINNED_PARAMS`** — `Osc 1 Pitch=0.5` (no transpose; pitch comes from MIDI note), `Amp Env Release=0.2` (prevents notes ringing through), `LFO 1 to Osc 1 Pitch=0.0` (no pitch wobble). Without these the surrogate AND the real-synth search both find degenerate solutions.
- **`surrogate_note`** — detected MIDI note snapped to nearest profile training note for the surrogate context; exact MIDI note used for DawDreamer render. Keeps the surrogate in-distribution.
- **Pitch bend via MIDI file** — fine pitch tracking (pyworld + autocorr fallback at 10ms) is written as a `.mid` file with calibrated ±6st bend range. This moves all oscillators together, unlike Osc 1 Pitch VST automation.
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
  notes: [24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96]
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
        pyworld F0 @ 5ms (full-audio, deterministic)
          → voiced/unvoiced tracking, no octave errors
          → fallback: per-frame autocorrelation
                      ↓
        energy + pitch-jump segmentation @ 10ms → note regions
          per region: midi_note (exact, → DawDreamer)
                      surrogate_note (snapped to profile, → surrogate)
                      ↓
        per coarse frame (100ms window, 50ms hop):
            grad_invert(surrogate, frame_emb, surrogate_note,
                        PINNED_PARAMS = {Osc 1 Pitch=0.5,
                                         Amp Env Release=0.2,
                                         LFO 1 to Osc 1 Pitch=0.0})
                      ↓
        stream_params.parquet  +  fine_pitch_trajectory.yaml
                      ↓
        render (DawDreamer):
          - MIDI file with note-on/off + pitch bend automation @ 10ms
            (calibrated ±6st range, moves all oscillators together)
          - per-param VST automation from stream_params
          - reset values applied before render
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
    │  score = EnCodec (40%) + MRSTFT (25%) + AP (20%) + SP (15%)        │
    │  for each unpinned param p:                                         │
    │      for offset in [-0.15, -0.05, +0.05, +0.15]:                  │
    │          trial = clip(all_frames[p] + offset, 0, 1)                │
    │          score = real_render_composite(trial)                       │
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
    │  parameter space: all surrogate params + cmaes_extra_params         │
    │      (profile-defined bounds constrain CMA-ES search range)         │
    │                                                                      │
    │  osc config scouting: saw / pulse / saw+pulse × short CMA-ES        │
    │  → best config (saw+pulse won on crane scream)                       │
    │                                                                      │
    │  Osc 2 interval scouting: 7 discrete intervals (oct-dn / 5th-dn /  │
    │      3rd-dn / unison / 3rd-up / 5th-up / oct-up) × short CMA-ES    │
    │  → winning interval pinned for main CMA-ES                          │
    │                                                                      │
    │  CMA-ES mode (hybrid default):                                       │
    │    Phase 1 — global: single pass over all frames (fast, ~0.029)    │
    │    Phase 2 — per-region: fine-tune regions ≥ 400ms where it        │
    │              beats global by >5%; crossfade at boundaries           │
    │  IPOP restart: 1.5× population if stagnating                        │
    │  → cmaes_log.yaml (param deltas, mode, osc config, n_renders)      │
    └────────────────────────────────────────────────────────────────────┘
                      ↓  [cosine dist ≈ 0.03–0.04]
                  rendered.wav  (timestamped subfolder)
```

`PINNED_PARAMS` is enforced at every stage. Without them the optimizer shifts pitch +24 semitones and holds notes indefinitely — both produce lower embedding distance but completely wrong audio.

Output is synth knob positions (0–1 per parameter). Load `best_patch.yaml` onto the synth and play the recovered MIDI note to hear the result.

---

---

## Accuracy improvements — quality roadmap

The current pipeline produces a recognizable approximation of the source: correct rhythm, correct pitch regions, some timbral overlap. Several root causes limit quality and point to concrete improvements:

### Richer synthesis

The surrogate was trained on static single-note patches. The most impactful changes require no retraining:

- ~~**Per-frame Filter Cutoff automation**~~ **Done.** pyworld SP + librosa centroid → per-frame trajectory written to `stream_params.parquet`, with crossfade at note boundaries.
- ~~**Oscillator interval snapping**~~ **Done.** `_scout_osc2_intervals` in `vst_cmaes.py` tries 7 discrete intervals (oct-dn through oct-up) before the main CMA-ES; winner is pinned.
- **Unison voices** — Unison Detune at 0.2–0.4 immediately makes any patch sound denser. Should be near-default, not CMA-ES-discovered.
- **Noise burst at attack** — route noise through a fast-decay envelope (short Filter Env Decay at high Filter Env Amount). Gives the percussive "click" at note onset most organic sounds have.

### Better target analysis

The current `target_analysis.py` answers 5 heuristic design questions. Grounding them in signal analysis would give more reliable warm-starts:

- **pyworld spectral envelope (SP)** as a direct filter target — SP is the smooth spectral shape per frame and maps directly to Filter Cutoff trajectory and Filter Resonance. Already available from `pyworld.wav2world()`, which the pipeline calls for F0/AP.
- **Attack transient decomposition** — `_detect_transient_min_ms()` in `target_analysis.py` uses librosa onset detection; the minimum inter-onset interval sets a per-target dynamic upper bound on Amp Env Attack in `cmaes_refine()`, replacing the previous static bound.
- **Harmonic structure classification** — use pyworld aperiodicity (AP) mean to gate oscillator selection explicitly: AP < 0.2 → saw/saw+pulse, AP 0.2–0.5 → saw+pulse + noise, AP > 0.5 → noise-dominant. Currently implicit in CMA-ES scoring.
- **Inharmonicity fingerprinting** — measure deviation of spectral peaks from integer ratios. Gives a direct objective for Ring Mod and Cross Modulation depth, which currently only emerge by accident.

### Better scoring

- **Per-frame SP distance** — `SP_WEIGHT = 0.15` in `audio_compare.py`; composite is `0.40×EnCodec + 0.25×MRSTFT + 0.20×AP + 0.15×SP`. Directly rewards the optimizer for matching filter movement frame-by-frame.
- **Pitch-invariant timbral scoring** — pitch-shift the render to match the source before embedding, so cosine distance measures timbre rather than penalizing small pitch errors.
- **LUFS normalization** — `_lufs_normalize()` in `audio_compare.py`; applied before all scoring comparisons.

### Better optimization

- **Hierarchical search** — separate coarse (discrete: oscillator type, Osc 2 interval, ring mod on/off) from medium (ADSR set analytically from target waveform) from fine (CMA-ES on the remaining 8–10 parameters). Shrinks the effective CMA-ES search space 3–5×.
- **DDSP analysis as CMA-ES warm-start** — Google's DDSP extracts continuous filter cutoff and harmonic amplitude trajectories analytically. Using those as x0 instead of the current target_analysis heuristics would give dramatically better convergence.
- **Surrogate retrained on production data** — after capture completes, rebuild S03 → S04 → S05. The surrogate's input dimension is read directly from the profile, so adding parameters to `obxf.yaml` automatically expands the model on the next train.

### Offline synth calibration

~~One-time parameter sweeps through OB-Xf~~ **Done.** `calibrate_synth.py` sweeps Filter Cutoff 0→1 and writes `s07_refine/obxf_calibration.npz`. Both `_centroid_hz_to_filter_cutoff` (stream_invert.py) and `target_analysis.py` now load this table via `np.interp`. Run once after installing the plugin: `conda run -n mimic-synth python calibrate_synth.py`

---

## License

MIT

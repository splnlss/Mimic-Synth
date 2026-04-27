# MimicSynth

A pipeline for building audio datasets from synthesizers, designed to train models that learn the parameter-to-timbre mapping of a synth. Currently targets the [OB-Xf](https://github.com/surge-synthesizer/OB-Xf) (free, open-source OB-Xa emulation) via DawDreamer, but the architecture is designed to extend to ASIO audio interfaces and hardware/analog synths with MIDI.

The pipeline has three stages:

| Stage | Folder | Purpose |
|---|---|---|
| S01 | `s01_profiles/` | Synth profile -- parameter definitions, importance weights, probe config |
| S02 | `s02_capture/` | Capture rig -- renders (param vector, note) to WAV via DawDreamer |
| S03 | `s03_dataset/` | Dataset builder -- Sobol sampling, quality gates, manifest, verifier |

---

## Requirements

- Python 3.10+
- [OB-Xf VST3](https://github.com/surge-synthesizer/OB-Xf/releases) installed at the system default path
- macOS, Windows, or Linux

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quickstart

### 1. Verify the plugin is found

```bash
.venv/bin/python enumerate_params.py
```

Lists all parameters exposed by the OB-Xf VST. Cross-check against `s01_profiles/obxf.yaml` if you update the plugin.

### 2. Capture raw audio (S02)

Renders 2^M Sobol-sampled parameter vectors x 5 notes into `s02_capture/data/`. Default M=10 (1,024 vectors = 5,120 captures).

```bash
cd s02_capture
../.venv/bin/python capture_v1_2.py
```

The script checkpoints every 50 vectors. If interrupted, re-run and choose **[c]ontinue** to resume.

Key capture features (v1.2):
- **Settle-before-patch-change**: drains the old patch's release tail before loading new parameters
- **Per-note settle**: drains between notes within the same parameter vector
- **Adaptive settle threshold**: uses measured self-noise floor so self-oscillating patches don't stall the settle loop
- **Self-noise baseline**: renders 200ms with no MIDI after each patch load; stored in parquet as `self_noise`
- **Hard reset fallback**: reloads the processing graph + second settle pass when the settle loop times out

### 3. Build the dataset (S03)

Wraps the capture rig with scrambled Sobol sampling, quality gates, and a reproducibility manifest.

```bash
.venv/bin/python -m s03_dataset.build_dataset \
    --profile s01_profiles/obxf.yaml \
    --m 10 \
    --out s03_dataset/data/
```

`--m 10` generates 2^10 = 1,024 vectors. Use `--m 14` (~16k) or higher for production runs.

Optional flags:
- `--seed INT` -- random seed (default `0`)
- `--importance-mode filter|scale` -- how importance weights are applied (default `filter`)

### 4. Verify the dataset

```bash
.venv/bin/python -m s03_dataset.verify_dataset \
    --dataset s02_capture/data/ \
    --profile s01_profiles/obxf.yaml
```

Checks every WAV for silence, clipping, stuck notes, and previous-note bleed. Exits non-zero if any failure rate exceeds 1%.

To dump a CSV of all failed captures:

```bash
.venv/bin/python -m s03_dataset.verify_dataset \
    --dataset s02_capture/data/ \
    --profile s01_profiles/obxf.yaml \
    --dump-failures
```

Report is written to `tests/reports/verify_dataset/data_<name>_<HHMMSS>.csv`.

### 5. Build sequence data (optional, for temporal models)

Generates interpolated parameter trajectories for training frame-to-frame dynamics.

```bash
.venv/bin/python -m s03_dataset.sequences \
    --profile s01_profiles/obxf.yaml \
    --out s03_dataset/data/sequences/ \
    --m 10 \
    --seconds 5.0 \
    --control-hz 100
```

Outputs `sequences.parquet`, `wav/<hash>.wav`, and `params/<hash>.npy`.

---

## Shared defaults

Project-wide constants live in `defaults.py` (sample rate, buffer size). Production code reads `sample_rate` from the profile YAML; `defaults.py` provides fallbacks for utility scripts and tests.

The canonical source of truth for audio settings is the profile:

```yaml
# s01_profiles/obxf.yaml
probe:
  sample_rate: 48000
```

---

## Run tests

```bash
.venv/bin/pytest                        # all tests
.venv/bin/pytest -m "not integration"   # skip tests that require DawDreamer + OB-Xf
```

---

## Project structure

```
defaults.py              # Shared constants (SAMPLE_RATE, BUFFER_SIZE)
enumerate_params.py      # Utility: list all VST parameters

s01_profiles/
  obxf.yaml              # Parameter profile for OB-Xf

s02_capture/
  capture_v1_2.py        # Current capture rig (v1.2)
  capture_v1.py          # Legacy v1 (no settle loop)
  capture_v1-1.py        # Legacy v1.1 (settle after patch change)
  data/                  # Output: samples.parquet + wav/ (gitignored)

s03_dataset/
  build_dataset.py       # Dataset builder CLI
  sampling.py            # Sobol sampling + importance weighting
  quality.py             # Per-capture quality gates (48 kHz)
  manifest.py            # Reproducibility manifest (manifest.yaml)
  sequences.py           # Sequence/trajectory dataset builder
  verify_dataset.py      # Post-hoc dataset auditor

tests/
  test_quality.py
  test_sampling.py
  test_manifest.py
  test_sequences.py
  test_verify_dataset.py
  test_capture_unit.py
  test_capture_v1_2.py
  test_integration.py
  reports/               # Generated failure CSVs (gitignored)
```

---

## Profile format

`s01_profiles/obxf.yaml` defines which parameters are sampled and how:

```yaml
parameters:
  "Filter Cutoff":
    encoding: vst
    range: [0.0, 1.0]
    continuous: true
    log_scale: true     # perceptual log mapping for frequency-type params
    importance: 1.0     # 0 = fixed at reset value, 1 = full range sampled
```

`importance: 0` excludes a parameter from sampling entirely. `log_scale: true` applies a perceptually uniform mapping so low values get finer resolution.

---

## Future: hardware synth capture

The pipeline is designed to extend beyond DawDreamer/VST hosting. For capturing hardware/analog synths:

- **MIDI out**: parameter vectors map to CC messages sent to the hardware synth
- **Audio in**: ASIO/CoreAudio interface captures the synth's analog output
- **Profile**: same YAML format, with `transport: midi_cc` instead of `vst_host`, plus CC mappings per parameter
- **Settle loop**: same concept (render/record silence until peak drops), but polls the audio input instead of `engine.get_audio()`

The quality gates (`quality.py`), dataset verifier, and Sobol sampling are transport-agnostic and work unchanged with hardware captures.

---

## License

MIT

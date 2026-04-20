# Dataset Verifier

Post-hoc quality audit for a s03_dataset (`samples.parquet` + `wav/*.wav`).
It loads every row, replays each WAV, runs the dataset quality gates
(silence, clipping, stuck note, previous-note bleed), and reports per-dataset
failure rates plus parameter-coverage stats.

s03_dataset target: every failure rate must be below **1%**. The verifier exits
non-zero if any rate exceeds the threshold.

## Prerequisites

- A built dataset directory containing:
  - `samples.parquet` — with columns `hash`, `note`, `wav`, and `p_*` parameters
  - `wav/` — the referenced WAV files (paths in parquet may be absolute or
    relative to the dataset directory)
- The profile YAML used to build the dataset (for `probe.sample_rate`,
  `probe.render_sec`, `probe.hold_sec`, `probe.release_sec`)
- Python env with `pandas`, `pyarrow`, `soundfile`, `numpy`, `pyyaml`
  (already in `requirements.txt`)

## Run

From the `MimicSynth/` project root:

```bash
.venv/bin/python -m s03_dataset.verify_dataset \
    --dataset data/ \
    --profile s01_profiles/obxf.yaml
```

With a custom failure threshold (default `0.01` = 1%):

```bash
.venv/bin/python -m s03_dataset.verify_dataset \
    --dataset data/ \
    --profile s01_profiles/obxf.yaml \
    --fail-threshold 0.005
```

Exit codes:
- `0` — all failure rates at or below threshold (**PASS**)
- `1` — at least one rate above threshold, or missing parquet (**FAIL**)

## What it checks

Per row:
- WAV exists and is readable
- Sample rate matches `profile.probe.sample_rate`
- Duration matches `profile.probe.render_sec` (±50 ms tolerance)
- Filename hash matches `hash` column (for `*_n<note>.wav` convention)
- `quality.analyse`: silence / clipping / stuck note / prev-note bleed

Aggregate:
- Counts + rates for each failure mode
- Parameter coverage: mean, std, min, max for every `p_*` column
- First 10 issues printed verbatim for debugging

## Example output

```
Total rows:         10000
WAVs readable:      10000
WAVs missing:       0  (0.00%)
WAVs unreadable:    0  (0.00%)
Wrong duration:     0  (0.00%)
Hash/filename diff: 0  (0.00%)
Silent:             12  (0.12%)
Clipped:            3  (0.03%)
Stuck notes:        0  (0.00%)
Prev-note bleed:    0  (0.00%)
Valid:              9985  (99.85%)

Parameter coverage:
  p_Filter Cutoff               mean=0.501 std=0.289 min=0.000 max=1.000
  p_Filter Resonance            mean=0.499 std=0.288 min=0.001 max=0.999
  ...

PASS: all failure rates ≤ 1.00%
```

## Programmatic use

```python
from pathlib import Path
from s03_dataset.verify_dataset import verify_dataset, print_report

report = verify_dataset(Path("data/"), Path("s01_profiles/obxf.yaml"))
ok = print_report(report, fail_threshold=0.01)
print(report.rate("silent"), report.param_coverage)
```

`Report` is a dataclass; inspect fields directly or call `report.rate("silent")`,
`report.rate("clipped")`, etc.

## CI integration

```bash
.venv/bin/python -m s03_dataset.verify_dataset \
    --dataset data/ --profile s01_profiles/obxf.yaml || exit 1
```

## Troubleshooting

- **`No samples.parquet found`** — wrong `--dataset` path, or capture hasn't
  written the parquet yet.
- **`missing_wav:...`** — parquet references WAVs that aren't on disk. Likely
  a partial run; rebuild or prune the parquet.
- **High `prev_bleed` rate** — the plugin is leaking previous-note state.
  Ensure `reset()` is called before each parameter vector and that
  `engine.load_graph` is called once (not per render).
- **High `stuck` rate** — `profile.probe.render_sec` < `hold_sec + release_sec`,
  or the synth's release time exceeds the configured release window.
- **Wrong sample rate** — dataset built with a different `SAMPLE_RATE` than
  `profile.probe.sample_rate`. Rebuild, or update the profile.

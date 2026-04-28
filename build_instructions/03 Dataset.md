---
tags: [build, 03-dataset, dataset, sampling, active-learning]
created: 2026-04-19
---

# 03 Dataset — Generation Overview

> [!info] Goal
> Take the "render one capture" primitive from [[02 Capture - VST|Bucket 2 V1]] or [[02 Capture - Hardware|Bucket 2 V2]] and use it to build the labelled dataset that [[04 Embed]] and [[05 Surrogate]] need: tens to hundreds of thousands of `(param_vector, note, audio)` tuples with good coverage of the modulated parameter space.

Bucket 3 is where the pipeline moves from "can I render one sample" to "do I have a dataset a model can actually learn from." It's mostly about sampling strategy, storage discipline, and knowing when you have enough.

## Where Bucket 3 sits

```
Bucket 2 (capture rig)           Bucket 3 (dataset)              Bucket 4+
  render_one(vec, note) ─┬───► sampler → capture loop ──► dataset/
                         │                                 ├── samples.parquet
                         │                                 └── wav/*.wav
                         │
                         └─► one primitive, two back-ends
                             (V1 = VST render, V2 = hardware roundtrip)
```

Bucket 3 is synth-agnostic by construction: it only needs the profile YAML and a `render_one(param_vector, note) -> audio` callable. Same code runs against OB-Xf (V1) or a Juno-106 / Peak / anything else (V2).

## Prerequisites

- [[01 Profile - OB-Xf|A Bucket 1 profile]] for the target synth.
- A working [[02 Capture - VST|Bucket 2 V1]] *or* [[02 Capture - Hardware|Bucket 2 V2]] capture rig — the `render_one` primitive must be reliable and deterministic-enough.
- For V2 specifically: latency calibration complete, silence/stuck-note detection in place, reset protocol verified.

## Dataset anatomy

Keep the layout boring and reproducible. Everything downstream assumes this shape.

```
dataset/
├── manifest.yaml            # seed, profile hash, sampler version, date, git sha
├── samples.parquet          # one row per capture
├── wav/
│   ├── 1a2b3c4d5e6f_n60.wav
│   ├── 1a2b3c4d5e6f_n72.wav
│   └── ...
└── coverage/
    ├── param_histograms.png
    └── sobol_projection.png
```

`samples.parquet` schema — minimum columns:

| Column          | Type     | Notes                                            |
| --------------- | -------- | ------------------------------------------------ |
| `hash`          | str      | Content hash of `(param_vec, note)` — 12 hex     |
| `note`          | int      | MIDI note number                                 |
| `velocity`      | int      | Usually fixed to profile probe velocity          |
| `wav`           | str      | Relative path to WAV                             |
| `duration_sec`  | float    | Render length (from profile)                     |
| `silent`        | bool     | True if peak amplitude < threshold               |
| `rms`           | float    | Post-render RMS in linear [0,1]                  |
| `peak`          | float    | Post-render peak                                 |
| `source_batch`  | str      | `sobol_v1`, `al_round_3`, etc.                   |
| `p_<name>`      | float    | One column per modulated parameter, normalised `[0,1]` |

Two invariants that matter: content-hashed filenames let retries be idempotent, and keeping every applied parameter value (including post-quantisation enum snapping) in the Parquet means you can always recover what the synth actually received, not just what the sampler proposed.

## Sampling strategy

### Baseline: Sobol in normalised space

Sobol sequences (scrambled) give you much better coverage per sample than uniform random — and the sequence is deterministic, so re-running with the same seed produces the same dataset. See [[Quasi-Random Sampling]] for the theory.

```python
from scipy.stats.qmc import Sobol

def cold_start_vectors(m, d, seed=0):
    """Generate 2**m Sobol points in d dimensions.
    Using exponent m (not sample count n) avoids the silent-truncation trap
    of `random_base2(m=int(np.log2(n)))` when n isn't a power of 2."""
    sobol = Sobol(d=d, scramble=True, seed=seed)
    return sobol.random_base2(m=m)   # returns 2**m samples
```

Prefer `random_base2` over `sobol.random(n)` when n is known to be a power of 2 — SciPy's own docs note that "most QMC constructions are designed for special values of n such as powers of 2 ... changing the sample size by even one can degrade performance." For non-power-of-2 budgets (e.g. 10k), `sobol.random(n)` still produces well-distributed samples with `scramble=True`; you just give up a bit of the theoretical low-discrepancy guarantee. In practice, for dataset generation this is a non-issue. The existing [[02 Capture - VST]] uses `sobol.random(n)`; Bucket 3 production runs should prefer powers of 2 via `random_base2`.

### Log-scale parameters

Filter cutoff, envelope times, LFO rate, etc. are perceptually logarithmic. Draw them uniform in `[0,1]` then map through a log transform when you hand them to the synth — or flag them in the profile and transform at apply time:

```python
def to_synth_value(u, spec):
    """u in [0,1] from Sobol; return actual normalised param value."""
    if spec.get("log_scale"):
        # u -> exponential mapping, tunable via spec["log_base"]
        return (np.expm1(u * np.log(1 + spec.get("log_base", 50)))
                / spec.get("log_base", 50))
    return u
```

The profile YAML already has the `log_scale: true` flag for the params that need it (see the Bucket 1 profiles for Juno, Peak, OB-Xf).

> [!note] Log-scale transform implemented in `s03_dataset`
> `s03_dataset.sampling.to_synth_value(u, spec)` handles the `log_scale` transform. The raw `capture_v1.py` still passes Sobol values straight through — use `s03_dataset.build_dataset` (which wraps `capture_v1`) so the transform is applied consistently. Datasets built with vs. without the transform are not comparable; commit the sampling version in `manifest.yaml`.

### Importance weighting

Some parameters matter perceptually more than others. The Bucket 1 profiles assign `importance` weights (Cutoff/Resonance = 1.0 for the filter-heavy synths).

Two distinct uses of the same field across the pipeline — don't conflate them:

1. **Membership filter (Bucket 2 V1, current):** `importance > 0` = "modulate this param at all; everything else is frozen at the reset value." This is how `capture_v1.py` uses it today.
2. **Range scale (Bucket 3, proposed):** `importance` acts as a *range* multiplier around the default: importance-1.0 params draw across the full `[0,1]` Sobol range; importance-0.3 params draw from a narrow band centred on the reset value.

Simple implementation for (2): `value = reset + importance * (sobol_u - 0.5)`, clipped to `[0,1]`. This preserves reset values as the centre of each param's sampled distribution and shrinks the range of lower-importance params proportionally.

Datasets built under interpretation (1) vs. (2) are *not* compatible — commit the interpretation in `manifest.yaml` so downstream buckets can detect mismatches.

### Probe-note strategy

Each parameter vector gets rendered at multiple MIDI notes (profile's `probe.notes`, typically C2/C3/C4/C5/C6). This serves two purposes: it captures pitch-dependent filter and envelope behaviour, and it amortises the reset cost across multiple captures on hardware. Don't sample the notes randomly — you want the same notes across all param vectors so downstream code can align them.

## Phases

### Phase A — Cold start (pure Sobol)

First 20–30% of the target dataset size. Pure Sobol in normalised parameter space, no active learning yet. This is the "just cover the space" pass.

Typical sizes:
- Software synth (V1, OB-Xf, 15 params): 10–20k cold-start vectors × 5 notes = 50–100k captures. ~30 min – 1 h on a modern laptop.
- Hardware synth (V2, Juno/Peak, 12–17 params): 3–5k vectors × 5 notes = 15–25k captures. ~12–20 h. Run overnight.

### Phase B — Active learning loop

After the cold start, train a weak surrogate (tiny MLP on CLAP embeddings is fine — see [[Active Learning for Synth Params]]) and use its *uncertainty* to pick the next batch of parameter vectors. Places the surrogate is confused are typically the high-resonance / self-oscillation / filter-mode-transition regions — exactly where the dataset needs more density.

Loop sketch:

```
while not converged and budget_remaining:
    surrogate = train_weak(dataset_so_far)
    candidates = sobol_pool(k_candidates)                   # big pool
    uncertainty = surrogate.predict_variance(candidates)     # ensemble / MC-dropout
    next_batch = top_n(candidates, by=uncertainty, n=batch_size)
    new_captures = capture_loop(next_batch, render_one)
    dataset_so_far.append(new_captures)
```

Typical `batch_size` is 500–2000. Iterate 3–5 rounds. Convergence = the top-uncertainty candidates no longer cluster in the same region.

Libraries: [modAL](https://modal-python.readthedocs.io/), [ALiPy](https://github.com/NUAA-AL/ALiPy), or just a hand-rolled loop — the architecture is simple.

### Phase C — Targeted finishing (optional)

Once you know what the model struggles with, run small targeted Sobol sub-draws in those regions (e.g. "all high-Q cutoff sweeps with envelope depth > 0.7"). This is worth doing for the final 10% of the budget.

## Addendum — sequence captures for Bucket 6b

> [!info] Required for [[06b Live|Bucket 6b]]
> The single-frame `(param, note, audio)` schema above is sufficient for Bucket 5's static surrogate and Bucket 6's offline inversion, but the streaming inverse in Bucket 6b needs time-aligned sequences. Run this pass **in addition** to the cold-start / AL phases — don't replace them.

Output schema, written alongside the single-frame dataset:

```
dataset/
├── sequences.parquet           # one row per sequence
├── wav/<seq_hash>.wav          # mono audio, T / control_hz seconds
└── params/<seq_hash>.npy       # float32 [T, d_params] trajectory
```

Each sequence = one pair of Sobol endpoints, linearly interpolated over T frames at a control rate `control_hz` (50–100 Hz). The synth is driven via DawDreamer's `set_automation` so param changes are time-aligned with the rendered audio. Log-scale transform is applied to the interpolated trajectory (not just endpoints) so the values that land in `params/*.npy` match what the synth actually received.

Implementation: [`s03_dataset/sequences.py`](file:///Users/splnlss/Claude/MimicSynth/s03_dataset/sequences.py). CLI:

```bash
python -m s03_dataset.sequences \
    --profile s01_profiles/obxf.yaml \
    --out dataset/obxf_2026-04-seq/ \
    --m 10 \
    --seconds 5.0 \
    --control-hz 100
```

That renders `2**(m-1)` = 512 sequences of 5 s each at 100 Hz (500 frames × 15 params per sequence). Budget ~1–2 hours for a ~2k-sequence production pass against the VST.

Considerations:
- **Endpoint pairing halves the Sobol count.** A pool of `2**m` points yields `2**(m-1)` pairs. Pick `m` accordingly.
- **Note handling.** Each sequence is a single sustained note covering the full duration (default = middle of `probe.notes`). Multi-note sequences are out of scope for V1; add as a follow-up if the model can't generalise across pitch.
- **Smoothness.** Linear interpolation is the floor. For a more musical eval set, replace with recorded performer trajectories (Bucket 6b § "Training the inverse" path 2); same schema, different source.
- **Compatibility.** The sequence dataset uses its own `sequences.parquet` + `manifest.yaml` (sampler: `sobol_interpolated`) so it never clashes with the single-frame dataset in the same directory tree.

## Quality gates

Every capture goes through the same lightweight checks before it's counted as valid:

1. **Silence check.** Peak amplitude < 1e-4 → `silent=True`, skip writing the WAV for software; on hardware, record it but mark it so the trainer can exclude.
2. **Clipping check.** Peak > 0.99 for more than a handful of samples → log as clipped. On software this usually means the probe velocity is too hot; on hardware, the interface gain is too high.
3. **Stuck-note detection.** After `hold_sec + release_sec` has fully elapsed, the remaining tail (last ~50 ms of the capture) should be near the noise floor. If its RMS stays within ~6 dB of the sustain-window RMS, the note never released — flag as stuck and optionally re-render. (An orthogonal "prev-note bleed" check: RMS of the first ~20 ms — before the probe's own attack — is non-trivial.)
4. **Coverage stats.** After each phase, plot param-wise histograms and a 2D Sobol projection of the most-important param pair (usually Cutoff × Resonance). Visually obvious gaps mean the sampler is broken.

On V2 specifically, also track MIDI/audio health stats from the capture rig itself (retries, reset failures, latency drift) — they're already exposed by the V2 `CaptureLoop` class.

## Splits

Before training anything, do a deterministic split so results are reproducible:

- **Train** — ~80%.
- **Validation** — ~10%. Used for [[05 Surrogate]] training.
- **Held-out test** — ~10%. Reserved for [[06 Invert]] evaluation and never touched during surrogate training.

Split by `hash` (modulo a small prime) rather than by row index — that way adding new samples later only grows the sets, doesn't reshuffle them.

## Scaling guidance

Rough rules of thumb that have held across a few different synth families:

| Synth complexity       | Params | Captures (cold + AL) | Time V1 (VST) | Time V2 (HW) |
| ---------------------- | ------ | -------------------- | ------------- | ------------ |
| Simple (Juno-like)     | 10–12  | ~50k                 | 15–30 min     | ~40 h        |
| Medium (OB-Xf, Peak)   | 15–18  | ~100k                | 30–60 min     | ~80 h        |
| Complex (wavetable/FM) | 20+    | 200–400k             | 2–4 h         | infeasible without cluster |

Hardware passes the "worth it" threshold around 50–100k samples — beyond that, each extra 10k buys less accuracy per hour of wall-clock time than a better embedding or surrogate.

## Reproducibility

Commit, with the dataset, a `manifest.yaml`:

```yaml
dataset:
  created: 2026-04-19T14:32:00Z
  seed: 0
  sampler: sobol_v1
  sampler_version: 1.2.0
  profile_hash: sha256:ab12...          # hash of the Bucket 1 profile YAML
  capture_rig: v1                        # or v2
  capture_rig_git_sha: 4f8e2c1
  phases:
    - {name: cold_start, n: 20000, seed: 0}
    - {name: al_round_1, n: 2000, uncertainty: mc_dropout}
    - {name: al_round_2, n: 2000, uncertainty: ensemble_var}
  counts:
    rendered: 120000
    silent: 423
    clipped: 18
    valid: 119559
```

This lets [[05 Surrogate]] know exactly which dataset revision it was trained against and makes regressions trivially diagnosable.

## Running it

> [!note] The `s03_dataset.build_dataset` CLI below is the target shape, not yet fully implemented.
> Today, dataset generation against V1 runs through [`capture_v1.py`](file:///Users/splnlss/Claude/MimicSynth/s02_capture/capture_v1.py) (cold-start Sobol + resume). The AL loop, manifest writer, and multi-backend `--backend` flag are future work in `s03_dataset/`.

Against V1 (software, OB-Xf):

```bash
python -m s03_dataset.build_dataset \
  --profile s01_profiles/obxf.yaml \
  --backend v1 \
  --n-cold 20000 \
  --al-rounds 3 \
  --al-batch 2000 \
  --out dataset/obxf_2026-04/
```

Against V2 (hardware, e.g. Peak):

```bash
python -m s03_dataset.build_dataset \
  --profile s01_profiles/novation_peak.yaml \
  --backend v2 \
  --n-cold 5000 \
  --al-rounds 3 \
  --al-batch 1000 \
  --out dataset/peak_2026-04/
```

The `--backend` flag is the only thing that differs — everything else is profile-driven.

## When Bucket 3 is done

You're ready for [[04 Embed]] when:

- Total valid captures ≥ the scaling-guidance target for this synth complexity.
- Coverage plots show no obvious holes in the modulated subspace.
- Silence / clipping / stuck-note rates are each < 1% of total.
- A tiny sanity surrogate (e.g. 3-layer MLP on multi-res STFT features) trains to non-trivial accuracy on the training split and generalises to validation.
- Manifest is written and committed alongside the dataset.

## Known pitfalls

- **Sobol's first few points are clumpy (unscrambled only).** With `scramble=True` (the default we use) this is largely non-issue — the random linear scrambling removes the clumpy-start pattern. Skip the burn-in unless you switch to `scramble=False`, in which case burn ~1024 points.
- **Non-power-of-2 batch sizes.** `sobol.random(n)` for non-power-of-2 `n` degrades low-discrepancy properties ("changing the sample size by even one can degrade performance," per SciPy docs) but doesn't break the sample — it's still well-distributed enough for dataset generation. For best theoretical coverage, prefer `random_base2(m)` with powers of 2.
- **Enum quantisation isn't stored.** If you sample an enum at 0.73, it gets snapped to the nearest category. Store the *applied* value in the Parquet, not the proposed `u ∈ [0,1]`.
- **Probe-note cross-talk on hardware.** Release tails from note N can leak into the start of note N+1's capture if `release_sec` in the probe is too short. Err long.
- **Dataset drift across sessions.** On V2, temperature drift between Tuesday and Wednesday captures is real. Log session timestamps and include them as a feature if drift matters; otherwise do large batches in single sessions.
- **Silent ≠ useless.** Some silent captures are correctly predicted silences (fully closed filter, zero envelope). Keep them with the flag set — the surrogate should learn to predict "silence" too.

## References

- [SciPy Sobol](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html) · [scikit-optimize Latin Hypercube](https://scikit-optimize.github.io/)
- [modAL — active learning in Python](https://modal-python.readthedocs.io/) · [ALiPy](https://github.com/NUAA-AL/ALiPy)
- [Polars](https://pola.rs/) · [PyArrow](https://arrow.apache.org/docs/python/) · [Parquet format](https://parquet.apache.org/)
- [[Quasi-Random Sampling]] · [[Active Learning for Synth Params]] · [[Synth-Mimic-Pipeline]]
- [[02 Capture - VST]] · [[02 Capture - Hardware]] · [[04 Embed]]

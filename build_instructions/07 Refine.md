---
tags: [build, 07-refine, refinement, hardware-loop, cma-es]
created: 2026-04-19
---

# 07 Refine — Hardware-Loop Refinement Overview

> [!info] Goal
> Close the last-mile gap between surrogate-predicted similarity and actual synth output by running a small number of CMA-ES iterations where each candidate is rendered on the **real hardware synth**, re-embedded, and re-scored. Only worth doing once [[06 Invert|Bucket 6]] has produced a surrogate-best patch that's already in the right neighbourhood.

The surrogate is an approximation. Real hardware has temperature drift, voice chip variance, analog noise, slight tuning offsets — things the surrogate can't model exactly. Bucket 7 spends a few hundred hardware renders to refine a patch from "surrogate-best" to "ear-acceptable on this specific unit."

## Where Bucket 7 sits

```
Bucket 6 output                                Bucket 7 (this doc)
top-K candidates ────► CMA-ES on real hardware ────► refined patch
     │                      ▲                              │
     │                      │                              │
     └─── surrogate score   └── real-synth render          │
          (cheap)               (via Bucket 2 V2)          ▼
                                                    Bucket 8 packaging
```

The *only* thing different from [[06 Invert|Bucket 6]] strategy 3 is that the objective function calls [[02 Capture - Hardware|Bucket 2 V2]]'s real-hardware render instead of the surrogate. Everything else — CMA-ES mechanics, enum handling, scoring — carries over.

## Prerequisites

- [[02 Capture - Hardware|Bucket 2 V2]] capture rig fully working: MIDI out, ASIO/Core Audio in, latency calibrated, reset protocol verified.
- [[06 Invert|Bucket 6]] produced a top-K candidate patch list for a target. "Close enough" means surrogate cosine distance is already below ~0.2.
- The *same* embedder [[04 Embed|Bucket 4]] used for training — don't swap to a different embedder here, or you'll refine against a different loss.
- Wall-clock time. This is the slowest bucket in the pipeline per target.

## The protocol

```python
# s07_refine/hw_refine.py — sketch
import cma
import numpy as np

def hw_refine(capture_loop, embedder, target_emb, note,
              x0, sigma0=0.05, popsize=16, maxiter=20, cache=None):
    """x0: best candidate from Bucket 6. Small sigma; we're near-optimal already."""
    es = cma.CMAEvolutionStrategy(
        x0=x0.tolist(),
        sigma0=sigma0,   # small — we trust the seed
        inopts={"bounds": [[0.0] * len(x0), [1.0] * len(x0)],
                "maxiter": maxiter, "popsize": popsize},
    )

    while not es.stop():
        xs = es.ask()
        scores = []
        for x in xs:
            key = _cache_key(x)
            if cache and key in cache:
                scores.append(cache[key])
                continue
            audio = capture_loop.render_one(x, note)      # <-- real synth
            emb   = embedder.encodec_embed(audio, 48000)
            score = 1 - float(np.dot(emb, target_emb) /
                              (np.linalg.norm(emb) * np.linalg.norm(target_emb)))
            if cache is not None:
                cache[key] = score
            scores.append(score)
        es.tell(xs, scores)

    return np.array(es.best.x), es.best.f
```

Key choices:

- **Small `sigma0`.** 0.05 (5% of normalised range) rather than the 0.15 used in Bucket 6. We're refining, not searching.
- **Tight `maxiter`.** 20 iterations × popsize 16 = 320 hardware renders per target. At ~3 s each, ~15 minutes per target.
- **Aggressive caching.** Same parameter vector → same audio (modulo temperature drift). Cache scores by a rounded param tuple; reuse on retry.
- **Keep enum quantisation visible.** Round enum dimensions on the candidate before rendering *and* store the rounded vector, not the raw CMA-ES suggestion, in the cache key.

## Budgeting

Rough wall-clock cost per target, on a healthy V2 rig:

| Knob                   | Low                | Default | High            |
| ---------------------- | ------------------ | ------- | --------------- |
| popsize                | 8                  | 16      | 24              |
| maxiter                | 10                 | 20      | 50              |
| render time (s/sample) | 2.5                | 3       | 4               |
| **total renders**      | 80                 | 320     | 1200            |
| **wall-clock**         | 3–5 min            | 15 min  | 60–80 min       |

15 minutes per target is a reasonable default. Going to the "high" column rarely improves scores by more than a few percent; diminishing returns set in quickly. If you're tempted to run more iterations, retrain the surrogate on an enlarged dataset instead.

## Caching and idempotency

Hardware renders are expensive. A good cache layout:

```
cache/
└── <synth_id>/
    └── <session_id>/
        └── scores.parquet          # (param_tuple_rounded, note, target_id, score)
```

Two reasons to namespace by session: temperature drift between sessions means a score from Monday can't be trusted on Wednesday to the 3rd decimal, and reset protocols occasionally change as you refine the V2 rig. Cross-session cache reuse is a productivity hack; be aware it's slightly lossy.

Round parameter tuples to ~3 decimal places before hashing for the cache key. That's ~1000× finer than the synth's 7-bit CC resolution (128 steps) so no real loss of fidelity.

## Temperature and warm-up

The V2 Juno profile warns about a 20-minute warmup. Bucket 7 reinforces this: if session 1's first 20 candidates are rendered cold and the rest warm, CMA-ES will learn "cold sound is worse" and bias against the cold-render parameter vectors — a bug, not a feature. Two mitigations:

- **Warm up before Bucket 7.** Play random notes through the synth for 20 minutes, discard audio.
- **Randomise candidate order within each generation.** Already the case in CMA-ES's default `ask()`, but worth confirming.

For fully-digital synths (OB-Xf, Peak) this is a non-issue.

## Cross-session refinement

If you want to squeeze more quality out of a target, you can run Bucket 7 across multiple sessions:

- Session 1: seed from Bucket 6, refine 20 iterations, save `best_x` and cache.
- Session 2: reseed from session 1's `best_x` with a small sigma, refine another 20, continuing the cache.

Each session converges to a slightly different local optimum because the synth's behaviour has drifted; averaging across sessions produces a patch that's robust to drift, which is what you want for a patch you'll play live.

## Validation

Per target:

1. **Score improvement.** Refined cosine distance < Bucket 6's surrogate-best cosine distance, evaluated on the real synth. If the hardware score is *higher* than the surrogate score, the surrogate is overfit — go back to Bucket 5.
2. **Audible improvement.** A/B comparison between the Bucket 6 patch (rendered on hardware) and the Bucket 7 patch. If you can't hear the difference, the refinement didn't help — save the compute and stop here next time.
3. **Stability.** Load the refined patch onto the synth cold from a saved SysEx / CC dump. The rendered audio should match the Bucket 7 cached audio to within a few percent cosine distance. Large mismatch means the reset protocol is incomplete.

## Outputs

Per target, extending Bucket 6's layout:

```
patches/bird_01/
├── target.wav
├── target_embedding.npy
├── candidates.parquet
├── best_patch.yaml                  # Bucket 6 output
├── rendered.wav
├── refined_patch.yaml               # NEW — Bucket 7 output
├── refined_patch.syx                # load-ready SysEx / init-patch dump
├── refined_rendered.wav
└── refinement_log.parquet           # every hardware render with score
```

## Dependencies

```bash
# Bucket 7 adds nothing beyond Bucket 6 + Bucket 2 V2 already installed.
pip install cma pandas numpy
```

## Uncertainties to flag

- **Per-target time cost adds up.** 15 minutes × 50 targets = 12.5 hours. If you're building a preset bank from a large animal-call library, budget cloud-free evenings. For single performance use cases (one bird call for one show) this is fine.
- **CMA-ES with noisy objective.** Analog synths add per-render noise (tuning drift, voice variance). CMA-ES assumes a deterministic objective; noise slows convergence. Mitigate by averaging 2–3 renders per candidate — but that doubles/triples the time cost. For the Juno-106 specifically, this matters; for OB-Xf it doesn't.
- **Cache invalidation across firmware changes.** If the Peak firmware is updated mid-project, old cache scores are stale. Version the cache directory by firmware revision from the Bucket 1 profile metadata.
- **Objective drift vs Bucket 6.** Bucket 6 optimises against the surrogate-predicted embedding. Bucket 7 optimises against the real-synth-then-embedder embedding. These are not the same optimum. Usually close, occasionally not — if your refined patch feels like a different musical idea than the Bucket 6 patch, that's expected, not a bug.
- **The "close enough to refine" threshold is a judgement call.** Cosine distance 0.2 is a default; tune per embedder. If Bucket 6 can't get below that threshold on a target, the surrogate or the dataset is the problem, not Bucket 7.
- **Live-use (Bucket 6b) doesn't benefit from Bucket 7.** The streaming path doesn't have a "one target, refine now" step — it's a continuous mapping. Bucket 7 is purely for the offline one-shot flow. Flag this so nobody tries to bolt it into the live loop.

## When Bucket 7 is done

- Refinement script runs end-to-end on a target in under ~20 min wall-clock on V2 hardware.
- Real-synth cosine distance improves measurably over Bucket 6's surrogate-best candidate.
- A/B comparison reveals audible improvement on a majority of spot-checked targets.
- Refined patch is persisted as both `.yaml` (human-readable) and `.syx` / `.json` (load-ready on the synth).
- Cache is populated and reused on re-runs.

## References

- [pycma](https://github.com/CMA-ES/pycma) · [Nevergrad](https://github.com/facebookresearch/nevergrad)
- [Hansen — CMA-ES tutorial](https://arxiv.org/abs/1604.00772)
- [[Hardware-in-the-Loop Capture]] · [[CMA-ES for Patch Search]]
- [[02 Capture - Hardware]] · [[06 Invert]] · [[08 Package]]

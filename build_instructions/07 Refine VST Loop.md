---
tags: [build, 07-refine, surrogate-to-real-gap, cma-es, vst-loop]
created: 2026-05-05
---

# 07 Refine — Closing the Surrogate-to-Real Gap

> [!info] Goal
> Reduce the ~0.19 cosine-distance gap between surrogate-predicted embeddings and real VST-rendered embeddings, so that inverted parameters actually sound like the target when rendered through OB-Xf.

## Problem Statement

The surrogate (S05) is trained on `(params, note) → EnCodec_embedding` using synthetic renders from the capture pipeline (S02). It predicts embeddings well (test cos-sim 0.9988), but **embeddings are not audio**.

When S06b inverts a target, it optimises against the surrogate — not the real synth. Different patches can produce similar embeddings while sounding completely different (phase, stereo image, aliasing, envelope shape). The result is a "surrogate-to-real gap" where:
- Surrogate score is low (good): e.g. cos-dist 0.0024
- Real-render score is high (bad): e.g. cos-dist 0.19

This gap was measured on the **crane bird scream** target:
- Surrogate-best score: 0.0024
- Real-render score after refinement: 0.1918
- Target RMS: 0.218, Rendered RMS: 0.046 (audible but poor timbre match)

## Root Causes

| # | Hypothesis | Evidence | Status |
|---|-----------|----------|--------|
| 1 | **Surrogate underfits timbre details** | Captures coarse spectrum, misses stereo width and phase | Likely — MLP hidden=512 may be too small for full timbre |
| 2 | **EnCodec embedding is lossy** | 48 kHz model at 150 Hz frame rate; fine temporal structure lost | Confirmed — embeddings capture "gist" not exact waveform |
| 3 | **Training data lacks diversity** | M=10 dev set (5,120 samples) covers limited timbre space | Likely — production M=14 (16,384 samples) may help |
| 4 | **VST render quality varies** | Linux OB-Xf v1.0.3 may differ from Windows/Mac builds | Investigated — not a WSL issue, but platform differences possible |
| 5 | **Refinement loop uses wrong objective** | S06b refinement uses surrogate gradient, not real render | Confirmed — the alpha-search uses surrogate-forward, not VST-render |

## Strategy: VST-Loop Refinement (Primary)

Replace the surrogate-based refinement in S06b with a **real VST render inside the optimization loop**:

```
S06b output (best params from surrogate)
        │
        ▼
┌─────────────────────────────┐
│  CMA-ES or grad descent     │
│  with REAL VST renderer     │
│  as objective function      │
└─────────────────────────────┘
        │
        ▼
   Render params → OB-Xf via DawDreamer
        │
        ▼
   Embed rendered audio with EnCodec
        │
        ▼
   Compare to target embedding
        │
        ▼
   Iterate until convergence
```

### Why CMA-ES over gradient descent?
- The VST renderer is not differentiable
- CMA-ES handles noisy objectives well (each render has slight variance)
- Population-based search explores parameter correlations

### Implementation sketch

```python
# s07_refine/vst_refine.py
import cma
import numpy as np
from pathlib import Path

def real_synth_score(params, note, target_emb, embedder, renderer):
    """Objective: render params through real VST, embed, score."""
    audio = renderer.render(params, note)          # DawDreamer VST render
    emb = embedder.encodec_embed(audio, sr=48000)
    score = 1 - cosine_similarity(emb, target_emb)
    return score


def vst_refine(x0, target_emb, note, embedder, renderer,
               sigma0=0.05, popsize=16, maxiter=20):
    """Refine S06b output using real VST renders."""
    es = cma.CMAEvolutionStrategy(
        x0=x0,
        sigma0=sigma0,
        inopts={
            "bounds": [[0.0] * len(x0), [1.0] * len(x0)],
            "maxiter": maxiter,
            "popsize": popsize,
        },
    )
    cache = {}
    while not es.stop():
        xs = es.ask()
        scores = []
        for x in xs:
            key = tuple(np.round(x, 3))
            if key in cache:
                scores.append(cache[key])
                continue
            s = real_synth_score(x, note, target_emb, embedder, renderer)
            cache[key] = s
            scores.append(s)
        es.tell(xs, scores)
    return np.array(es.best.x), es.best.f
```

### Key parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `sigma0` | 0.05 | Already near-optimal from S06b; small search radius |
| `popsize` | 16 | Good exploration without excessive renders |
| `maxiter` | 20 | 320 renders total; ~15 min at 3s/render |
| `cache_key_rounding` | 3 decimals | Finer than synth resolution (~7-bit CC) |

## Strategy B: Surrogate Retraining with Real Embeddings

If the VST-loop is too slow, generate a **second training set** where:
1. Sample random params + notes
2. Render through real VST (not just capture pipeline's cached renders)
3. Re-embed with EnCodec
4. Retrain surrogate on `(params, note) → real_embedding`

This "closes the loop" by training the surrogate on the same renderer used at test time.

**Trade-off**: Requires ~10k real renders (slow) but eliminates systematic bias.

## Strategy C: Multi-Objective Surrogate Loss

Add an **audio-domain loss** to the surrogate training:
- Current: `loss = MSE(emb_pred, emb_true) + 0.3 * cosine_distance(emb_pred, emb_true)`
- Proposed: add `0.1 * MRSTFT_distance(audio_pred, audio_true)`

Requires the surrogate to predict audio, not just embeddings. More complex but preserves fine structure.

## Expected Metrics

| Metric | Before (S06b) | Target (S07) |
|--------|--------------|--------------|
| Surrogate score | 0.0024 | — |
| Real-render score | 0.19 | < 0.05 |
| A/B listenability | "similar pitch, wrong timbre" | "recognisably the target" |
| Time per target | ~5 min (S06b alone) | ~20 min (S06b + S07) |

## Files to create

```
s07_refine/
├── __init__.py
├── vst_refine.py          # CMA-ES on real VST renderer
├── audio_compare.py       # Utility: render + score a candidate
├── cache_manager.py       # SQLite cache for (param_key, note, score)
└── README.md              # Usage and calibration
```

## Integration with S06b

S06b should accept `--refine-mode {surrogate, vst}` where:
- `surrogate` (default): fast, uses current alpha-search loop
- `vst`: slow but accurate, calls `s07_refine.vst_refine()`

## References

- Existing design in `build_instructions/07 Refine.md` (hardware-loop variant)
- S06b pipeline: `s06b_live/stream_invert.py`
- Surrogate model: `s05_surrogate/model.py`
- OB-Xf Linux VST3: https://github.com/surge-synthesizer/OB-Xf/releases/tag/v1.0.3

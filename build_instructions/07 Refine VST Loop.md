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

The core idea: **stop using the surrogate gradient at refinement time.** The surrogate's predicted embedding is close (cos-sim 0.9988 on test split) but its gradient direction in parameter space is not exactly aligned with the real-synth gradient direction — close enough to seed a good initial guess, not close enough to converge.

So the refinement loop should evaluate every candidate by *actually rendering through OB-Xf*, embed the rendered audio with EnCodec, and compare to the target embedding. The S06b initial inversion provides a good seed; the refinement walks it closer using only real-synth feedback.

```
S06b output (best params from surrogate seed)
        │
        ▼
┌─────────────────────────────────────────────┐
│  Search over params (no surrogate)          │
│  Strategy 1 (Hill-climb): per-param probe   │
│  Strategy 2 (CMA-ES):    population search  │
└─────────────────────────────────────────────┘
        │   (every candidate evaluated this way:)
        ▼
   Render params → OB-Xf via DawDreamer
        │
        ▼
   Embed rendered audio with EnCodec (mean pool, 128-d)
        │
        ▼
   Cosine distance to target embedding → score
        │
        ▼
   Iterate
```

Both strategies must respect the `PINNED_PARAMS` set from `s06b_live/stream_invert.py` (Osc 1 Pitch, Amp Env Release, LFO 1 to Osc 1 Pitch). Those pins exist precisely *because* freeing them lets the optimizer find degenerate solutions; that hazard applies just as much to a real-synth optimizer.

The S06b output is a *per-frame parameter trajectory*, not a single static patch. Both strategies preserve that trajectory shape by optimizing a **global per-param offset** that is added uniformly across all frames. This keeps the dynamic timbre information from the surrogate while letting the real-synth loop correct systematic biases.

---

### Strategy 1: Per-param hill-climbing (recommend starting here)

Coordinate-descent on global offsets. For each unpinned param `p`, sweep a small list of offsets, render through the real VST for each, keep the offset that lowers the real-synth cosine distance most. Iterate until no param improves.

**Pseudocode**

```python
# s07_refine/vst_hill_climb.py
def hill_climb(df, note_regions, param_cols, pinned_cols,
               profile_path, target_emb, embedder, total_sec, device,
               offsets=(-0.15, -0.05, 0.05, 0.15), n_passes=2):
    candidates = [c for c in param_cols if c not in pinned_cols]
    audio = render_trajectory(df, note_regions, param_cols, profile_path, total_sec)
    current_score = score(audio, target_emb, embedder, device)

    for pass_i in range(n_passes):
        improved = False
        for col in candidates:
            best_offset, best_score = 0.0, current_score
            for off in offsets:
                trial = df.copy()
                trial[col] = np.clip(trial[col] + off, 0.0, 1.0)
                audio = render_trajectory(trial, note_regions, param_cols, profile_path, total_sec)
                s = score(audio, target_emb, embedder, device)
                if s < best_score:
                    best_score, best_offset = s, off
            if best_offset != 0.0:
                df[col] = np.clip(df[col] + best_offset, 0.0, 1.0)
                current_score = best_score
                improved = True
        if not improved:
            break
    return df, current_score
```

**Tradeoffs**

| | Hill-climbing |
|---|---|
| Renders per pass | n_unpinned_params × len(offsets) ≈ 12 × 4 = 48 |
| Default schedule | 2 passes ≈ 96 renders ≈ 3–5 minutes |
| Quality | Greedy / coordinate-bias; can plateau early |
| Dependencies | None beyond DawDreamer (already required) |
| Interpretability | High — log shows which param moved by how much |

**When to use** — first pass refinement, baseline, or when you want the diagnostic value of a per-param breakdown ("Filter Cutoff -0.10 helped, Filter Resonance +0.05 helped, others didn't move"). Use as the default in `s06b` until empirically beaten.

---

### Strategy 2: CMA-ES (escalation when hill-climbing plateaus)

Population-based gradient-free search. Maintains a covariance matrix of parameter correlations, so it discovers things like "lowering Filter Cutoff also wants slightly more Filter Resonance" without being told. Better-quality result; ~4× more expensive than hill-climbing.

```python
# s07_refine/vst_cmaes.py
import cma
import numpy as np

def vst_refine(x0, target_emb, note, embedder, renderer,
               sigma0=0.05, popsize=16, maxiter=20):
    """CMA-ES over a single param vector evaluated by real VST renders."""
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

**CMA-ES key parameters**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `sigma0` | 0.05 | Already near-optimal from S06b; small search radius |
| `popsize` | 16 | Good exploration without excessive renders |
| `maxiter` | 20 | 320 renders total; ~15 min at 3s/render |
| `cache_key_rounding` | 3 decimals | Finer than synth resolution (~7-bit CC) |

**When to use** — escalate to CMA-ES after hill-climbing plateaus. Particularly valuable when several params are entangled (filter cutoff/resonance/env-amount are a notorious trio). The covariance learning is the win.

---

### Strategy 1 vs Strategy 2: comparison

|  | Hill-climb (Strategy 1) | CMA-ES (Strategy 2) |
|---|---|---|
| Cost | ~96 renders / 3–5 min | ~320 renders / ~15 min |
| Convergence quality | Greedy plateau | Globally smarter |
| Handles param correlations | No | Yes (covariance matrix) |
| External deps | None | `cma` package |
| Interpretable log | Yes (per-param deltas) | Less direct |
| Resumable | Trivial (per-pass checkpoint) | Possible but messier |

Run **Strategy 1 by default**; flip to Strategy 2 only when the hill-climb result is unsatisfying.

---

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

| Metric | Before refinement | Strategy 1 (hill) | Strategy 2 (CMA-ES) |
|---|---|---|---|
| Real-render cos-dist (crane scream) | 0.10 | ~0.06 (target) | ~0.04 (target) |
| Time per target | ~30 s (s06b alone) | ~5 min (s06b + hill) | ~15 min (s06b + CMA-ES) |
| A/B listenability | "right notes, wrong timbre" | "closer timbre" | "recognisably the target" |

## Files to create

```
s07_refine/
├── __init__.py
├── vst_hill_climb.py      # Strategy 1: per-param coordinate descent (PRIMARY)
├── vst_cmaes.py           # Strategy 2: CMA-ES on real VST renders
└── audio_compare.py       # Shared: render trajectory + EnCodec score
```

## Integration with S06b

S06b accepts these new CLI flags:

| Flag | Default | Effect |
|---|---|---|
| `--hill-iterations N` | 2 | N hill-climb passes after α-refinement; 0 to disable |
| `--hill-offsets a,b,c,d` | `-0.15,-0.05,0.05,0.15` | Comma-list of offsets to try per param per pass |
| `--cmaes` | off | Run CMA-ES after hill-climbing (requires `cma` package) |

The pipeline becomes: `analyze → invert → render → α-refine → re-render → hill-climb → re-render → [optional CMA-ES → re-render]`.

## References

- Existing design in `build_instructions/07 Refine.md` (hardware-loop variant)
- S06b pipeline: `s06b_live/stream_invert.py`
- Surrogate model: `s05_surrogate/model.py`
- OB-Xf Linux VST3: https://github.com/surge-synthesizer/OB-Xf/releases/tag/v1.0.3

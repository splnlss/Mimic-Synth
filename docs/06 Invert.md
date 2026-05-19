---
tags: [build, 06-invert, inversion, cma-es, patch-search]
created: 2026-04-19
---

# 06 Invert / Patch Search Overview

> [!info] Goal
> Given a target sound (the bird, the voice, the animal call), find the parameter vector whose surrogate-predicted embedding is closest to the target's embedding. Output: a patch — a concrete list of MIDI CC / NRPN / SysEx values — that can be loaded onto the synth to reproduce the target as closely as the surrogate + synth permit.

Bucket 6 is the offline, one-shot half of the system. [[06b Live|Bucket 6b]] is the streaming sibling. Both rely on the surrogate from [[05 Surrogate]], but their inference paths differ.

## Where Bucket 6 sits

```
target audio clip ─► Bucket 4 embed ─► target_embedding
                                             │
                                             ▼
                            ┌─── (1) learned inverse g(·) ─────┐
                            │                                  │
Bucket 5 surrogate f(·) ────┼─── (2) gradient descent on f ────┼──► candidate patches
                            │                                  │
                            └─── (3) CMA-ES on f(candidate) ───┘
                                             │
                                             ▼
                            top-K ranked by surrogate similarity
                                             │
                                             ▼
                      (optional) [[07 Refine]]
```

Three strategies that compose. Use all three in the order shown.

## Prerequisites

- [[05 Surrogate|Bucket 5]] surrogate trained and passing its sanity checks.
- [[04 Embed|Bucket 4]] `embed()` callable available.
- A target audio clip (WAV / FLAC / bird recording / voice snippet).
- For strategy 1: reversed-direction training data (same as Bucket 3, just with inputs and outputs swapped).

## Strategy 1 — Learned inverse model `g: embedding → params`

Train a second network on the same dataset with inputs and outputs swapped. One forward pass gives a seed patch in milliseconds — imprecise but great as a starting point for strategies 2 and 3.

Architecture mirror of [[05 Surrogate|Bucket 5]]'s MLP:

```
inputs:
  embedding:  [d_embed]     (128 from EnCodec mean-pool, or 256 with mean+std)
  note:       [1]           (target's dominant note from a pitch tracker)

concat → Linear(d_embed + 1 → 512) → SiLU → ... → Linear(hidden → d_params)
       → sigmoid (to keep outputs in [0, 1])

output: parameter vector in [0, 1]^d_params
```

Loss: MSE on parameter values (plus a small term for enum correctness if present). Training takes the same ~10–30 min as the surrogate.

**Important caveat:** the mapping embedding → params is **one-to-many**. Two very different patches can produce nearly-identical embeddings. A plain MSE inverse will predict a blurry average of those modes and sound bad. Two mitigations:

- **Use it only as a seed.** Strategies 2 and 3 refine it. Never use the inverse's raw output as the final patch in v1.
- **Variational / mixture density variants.** Train a mixture-density network or CVAE to output several candidate modes. Post-v1 — flagged in [[Inverse Synth Models]].

Skip this strategy entirely in v1 if you want; [[06b Live|Bucket 6b]] will force you to train one eventually, but offline Bucket 6 works fine without it.

## Strategy 2 — Gradient descent through the frozen surrogate

Treat the parameter vector as a learnable tensor; backprop from embedding distance through the surrogate.

```python
# s06_invert/grad_search.py
import torch
import torch.nn.functional as F

def grad_invert(surrogate, target_emb, note, d_params,
                n_starts=16, steps=300, lr=5e-2, device="cuda"):
    surrogate.eval()
    target = target_emb.to(device)
    note_t = torch.tensor([[note / 127.0]], device=device)

    best = None
    for _ in range(n_starts):
        # random start in [0, 1]^d
        params = torch.rand(1, d_params, device=device, requires_grad=True)
        opt = torch.optim.Adam([params], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            clamped = params.clamp(0.0, 1.0)
            pred = surrogate(clamped, note_t)
            loss = 1 - F.cosine_similarity(pred, target.unsqueeze(0), dim=-1).mean()
            loss.backward()
            opt.step()
        final = params.detach().clamp(0.0, 1.0)
        with torch.no_grad():
            score = 1 - F.cosine_similarity(
                surrogate(final, note_t), target.unsqueeze(0), dim=-1
            ).item()
        if best is None or score < best[0]:
            best = (score, final.squeeze(0).cpu())
    return best
```

Notes:

- **Multi-start is mandatory.** The loss surface is non-convex; a single start will get stuck. 16 starts is a reasonable default; 64 for difficult targets.
- **Clamp aggressively.** Params drift outside `[0, 1]` during Adam steps; clamping every forward pass keeps the surrogate in-distribution.
- **Enum params leak through.** Gradient descent happily produces 0.73 for a 3-value enum. Post-process by snapping to the nearest category before sending to the synth. Use a straight-through estimator if you want to round-trip properly.

Runtime: ~1–5 seconds for 16 starts × 300 steps on a GPU. Fast enough to run interactively.

## Strategy 3 — Gradient-free refinement (CMA-ES)

Some regions of parameter space have discontinuous gradients (high-Q filter self-oscillation thresholds, filter-type transitions). CMA-ES is the standard tool — it handles non-smooth surfaces and adapts its search covariance on the fly. See [[CMA-ES for Patch Search]] for background.

```python
import cma
import numpy as np

def cmaes_invert(surrogate, target_emb, note, d_params, x0, sigma0=0.15,
                 maxiter=200, popsize=None, device="cuda"):
    """x0: seed from strategy 1 or 2. sigma0: initial step (0.15 is generous)."""
    es = cma.CMAEvolutionStrategy(
        x0=x0.cpu().numpy().tolist(),
        sigma0=sigma0,
        inopts={"bounds": [[0.0] * d_params, [1.0] * d_params],
                "maxiter": maxiter, "popsize": popsize},
    )

    while not es.stop():
        xs = es.ask()                                    # batch of candidates
        batch = torch.tensor(np.array(xs), dtype=torch.float32, device=device)
        note_t = torch.full((len(xs), 1), note / 127.0, device=device)
        with torch.no_grad():
            preds = surrogate(batch, note_t)
            scores = 1 - torch.nn.functional.cosine_similarity(
                preds, target_emb.unsqueeze(0).to(device), dim=-1
            )
        es.tell(xs, scores.cpu().numpy().tolist())       # lower = better

    return np.array(es.best.x), es.best.f
```

CMA-ES at popsize 16, 200 iterations = 3200 surrogate forward passes. On a GPU that's sub-second total. Hardware-loop CMA-ES (Bucket 7) is where this gets expensive.

[Nevergrad](https://github.com/facebookresearch/nevergrad) is an alternative wrapper with a unified interface to CMA-ES, DE, PSO. Useful if you want to try other algorithms without rewriting the loop.

## Handling enumerated parameters

Waveform select, filter type, PWM source — these are categorical and the three search strategies handle them differently:

- **Strategy 1** predicts a continuous value; snap to nearest category on output. Training loss can use cross-entropy on the enum dimensions and MSE on the rest.
- **Strategy 2** needs straight-through: forward pass uses snapped value, backward pass pretends it was continuous. `torch.nn.functional.one_hot` + Gumbel-softmax works; the naive "round in forward, pass gradient in backward" also works for v1.
- **Strategy 3** handles them natively. Treat enum dimensions as continuous in `[0, 1]` and snap after `es.ask()`, before feeding to the surrogate. CMA-ES doesn't care about discontinuity.

In practice: use strategy 3 for enum-heavy synths and skip the straight-through complications of strategy 2.

## The pipeline end-to-end

```python
def invert(target_wav, synth_profile, surrogate, embedder):
    # 1. embed the target
    audio, sr = sf.read(target_wav)
    target_emb = torch.tensor(embedder.encodec_embed(audio, sr))

    # 2. pick a representative note (pitch tracker output, or profile midpoint)
    note = estimate_dominant_note(audio, sr)            # e.g. 60

    # 3. seed (skip if no inverse model)
    seed_score, seed_x = grad_invert(
        surrogate, target_emb, note, d_params=len(synth_profile.modulated)
    )

    # 4. CMA-ES refine
    best_x, best_loss = cmaes_invert(
        surrogate, target_emb, note, d_params=len(synth_profile.modulated),
        x0=seed_x
    )

    # 5. snap enums, scale each param to synth units, produce MIDI patch
    patch = synth_profile.from_normalised(best_x)
    return patch, best_loss
```

Output is a `patch` object that [[02 Capture - Hardware|Bucket 2 V2]]'s `MidiSender` can push onto the real synth in seconds.

## Outputs you produce

Per target clip:

```
patches/bird_01/
├── target.wav              # the input
├── target_embedding.npy
├── candidates.parquet      # top-K patches with surrogate scores
├── best_patch.syx          # load-ready SysEx / CC dump for the synth
├── best_patch.yaml         # human-readable parameter values
└── rendered.wav            # (from Bucket 7) actual synth output using best_patch
```

## Validation

1. **Held-out captures round-trip.** Take a test-set capture from Bucket 3, treat it as a target, run inversion. The recovered params should be close (not necessarily equal — see "one-to-many" above) to the true params. Cosine distance of recovered-embedding vs target-embedding under ~0.1 is a success.
2. **Real target clip.** Pick an actual target (bird recording). Run inversion, push the patch to the synth via V2, record the result. Listen and compare.
3. **Stability.** Run inversion 10 times on the same target with different random seeds. Best-of-10 scores should vary by <10%. Larger variance means the surrogate has too many local minima or CMA-ES is under-budgeted.

## Dependencies

```bash
pip install cma nevergrad torch numpy soundfile
# optional
pip install crepe   # for the dominant-note estimator
```

## Uncertainties to flag

- **Dominant-note estimation is crude.** `estimate_dominant_note()` is not trivial for non-musical sounds (birds, rustling). Options: CREPE (high quality, heavy), pYIN (medium, cheap), or brute-forcing inversion across all candidate notes and picking the best. The brute-force path is simpler and probably what v1 wants.
- **One-to-many inversion is real.** Multiple different patches can map to the same embedding. The "best" patch by surrogate is not unique. If reproducibility across runs matters (e.g. you're building a preset library), fix the random seed; if you want diverse mimics of the same target, don't.
- **Surrogate ≠ real synth.** Everything in Bucket 6 optimises against the *surrogate's* predicted embedding, not the real synth's. A patch with surrogate score 0.02 may render on the real synth at score 0.15. That gap is exactly what [[07 Refine|Bucket 7]] is for. Flag it explicitly so expectations match reality.
- **CMA-ES dimensionality.** CMA-ES works well up to ~50 dimensions; above that it slows markedly. For a 15-param synth this is fine; for 30+ params (complex FM or wavetable), expect longer runs and consider Latin-Hypercube seeded populations.
- **Multi-target / sequence inversion is out of scope here.** Inverting a full bird song (not a single call) is [[06b Live|Bucket 6b]]'s job — the frame-wise streaming path. Bucket 6 is one-target, one-patch. Don't try to extend it to sequences; the architectures are different enough to warrant separate code paths.

## When Bucket 6 is done

- `invert(target_wav, profile, surrogate, embedder) -> patch` runs end-to-end.
- Held-out round-trip validation passes (cosine distance < 0.1 on recovered embedding).
- A real target clip produces a patch that, when loaded on the synth, is audibly in the neighbourhood of the target.
- Outputs are written to `patches/<target>/` with best_patch + rendered + scores.

## References

- [pycma](https://github.com/CMA-ES/pycma) · [Nevergrad](https://github.com/facebookresearch/nevergrad) · [BoTorch](https://botorch.org/) · [Optuna](https://optuna.org/)
- [InverSynth (Barkan 2019)](https://arxiv.org/abs/1812.06349) · [Sound2Synth (Chen 2022)](https://sound2synth.github.io/)
- [Han et al. — Synth matching with differentiable DSP](https://arxiv.org/abs/2309.16709)
- [[Inverse Synth Models]] · [[CMA-ES for Patch Search]] · [[Differentiable Audio Loss]]
- [[05 Surrogate]] · [[06b Live]] · [[07 Refine]]

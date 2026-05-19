---
tags: [build, 05-surrogate, surrogate, forward-model, training]
created: 2026-04-19
updated: 2026-04-19
---

# 05 Surrogate — Forward Model Overview

> [!info] Goal
> Train a fast, differentiable neural approximation of the target synth. Input: `(param_vector, note)` conditioning. Output: predicted audio embedding **in EnCodec latent space** (see [[04 Embed|Bucket 4]]). Used by [[06 Invert]] for gradient-based patch search — no real-time synthesis required at inference time.

The surrogate is the pipeline's workhorse. Training a dataset of captures into a network that answers "what does this patch sound like" in milliseconds is what makes the whole inversion loop tractable.

> [!note] 2026-04-19 — targets updated for EnCodec
> Earlier drafts trained the surrogate against L2-normed CLAP vectors (`d_embed=512`, `F.normalize` on the head). [[04 Embed|Bucket 4]] now uses EnCodec's continuous latents (`d_embed=128` with mean pooling, or `d_embed=256` with mean+std), which are **not** L2-normed and encode meaningful magnitude. This doc has been updated accordingly. If you were mid-training on CLAP targets, nothing is broken — swap `d_embed`, drop the normalise line, and retrain against the EnCodec embeddings file.

## Where Bucket 5 sits

```
Bucket 3 dataset (samples.parquet)   Bucket 4 (encodec_embeddings.npy)
           │                                      │
           └──────────────┬───────────────────────┘
                          ▼
                 Bucket 5 surrogate
                    f(p, note) ≈ encodec_embed(synth(p, note))
                          │
                          ▼
                  frozen checkpoint
                          │
                          ▼
               Bucket 6 inversion search
```

## Prerequisites

- [[03 Dataset|Bucket 3]]: `samples.parquet` with a `p_*` column per modulated parameter.
- [[04 Embed|Bucket 4]]: aligned `encodec_embeddings.npy` (128-d mean-pooled, or 256-d with mean+std).
- GPU with ≥8 GB VRAM for comfortable training. CPU works for the v1 MLP but is 10–20× slower. EnCodec's 128-d targets are smaller than the old CLAP 512-d, so memory pressure is lower — a 6 GB card is fine.

## Model design

### v1: plain MLP (start here)

```
inputs:
  params:  [d_params]  in [0, 1]           (e.g. 15 for OB-Xf / Peak)
  note:    [1]         normalised to [0, 1]   (or a small learned embedding)
  velocity: [1]        normalised             (optional; fix to constant in v1)

concat → Linear(d+2 → 256) → SiLU → Linear(256 → 512) → SiLU
       → Linear(512 → 512) → SiLU → Linear(512 → d_embed)
       → (no L2 normalise — EnCodec latents encode magnitude)

output: predicted EnCodec latent, shape [d_embed]   (128 for mean-pool, 256 for mean+std)
```

Roughly 300k–700k params with `d_embed=128`. Trains in 10–30 min on a laptop GPU for a 100k-sample dataset. This is not a placeholder — a plain MLP genuinely gets you 80% of the way there. Transformers are not required for Bucket 5 v1.

If you opted for `mean+std` pooling in Bucket 4, just change `d_embed=256`. Everything else stays.

### v2: note conditioning via a small embedding

MIDI note as a scalar works but loses the periodic structure. A cheap upgrade:

```python
note_emb = nn.Embedding(128, 16)            # one row per MIDI note
x = torch.cat([params, note_emb(note_idx), velocity], dim=-1)
```

The embedding learns that note 60 and 72 are related (octaves) in a way the scalar representation doesn't force. Worth the ~2k extra parameters.

### v3: temporal variant (skip until v1 works)

For sounds where timbre evolves within the probe window (bird calls, modulated patches), predict a sequence of frame embeddings rather than a single vector:

```
params + note → linear projection → GRU decoder (or small transformer)
             → per-frame EnCodec-latent sequence [T, 128]
```

Train against the frame-wise EnCodec sequences from [[04 Embed|Bucket 4]] (`encodec_sequence(...)`). EnCodec runs at 75 Hz natively, so a 2.5 s capture is ~188 frames. This is strictly a post-v1 upgrade — get the static case converging first. The temporal surrogate also feeds [[06b Live|Bucket 6b]]'s sequence-aware inverse training, so v3 + the Bucket 3 sequence-capture addendum land at the same time.

## Training recipe (v1)

```python
# s05_surrogate/train.py — annotated sketch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd


class SurrogateDataset(Dataset):
    def __init__(self, parquet_path, embeddings_path, param_names):
        self.df = pd.read_parquet(parquet_path)
        self.emb = np.load(embeddings_path, mmap_mode="r")
        self.param_cols = [f"p_{n}" for n in param_names]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        params = torch.tensor(row[self.param_cols].values, dtype=torch.float32)
        note   = torch.tensor(row["note"] / 127.0, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(self.emb[i], dtype=torch.float32)
        return params, note, target


class Surrogate(nn.Module):
    def __init__(self, d_params, d_embed=128, hidden=512):
        """d_embed=128 for mean-pooled EnCodec latents; set to 256 for mean+std pooling."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_params + 1, 256), nn.SiLU(),
            nn.Linear(256, hidden),      nn.SiLU(),
            nn.Linear(hidden, hidden),   nn.SiLU(),
            nn.Linear(hidden, d_embed),
        )

    def forward(self, params, note):
        x = torch.cat([params, note], dim=-1)
        return self.net(x)                # no L2 normalise — EnCodec latents carry magnitude


def loss_fn(pred, target):
    mse = F.mse_loss(pred, target)
    cos = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
    return mse + 0.3 * cos                # mse dominates; cos keeps direction honest
```

> [!tip] Per-dim target standardisation
> EnCodec latents can have per-dimension magnitude variance spanning 1–2 orders. Log mean/std per dim over the training set once, and either (a) standardise targets to zero-mean unit-std before training and invert on inference, or (b) leave targets raw and trust the MSE loss to sort it out. (a) is cleaner; (b) is fine and avoids a preprocessing step. Pick one and document it in the manifest.

Hyperparameters that work:

- Optimiser: AdamW, lr = 3e-4, weight_decay = 1e-4.
- Batch size: 1024 on a 12 GB GPU. Scales linearly with VRAM.
- Epochs: 30–50. Early-stop on validation loss plateau.
- LR schedule: cosine decay to 1e-5.
- Augmentation: none. The data is deterministic; augmentation hurts.

Expected validation cosine-distance (lower is better). Numbers below are **rough estimates** against EnCodec latents — the previous table was calibrated against L2-normed CLAP. EnCodec cosine distances generally sit in similar ranges but are slightly looser (the target space is richer, so hitting it exactly is harder). Track your own baseline after the first full training run and use that as the sanity anchor going forward:

| Synth               | Params | Dataset | MLP v1 cos-dist (est.) | Notes |
| ------------------- | ------ | ------- | ---------------------- | ----- |
| OB-Xf (V1 software) | 15     | 100k    | 0.04–0.08              | Deterministic; tighter fits possible. |
| Juno-106 (hardware) | 12     | 50k     | 0.06–0.12              | Hardware noise limits the floor. |
| Novation Peak       | 17     | 100k    | 0.05–0.10              | FPGA oscillators help. |

Also track **relative MSE** — since EnCodec latents aren't unit-norm, MSE is the primary loss signal; cosine is a directional sanity check. A well-trained v1 surrogate should have val-MSE roughly an order of magnitude below the target variance.

If your cosine numbers are 2–3× worse than this range, something is off — either the dataset is undersampled, the embedding path is miswired, or the label/input alignment in the DataLoader is wrong. Worth spending an hour debugging before training longer.

## Splits and reproducibility

Re-use the split strategy from [[03 Dataset|Bucket 3]] (`hash` modulo a small prime → train/val/test). Never touch the test set during surrogate training — it's reserved for [[06 Invert]] inversion accuracy evaluation.

Log to Weights & Biases / MLflow / a simple CSV:

- Training config hash (model class + hyperparams).
- Dataset manifest hash.
- Embedding manifest hash (EnCodec library version + model variant, e.g. `encodec==0.1.1` + `encodec_model_48khz` + pool mode).
- Git SHA.
- Final train / val / test loss (MSE + cosine).

This trio lets "which surrogate ran against which dataset against which embedding" stay legible six months later.

## Export for downstream use

Three consumers; each needs a different format:

1. **[[06 Invert|Bucket 6]]** — gradient descent + CMA-ES both want the PyTorch model in-process. Save as `state_dict.pt` plus the model class definition.
2. **[[06b Live|Bucket 6b]]** — the live path uses the *inverse* model, not the surrogate, but sanity runs benefit from the surrogate in ONNX for cheap round-tripping. Export via `torch.onnx.export`.
3. **Max / Pd integration (Bucket 8)** — via [nn~](https://github.com/acids-ircam/nn_tilde). Same ONNX artefact plus a small JSON descriptor.

```python
torch.onnx.export(
    model.cpu().eval(),
    (torch.zeros(1, d_params), torch.zeros(1, 1)),
    "surrogate.onnx",
    input_names=["params", "note"],
    output_names=["encodec_latent"],       # 128-d mean-pooled, or 256-d mean+std
    dynamic_axes={"params": {0: "batch"}, "note": {0: "batch"}},
    opset_version=17,
)
```

## Validation sanity checks

Before declaring done:

1. **Per-parameter sweep.** Hold all params at reset values, sweep one param from 0 → 1, plot cosine distance from the reset-capture EnCodec latent. Should be monotonic-ish and smooth for continuous params, step-ish for enums.
2. **Round-trip a known capture.** Pick a test-set row, predict the latent from `(params, note)`, compare to the ground-truth EnCodec latent. Cosine similarity ≥ 0.9 and MSE below the per-dim target variance means the surrogate is behaving. Cosine around 0.7 means something's wrong — check DataLoader alignment first.
3. **Gradient sanity.** Backprop `d(cosine distance)/d(params)` from a random target. The gradient should be non-trivially non-zero and not all clustered on one param.

Failing any of these means [[06 Invert]] will struggle — fix before progressing.

## Dependencies

```bash
pip install torch torchvision lightning wandb pandas pyarrow numpy
# optional
pip install onnx onnxruntime
```

## Uncertainties to flag

- **No L2 normalisation on EnCodec targets.** Previous drafts kept `F.normalize(y, dim=-1)` on the surrogate head to match CLAP's unit-sphere convention. EnCodec latents are not unit-norm and the magnitude carries signal — normalising away the magnitude throws away fine timbral information. Keep the head linear, let MSE dominate the loss, and use cosine only as a directional regulariser (weight 0.3).
- **Per-dim target scale.** EnCodec latents have uneven per-dimension variance. Either standardise targets (mean-zero, unit-std per dim) before training, or verify empirically that the raw MSE loss converges; pick one and document it. Mixing the two across runs will make cosine-distance numbers non-comparable.
- **Temporal modelling need.** For bird-like targets, the v1 static surrogate is likely insufficient; the v3 per-frame EnCodec-sequence variant with a GRU/transformer decoder will probably be needed. The jump from v1 to v3 is the largest latent-cost item in this bucket. Flag it as a probable, not certain, upgrade — and note it's the same capture pass that Bucket 6b needs ("Bucket 3.5").
- **Dataset size vs surrogate capacity.** The MLP sizes above are calibrated for 50k–200k samples and 128-d EnCodec targets. 128-d is easier to fit than 512-d CLAP, so a smaller hidden dim (256–384) may be enough — measure before growing. Scaling to 500k samples won't help an undersized MLP; grow `hidden` before throwing more data at it. Past 5M params you're in transformer-residual territory.
- **Enum parameters leak gradient.** The surrogate sees enums as continuous `[0,1]` inputs, but the synth snaps them to categories. The gradient through those dimensions is a lie. Bucket 6 handles this explicitly; flag it here so you remember.
- **Note conditioning scaling.** Note/127 is a cheap hack. For V2 hardware synths with stretched tuning or non-linear pitch response, a learned 128-entry embedding may materially outperform the scalar. Measure before deciding.
- **Distribution shift between V1 and V2.** A surrogate trained on OB-Xf captures (V1) will not work on Juno-106 captures (V2) — each synth gets its own surrogate. Don't share weights across synths until/unless you deliberately design a multi-synth conditioning scheme (an open question noted in the pipeline doc).
- **Expected-loss numbers above are rough.** The cosine-distance ranges in the table were sketched against CLAP targets and then nudged for EnCodec. Treat them as order-of-magnitude guidance; calibrate to your own first training run, then use that baseline going forward.

## When Bucket 5 is done

- Validation MSE and cosine distance against EnCodec targets in the expected range for your synth (calibrated to your own baseline).
- Gradient / sweep / round-trip sanity checks all pass.
- Checkpoint + ONNX + manifest committed to `surrogate/<synth_id>_<timestamp>/`. Manifest records the EnCodec library version, model variant (`encodec_model_48khz`), pool mode (`mean` or `meanstd`), and target standardisation choice.
- Test set is untouched.

## References

- [PyTorch](https://pytorch.org/) · [Lightning](https://lightning.ai/) · [Weights & Biases](https://wandb.ai/)
- [ONNX opset 17 notes](https://onnx.ai/onnx/operators/) · [nn~ / nn_tilde](https://github.com/acids-ircam/nn_tilde)
- [EnCodec (Meta)](https://github.com/facebookresearch/encodec) · [paper](https://arxiv.org/abs/2210.13438) — the embedding target space.
- [Yee-King & Roth 2018](https://ieeexplore.ieee.org/document/8521773) — the foundational "surrogate + inversion" architecture for VST synths.
- [InverSynth (Barkan et al. 2019)](https://arxiv.org/abs/1812.06349)
- [[Neural Surrogate for Synth]] · [[Differentiable Audio Loss]]
- [[03 Dataset]] · [[04 Embed]] · [[06 Invert]] · [[06b Live]]

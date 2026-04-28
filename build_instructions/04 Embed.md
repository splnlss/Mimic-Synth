---
tags: [build, 04-embed, embedding, loss, audio-ml]
created: 2026-04-19
updated: 2026-04-19
---

# 04 Embed — Audio Embedding Overview

> [!info] Goal
> Convert every audio capture from [[03 Dataset|Bucket 3]] into a fixed-size vector (or short sequence of vectors) such that "sounds perceptually similar" ≈ "close in this vector space." This embedding is effectively the loss function for [[05 Surrogate]] and [[06 Invert]].

The embedding is the most quietly consequential choice in the whole project. It decides what "similar" means. Different embeddings have different blind spots — a semantic text-aligned model like CLAP may nail instrument-family similarity but be tone-deaf to the microsecond transients that make a bird call feel alive.

> [!note] 2026-04-19 — primary embedder switched to EnCodec
> Earlier drafts of this bucket used CLAP as the primary embedder. For this project the loss function's job is to tell "how similar do two synth renders sound", not semantic retrieval. Reconstruction-trained latents (EnCodec) preserve fine timbral detail that semantic-retrieval embeddings average away. CLAP is retained below as an optional semantic lens, not the main loss.

## Where Bucket 4 sits

```
Bucket 3 dataset                Bucket 4 (this doc)                  Bucket 5 + 6
  wav/*.wav ─────► embedding model ─────► embeddings.parquet ─────► loss / target
  samples.parquet                          (or .npy memmap)
```

Bucket 4 has two outputs: a function `embed(audio) -> vector` that runs on any new audio, and a pre-computed embedding file aligned 1-to-1 with `samples.parquet` so the trainer doesn't re-embed the training set every epoch.

## Prerequisites

- [[03 Dataset|Bucket 3]] complete: `samples.parquet` + `wav/` directory on disk at **48 kHz** (Bucket 2 V1 is already set to 48 kHz; V2 Hardware profiles should match).
- GPU optional but helpful. EnCodec on CPU is ~5 ms/clip; on GPU ~1 ms/clip. For a 100k dataset that's 10 min vs. 2 min. Both tractable.

## Options and when to pick each

| Embedding               | Dim        | Strength                                   | Weakness for this task                        | v1 recommendation |
| ----------------------- | ---------- | ------------------------------------------ | --------------------------------------------- | ----------------- |
| [EnCodec](https://github.com/facebookresearch/encodec) latents (continuous) | 128 × T    | Reconstruction-oriented, great fine timbral detail, streaming-friendly, no training needed | Not perceptually normalised; 128×T is higher than a CLAP vector | ✅ **default**    |
| Multi-resolution STFT   | ~5–10 k    | Cheap, differentiable, classic             | Not a learned embedding — no semantics        | ✅ as regulariser |
| [LAION-CLAP](https://github.com/LAION-AI/CLAP) | 512 | Text-aligned, semantically rich | Blurs fine timbral texture; ~10 Hz-ish frame rate | optional auxiliary |
| [DAC](https://github.com/descriptinc/descript-audio-codec) | 1024 × T | Highest-fidelity codec; excellent timbre  | Big, slow; overkill for training-time loss   | later             |
| [OpenL3](https://github.com/marl/openl3) / [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) / [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) | 512–2048  | Old reliable, tested                       | Less aligned with synth matching than EnCodec | fallback          |
| [BEATs](https://github.com/microsoft/unilm/tree/master/beats) / [MERT](https://github.com/yizhilll/MERT) | 768        | Music-trained; better on pitched content   | Less evaluated for synth timbre               | experiment        |

See [[CLAP vs EnCodec for Synth Matching]] for the deeper tradeoff analysis.

### v1 recommendation

- **Primary:** EnCodec 48 kHz encoder (continuous latents, pre-quantiser). Time-averaged for static Bucket 3 captures; frame-wise for live use in [[06b Live|Bucket 6b]].
- **Secondary loss term:** multi-resolution STFT (via [auraloss](https://github.com/csteinmetz1/auraloss) or the NumPy one-liner below) — cheap, differentiable, catches transients the latents smooth over.
- **Held in reserve:** CLAP as an auxiliary semantic term if you need retrieval-style matching (e.g. "give me five patches similar in *category* to this target"). Not needed for the core inversion loop.

## Why EnCodec over CLAP for this use case

The embedding's only job in this pipeline is to act as a perceptual loss between two synth renders. CLAP is trained to put "bird call" and "canary" near each other — semantic abstractions that compress timbral detail. EnCodec is trained to reconstruct the input waveform from the latents — it must preserve the fine structure (transient shape, harmonic balance, amplitude envelope) that makes one bird call sound different from another. For loss-between-two-renders, reconstruction fidelity beats semantic coverage.

Other practical wins:

- **Same embedder in training (Bucket 5) and live (Bucket 6b).** EnCodec is already a streaming encoder. No separate "distil CLAP into a student" step; no risk of the surrogate and the live model disagreeing about what "similar" means.
- **No checkpoint hunting.** `pip install encodec` → one line → works. CLAP's LAION-CLAP ships multiple checkpoints with non-obvious naming; the right one varies per task.
- **48 kHz native.** Matches the Bucket 2 V1 capture rate (now set to 48 kHz). No resample step.
- **Frame rate.** EnCodec 48 kHz model produces latents at 75 Hz — natural fit for the Bucket 6b live path's control-rate needs.

The tradeoff worth knowing: EnCodec latents aren't perceptually metric the way CLAP embeddings (roughly) are — pairwise distances don't cleanly map onto "how different do these sound to a human." For a loss function between two synth outputs of *similar* character that's not a blocker. If you later need semantic retrieval across the dataset, layer CLAP on as a second embedding in parallel.

## Single-vector vs frame-wise embeddings

EnCodec always produces a sequence `[128, T]`. For Bucket 3 and 5, collapse it to a single vector via time-averaging (or mean + std concatenated) so the surrogate has a fixed-size regression target. For Bucket 6b, keep the frame-wise sequence:

```
full WAV ─► EnCodec encoder ─► [128, T]
                                     ├─► time-average → [128]         (Bucket 5 target)
                                     ├─► mean ⊕ std  → [256]         (richer static; optional)
                                     └─► raw sequence → streaming inverse (Bucket 6b)
```

For [[03 Dataset|Bucket 3]] captures (2.5 s of held note), time-average is fine. For live frames (100–400 ms hop), use the sequence. Same weights, same API, different pooling.

## Implementation sketch

```python
# s04_embed/embed.py
"""EnCodec primary + multi-res STFT auxiliary, single API for static and live use."""
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio

SR = 48000            # EnCodec 48 k native; matches Bucket 2 V1 capture rate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(self):
        # 48 kHz stereo-trained encoder; we'll feed it mono-as-stereo then pool.
        self.enc = EncodecModel.encodec_model_48khz().to(DEVICE)
        self.enc.set_target_bandwidth(6.0)    # irrelevant for our use; we drop the quantiser
        self.enc.eval()

    @torch.no_grad()
    def encodec_embed(self, audio: np.ndarray, sr: int, pool: str = "mean") -> np.ndarray:
        """Returns a 128-d (pool='mean') or 256-d (pool='meanstd') vector."""
        x = torch.from_numpy(audio).float()
        if x.ndim == 1:                       # mono → (1, 1, T)
            x = x.unsqueeze(0).unsqueeze(0)
        x = convert_audio(x, sr, SR, self.enc.channels).to(DEVICE)
        # Grab the encoder's continuous latents *before* residual VQ.
        # The library exposes this as ._encode -> returns (emb, codes) pair in some versions.
        emb = self.enc.encoder(x)             # [B, 128, T]
        emb = emb.squeeze(0).cpu().numpy()    # [128, T]
        if pool == "mean":
            v = emb.mean(axis=1)                              # [128]
        elif pool == "meanstd":
            v = np.concatenate([emb.mean(axis=1), emb.std(axis=1)])  # [256]
        elif pool == "none":
            return emb                                        # [128, T]
        else:
            raise ValueError(pool)
        return v.astype(np.float32)

    @torch.no_grad()
    def encodec_sequence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Frame-wise latents for Bucket 6b. Returns [128, T] at 75 Hz frame rate."""
        return self.encodec_embed(audio, sr, pool="none")

    def mrstft_feats(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Concatenated log-magnitude spectra at multiple FFT sizes.
        Stays in NumPy — used as an auxiliary loss, not a target."""
        feats = []
        for n_fft in (256, 512, 1024, 2048):
            spec = np.abs(_stft(audio, n_fft=n_fft, hop=n_fft // 4))
            feats.append(np.log1p(spec).mean(axis=1))  # collapse time
        return np.concatenate(feats).astype(np.float32)
```

Then a one-shot indexer:

```python
# s04_embed/index_dataset.py
import pandas as pd, numpy as np
from tqdm import tqdm
from .embed import Embedder

df = pd.read_parquet("dataset/samples.parquet")
emb = Embedder()
out = np.zeros((len(df), 128), dtype=np.float32)   # or 256 if pool="meanstd"

for i, row in enumerate(tqdm(df.itertuples(), total=len(df))):
    audio, sr = sf.read(row.wav)
    out[i] = emb.encodec_embed(audio, sr, pool="mean")

np.save("dataset/encodec_embeddings.npy", out)
# optional: also save mrstft_features.npy for the aux loss term
```

Memory-mapped `.npy` is fine up to ~1M embeddings; switch to LMDB / Zarr beyond that.

> [!warning] Check EnCodec's API surface
> The `encodec` pip package has gone through minor renames (`.encoder(x)`, `._encode(x)`, `.encode_to_latent(x)` depending on version). Pin a specific version in the manifest and verify the continuous-latent path works before building the whole dataset. As of time-of-writing, grabbing `self.enc.encoder(x)` directly on the 48 kHz model returns the pre-quantiser latents.

## Loss for Bucket 5 surrogate training

With EnCodec as the target, the surrogate regresses onto a 128-d (or 256-d) unnormalised vector. Unlike L2-normed CLAP, **do not** normalise EnCodec latents to the unit sphere — they encode magnitude information that matters. Use:

```python
def loss_fn(pred, target):
    mse = F.mse_loss(pred, target)                                # main term
    cos = 1 - F.cosine_similarity(pred, target, dim=-1).mean()    # shape term
    # mrstft aux term uses the raw audio, not the embedding — computed in a separate path
    return mse + 0.3 * cos
```

The cosine term stays useful as a shape prior even when the targets aren't L2-normed. Adjust [[05 Surrogate]]'s surrogate head accordingly: drop the `F.normalize(y, dim=-1)` line that was there for CLAP.

## Validation

Before moving on, spot-check that the embedding actually ranks similar-sounding captures close and dissimilar ones far:

1. Pick 10 random "anchor" captures from the dataset.
2. For each, find the 5 nearest and 5 farthest by cosine distance in embedding space.
3. Listen to them back-to-back. Near ones should sound obviously similar; far ones obviously different.

If step 3 fails (near ≈ far), the embedding isn't capturing what you care about and the surrogate will never train well. EnCodec's reconstruction objective usually makes this pass easily; if it doesn't, check that you're pulling the pre-quantiser latents, not the quantised codes.

Additional: a 2D UMAP / t-SNE of a sampled 10k subset, coloured by the most important parameter (e.g. Cutoff). A good embedding will show a visible gradient along that axis.

## Dependencies

```bash
pip install encodec soundfile librosa torch tqdm pandas pyarrow
# for the aux STFT loss during Bucket 5 training
pip install auraloss
# optional — only if you want CLAP as a secondary semantic lens
pip install laion-clap
# optional alternatives
pip install descript-audio-codec openl3 umap-learn
```

The EnCodec checkpoint (~80 MB for the 48 kHz model) downloads on first use; pin the library version in the manifest.

## Live-mode notes (for [[06b Live|Bucket 6b]])

Same module, different pooling:

- **Hop:** ~13 ms (75 Hz frame rate, EnCodec's native) up to 100–300 ms windows for averaged stability. Latency isn't the driver here (see Bucket 6b) — stability is. Average the last N=4–16 frames before feeding the inverse.
- **Inference budget:** EnCodec encoder runs at ~3 ms/frame on CPU, ~1 ms on GPU. Well within any realistic budget.
- **Same weights.** No distillation step. The surrogate was trained in EnCodec space, the inverse is trained in EnCodec space, the live encoder *is* EnCodec. One embedder, end-to-end.

This is the biggest advantage of the EnCodec choice: it collapses the entire "two-embedder" architecture that CLAP would have required.

## Uncertainties to flag

- **EnCodec's continuous-latent API depends on library version.** The public API exposes quantised codes first-class; the continuous pre-quantiser output requires reaching into `.encoder(x)` or equivalent. Pin version and unit-test that you're getting `[128, T]`, not `[n_codebooks, T]` integers.
- **48 kHz mono-as-stereo packaging.** EnCodec 48 k was trained on stereo. Feeding mono as a `(1, 1, T)` tensor works but might nudge the latents compared to genuine stereo. If your capture rig is strictly mono, this is a constant offset, not a bug. If you later capture stereo (filter sweeps with stereo chorus) the embedding will differ — consistency matters more than absolute correctness for this project.
- **Latent range is unbounded.** Unlike L2-normed CLAP, EnCodec latents can have large-magnitude outliers. Log the target statistics (min/max/mean/std per dim across the dataset) once indexing completes; if any dim is orders of magnitude larger than others, per-dim standardisation before feeding the surrogate is worth considering.
- **CLAP is still worth having around as a second opinion.** For the spot-check step ("does this embedding say these two birds are similar?"), CLAP's semantic prior is a useful sanity lens. Nothing in the pipeline forbids saving both an `encodec_embeddings.npy` and a `clap_embeddings.npy` — the second one just isn't the training target.
- **Dataset size vs EnCodec dim.** 128-d is modest; 256-d with mean+std pooling is still tractable. Don't default to the full frame sequence as the surrogate target — it blows up Bucket 5's memory and training time for marginal gain on the static case.
- **DAC as the escape hatch, not EnCodec → CLAP.** If EnCodec falls short on timbral precision (unlikely but possible), the next step is DAC — also reconstruction-trained, higher fidelity, higher compute cost. Don't reach for CLAP as "more precise"; it's more *semantic*, which is the opposite direction.

## When Bucket 4 is done

- `embed(audio, sr) -> vector` function runs end-to-end on a held-out WAV and returns a 128-d (or 256-d) float32 vector.
- `encodec_embeddings.npy` aligned 1-to-1 with `samples.parquet` is written to disk.
- Nearest / farthest spot-check passes qualitatively.
- Embedding choice + `encodec` library version + pool mode are committed to the dataset manifest.
- `Embedder` class exposes both `encodec_embed` (static pooled) and `encodec_sequence` (frame-wise) so Bucket 6b can reuse it without modification.

## References

- [EnCodec (Meta)](https://github.com/facebookresearch/encodec) · [paper](https://arxiv.org/abs/2210.13438)
- [DAC (Descript)](https://github.com/descriptinc/descript-audio-codec)
- [LAION-CLAP](https://github.com/LAION-AI/CLAP) · [paper](https://arxiv.org/abs/2211.06687) — kept as optional semantic lens.
- [OpenL3](https://github.com/marl/openl3) · [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) · [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- [Multi-res STFT loss reference implementation (auraloss)](https://github.com/csteinmetz1/auraloss)
- [[Audio Embeddings]] · [[CLAP vs EnCodec for Synth Matching]] · [[Differentiable Audio Loss]]
- [[03 Dataset]] · [[05 Surrogate]] · [[06b Live]]

"""
Pre-compute EnCodec embeddings for every capture in a dataset.

Reads samples.parquet + WAVs, runs the Embedder on each, and writes
embeddings as a .npy file aligned 1-to-1 with the parquet.

Supports checkpoint/resume: if interrupted, re-run and choose [c]ontinue
to skip already-embedded rows. Checkpoints every 500 rows.

Usage:
    python -m s04_embed.index_dataset \
        --dataset s02_capture/data/ \
        --out s04_embed/data/ \
        --pool mean
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from .embed import Embedder

CHECKPOINT_EVERY = 500
NPY_FILENAME = "encodec_embeddings.npy"
DONE_FILENAME = "encodec_embeddings_done.npy"


def _prompt_resume_or_overwrite(npy_path: Path, n_done: int, n_total: int) -> str:
    while True:
        ans = input(
            f"\nExisting embeddings found at {npy_path}\n"
            f"  {n_done}/{n_total} rows complete.\n"
            "  [c]ontinue (skip already-embedded rows)\n"
            "  [o]verwrite (delete and start fresh)\n"
            "  [a]bort\n"
            "Choice [c/o/a]: "
        ).strip().lower()
        if ans in ("c", "continue"):
            return "resume"
        if ans in ("o", "overwrite"):
            return "overwrite"
        if ans in ("a", "abort", ""):
            return "abort"
        print("  (please answer c, o, or a)")


def _flush(npy_path: Path, done_path: Path, arr: np.ndarray, done: np.ndarray) -> None:
    """Atomic checkpoint: write to .tmp then rename."""
    tmp_npy = npy_path.parent / (npy_path.stem + "_tmp.npy")
    tmp_done = done_path.parent / (done_path.stem + "_tmp.npy")
    np.save(str(tmp_npy), arr)
    np.save(str(tmp_done), done)
    tmp_npy.replace(npy_path)
    tmp_done.replace(done_path)


def _resolve_wav_root(dataset_dir: Path, df: pd.DataFrame) -> Path:
    """Find the root directory for WAV paths in the parquet."""
    wav_root = dataset_dir
    if not df.empty:
        sample_wav = df["wav"].iloc[0]
        if not (dataset_dir / sample_wav).exists():
            if (dataset_dir.parent / sample_wav).exists():
                wav_root = dataset_dir.parent
    return wav_root


def _log_latent_stats(arr: np.ndarray, done: np.ndarray, dim: int) -> None:
    valid = arr[done]
    if len(valid) == 0:
        return
    print(f"\nLatent statistics ({len(valid)} valid embeddings, {dim}-d):")
    print(f"  Per-dim mean:  [{valid.mean(axis=0).min():.3f}, {valid.mean(axis=0).max():.3f}]")
    print(f"  Per-dim std:   [{valid.std(axis=0).min():.3f}, {valid.std(axis=0).max():.3f}]")
    print(f"  Global range:  [{valid.min():.3f}, {valid.max():.3f}]")


def index_dataset(
    dataset_dir: Path | str,
    out_dir: Path | str,
    pool: str = "mean",
) -> Path:
    """Embed all captures and write encodec_embeddings.npy.

    Supports resume: if a partial .npy + _done.npy exist, prompts to
    continue, overwrite, or abort. Checkpoints every 500 rows.

    Returns the path to the output .npy file.
    """
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = dataset_dir / "samples.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No samples.parquet in {dataset_dir}")
    df = pd.read_parquet(parquet_path)

    if df.empty:
        raise ValueError("samples.parquet is empty — nothing to embed")

    wav_root = _resolve_wav_root(dataset_dir, df)

    dim = 128 if pool == "mean" else 256 if pool == "meanstd" else None
    if dim is None:
        raise ValueError("index_dataset requires pool='mean' or 'meanstd', not 'none'")

    npy_path = out_dir / NPY_FILENAME
    done_path = out_dir / DONE_FILENAME

    # ── Resume / overwrite logic ─────────────────────────────────────────
    out_arr = np.zeros((len(df), dim), dtype=np.float32)
    done_mask = np.zeros(len(df), dtype=bool)

    if npy_path.exists() and done_path.exists():
        existing_arr = np.load(npy_path)
        existing_done = np.load(done_path)
        # Validate shape compatibility with current parquet
        if existing_arr.shape == (len(df), dim) and existing_done.shape == (len(df),):
            n_done = int(existing_done.sum())
            choice = _prompt_resume_or_overwrite(npy_path, n_done, len(df))
            if choice == "abort":
                print("Aborted.")
                return npy_path
            elif choice == "resume":
                out_arr = existing_arr
                done_mask = existing_done
                print(f"Resuming — {n_done} embeddings already complete.")
            else:  # overwrite
                npy_path.unlink(missing_ok=True)
                done_path.unlink(missing_ok=True)
                print("Cleared existing embeddings.")
        else:
            print(f"Existing embeddings shape mismatch (expected ({len(df)}, {dim}), "
                  f"got {existing_arr.shape}) — starting fresh.")
    elif npy_path.exists():
        # .npy without done mask — can't tell what's done, start fresh
        npy_path.unlink(missing_ok=True)

    # ── Embedding loop ───────────────────────────────────────────────────
    n_already = int(done_mask.sum())
    n_remaining = len(df) - n_already
    emb = Embedder()
    skipped = 0
    newly_done = 0

    pbar = tqdm(total=len(df), initial=n_already, desc="embedding")

    for i, row in enumerate(df.itertuples()):
        if done_mask[i]:
            continue

        wav_path = Path(row.wav)
        if not wav_path.is_absolute():
            wav_path = wav_root / wav_path
        if not wav_path.exists():
            skipped += 1
            done_mask[i] = True  # mark as done (zero vector — WAV missing)
            pbar.update(1)
            continue

        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        out_arr[i] = emb.encodec_embed(audio, sr, pool=pool)
        done_mask[i] = True
        newly_done += 1
        pbar.update(1)

        if newly_done % CHECKPOINT_EVERY == 0:
            _flush(npy_path, done_path, out_arr, done_mask)

    pbar.close()

    # Final save
    _flush(npy_path, done_path, out_arr, done_mask)

    _log_latent_stats(out_arr, done_mask, dim)
    if skipped:
        print(f"Warning: {skipped}/{len(df)} WAVs not found — zeros in output")
    print(f"Saved {npy_path} — shape {out_arr.shape}, "
          f"{int(done_mask.sum())}/{len(df)} complete")
    return npy_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pre-compute EnCodec embeddings for a capture dataset."
    )
    ap.add_argument("--dataset", required=True,
                    help="Path to dataset dir (with samples.parquet + wav/)")
    ap.add_argument("--out", required=True,
                    help="Output directory for encodec_embeddings.npy")
    ap.add_argument("--pool", choices=["mean", "meanstd"], default="mean",
                    help="Pooling mode: mean (128-d) or meanstd (256-d)")
    args = ap.parse_args()

    index_dataset(args.dataset, args.out, pool=args.pool)
    return 0


if __name__ == "__main__":
    sys.exit(main())

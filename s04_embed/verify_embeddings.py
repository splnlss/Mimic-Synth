"""
Post-hoc embedding verifier.

Checks that encodec_embeddings.npy is complete, well-formed, and
qualitatively sensible (no NaN/Inf, reasonable latent range, nearest
neighbours sound similar).

Usage:
    python -m s04_embed.verify_embeddings \
        --embeddings s04_embed/data/encodec_embeddings.npy \
        --dataset s02_capture/data/
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import defaults as _defs


@dataclass
class EmbeddingReport:
    n_rows: int = 0
    n_parquet: int = 0
    dim: int = 0
    n_complete: int = 0
    n_zero: int = 0
    has_nan: bool = False
    has_inf: bool = False
    global_min: float = 0.0
    global_max: float = 0.0
    per_dim_mean_range: tuple[float, float] = (0.0, 0.0)
    per_dim_std_range: tuple[float, float] = (0.0, 0.0)


def verify_embeddings(
    npy_path: Path | str,
    dataset_dir: Path | str,
) -> tuple[EmbeddingReport, bool]:
    """Verify embedding file against its dataset.

    Returns (report, passed).
    """
    npy_path = Path(npy_path)
    dataset_dir = Path(dataset_dir)

    if not npy_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {npy_path}")

    parquet_path = dataset_dir / "samples.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No samples.parquet in {dataset_dir}")

    arr = np.load(npy_path)
    df = pd.read_parquet(parquet_path)

    report = EmbeddingReport()
    report.n_rows = arr.shape[0]
    report.n_parquet = len(df)
    report.dim = arr.shape[1] if arr.ndim == 2 else 0

    passed = True

    # Shape alignment
    if report.n_rows != report.n_parquet:
        passed = False

    # Dimension check
    if report.dim not in (128, 256):
        passed = False

    # NaN / Inf
    report.has_nan = bool(np.any(np.isnan(arr)))
    report.has_inf = bool(np.any(np.isinf(arr)))
    if report.has_nan or report.has_inf:
        passed = False

    # Completeness — rows that are all zeros are either missing WAVs or unembed
    row_norms = np.linalg.norm(arr, axis=1)
    report.n_zero = int(np.sum(row_norms == 0))
    report.n_complete = report.n_rows - report.n_zero

    # Check done mask if available
    done_path = npy_path.parent / "encodec_embeddings_done.npy"
    if done_path.exists():
        done = np.load(done_path)
        if done.shape == (report.n_rows,):
            n_done = int(done.sum())
            if n_done < report.n_rows:
                passed = False  # incomplete run

    # Latent statistics (on non-zero rows)
    valid = arr[row_norms > 0]
    if len(valid) > 0:
        report.global_min = float(valid.min())
        report.global_max = float(valid.max())
        per_dim_mean = valid.mean(axis=0)
        per_dim_std = valid.std(axis=0)
        report.per_dim_mean_range = (float(per_dim_mean.min()), float(per_dim_mean.max()))
        report.per_dim_std_range = (float(per_dim_std.min()), float(per_dim_std.max()))

        # Sanity: if all dims have identical values, something is wrong
        if per_dim_std.max() < 1e-6:
            passed = False

    return report, passed


def print_report(report: EmbeddingReport, passed: bool) -> None:
    print(f"\n{'=' * 60}")
    print(f"Embedding Verification Report")
    print(f"{'=' * 60}")
    print(f"  Rows:          {report.n_rows}")
    print(f"  Parquet rows:  {report.n_parquet}")
    print(f"  Aligned:       {'YES' if report.n_rows == report.n_parquet else 'NO'}")
    print(f"  Dimension:     {report.dim}")
    print(f"  Complete:      {report.n_complete}/{report.n_rows} "
          f"({report.n_zero} zero-rows)")
    print(f"  NaN:           {'YES' if report.has_nan else 'no'}")
    print(f"  Inf:           {'YES' if report.has_inf else 'no'}")
    if report.n_complete > 0:
        print(f"  Global range:  [{report.global_min:.3f}, {report.global_max:.3f}]")
        print(f"  Per-dim mean:  [{report.per_dim_mean_range[0]:.3f}, "
              f"{report.per_dim_mean_range[1]:.3f}]")
        print(f"  Per-dim std:   [{report.per_dim_std_range[0]:.3f}, "
              f"{report.per_dim_std_range[1]:.3f}]")
    print(f"\n  Result:        {'PASS' if passed else 'FAIL'}")
    print(f"{'=' * 60}")


def neighbor_spot_check(
    npy_path: Path | str,
    dataset_dir: Path | str,
    n_anchors: int = 10,
    k: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Pick random anchors, find nearest/farthest by cosine distance.

    Returns a list of dicts with anchor index, nearest indices, farthest
    indices, and their distances. Useful for qualitative listening tests.
    """
    arr = np.load(npy_path)
    df = pd.read_parquet(Path(dataset_dir) / "samples.parquet")

    # Only consider non-zero rows
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    valid_mask = norms.squeeze() > 0
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < n_anchors + k:
        raise ValueError(f"Not enough valid embeddings ({len(valid_indices)}) "
                         f"for spot check with {n_anchors} anchors and k={k}")

    # L2-normalise for cosine similarity
    arr_norm = arr.copy()
    arr_norm[valid_mask] = arr[valid_mask] / norms[valid_mask]

    rng = np.random.default_rng(seed)
    anchor_indices = rng.choice(valid_indices, size=n_anchors, replace=False)

    results = []
    for anchor_idx in anchor_indices:
        # Cosine similarity with all valid rows
        sims = arr_norm[valid_indices] @ arr_norm[anchor_idx]
        sorted_order = np.argsort(sims)

        # Nearest (highest similarity, excluding self)
        nearest_local = sorted_order[-(k + 1):][::-1]
        nearest_local = [j for j in nearest_local if valid_indices[j] != anchor_idx][:k]
        nearest_global = valid_indices[nearest_local]

        # Farthest (lowest similarity)
        farthest_local = sorted_order[:k]
        farthest_global = valid_indices[farthest_local]

        results.append({
            "anchor_idx": int(anchor_idx),
            "anchor_hash": str(df.iloc[anchor_idx]["hash"]),
            "anchor_note": int(df.iloc[anchor_idx]["note"]),
            "nearest": [
                {"idx": int(idx), "hash": str(df.iloc[idx]["hash"]),
                 "note": int(df.iloc[idx]["note"]),
                 "sim": float(sims[np.where(valid_indices == idx)[0][0]])}
                for idx in nearest_global
            ],
            "farthest": [
                {"idx": int(idx), "hash": str(df.iloc[idx]["hash"]),
                 "note": int(df.iloc[idx]["note"]),
                 "sim": float(sims[np.where(valid_indices == idx)[0][0]])}
                for idx in farthest_global
            ],
        })

    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify EnCodec embeddings against dataset."
    )
    ap.add_argument("--embeddings", default=str(_defs.S04_EMBEDDINGS),
                    help="Path to encodec_embeddings.npy")
    ap.add_argument("--dataset", default=str(_defs.S02_DIR),
                    help="Path to dataset dir (with samples.parquet)")
    ap.add_argument("--spot-check", action="store_true",
                    help="Run nearest/farthest neighbor spot-check")
    args = ap.parse_args()

    report, passed = verify_embeddings(args.embeddings, args.dataset)
    print_report(report, passed)

    if args.spot_check and passed:
        print("\nNearest / farthest neighbor spot-check:")
        results = neighbor_spot_check(args.embeddings, args.dataset)
        for r in results:
            print(f"\n  Anchor: idx={r['anchor_idx']} "
                  f"hash={r['anchor_hash']} note={r['anchor_note']}")
            for n in r["nearest"][:3]:
                print(f"    Near:  idx={n['idx']} hash={n['hash']} "
                      f"note={n['note']} sim={n['sim']:.3f}")
            for f in r["farthest"][:3]:
                print(f"    Far:   idx={f['idx']} hash={f['hash']} "
                      f"note={f['note']} sim={f['sim']:.3f}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

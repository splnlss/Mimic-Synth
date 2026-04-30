"""Held-out validation of the full inversion pipeline.

Reproduces the train/val/test split from s05_surrogate/train.py (random_split
seed=42, 80/10/10), then runs inversion on each test-split capture and reports
cosine distance of the recovered embedding vs the ground-truth embedding.

Pass criterion (from build doc): mean cosine distance < 0.1.

Stability check (--stability): runs 10 independent inversions on the first
test sample and reports best-score variance. Should be < 10% of the mean.

Usage:
    conda activate mimic-synth
    python -m s06_invert.validate \\
        --surrogate s05_surrogate/runs/run_20260429_145056/state_dict.pt \\
        --dataset s03_dataset/data/samples.parquet \\
        --embeddings s04_embed/data/encodec_embeddings.npy \\
        --profile s01_profiles/obxf.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import random_split
from tqdm import tqdm

from s05_surrogate.model import Surrogate
from s06_invert.grad_search import grad_invert
from s06_invert.cmaes_search import cmaes_invert


def _load_surrogate(checkpoint_path: Path, device: str) -> tuple[Surrogate, dict]:
    manifest_path = checkpoint_path.parent / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    model = Surrogate(input_dim=manifest["input_dim"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model, manifest


def _test_indices(n_total: int) -> list[int]:
    """Reproduce the test split from train.py."""
    train_size = int(0.8 * n_total)
    val_size = int(0.1 * n_total)
    test_size = n_total - train_size - val_size
    _, _, test_ds = random_split(
        range(n_total), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    return list(test_ds)


def validate(
    surrogate_checkpoint: Path,
    dataset_path: Path,
    embeddings_path: Path,
    profile_path: Path,
    n_starts: int = 16,
    grad_steps: int = 300,
    cmaes_maxiter: int = 200,
    stability: bool = False,
    device: str | None = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    surrogate, manifest = _load_surrogate(surrogate_checkpoint, device)
    param_cols = manifest["param_cols"]
    d_params = len(param_cols)

    df = pd.read_parquet(dataset_path)
    embeddings = np.load(embeddings_path, mmap_mode="r")

    test_idx = _test_indices(len(df))
    print(f"Test split: {len(test_idx)} samples")

    scores = []
    for i in tqdm(test_idx, desc="validating"):
        row = df.iloc[i]
        target_emb = torch.tensor(embeddings[i], dtype=torch.float32)
        note = int(row["note"])

        grad_score, grad_params = grad_invert(
            surrogate, target_emb, note, d_params,
            n_starts=n_starts, steps=grad_steps, device=device,
        )
        _, cmaes_params = cmaes_invert(
            surrogate, target_emb, note, d_params, grad_params,
            maxiter=cmaes_maxiter, device=device,
        )

        final = torch.tensor(cmaes_params, dtype=torch.float32, device=device).unsqueeze(0)
        note_t = torch.full((1,), note / 127.0, device=device)
        with torch.no_grad():
            pred = surrogate(final, note_t)
        recovered_score = (
            1.0 - F.cosine_similarity(pred, target_emb.to(device).unsqueeze(0), dim=-1)
        ).item()
        scores.append(recovered_score)

    mean_score = float(np.mean(scores))
    p90_score = float(np.percentile(scores, 90))
    passed = mean_score < 0.1

    print(f"\nValidation results ({len(scores)} test samples):")
    print(f"  Mean cosine distance : {mean_score:.4f}  {'PASS' if passed else 'FAIL'} (threshold: 0.1)")
    print(f"  p90 cosine distance  : {p90_score:.4f}")

    result = {"mean": mean_score, "p90": p90_score, "passed": passed, "n": len(scores)}

    if stability and len(test_idx) > 0:
        i0 = test_idx[0]
        target_emb = torch.tensor(embeddings[i0], dtype=torch.float32)
        note = int(df.iloc[i0]["note"])
        stab_scores = []
        print("\nStability check (10 independent inversions)...")
        for _ in tqdm(range(10), desc="stability"):
            s, p = grad_invert(
                surrogate, target_emb, note, d_params,
                n_starts=n_starts, steps=grad_steps, device=device,
            )
            _, cp = cmaes_invert(
                surrogate, target_emb, note, d_params, p,
                maxiter=cmaes_maxiter, device=device,
            )
            final = torch.tensor(cp, dtype=torch.float32, device=device).unsqueeze(0)
            note_t = torch.full((1,), note / 127.0, device=device)
            with torch.no_grad():
                pred = surrogate(final, note_t)
            stab_scores.append(
                (1.0 - F.cosine_similarity(pred, target_emb.to(device).unsqueeze(0), dim=-1)).item()
            )
        stab_mean = float(np.mean(stab_scores))
        stab_std = float(np.std(stab_scores))
        stab_pct = stab_std / stab_mean * 100 if stab_mean > 0 else 0
        stab_passed = stab_pct < 10.0
        print(f"  Stability: mean={stab_mean:.4f} std={stab_std:.4f} "
              f"({stab_pct:.1f}% of mean)  {'PASS' if stab_passed else 'FAIL'}")
        result["stability_pct"] = stab_pct
        result["stability_passed"] = stab_passed

    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate surrogate inversion on held-out test split."
    )
    ap.add_argument("--surrogate", required=True, help="Path to state_dict.pt")
    ap.add_argument("--dataset", required=True, help="Path to samples.parquet")
    ap.add_argument("--embeddings", required=True, help="Path to encodec_embeddings.npy")
    ap.add_argument("--profile", required=True, help="Path to synth profile YAML")
    ap.add_argument("--n-starts", type=int, default=16)
    ap.add_argument("--grad-steps", type=int, default=300)
    ap.add_argument("--cmaes-maxiter", type=int, default=200)
    ap.add_argument("--stability", action="store_true",
                    help="Run stability check on first test sample")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    result = validate(
        surrogate_checkpoint=Path(args.surrogate),
        dataset_path=Path(args.dataset),
        embeddings_path=Path(args.embeddings),
        profile_path=Path(args.profile),
        n_starts=args.n_starts,
        grad_steps=args.grad_steps,
        cmaes_maxiter=args.cmaes_maxiter,
        stability=args.stability,
        device=args.device,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

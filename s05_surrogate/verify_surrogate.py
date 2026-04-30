"""Sanity checks for a trained surrogate.

Three checks, in order of decisiveness:

1. **Round-trip on held-out test split** (primary). For each test capture,
   forward-pass the ground-truth (params, note) and compare to the real
   EnCodec embedding. Spec criterion: mean cosine similarity ≥ 0.9.
   Requires --dataset and --embeddings.

2. **Per-parameter sweep**. Hold all params at reset values from the
   profile, sweep one param 0→1, check output actually moves. Optional
   --profile; falls back to all-zeros baseline if omitted.

3. **Gradient non-degeneracy**. Random params + random target — gradients
   should flow non-trivially. Catches dead-relu / collapsed models.

Usage:
    python -m s05_surrogate.verify_surrogate \\
        --checkpoint s05_surrogate/runs/<run_id>/state_dict.pt \\
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

from .model import Surrogate


def _load_surrogate(checkpoint_path: Path, input_dim_override: int | None, device: str) -> tuple[Surrogate, dict | None]:
    manifest_path = checkpoint_path.parent / "manifest.json"
    manifest = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        input_dim = manifest["input_dim"]
    elif input_dim_override is not None:
        input_dim = input_dim_override
    else:
        raise ValueError(
            f"manifest.json not found alongside {checkpoint_path}; pass --input-dim explicitly"
        )

    model = Surrogate(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model, manifest


def _test_indices(n_total: int) -> list[int]:
    """Reproduce the test split from train.py (random_split seed=42, 80/10/10)."""
    train_size = int(0.8 * n_total)
    val_size = int(0.1 * n_total)
    test_size = n_total - train_size - val_size
    _, _, test_ds = random_split(
        range(n_total), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    return list(test_ds)


def round_trip_check(
    model: Surrogate,
    param_cols: list[str],
    dataset_path: Path,
    embeddings_path: Path,
    device: str,
) -> dict:
    df = pd.read_parquet(dataset_path)
    emb = np.load(embeddings_path)
    test_idx = _test_indices(len(df))
    test_df = df.iloc[test_idx]

    params_t = torch.tensor(test_df[param_cols].values.astype(np.float32), dtype=torch.float32).to(device)
    notes_t = torch.tensor(test_df["note"].values.astype(np.float32) / 127.0, dtype=torch.float32).to(device)
    target_t = torch.tensor(emb[test_idx], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(params_t, notes_t)

    cos_sim = F.cosine_similarity(pred, target_t, dim=-1)
    mse_per = F.mse_loss(pred, target_t, reduction="none").mean(dim=-1)
    target_var = target_t.var(dim=0).mean().item()

    mean_sim = float(cos_sim.mean())
    p10_sim = float(cos_sim.quantile(0.1))
    min_sim = float(cos_sim.min())
    mean_mse = float(mse_per.mean())
    mse_ratio = mean_mse / target_var if target_var > 0 else float("inf")

    passed = mean_sim >= 0.9
    print("[1] Round-trip on test split (ground-truth params → predicted embedding)")
    print(f"    test samples         : {len(test_idx)}")
    print(f"    cosine similarity    : mean={mean_sim:.4f}  p10={p10_sim:.4f}  min={min_sim:.4f}")
    print(f"    MSE / target variance: {mse_ratio:.4f}  (smaller = better)")
    print(f"    {'PASS' if passed else 'FAIL'}  (spec: mean cosine sim ≥ 0.9, got {mean_sim:.4f})")
    print()
    return {"check": "round_trip", "passed": passed, "mean_sim": mean_sim,
            "p10_sim": p10_sim, "min_sim": min_sim, "mse_ratio": mse_ratio,
            "n": len(test_idx)}


def sweep_check(
    model: Surrogate,
    param_cols: list[str],
    profile_path: Path | None,
    device: str,
    sweep_param_idx: int = 0,
    n_steps: int = 21,
) -> dict:
    d_params = len(param_cols)
    baseline = torch.zeros(d_params, device=device)

    if profile_path is not None:
        with open(profile_path) as f:
            profile = yaml.safe_load(f)
        reset = profile.get("reset", {})
        for i, col in enumerate(param_cols):
            name = col.removeprefix("p_")
            if name in reset:
                baseline[i] = float(reset[name])

    sweep_vals = torch.linspace(0.0, 1.0, n_steps, device=device)
    sweep_p = baseline.unsqueeze(0).repeat(n_steps, 1)
    sweep_p[:, sweep_param_idx] = sweep_vals
    sweep_n = torch.full((n_steps,), 60.0 / 127.0, device=device)

    with torch.no_grad():
        out = model(sweep_p, sweep_n)
        ref = out[n_steps // 2]
        cos_dist = (1.0 - F.cosine_similarity(out, ref.unsqueeze(0).expand_as(out), dim=-1))

    delta_norm = float((out[1:] - out[:-1]).norm(dim=-1).mean())
    cos_range = float(cos_dist.max() - cos_dist.min())
    passed = delta_norm > 1e-3

    print(f"[2] Per-parameter sweep ({param_cols[sweep_param_idx]})")
    print(f"    baseline             : {'reset values from profile' if profile_path else 'all zeros'}")
    print(f"    mean step-to-step Δ  : {delta_norm:.4f}")
    print(f"    cosine dist range    : {cos_range:.4f}")
    print(f"    {'PASS' if passed else 'FAIL'}  (output should move with param; static output = dead model)")
    print()
    return {"check": "sweep", "passed": passed, "delta_norm": delta_norm,
            "cos_range": cos_range, "param": param_cols[sweep_param_idx]}


def gradient_check(model: Surrogate, d_params: int, device: str) -> dict:
    params = torch.rand(1, d_params, requires_grad=True, device=device)
    note = torch.tensor([60.0 / 127.0], requires_grad=True, device=device)
    target = torch.randn(1, 128, device=device)

    pred = model(params, note)
    loss = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    loss.backward()

    grad_norm = float(params.grad.norm())
    n_active = int((params.grad.abs() > 1e-6).sum())
    passed = grad_norm > 1e-5 and n_active >= d_params // 2

    print("[3] Gradient non-degeneracy (random target)")
    print(f"    gradient norm        : {grad_norm:.4f}")
    print(f"    active dims          : {n_active}/{d_params}")
    print(f"    {'PASS' if passed else 'FAIL'}  (low grad norm or many dead dims = inversion will fail)")
    print()
    return {"check": "gradient", "passed": passed, "grad_norm": grad_norm,
            "active_dims": n_active}


def verify(
    checkpoint_path: Path,
    dataset_path: Path | None,
    embeddings_path: Path | None,
    profile_path: Path | None,
    input_dim_override: int | None = None,
    device: str | None = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, manifest = _load_surrogate(checkpoint_path, input_dim_override, device)
    if manifest is not None:
        param_cols = manifest["param_cols"]
    elif dataset_path is not None:
        df = pd.read_parquet(dataset_path)
        param_cols = [c for c in df.columns if c.startswith("p_")]
    else:
        param_cols = [f"p_{i}" for i in range(model.net[0].in_features - 1)]

    d_params = len(param_cols)
    print(f"--- Surrogate Verification ({checkpoint_path}) ---")
    print(f"    d_params={d_params}  device={device}")
    print()

    results = []

    if dataset_path is not None and embeddings_path is not None:
        results.append(round_trip_check(model, param_cols, dataset_path, embeddings_path, device))
    else:
        print("[1] Round-trip on test split — SKIPPED (pass --dataset and --embeddings)")
        print()

    results.append(sweep_check(model, param_cols, profile_path, device))
    results.append(gradient_check(model, d_params, device))

    all_passed = all(r["passed"] for r in results)
    print(f"--- Overall: {'PASS' if all_passed else 'FAIL'} ---")
    return {"all_passed": all_passed, "checks": results}


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a trained surrogate.")
    ap.add_argument("--checkpoint", required=True, help="Path to state_dict.pt")
    ap.add_argument("--dataset", default=None,
                    help="Path to samples.parquet (enables round-trip check)")
    ap.add_argument("--embeddings", default=None,
                    help="Path to encodec_embeddings.npy (enables round-trip check)")
    ap.add_argument("--profile", default=None,
                    help="Synth profile YAML (enables reset-value sweep baseline)")
    ap.add_argument("--input-dim", type=int, default=None,
                    help="Override input_dim if manifest.json is missing")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    result = verify(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=Path(args.dataset) if args.dataset else None,
        embeddings_path=Path(args.embeddings) if args.embeddings else None,
        profile_path=Path(args.profile) if args.profile else None,
        input_dim_override=args.input_dim,
        device=args.device,
    )
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

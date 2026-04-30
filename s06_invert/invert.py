"""End-to-end patch inversion: target WAV → best synth parameter vector.

Pipeline:
  1. Embed target audio with EnCodec (same embedder as Bucket 4).
  2. For each candidate note (profile notes, or --note if supplied):
       a. Multi-start gradient descent through frozen surrogate.
       b. CMA-ES refinement seeded from grad result.
  3. Pick note/params with lowest cosine distance.
  4. Write patches/<target_stem>/ with best_patch.yaml, candidates.parquet,
     target_embedding.npy.

Usage:
    conda activate mimic-synth
    python -m s06_invert.invert \\
        --target path/to/target.wav \\
        --surrogate s05_surrogate/runs/run_20260429_145056/state_dict.pt \\
        --profile s01_profiles/obxf.yaml \\
        --out patches/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import torch
import yaml

from s05_surrogate.model import Surrogate
import defaults as _defs
from s06_invert.grad_search import grad_invert
from s06_invert.cmaes_search import cmaes_invert


def _load_surrogate(checkpoint_path: Path, device: str) -> tuple[Surrogate, dict]:
    manifest_path = checkpoint_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found alongside {checkpoint_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    input_dim = manifest["input_dim"]
    model = Surrogate(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model, manifest


def _embed_target(wav_path: Path, device: str) -> np.ndarray:
    from s04_embed.embed import Embedder
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    embedder = Embedder(device=device)
    return embedder.encodec_embed(audio, sr, pool="mean")  # [128]


def _estimate_best_note(wav_path: Path, profile_notes: list[int]) -> int:
    """Estimate the best MIDI note for the target audio based on frequency analysis."""
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Simple peak frequency detection on the middle 50% of the sample
    mid_start = len(audio) // 4
    mid_end = 3 * len(audio) // 4
    segment = audio[mid_start:mid_end]
    
    # Compute Power Spectral Density
    n_fft = min(len(segment), 2048)
    freqs, psd = signal.welch(segment, sr, nperseg=n_fft)
    peak_freq = freqs[np.argmax(psd)]
    
    if peak_freq <= 0:
        return profile_notes[len(profile_notes)//2]
    
    target_midi = 12 * np.log2(peak_freq / 440.0) + 69
    best_note = min(profile_notes, key=lambda n: abs(n - target_midi))
    
    print(f"Detected peak frequency: {peak_freq:.1f} Hz (~MIDI {target_midi:.1f})")
    print(f"Auto-selected closest profile note: {best_note}")
    return best_note


def invert(
    target_wav: Path,
    surrogate_checkpoint: Path,
    profile_path: Path,
    out_dir: Path,
    note: int | None = None,
    n_starts: int = 32,
    grad_steps: int = 500,
    cmaes_maxiter: int = 400,
    device: str | None = None,
) -> dict:
    """Run inversion and write output files. Returns best result dict."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    surrogate, manifest = _load_surrogate(surrogate_checkpoint, device)
    param_cols = manifest["param_cols"]  # ["p_Osc 1 Pitch", ...]
    d_params = len(param_cols)

    target_emb_np = _embed_target(target_wav, device)
    target_emb = torch.tensor(target_emb_np, dtype=torch.float32)

    profile_notes = profile["probe"]["notes"]
    if note is None:
        note = _estimate_best_note(target_wav, profile_notes)
    
    candidate_notes = [note]

    candidates = []
    for n in candidate_notes:
        grad_score, grad_params = grad_invert(
            surrogate, target_emb, n, d_params,
            n_starts=n_starts, steps=grad_steps, device=device,
        )
        cmaes_score, cmaes_params_np = cmaes_invert(
            surrogate, target_emb, n, d_params, grad_params,
            maxiter=cmaes_maxiter, device=device,
        )
        candidates.append({
            "note": n,
            "method": "grad",
            "score": grad_score,
            "params": grad_params.numpy(),
        })
        candidates.append({
            "note": n,
            "method": "cmaes",
            "score": cmaes_score,
            "params": cmaes_params_np,
        })

    candidates.sort(key=lambda c: c["score"])
    best = candidates[0]

    target_stem = target_wav.stem
    run_dir = out_dir / target_stem
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(run_dir / "target_embedding.npy"), target_emb_np)

    rows = []
    for c in candidates:
        row = {"note": c["note"], "method": c["method"], "score": c["score"]}
        for col, val in zip(param_cols, c["params"]):
            row[col] = float(val)
        rows.append(row)
    pd.DataFrame(rows).to_parquet(run_dir / "candidates.parquet")

    best_patch = {
        "target": str(target_wav),
        "note": int(best["note"]),
        "method": best["method"],
        "score": float(best["score"]),
        "params": {col.removeprefix("p_"): float(v)
                   for col, v in zip(param_cols, best["params"])},
    }
    with open(run_dir / "best_patch.yaml", "w") as f:
        yaml.dump(best_patch, f, default_flow_style=False, sort_keys=False)

    print(f"Best score: {best['score']:.4f}  note={best['note']}  method={best['method']}")
    print(f"Output: {run_dir}")
    return best_patch


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Invert a target audio clip to synth parameters."
    )
    ap.add_argument("--target", required=True, help="Path to target WAV/FLAC")
    ap.add_argument("--surrogate", default=None,
                    help="Path to surrogate state_dict.pt (default: latest run in S05_RUNS_DIR)")
    ap.add_argument("--profile", default=str(_defs.PROFILE_PATH), help="Path to synth profile YAML")
    ap.add_argument("--out", default=str(_defs.S06_PATCHES_DIR), help="Output directory for patches/")
    ap.add_argument("--note", type=int, default=None,
                    help="MIDI note (default: brute-force over profile notes)")
    ap.add_argument("--n-starts", type=int, default=32,
                    help="Gradient descent random starts per note (default: 32)")
    ap.add_argument("--grad-steps", type=int, default=500,
                    help="Adam steps per start (default: 500)")
    ap.add_argument("--cmaes-maxiter", type=int, default=400,
                    help="CMA-ES iterations per note (default: 400)")
    ap.add_argument("--device", default=None,
                    help="Torch device (default: cuda if available)")
    args = ap.parse_args()

    if args.surrogate is None:
        runs = sorted(_defs.S05_RUNS_DIR.glob("run_*")) if _defs.S05_RUNS_DIR.exists() else []
        if not runs:
            ap.error("No surrogate runs found in S05_RUNS_DIR; pass --surrogate explicitly.")
        args.surrogate = str(runs[-1] / "state_dict.pt")

    invert(
        target_wav=Path(args.target),
        surrogate_checkpoint=Path(args.surrogate),
        profile_path=Path(args.profile),
        out_dir=Path(args.out),
        note=args.note,
        n_starts=args.n_starts,
        grad_steps=args.grad_steps,
        cmaes_maxiter=args.cmaes_maxiter,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

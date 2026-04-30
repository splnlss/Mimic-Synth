import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

from s05_surrogate.model import Surrogate
from s06_invert.invert import _load_surrogate, _estimate_best_note
import defaults as _defs

def stream_invert(
    target_wav: Path,
    surrogate_checkpoint: Path,
    profile_path: Path,
    out_dir: Path,
    win_sec: float = 0.1,
    hop_sec: float = 0.05,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    audio, sr = sf.read(str(target_wav), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    
    surrogate, manifest = _load_surrogate(surrogate_checkpoint, device)
    param_cols = manifest["param_cols"]
    d_params = len(param_cols)
    
    from s04_embed.embed import Embedder
    embedder = Embedder(device=device)

    win_samples = int(win_sec * sr)
    hop_samples = int(hop_sec * sr)
    
    results = []
    
    # We use a sliding window to track changes
    for start in tqdm(range(0, len(audio) - win_samples, hop_samples), desc="Streaming Inversion"):
        window = audio[start : start + win_samples]
        timestamp = start / sr
        
        # 1. Embed window
        emb = embedder.encodec_embed(window, sr, pool="mean")
        emb_torch = torch.tensor(emb, dtype=torch.float32).to(device)
        
        # 2. Local pitch detection for this window
        # (Writing window to temp is slow, ideally Embedder takes raw buffer)
        # For now, let's just use a fixed note or reuse the first detection
        # Improving pitch detection to be window-based in next step
        target_midi = 12 * np.log2(np.max(np.abs(np.fft.rfft(window * np.hanning(len(window))))) * (sr/len(window)) / 440.0) + 69
        # (placeholder crude pitch)
        
        # 3. Fast Inversion: just Gradient Descent for speed (no CMA-ES here)
        from s06_invert.grad_search import grad_invert
        # Reduced starts and steps for "real-time-ish" tracking
        score, params = grad_invert(
            surrogate, emb_torch, 84, d_params, 
            n_starts=4, steps=100, device=device
        )
        
        res = {"timestamp": timestamp, "score": float(score)}
        for col, val in zip(param_cols, params.numpy()):
            res[col] = float(val)
        results.append(res)

    df = pd.DataFrame(results)
    run_dir = out_dir / target_wav.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(run_dir / "stream_params.parquet")
    print(f"Streaming tracking complete. Saved to {run_dir / 'stream_params.parquet'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--surrogate", required=True)
    ap.add_argument("--profile", default=str(_defs.PROFILE_PATH))
    ap.add_argument("--out", default=str(_defs.S06_PATCHES_DIR))
    ap.add_argument("--win-sec", type=float, default=0.05) # 20fps tracking
    ap.add_argument("--hop-sec", type=float, default=0.02)
    args = ap.parse_args()
    
    stream_invert(
        Path(args.target), Path(args.surrogate), Path(args.profile), Path(args.out),
        win_sec=args.win_sec, hop_sec=args.hop_sec
    )

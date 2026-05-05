import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import torch
import yaml
from tqdm import tqdm

from s05_surrogate.model import Surrogate
from s06_invert.invert import _load_surrogate, _estimate_best_note
import defaults as _defs

def detect_pitch_autocorr(audio, sr):
    """Detect pitch using autocorrelation method (more robust for music)."""
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    
    audio = audio * np.hanning(len(audio))
    
    autocorr = signal.correlate(audio, audio, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    indices = np.arange(len(autocorr))
    
    def find_peaks_autocorr(autocorr, min_freq=50, max_freq=2000):
        window_size = int(sr / min_freq)
        peaks, _ = signal.find_peaks(autocorr, distance=window_size)
        if len(peaks) == 0:
            return None
        
        peak_heights = autocorr[peaks]
        best_peak = peaks[np.argmax(peak_heights)]
        
        for i in range(1, len(autocorr)):
            if autocorr[i] > autocorr[best_peak]:
                best_peak = i
        
        if best_peak < 1:
            return None
        
        freq = sr / best_peak
        if min_freq <= freq <= max_freq:
            return freq
        return None
    
    pitch_hz = find_peaks_autocorr(autocorr)
    
    if pitch_hz is None:
        freqs, psd = signal.welch(audio, sr, nperseg=min(len(audio), 2048))
        pitch_hz = freqs[np.argmax(psd)]
    
    return pitch_hz

def smooth_trajectory(values, window_size=3):
    """Smooth parameter trajectory using moving average."""
    values = np.array(values)
    if len(values) <= window_size:
        return values
    
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(values, kernel, mode='same')
    
    edges = window_size // 2
    smoothed[:edges] = values[:edges]
    smoothed[-edges:] = values[-edges:]
    
    return smoothed

def stream_invert(
    target_wav: Path,
    surrogate_checkpoint: Path,
    profile_path: Path,
    out_dir: Path,
    win_sec: float = 0.1,
    hop_sec: float = 0.05,
    n_starts: int = 8,
    grad_steps: int = 200,
    smooth_window: int = 3,
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
    profile_notes = profile["probe"]["notes"]

    win_samples = int(win_sec * sr)
    hop_samples = int(hop_sec * sr)
    
    results = []
    prev_params = None
    prev_note = profile_notes[len(profile_notes)//2]
    
    for start in tqdm(range(0, len(audio) - win_samples, hop_samples), desc="Streaming Inversion"):
        window = audio[start : start + win_samples]
        timestamp = start / sr
        
        pitch_hz = detect_pitch_autocorr(window, sr)
        
        if pitch_hz is not None and pitch_hz > 0:
            target_midi = 12 * np.log2(pitch_hz / 440.0) + 69
            note = int(round(target_midi))
        else:
            note = prev_note
        
        if note not in profile_notes:
            note = min(profile_notes, key=lambda n: abs(n - note))
        
        prev_note = note
        
        emb = embedder.encodec_embed(window, sr, pool="mean")
        emb_torch = torch.tensor(emb, dtype=torch.float32).to(device)
        
        from s06_invert.grad_search import grad_invert
        
        if prev_params is not None:
            score, params = grad_invert(
                surrogate, emb_torch, note, d_params,
                n_starts=1, steps=50, device=device,
                init_params=prev_params
            )
        else:
            score, params = grad_invert(
                surrogate, emb_torch, note, d_params,
                n_starts=4, steps=50, device=device
            )
        
        prev_params = params
        
        res = {
            "timestamp": timestamp,
            "pitch_hz": float(pitch_hz) if pitch_hz else np.nan,
            "note": note,
            "score": float(score)
        }
        for col, val in zip(param_cols, params.numpy()):
            res[col] = float(val)
        results.append(res)

    df = pd.DataFrame(results)
    
    for col in param_cols:
        if col in df.columns:
            df[col] = smooth_trajectory(df[col].values, smooth_window)
    
    run_dir = out_dir / target_wav.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(run_dir / "stream_params.parquet")
    
    best_row = df.loc[df["score"].idxmin()]
    best_patch = {
        "target": str(target_wav),
        "note": int(best_row["note"]),
        "pitch_hz": float(best_row["pitch_hz"]) if not np.isnan(best_row["pitch_hz"]) else None,
        "score": float(best_row["score"]),
        "params": {col.removeprefix("p_"): float(best_row[col]) for col in param_cols},
    }
    
    with open(run_dir / "best_patch.yaml", "w") as f:
        yaml.dump(best_patch, f, default_flow_style=False, sort_keys=False)
    
    with open(run_dir / "trajectory.yaml", "w") as f:
        yaml.dump({
            "target": str(target_wav),
            "sample_rate": sr,
            "window_sec": win_sec,
            "hop_sec": hop_sec,
            "num_frames": len(df),
            "params": {col.removeprefix("p_"): [float(v) for v in df[col].values] for col in param_cols}
        }, f, default_flow_style=False, sort_keys=False)
    
    print(f"Streaming tracking complete. {len(df)} frames.")
    print(f"Best score: {best_row['score']:.4f}  note={int(best_row['note'])}")
    print(f"Saved to {run_dir / 'stream_params.parquet'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--surrogate", required=True)
    ap.add_argument("--profile", default=str(_defs.PROFILE_PATH))
    ap.add_argument("--out", default=str(_defs.S06_PATCHES_DIR))
    ap.add_argument("--win-sec", type=float, default=0.2)
    ap.add_argument("--hop-sec", type=float, default=0.1)
    ap.add_argument("--n-starts", type=int, default=4)
    ap.add_argument("--grad-steps", type=int, default=50)
    ap.add_argument("--smooth-window", type=int, default=3)
    args = ap.parse_args()
    
    stream_invert(
        Path(args.target), Path(args.surrogate), Path(args.profile), Path(args.out),
        win_sec=args.win_sec, hop_sec=args.hop_sec,
        n_starts=args.n_starts, grad_steps=args.grad_steps,
        smooth_window=args.smooth_window
    )

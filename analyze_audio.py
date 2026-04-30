#!/usr/bin/env python3
"""Compare target and rendered audio"""

import numpy as np
import soundfile as sf
from pathlib import Path
from defaults import TARGETS_DIR, S06_PATCHES_DIR

def analyze_audio(path, name):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo to mono
    
    print(f"\n{name}:")
    print(f"  Path: {path}")
    print(f"  Duration: {len(audio)/sr:.3f}s")
    print(f"  Sample rate: {sr}")
    print(f"  Max amplitude: {np.abs(audio).max():.6f}")
    print(f"  RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    print(f"  Peak dB: {20*np.log10(max(np.abs(audio).max(), 1e-10)):.1f}")
    
    # Check if normalized
    if np.abs(audio).max() > 0.95:
        print(f"  ⚠️  Audio might be clipping!")
    elif np.abs(audio).max() < 0.01:
        print(f"  ⚠️  Audio is very quiet")
    
    return audio, sr

def main():
    target_path = TARGETS_DIR / "613846_bird-call-funny.wav"
    rendered_path = S06_PATCHES_DIR / "613846_bird-call-funny" / "rendered.wav"
    
    target_audio, sr1 = analyze_audio(target_path, "TARGET")
    rendered_audio, sr2 = analyze_audio(rendered_path, "RENDERED")
    
    # Normalize rendered to match target RMS for comparison
    target_rms = np.sqrt(np.mean(target_audio**2))
    rendered_rms = np.sqrt(np.mean(rendered_audio**2))
    
    if rendered_rms > 0:
        normalized_rendered = rendered_audio * (target_rms / rendered_rms)
        print(f"\nNormalizing rendered audio: RMS {rendered_rms:.6f} -> {target_rms:.6f}")
        print(f"  Scale factor: {target_rms/rendered_rms:.3f}")
    else:
        normalized_rendered = rendered_audio
        print(f"\nRendered audio has zero RMS!")
    
    # Save normalized version
    if rendered_rms > 0 and target_rms > 0:
        normalized_path = Path("patches/613846_bird-call-funny/rendered_normalized.wav")
        sf.write(normalized_path, normalized_rendered, sr2)
        print(f"Saved normalized audio to {normalized_path}")
    
    # Compare waveforms (first 0.5s)
    n_samples = min(len(target_audio), len(rendered_audio), int(sr1 * 0.5))
    if n_samples > 0:
        print(f"\nFirst {n_samples/sr1:.3f}s comparison:")
        print(f"  Target max: {np.abs(target_audio[:n_samples]).max():.6f}")
        print(f"  Rendered max: {np.abs(rendered_audio[:n_samples]).max():.6f}")
        
        # Simple correlation
        if n_samples > 10:
            corr = np.corrcoef(target_audio[:n_samples], rendered_audio[:n_samples])[0,1]
            print(f"  Correlation: {corr:.6f}")

if __name__ == "__main__":
    main()
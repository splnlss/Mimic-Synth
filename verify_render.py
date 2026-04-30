#!/usr/bin/env python3
"""Verify the rendered audio against target"""

import numpy as np
import soundfile as sf
from pathlib import Path
import torch

# Import embedder from s04
import sys
sys.path.insert(0, str(Path(__file__).parent))
from s04_embed.embed import Embedder
from defaults import TARGETS_DIR, S06_PATCHES_DIR

def main():
    # Paths
    target_path = TARGETS_DIR / "613846_bird-call-funny.wav"
    rendered_path = S06_PATCHES_DIR / "613846_bird-call-funny" / "rendered.wav"
    target_embed_path = S06_PATCHES_DIR / "613846_bird-call-funny" / "target_embedding.npy"
    
    print(f"Target: {target_path}")
    print(f"Rendered: {rendered_path}")
    
    # Load target embedding (pre-computed during inversion)
    target_embed = np.load(target_embed_path)
    print(f"Target embedding shape: {target_embed.shape}")
    
    # Compute rendered embedding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = Embedder(device=device)
    
    audio, sr = sf.read(str(rendered_path), dtype="float32", always_2d=False)
    rendered_embed = embedder.encodec_embed(audio, sr, pool="mean")
    
    print(f"Rendered embedding shape: {rendered_embed.shape}")
    
    # Cosine similarity
    cos_sim = np.dot(target_embed, rendered_embed) / (
        np.linalg.norm(target_embed) * np.linalg.norm(rendered_embed)
    )
    cos_dist = 1 - cos_sim
    
    print(f"\nCosine similarity: {cos_sim:.6f}")
    print(f"Cosine distance: {cos_dist:.6f}")
    print(f"Original patch score: 0.0011369")
    
    # Also compute MSE
    mse = np.mean((target_embed - rendered_embed) ** 2)
    print(f"MSE: {mse:.6f}")
    
    # Check if it's close
    if cos_dist < 0.01:
        print("\n✅ Success! Rendered audio matches target well.")
    elif cos_dist < 0.05:
        print("\n⚠️  Moderate match.")
    else:
        print("\n❌ Poor match.")
        
    return cos_dist

if __name__ == "__main__":
    main()
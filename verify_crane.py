import numpy as np
import soundfile as sf
from pathlib import Path
import torch
import sys
sys.path.insert(0, str(Path(__file__).parent))
from s04_embed.embed import Embedder
from defaults import S06_PATCHES_DIR

patch_dir = S06_PATCHES_DIR / "816426_crane-bird-scream_mono"
target_embed = np.load(patch_dir / "target_embedding.npy")
print(f"Target embedding shape: {target_embed.shape}")

rendered_path = patch_dir / "rendered.wav"
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = Embedder(device=device)
audio, sr = sf.read(str(rendered_path), dtype="float32", always_2d=False)
rendered_embed = embedder.encodec_embed(audio, sr, pool="mean")
print(f"Rendered embedding shape: {rendered_embed.shape}")

cos_sim = np.dot(target_embed, rendered_embed) / (
    np.linalg.norm(target_embed) * np.linalg.norm(rendered_embed)
)
cos_dist = 1 - cos_sim
print(f"Cosine similarity: {cos_sim:.6f}")
print(f"Cosine distance: {cos_dist:.6f}")
print(f"Patch score: 0.114428")
print(f"Difference: {cos_dist - 0.114428:.6f}")

# Also compute MSE
mse = np.mean((target_embed - rendered_embed) ** 2)
print(f"MSE: {mse:.6f}")

if cos_dist < 0.01:
    print("✅ Excellent match")
elif cos_dist < 0.05:
    print("⚠️ Moderate match")
elif cos_dist < 0.1:
    print("❌ Some match")
else:
    print("❌ Poor match")
import soundfile as sf
import numpy as np
from defaults import TARGETS_DIR
path = str(TARGETS_DIR / "816426_crane-bird-scream.wav")
audio, sr = sf.read(path, dtype="float32", always_2d=False)
if audio.ndim == 2:
    audio = audio.mean(axis=1)
out_path = str(TARGETS_DIR / "816426_crane-bird-scream_mono.wav")
sf.write(out_path, audio, sr)
print(f"Saved mono version to {out_path}")
print(f"Shape: {audio.shape}, SR: {sr}")
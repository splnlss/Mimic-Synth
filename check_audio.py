import soundfile as sf
import numpy as np
from defaults import TARGETS_DIR
path = str(TARGETS_DIR / "816426_crane-bird-scream.wav")
audio, sr = sf.read(path)
print(f"Shape: {audio.shape}")
print(f"Sample rate: {sr}")
print(f"Duration: {len(audio)/sr:.2f}s")
print(f"Channels: {audio.ndim}")
if audio.ndim == 2:
    print(f"Channel shape: {audio.shape[1]}")
print(f"Data type: {audio.dtype}")
print(f"Max amplitude: {np.abs(audio).max()}")
import numpy as np
import soundfile as sf
from pathlib import Path
import scipy.signal as signal
import scipy.fft as fft
from defaults import TARGETS_DIR, S06_PATCHES_DIR

def analyze(name, path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    dur = len(audio) / sr
    rms = np.sqrt(np.mean(audio**2))
    peak = np.abs(audio).max()
    # spectral centroid
    freqs = fft.rfftfreq(len(audio), 1/sr)
    spectrum = np.abs(fft.rfft(audio))
    if spectrum.sum() > 0:
        centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
    else:
        centroid = 0
    # zero crossing rate
    zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2
    print(f"{name}:")
    print(f"  Duration: {dur:.3f}s")
    print(f"  RMS: {rms:.6f}")
    print(f"  Peak: {peak:.6f} ({20*np.log10(max(peak,1e-10)):.1f} dB)")
    print(f"  Spectral centroid: {centroid:.1f} Hz")
    print(f"  Zero-crossing rate: {zcr:.3f}")
    return audio, sr

print("="*50)
target_path = TARGETS_DIR / "816426_crane-bird-scream.wav"
rendered_path = S06_PATCHES_DIR / "816426_crane-bird-scream_mono" / "rendered.wav"
norm_path = S06_PATCHES_DIR / "816426_crane-bird-scream_mono" / "rendered_normalized.wav"

target_audio, sr = analyze("TARGET (original stereo)", target_path)
print()
rendered_audio, sr = analyze("RENDERED (synthesized)", rendered_path)
print()
norm_audio, sr = analyze("RENDERED NORMALIZED (RMS matched)", norm_path)
print()

# Compare waveforms (first second)
n = min(len(target_audio), len(rendered_audio), int(sr * 1.0))
if n > 10:
    corr = np.corrcoef(target_audio[:n], rendered_audio[:n])[0,1]
    print(f"Waveform correlation (first {n/sr:.2f}s): {corr:.4f}")
    
# Compare spectra
def compute_spectrum(audio, sr):
    freqs = fft.rfftfreq(len(audio), 1/sr)
    spectrum = np.abs(fft.rfft(audio))
    return freqs, spectrum

f1, s1 = compute_spectrum(target_audio, sr)
f2, s2 = compute_spectrum(rendered_audio, sr)
# Interpolate to common freq axis
f_min = min(f1[-1], f2[-1])
n_freq = min(len(s1), len(s2))
s1 = s1[:n_freq]
s2 = s2[:n_freq]
if np.sum(s1) > 0 and np.sum(s2) > 0:
    # Normalize spectra
    s1_norm = s1 / np.sum(s1)
    s2_norm = s2 / np.sum(s2)
    # Spectral similarity (earth mover's distance approx)
    diff = np.abs(s1_norm - s2_norm).sum() / 2
    print(f"Spectral divergence (approx): {diff:.4f}")

print("="*50)
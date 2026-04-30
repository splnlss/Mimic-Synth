import numpy as np
import soundfile as sf
import scipy.signal as signal
from defaults import TARGETS_DIR

# Load audio
path = str(TARGETS_DIR / "816426_crane-bird-scream_mono.wav")
audio, sr = sf.read(path, dtype="float32")
if audio.ndim == 2:
    audio = audio.mean(axis=1)

print(f"Duration: {len(audio)/sr:.3f}s, SR: {sr}")

# Compute spectrogram
n_fft = 2048
hop = 512
freqs = np.fft.rfftfreq(n_fft, 1/sr)
spectrogram = []
for i in range(0, len(audio) - n_fft, hop):
    frame = audio[i:i+n_fft] * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frame))
    spectrogram.append(spec)
spectrogram = np.array(spectrogram).T  # freq x time

# Find peak frequency per frame
peak_freqs = []
for t in range(spectrogram.shape[1]):
    peak_idx = np.argmax(spectrogram[:, t])
    peak_freqs.append(freqs[peak_idx])

peak_freqs = np.array(peak_freqs)
print(f"Peak frequencies range: {peak_freqs.min():.1f} - {peak_freqs.max():.1f} Hz")
print(f"Median peak frequency: {np.median(peak_freqs):.1f} Hz")

# Convert to MIDI
def hz_to_midi(freq):
    return 12 * np.log2(freq / 440.0) + 69

midi_notes = hz_to_midi(peak_freqs[peak_freqs > 0])
print(f"MIDI notes range: {midi_notes.min():.1f} - {midi_notes.max():.1f}")
print(f"Median MIDI note: {np.median(midi_notes):.1f}")

# Candidate notes in profile
candidates = [36, 48, 60, 72, 84]
print(f"\nProfile candidate notes: {candidates}")
print("Closest candidate to median:")
closest = min(candidates, key=lambda n: abs(n - np.median(midi_notes)))
print(f"  {closest} (difference: {abs(closest - np.median(midi_notes)):.1f} semitones)")

# Also compute fundamental via autocorrelation (crude)
def estimate_fundamental(audio_segment, sr):
    # autocorrelation
    corr = np.correlate(audio_segment, audio_segment, mode='full')
    corr = corr[len(corr)//2:]
    # find first peak after zero lag
    peak = np.argmax(corr[20:]) + 20
    if peak > 0:
        freq = sr / peak
        return freq
    return 0

# Try on middle segment
segment = audio[len(audio)//4:3*len(audio)//4]
f0 = estimate_fundamental(segment, sr)
print(f"\nAutocorrelation fundamental estimate: {f0:.1f} Hz -> MIDI {hz_to_midi(f0):.1f}")
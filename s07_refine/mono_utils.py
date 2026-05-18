"""Enforce mono WAV for all target inputs.

VST synthesis is mono/stereo at output but target matching works on mono
embeddings. Keeping all targets mono eliminates channel-layout ambiguity and
ensures EnCodec receives a consistent input shape.

Rules:
  1-channel  → pass through unchanged; return same path
  2-channel  → downmix L+R by averaging (standard center-mix ITU-R BS.775);
               save <stem>_mono.wav alongside the original
  3+-channel → raise ValueError; the caller must ask the user for
               instructions (5.1, ambisonic, etc. require different treatment)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import soundfile as sf

from pathlib import Path


def ensure_mono(wav_path: Path | str) -> tuple[np.ndarray, int, Path]:
    """Load `wav_path` and return a mono float32 signal.

    Args:
        wav_path: path to any WAV file (1, 2, or more channels).

    Returns:
        (audio_mono, sample_rate, mono_path) where:
            audio_mono  float32 numpy array, shape (N,)
            sample_rate integer sample rate
            mono_path   Path to the (possibly newly written) mono WAV. Equal
                        to wav_path when the input is already mono; otherwise
                        `<parent>/<stem>_mono.wav`.

    Raises:
        ValueError: if the file has more than 2 channels. The caller should
            present the user with options (e.g., pick a channel subset, use
            a downmix matrix, etc.) before retrying.
    """
    wav_path = Path(wav_path)
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    # audio shape: (samples, channels)
    n_channels = audio.shape[1]

    if n_channels == 1:
        mono = audio[:, 0]
        return mono, sr, wav_path

    if n_channels == 2:
        # Standard ITU-R BS.775 center-mix: simple L+R average is sufficient
        # for a subtractive-synth patch search where phase coherence matters
        # less than spectral content.
        mono = audio.mean(axis=1)
        mono_path = wav_path.parent / f"{wav_path.stem}_mono{wav_path.suffix}"
        sf.write(str(mono_path), mono, sr)
        print(f"[mono_utils] Stereo → mono: saved {mono_path.name}")
        return mono, sr, mono_path

    raise ValueError(
        f"{wav_path.name} has {n_channels} channels. "
        "Please provide instructions: which channels to mix, a downmix "
        "matrix, or a pre-converted mono file."
    )

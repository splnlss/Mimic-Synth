"""
EnCodec primary embedder + multi-resolution STFT auxiliary.

Single API for static (Bucket 5 surrogate target) and live (Bucket 6b) use.
The 48 kHz EnCodec encoder produces continuous pre-quantiser latents at
150 Hz frame rate (128-d per frame). For static captures these are
time-averaged to a fixed-size vector; for live use the frame sequence
is returned directly.
"""
from __future__ import annotations

import numpy as np
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio

from defaults import SAMPLE_RATE

ENCODEC_SR = 48000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _stft(audio: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Real-valued STFT returning complex magnitudes [freq_bins, frames]."""
    window = np.hanning(n_fft)
    n_frames = 1 + (len(audio) - n_fft) // hop
    out = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float64)
    for i in range(n_frames):
        frame = audio[i * hop : i * hop + n_fft] * window
        out[:, i] = np.abs(np.fft.rfft(frame))
    return out


class Embedder:
    """EnCodec 48 kHz encoder for audio embedding.

    Exposes both pooled (static) and frame-wise (streaming) outputs
    from the same weights, so Bucket 5 training and Bucket 6b live
    inference use identical embeddings.
    """

    def __init__(self, device: str | None = None, compile: bool = True):
        self.device = device or DEVICE
        self.enc = EncodecModel.encodec_model_48khz().to(self.device)
        self.enc.set_target_bandwidth(6.0)
        self.enc.eval()
        self._channels = self.enc.channels  # cache before possible compile replacement
        self._amp = self.device.startswith("cuda")
        if compile and self._amp:
            # Compile the encoder submodule only — that's what we call directly.
            # First forward pass triggers ~30s JIT warmup; subsequent calls are faster.
            self.enc.encoder = torch.compile(self.enc.encoder, mode="reduce-overhead")

    @torch.no_grad()
    def encodec_embed(
        self, audio: np.ndarray, sr: int = SAMPLE_RATE, pool: str = "mean"
    ) -> np.ndarray:
        """Embed audio to a fixed-size vector.

        Args:
            audio: 1-D float32 mono waveform.
            sr: Sample rate of the input audio.
            pool: Pooling mode — "mean" (128-d), "meanstd" (256-d),
                  or "none" (returns raw [128, T] frame sequence).

        Returns:
            np.ndarray of shape [128], [256], or [128, T] depending on pool.
        """
        x = torch.from_numpy(audio).float()
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        x = convert_audio(x, sr, ENCODEC_SR, self._channels).to(self.device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self._amp):
            emb = self.enc.encoder(x)  # [B, 128, T]
        emb = emb.float().squeeze(0).cpu().numpy()  # [128, T]

        if pool == "mean":
            return emb.mean(axis=1).astype(np.float32)  # [128]
        elif pool == "meanstd":
            return np.concatenate(
                [emb.mean(axis=1), emb.std(axis=1)]
            ).astype(np.float32)  # [256]
        elif pool == "none":
            return emb.astype(np.float32)  # [128, T]
        else:
            raise ValueError(f"Unknown pool mode: {pool!r}")

    @torch.no_grad()
    def encodec_sequence(
        self, audio: np.ndarray, sr: int = SAMPLE_RATE
    ) -> np.ndarray:
        """Frame-wise latents for Bucket 6b live use. Returns [128, T]."""
        return self.encodec_embed(audio, sr, pool="none")

    @torch.no_grad()
    def encodec_embed_batch(
        self,
        audios: list[np.ndarray],
        sr: int = SAMPLE_RATE,
        pool: str = "mean",
    ) -> np.ndarray:
        """Batch-embed multiple audio clips in a single forward pass.

        All clips are padded to the length of the longest clip in the batch.
        Returns [B, 128] (mean), [B, 256] (meanstd).
        """
        if pool == "none":
            raise ValueError("Batch mode requires pool='mean' or 'meanstd'")

        max_len = max(len(a) for a in audios)
        lengths = np.zeros(len(audios), dtype=np.int64)

        # convert_audio expects (channels, T), so convert each clip individually
        # then stack into a batch
        converted = []
        for i, a in enumerate(audios):
            lengths[i] = len(a)
            padded = np.zeros(max_len, dtype=np.float32)
            padded[:len(a)] = a
            x_i = torch.from_numpy(padded).float().unsqueeze(0)  # (1, T)
            x_i = convert_audio(x_i, sr, ENCODEC_SR, self._channels)  # (2, T')
            converted.append(x_i)
        x = torch.stack(converted, dim=0).to(self.device)  # (B, 2, T')

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self._amp):
            emb = self.enc.encoder(x)  # [B, 128, T_enc]
        emb = emb.float().cpu().numpy()    # [B, 128, T_enc]

        # Compute per-sample frame counts to mask padding
        enc_stride = max_len / emb.shape[2] if emb.shape[2] > 0 else 1
        frame_counts = np.maximum(1, (lengths / enc_stride).astype(np.int64))

        dim = 128 if pool == "mean" else 256
        out = np.zeros((len(audios), dim), dtype=np.float32)
        for i in range(len(audios)):
            t = frame_counts[i]
            e = emb[i, :, :t]  # [128, t]
            if pool == "mean":
                out[i] = e.mean(axis=1)
            else:  # meanstd
                out[i] = np.concatenate([e.mean(axis=1), e.std(axis=1)])
        return out

    def mrstft_feats(self, audio: np.ndarray) -> np.ndarray:
        """Multi-resolution STFT features (time-collapsed log-magnitude).

        Returns a 1-D vector of concatenated log-magnitude spectra at
        multiple FFT sizes. Stays in NumPy — used as an auxiliary loss
        term, not as the primary embedding target.
        """
        feats = []
        for n_fft in (256, 512, 1024, 2048):
            hop = n_fft // 4
            spec = _stft(audio, n_fft=n_fft, hop=hop)
            feats.append(np.log1p(spec).mean(axis=1))  # collapse time
        return np.concatenate(feats).astype(np.float32)

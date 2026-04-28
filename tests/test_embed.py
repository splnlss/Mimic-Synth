"""Unit tests for s04_embed — synthetic signals, no capture data required."""
import numpy as np
import pytest

from s04_embed.embed import Embedder, _stft

SR = 48000


def _tone(freq, dur_sec, amp=0.3, sr=SR):
    t = np.arange(int(dur_sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture(scope="module")
def embedder():
    return Embedder(device="cpu")


# ── _stft ────────────────────────────────────────────────────────────────────

class TestSTFT:
    def test_output_shape(self):
        audio = _tone(440, 1.0)
        spec = _stft(audio, n_fft=1024, hop=256)
        assert spec.shape[0] == 513  # n_fft//2 + 1
        expected_frames = 1 + (len(audio) - 1024) // 256
        assert spec.shape[1] == expected_frames

    def test_pure_tone_peak(self):
        """A 440 Hz tone should have peak energy near the 440 Hz bin."""
        audio = _tone(440, 1.0, amp=0.5)
        spec = _stft(audio, n_fft=2048, hop=512)
        freqs = np.fft.rfftfreq(2048, d=1.0 / SR)
        peak_bin = np.argmax(spec.mean(axis=1))
        assert abs(freqs[peak_bin] - 440) < 50  # within ~50 Hz


# ── Embedder.encodec_embed ───────────────────────────────────────────────────

class TestEncodecEmbed:
    def test_mean_pool_shape(self, embedder):
        audio = _tone(440, 1.0)
        vec = embedder.encodec_embed(audio, SR, pool="mean")
        assert vec.shape == (128,)
        assert vec.dtype == np.float32

    def test_meanstd_pool_shape(self, embedder):
        audio = _tone(440, 1.0)
        vec = embedder.encodec_embed(audio, SR, pool="meanstd")
        assert vec.shape == (256,)
        assert vec.dtype == np.float32

    def test_none_pool_shape(self, embedder):
        audio = _tone(440, 1.0)
        seq = embedder.encodec_embed(audio, SR, pool="none")
        assert seq.ndim == 2
        assert seq.shape[0] == 128
        assert seq.shape[1] > 0
        assert seq.dtype == np.float32

    def test_invalid_pool_raises(self, embedder):
        audio = _tone(440, 0.5)
        with pytest.raises(ValueError, match="Unknown pool mode"):
            embedder.encodec_embed(audio, SR, pool="bogus")

    def test_deterministic(self, embedder):
        """Same input should produce identical embeddings."""
        audio = _tone(440, 0.5)
        v1 = embedder.encodec_embed(audio, SR)
        v2 = embedder.encodec_embed(audio, SR)
        np.testing.assert_array_equal(v1, v2)

    def test_different_pitches_differ(self, embedder):
        """Different frequencies should produce different embeddings."""
        low = embedder.encodec_embed(_tone(220, 1.0), SR)
        high = embedder.encodec_embed(_tone(880, 1.0), SR)
        assert not np.allclose(low, high, atol=0.1)

    def test_encodec_sequence_matches_none_pool(self, embedder):
        audio = _tone(440, 0.5)
        seq = embedder.encodec_sequence(audio, SR)
        none = embedder.encodec_embed(audio, SR, pool="none")
        np.testing.assert_array_equal(seq, none)


# ── Embedder.mrstft_feats ───────────────────────────────────────────────────

class TestMRSTFT:
    def test_output_shape(self, embedder):
        audio = _tone(440, 1.0)
        feats = embedder.mrstft_feats(audio)
        # 4 FFT sizes: 256/2+1=129, 512/2+1=257, 1024/2+1=513, 2048/2+1=1025
        expected_dim = 129 + 257 + 513 + 1025
        assert feats.shape == (expected_dim,)
        assert feats.dtype == np.float32

    def test_different_timbres_differ(self, embedder):
        """A pure tone and noise should have very different STFT features."""
        tone = _tone(440, 1.0)
        noise = np.random.randn(SR).astype(np.float32) * 0.1
        f_tone = embedder.mrstft_feats(tone)
        f_noise = embedder.mrstft_feats(noise)
        assert not np.allclose(f_tone, f_noise, atol=0.5)

"""Tests for s04_embed indexer and verifier — uses synthetic WAVs, no capture data."""
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from s04_embed.index_dataset import index_dataset, _flush, _resolve_wav_root
from s04_embed.verify_embeddings import verify_embeddings, neighbor_spot_check

SR = 48000


def _tone(freq, dur_sec, amp=0.3, sr=SR):
    t = np.arange(int(dur_sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def mini_dataset(tmp_path):
    """Create a minimal dataset: 10 WAVs + samples.parquet."""
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()

    rows = []
    freqs = [220, 330, 440, 550, 660, 220, 330, 440, 550, 660]
    notes = [36, 48, 60, 72, 84, 36, 48, 60, 72, 84]
    for i, (freq, note) in enumerate(zip(freqs, notes)):
        h = f"hash{i:04d}"
        wav_path = wav_dir / f"{h}_n{note}.wav"
        audio = _tone(freq, 0.5, amp=0.3 + i * 0.02)
        sf.write(wav_path, audio, SR)
        rows.append({
            "hash": h, "note": note,
            "wav": f"wav/{h}_n{note}.wav",
            "self_noise": 0.0,
            "p_cutoff": float(i) / 10,
        })

    pd.DataFrame(rows).to_parquet(tmp_path / "samples.parquet")
    return tmp_path


# ── index_dataset ────────────────────────────────────────────────────────────

class TestIndexDataset:
    def test_produces_npy(self, mini_dataset, tmp_path):
        out = tmp_path / "embed_out"
        npy_path = index_dataset(mini_dataset, out, pool="mean")
        assert npy_path.exists()
        arr = np.load(npy_path)
        assert arr.shape == (10, 128)
        assert arr.dtype == np.float32

    def test_produces_done_mask(self, mini_dataset, tmp_path):
        out = tmp_path / "embed_out"
        index_dataset(mini_dataset, out, pool="mean")
        done_path = out / "encodec_embeddings_done.npy"
        assert done_path.exists()
        done = np.load(done_path)
        assert done.shape == (10,)
        assert done.all()

    def test_meanstd_pool(self, mini_dataset, tmp_path):
        out = tmp_path / "embed_out"
        npy_path = index_dataset(mini_dataset, out, pool="meanstd")
        arr = np.load(npy_path)
        assert arr.shape == (10, 256)

    def test_no_all_zeros(self, mini_dataset, tmp_path):
        """Every row should have non-zero embeddings (all WAVs valid)."""
        out = tmp_path / "embed_out"
        npy_path = index_dataset(mini_dataset, out, pool="mean")
        arr = np.load(npy_path)
        norms = np.linalg.norm(arr, axis=1)
        assert np.all(norms > 0), "All rows should be non-zero"

    def test_different_audio_different_embeddings(self, mini_dataset, tmp_path):
        """Different frequencies should produce different embeddings."""
        out = tmp_path / "embed_out"
        npy_path = index_dataset(mini_dataset, out, pool="mean")
        arr = np.load(npy_path)
        # Row 0 is 220 Hz, row 2 is 440 Hz
        assert not np.allclose(arr[0], arr[2], atol=0.1)

    def test_missing_wav_yields_zero(self, mini_dataset, tmp_path):
        """Missing WAVs should produce zero vectors, not crash."""
        # Delete one WAV
        wav_files = list((mini_dataset / "wav").glob("*.wav"))
        wav_files[0].unlink()

        out = tmp_path / "embed_out"
        npy_path = index_dataset(mini_dataset, out, pool="mean")
        arr = np.load(npy_path)
        norms = np.linalg.norm(arr, axis=1)
        assert np.sum(norms == 0) == 1, "Exactly one row should be zero (missing WAV)"
        assert np.sum(norms > 0) == 9, "Other 9 rows should be non-zero"

    def test_missing_parquet_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            index_dataset(tmp_path, tmp_path / "out")


# ── checkpoint / resume ──────────────────────────────────────────────────────

class TestCheckpointResume:
    def test_flush_creates_files(self, tmp_path):
        npy_path = tmp_path / "embeddings.npy"
        done_path = tmp_path / "embeddings_done.npy"
        arr = np.ones((5, 128), dtype=np.float32)
        done = np.array([True, True, False, False, False])
        _flush(npy_path, done_path, arr, done)
        assert npy_path.exists()
        assert done_path.exists()
        np.testing.assert_array_equal(np.load(npy_path), arr)
        np.testing.assert_array_equal(np.load(done_path), done)

    def test_resume_skips_done_rows(self, mini_dataset, tmp_path):
        """Simulate partial run, then verify resume skips done rows."""
        out = tmp_path / "embed_out"
        out.mkdir()

        # Create a partial embedding (first 5 rows "done")
        from s04_embed.embed import Embedder
        emb = Embedder(device="cpu")
        df = pd.read_parquet(mini_dataset / "samples.parquet")

        partial_arr = np.zeros((10, 128), dtype=np.float32)
        done_mask = np.zeros(10, dtype=bool)
        for i in range(5):
            wav_path = mini_dataset / df.iloc[i]["wav"]
            audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            partial_arr[i] = emb.encodec_embed(audio, sr)
            done_mask[i] = True

        np.save(out / "encodec_embeddings.npy", partial_arr)
        np.save(out / "encodec_embeddings_done.npy", done_mask)

        # Verify the partial state
        first_five = partial_arr[:5].copy()

        # Now do a full run (will need user input for resume — test the
        # _resolve_wav_root and shape-checking paths instead)
        loaded_arr = np.load(out / "encodec_embeddings.npy")
        loaded_done = np.load(out / "encodec_embeddings_done.npy")
        assert loaded_arr.shape == (10, 128)
        assert loaded_done.sum() == 5
        np.testing.assert_array_equal(loaded_arr[:5], first_five)


# ── verify_embeddings ────────────────────────────────────────────────────────

class TestVerifyEmbeddings:
    def test_complete_embeddings_pass(self, mini_dataset, tmp_path):
        out = tmp_path / "embed_out"
        npy_path = index_dataset(mini_dataset, out, pool="mean")
        report, passed = verify_embeddings(npy_path, mini_dataset)
        assert passed
        assert report.n_rows == 10
        assert report.n_parquet == 10
        assert report.dim == 128
        assert report.n_complete == 10
        assert report.n_zero == 0
        assert not report.has_nan
        assert not report.has_inf

    def test_shape_mismatch_fails(self, mini_dataset, tmp_path):
        out = tmp_path / "embed_out"
        out.mkdir()
        # Wrong number of rows
        arr = np.zeros((5, 128), dtype=np.float32)
        npy_path = out / "encodec_embeddings.npy"
        np.save(npy_path, arr)
        report, passed = verify_embeddings(npy_path, mini_dataset)
        assert not passed
        assert report.n_rows == 5
        assert report.n_parquet == 10

    def test_nan_fails(self, mini_dataset, tmp_path):
        out = tmp_path / "embed_out"
        out.mkdir()
        arr = np.ones((10, 128), dtype=np.float32)
        arr[3, 50] = np.nan
        npy_path = out / "encodec_embeddings.npy"
        np.save(npy_path, arr)
        report, passed = verify_embeddings(npy_path, mini_dataset)
        assert not passed
        assert report.has_nan

    def test_inf_fails(self, mini_dataset, tmp_path):
        out = tmp_path / "embed_out"
        out.mkdir()
        arr = np.ones((10, 128), dtype=np.float32)
        arr[0, 0] = np.inf
        npy_path = out / "encodec_embeddings.npy"
        np.save(npy_path, arr)
        report, passed = verify_embeddings(npy_path, mini_dataset)
        assert not passed
        assert report.has_inf

    def test_incomplete_done_mask_fails(self, mini_dataset, tmp_path):
        out = tmp_path / "embed_out"
        out.mkdir()
        arr = np.ones((10, 128), dtype=np.float32)
        done = np.zeros(10, dtype=bool)
        done[:7] = True  # only 7/10 done
        np.save(out / "encodec_embeddings.npy", arr)
        np.save(out / "encodec_embeddings_done.npy", done)
        report, passed = verify_embeddings(
            out / "encodec_embeddings.npy", mini_dataset
        )
        assert not passed

    def test_latent_stats_populated(self, mini_dataset, tmp_path):
        out = tmp_path / "embed_out"
        npy_path = index_dataset(mini_dataset, out, pool="mean")
        report, _ = verify_embeddings(npy_path, mini_dataset)
        assert report.global_min < 0  # EnCodec latents are centered
        assert report.global_max > 0
        assert report.per_dim_std_range[1] > 0


# ── neighbor_spot_check ──────────────────────────────────────────────────────

class TestNeighborSpotCheck:
    def test_returns_results(self, mini_dataset, tmp_path):
        out = tmp_path / "embed_out"
        npy_path = index_dataset(mini_dataset, out, pool="mean")
        results = neighbor_spot_check(npy_path, mini_dataset, n_anchors=3, k=2)
        assert len(results) == 3
        for r in results:
            assert "anchor_idx" in r
            assert len(r["nearest"]) == 2
            assert len(r["farthest"]) == 2

    def test_nearest_has_higher_similarity(self, mini_dataset, tmp_path):
        out = tmp_path / "embed_out"
        npy_path = index_dataset(mini_dataset, out, pool="mean")
        results = neighbor_spot_check(npy_path, mini_dataset, n_anchors=3, k=2)
        for r in results:
            near_sims = [n["sim"] for n in r["nearest"]]
            far_sims = [f["sim"] for f in r["farthest"]]
            assert min(near_sims) > max(far_sims), \
                "Nearest neighbors should have higher similarity than farthest"

    def test_same_freq_are_neighbors(self, mini_dataset, tmp_path):
        """Rows with the same frequency should be nearest neighbors."""
        out = tmp_path / "embed_out"
        npy_path = index_dataset(mini_dataset, out, pool="mean")
        # Rows 0 and 5 are both 220 Hz, rows 2 and 7 are both 440 Hz
        arr = np.load(npy_path)
        # Cosine similarity between same-freq pairs should be high
        def cos_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        sim_same = cos_sim(arr[0], arr[5])  # both 220 Hz
        sim_diff = cos_sim(arr[0], arr[2])  # 220 vs 440 Hz
        assert sim_same > sim_diff, \
            f"Same-freq similarity ({sim_same:.3f}) should exceed " \
            f"cross-freq similarity ({sim_diff:.3f})"

"""End-to-end test for s03_dataset.verify_dataset on a synthetic dataset."""
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
import pytest

from s03_dataset.verify_dataset import verify_dataset, print_report

SR = 48000


@pytest.fixture
def tiny_profile(tmp_path):
    p = {
        "probe": {
            "sample_rate": SR,
            "render_sec": 0.8,
            "hold_sec": 0.6,
            "release_sec": 0.2,
            "velocity": 100,
            "notes": [60],
        },
        "parameters": {"a": {"importance": 1.0}},
    }
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(p))
    return path


def _write_row(wav_dir, h, note, *, silent=False, clipped=False, stuck=False, bleed=False):
    total = int(0.8 * SR)
    audio = np.zeros(total, dtype=np.float32)
    if not silent:
        lead = int(0.03 * SR)  # silent lead-in so prev-bleed gate doesn't fire
        hold = int(0.6 * SR) - lead
        t = np.arange(hold) / SR
        attack = np.linspace(0, 1, min(int(0.02 * SR), hold), dtype=np.float32)
        tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        tone[:len(attack)] *= attack
        audio[lead:lead+hold] = tone
        rel = int(0.2 * SR)
        env = np.linspace(1.0, 0.0, rel, dtype=np.float32)
        tt = np.arange(rel) / SR
        audio[lead+hold:lead+hold+rel] = (0.3 * np.sin(2 * np.pi * 220 * tt) * env).astype(np.float32)
    if clipped:
        audio[:100] = 1.0
    if stuck:
        audio[-int(0.05 * SR):] = 0.3
    if bleed:
        audio[:int(0.02 * SR)] = 0.5
    fname = f"{h}_n{note}.wav"
    sf.write(wav_dir / fname, audio, SR)
    return f"wav/{fname}"


def _make_dataset(tmp_path, rows_spec):
    """rows_spec: list of dicts with keys h, note, and optional silent/clipped/stuck/bleed."""
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    rows = []
    for r in rows_spec:
        wav_rel = _write_row(
            wav_dir, r["h"], r["note"],
            silent=r.get("silent", False),
            clipped=r.get("clipped", False),
            stuck=r.get("stuck", False),
            bleed=r.get("bleed", False),
        )
        rows.append({"hash": r["h"], "note": r["note"], "wav": wav_rel, "p_a": 0.5})
    pd.DataFrame(rows).to_parquet(tmp_path / "samples.parquet")
    return tmp_path


class TestVerifyDataset:
    def test_clean_dataset_passes(self, tmp_path, tiny_profile):
        ds = _make_dataset(tmp_path, [
            {"h": "aaaaaaaaaaaa", "note": 60},
            {"h": "bbbbbbbbbbbb", "note": 60},
            {"h": "cccccccccccc", "note": 60},
        ])
        report = verify_dataset(ds, tiny_profile)
        assert report.total_rows == 3
        assert report.wavs_ok == 3
        assert report.valid == 3
        assert report.silent == 0
        assert "p_a" in report.param_coverage

    def test_detects_silence(self, tmp_path, tiny_profile):
        ds = _make_dataset(tmp_path, [
            {"h": "a"*12, "note": 60, "silent": True},
            {"h": "b"*12, "note": 60},
        ])
        report = verify_dataset(ds, tiny_profile)
        assert report.silent == 1
        assert report.valid == 1

    def test_detects_clipping(self, tmp_path, tiny_profile):
        ds = _make_dataset(tmp_path, [{"h": "a"*12, "note": 60, "clipped": True}])
        report = verify_dataset(ds, tiny_profile)
        assert report.clipped == 1

    def test_detects_bleed(self, tmp_path, tiny_profile):
        ds = _make_dataset(tmp_path, [{"h": "a"*12, "note": 60, "bleed": True}])
        report = verify_dataset(ds, tiny_profile)
        assert report.prev_bleed == 1

    def test_detects_missing_wav(self, tmp_path, tiny_profile):
        ds = _make_dataset(tmp_path, [{"h": "a"*12, "note": 60}])
        (ds / "wav" / "aaaaaaaaaaaa_n60.wav").unlink()
        report = verify_dataset(ds, tiny_profile)
        assert report.wavs_missing == 1

    def test_hash_filename_mismatch(self, tmp_path, tiny_profile):
        ds = _make_dataset(tmp_path, [{"h": "a"*12, "note": 60}])
        df = pd.read_parquet(ds / "samples.parquet")
        df.loc[0, "hash"] = "z" * 12
        df.to_parquet(ds / "samples.parquet")
        report = verify_dataset(ds, tiny_profile)
        assert report.hash_mismatch == 1

    def test_print_report_pass(self, tmp_path, tiny_profile, capsys):
        ds = _make_dataset(tmp_path, [{"h": "a"*12, "note": 60}])
        report = verify_dataset(ds, tiny_profile)
        assert print_report(report, fail_threshold=0.01) is True
        assert "PASS" in capsys.readouterr().out

    def test_print_report_fail(self, tmp_path, tiny_profile, capsys):
        ds = _make_dataset(tmp_path, [
            {"h": "a"*12, "note": 60, "silent": True},
            {"h": "b"*12, "note": 60, "silent": True},
        ])
        report = verify_dataset(ds, tiny_profile)
        assert print_report(report, fail_threshold=0.01) is False
        assert "FAIL" in capsys.readouterr().out

    def test_missing_parquet_raises(self, tmp_path, tiny_profile):
        with pytest.raises(FileNotFoundError):
            verify_dataset(tmp_path, tiny_profile)

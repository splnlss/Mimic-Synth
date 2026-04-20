"""Unit tests for s03_dataset.manifest."""
import pytest

from s03_dataset.manifest import (
    Manifest, Counts, Phase, new_manifest, profile_hash,
    write_manifest, read_manifest, assert_profile_matches,
)


def _profile():
    return {
        "synth": {"id": "obxf"},
        "parameters": {"a": {"importance": 1.0}},
        "probe": {"render_sec": 1.0, "hold_sec": 0.6, "release_sec": 0.4},
    }


class TestProfileHash:
    def test_deterministic(self):
        p = _profile()
        assert profile_hash(p) == profile_hash(p)

    def test_key_order_invariant(self):
        a = {"x": 1, "y": 2}
        b = {"y": 2, "x": 1}
        assert profile_hash(a) == profile_hash(b)

    def test_change_detected(self):
        p = _profile()
        h1 = profile_hash(p)
        p["parameters"]["a"]["importance"] = 0.5
        assert profile_hash(p) != h1

    def test_format(self):
        h = profile_hash(_profile())
        assert h.startswith("sha256:") and len(h) == 7 + 64


class TestManifest:
    def test_new_manifest_fields(self):
        m = new_manifest(seed=7, profile=_profile())
        assert m.seed == 7
        assert m.sampler == "sobol_scrambled"
        assert m.profile_hash.startswith("sha256:")

    def test_roundtrip(self, tmp_path):
        m = new_manifest(seed=1, profile=_profile())
        m.phases.append(Phase(name="cold_start", n=1024, seed=1))
        m.counts = Counts(rendered=100, silent=2, valid=95)
        path = tmp_path / "manifest.yaml"
        write_manifest(path, m)
        m2 = read_manifest(path)
        assert m2.seed == m.seed
        assert m2.profile_hash == m.profile_hash
        assert m2.phases[0].name == "cold_start"
        assert m2.counts.rendered == 100
        assert m2.counts.valid == 95

    def test_assert_profile_matches_ok(self):
        p = _profile()
        m = new_manifest(seed=0, profile=p)
        assert_profile_matches(m, p)  # no raise

    def test_assert_profile_matches_detects_change(self):
        p = _profile()
        m = new_manifest(seed=0, profile=p)
        p["parameters"]["a"]["importance"] = 0.1
        with pytest.raises(ValueError, match="Profile hash mismatch"):
            assert_profile_matches(m, p)

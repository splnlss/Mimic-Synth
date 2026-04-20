"""Unit tests for s03_dataset.sequences — no DawDreamer required."""
import numpy as np
import pytest

from s03_dataset.sequences import (
    interpolated_trajectory, endpoint_pairs, apply_log_scale, sequence_hash,
)


class TestInterpolatedTrajectory:
    def test_shape(self):
        traj = interpolated_trajectory(np.zeros(15), np.ones(15), n_frames=500)
        assert traj.shape == (500, 15)
        assert traj.dtype == np.float32

    def test_endpoints(self):
        a = np.array([0.1, 0.9, 0.5], dtype=np.float32)
        b = np.array([0.9, 0.1, 0.5], dtype=np.float32)
        traj = interpolated_trajectory(a, b, n_frames=10)
        np.testing.assert_allclose(traj[0], a, atol=1e-6)
        np.testing.assert_allclose(traj[-1], b, atol=1e-6)

    def test_midpoint(self):
        a = np.zeros(3, dtype=np.float32)
        b = np.ones(3, dtype=np.float32)
        traj = interpolated_trajectory(a, b, n_frames=101)
        np.testing.assert_allclose(traj[50], 0.5 * np.ones(3), atol=1e-6)

    def test_monotonic_when_ordered(self):
        a = np.array([0.0])
        b = np.array([1.0])
        traj = interpolated_trajectory(a, b, n_frames=50)
        diffs = np.diff(traj[:, 0])
        assert np.all(diffs >= 0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            interpolated_trajectory(np.zeros(3), np.zeros(4), n_frames=10)

    def test_too_few_frames_raises(self):
        with pytest.raises(ValueError):
            interpolated_trajectory(np.zeros(3), np.ones(3), n_frames=1)


class TestEndpointPairs:
    def test_shape(self):
        pairs = endpoint_pairs(m=4, d=6, seed=0)
        assert pairs.shape == (8, 2, 6)  # 2**4 = 16 points → 8 pairs

    def test_determinism(self):
        a = endpoint_pairs(m=3, d=4, seed=7)
        b = endpoint_pairs(m=3, d=4, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_pairs_in_unit_cube(self):
        pairs = endpoint_pairs(m=5, d=3, seed=0)
        assert pairs.min() >= 0.0 and pairs.max() <= 1.0


class TestApplyLogScale:
    def test_no_op_when_linear(self):
        traj = np.random.default_rng(0).random((10, 3)).astype(np.float32)
        specs = [{"log_scale": False}, {}, {"log_scale": False}]
        out = apply_log_scale(traj, specs)
        np.testing.assert_array_equal(out, traj)

    def test_log_scale_endpoints_preserved(self):
        traj = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
        specs = [{"log_scale": True, "log_base": 50}]
        out = apply_log_scale(traj, specs)
        assert out[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert out[2, 0] == pytest.approx(1.0, abs=1e-6)
        assert out[1, 0] < 0.5  # log mapping compresses mid-range toward zero

    def test_mixed_columns(self):
        traj = np.full((5, 2), 0.5, dtype=np.float32)
        specs = [{"log_scale": True, "log_base": 50}, {}]
        out = apply_log_scale(traj, specs)
        assert out[0, 0] < 0.5
        assert out[0, 1] == 0.5


class TestSequenceHash:
    def test_deterministic(self):
        traj = np.linspace(0, 1, 20, dtype=np.float32).reshape(10, 2)
        assert sequence_hash(traj, 60) == sequence_hash(traj, 60)

    def test_different_notes_differ(self):
        traj = np.linspace(0, 1, 20, dtype=np.float32).reshape(10, 2)
        assert sequence_hash(traj, 60) != sequence_hash(traj, 72)

    def test_different_trajectories_differ(self):
        a = np.linspace(0, 1, 20, dtype=np.float32).reshape(10, 2)
        b = np.linspace(0, 0.5, 20, dtype=np.float32).reshape(10, 2)
        assert sequence_hash(a, 60) != sequence_hash(b, 60)

    def test_format(self):
        traj = np.zeros((5, 3), dtype=np.float32)
        h = sequence_hash(traj, 60)
        assert len(h) == 12 and all(c in "0123456789abcdef" for c in h)

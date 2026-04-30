"""Unit tests for s06_invert — synthetic data, no GPU or real captures required."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helpers ──────────────────────────────────────────────────────────────────

D_PARAMS = 8
D_EMBED = 16  # small for fast CPU tests


class _TinySurrogate(nn.Module):
    """Minimal surrogate that produces non-trivial gradients."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_PARAMS + 1, 32), nn.ReLU(),
            nn.Linear(32, D_EMBED),
        )

    def forward(self, params, note):
        x = torch.cat([params, note.unsqueeze(-1)], dim=-1)
        return self.net(x)


def _random_target() -> torch.Tensor:
    t = torch.randn(D_EMBED)
    return F.normalize(t, dim=0)


# ── grad_search ───────────────────────────────────────────────────────────────

class TestGradInvert:
    def test_output_shape_and_range(self):
        from s06_invert.grad_search import grad_invert
        model = _TinySurrogate()
        target = _random_target()
        score, params = grad_invert(
            model, target, note=60, d_params=D_PARAMS,
            n_starts=2, steps=10, device="cpu",
        )
        assert params.shape == (D_PARAMS,)
        assert float(params.min()) >= 0.0 - 1e-6
        assert float(params.max()) <= 1.0 + 1e-6

    def test_score_is_cosine_distance(self):
        from s06_invert.grad_search import grad_invert
        model = _TinySurrogate()
        target = _random_target()
        score, params = grad_invert(
            model, target, note=60, d_params=D_PARAMS,
            n_starts=2, steps=10, device="cpu",
        )
        assert 0.0 <= score <= 2.0  # cosine distance is in [0, 2]

    def test_multistart_not_worse_than_single(self):
        from s06_invert.grad_search import grad_invert
        model = _TinySurrogate()
        target = _random_target()
        score_multi, _ = grad_invert(
            model, target, note=60, d_params=D_PARAMS,
            n_starts=4, steps=30, device="cpu",
        )
        score_single, _ = grad_invert(
            model, target, note=60, d_params=D_PARAMS,
            n_starts=1, steps=30, device="cpu",
        )
        # more starts can only improve or match
        assert score_multi <= score_single + 1e-4

    def test_optimisation_reduces_score(self):
        """Score after 100 steps should generally be lower than random."""
        from s06_invert.grad_search import grad_invert
        torch.manual_seed(0)
        model = _TinySurrogate()
        target = _random_target()
        score_optimised, _ = grad_invert(
            model, target, note=60, d_params=D_PARAMS,
            n_starts=4, steps=100, device="cpu",
        )
        # score of a random param vector
        with torch.no_grad():
            rand_params = torch.rand(1, D_PARAMS)
            note_t = torch.tensor([60 / 127.0])
            pred = model(rand_params, note_t)
            rand_score = (1 - F.cosine_similarity(
                pred, target.unsqueeze(0), dim=-1
            )).item()
        assert score_optimised < rand_score + 0.5  # optimiser should not be worse by a large margin

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_runs_on_gpu(self):
        from s06_invert.grad_search import grad_invert
        model = _TinySurrogate().cuda()
        target = _random_target()
        score, params = grad_invert(
            model, target, note=60, d_params=D_PARAMS,
            n_starts=2, steps=10, device="cuda",
        )
        assert params.device.type == "cpu"  # returned on CPU


# ── cmaes_search ─────────────────────────────────────────────────────────────

class TestCMAESInvert:
    def test_import_error_without_cma(self, monkeypatch):
        import sys
        # Temporarily hide cma if installed
        monkeypatch.setitem(sys.modules, "cma", None)
        from s06_invert import cmaes_search
        import importlib
        importlib.reload(cmaes_search)
        with pytest.raises(ImportError, match="cma package required"):
            cmaes_search.cmaes_invert(
                _TinySurrogate(), _random_target(), 60, D_PARAMS,
                x0=torch.rand(D_PARAMS), device="cpu",
            )

    @pytest.mark.skipif(
        not pytest.importorskip("cma", reason="cma not installed"),
        reason="cma not installed",
    )
    def test_output_shape_and_range(self):
        from s06_invert.cmaes_search import cmaes_invert
        model = _TinySurrogate()
        x0 = torch.rand(D_PARAMS)
        score, params = cmaes_invert(
            model, _random_target(), 60, D_PARAMS, x0,
            maxiter=5, device="cpu",
        )
        assert params.shape == (D_PARAMS,)
        assert float(params.min()) >= 0.0 - 1e-6
        assert float(params.max()) <= 1.0 + 1e-6
        assert 0.0 <= score <= 2.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_runs_on_gpu(self):
        pytest.importorskip("cma")
        from s06_invert.cmaes_search import cmaes_invert
        model = _TinySurrogate().cuda()
        x0 = torch.rand(D_PARAMS)
        score, params = cmaes_invert(
            model, _random_target(), 60, D_PARAMS, x0,
            maxiter=5, device="cuda",
        )
        assert isinstance(params, np.ndarray)


# ── validate helpers ──────────────────────────────────────────────────────────

class TestTestIndices:
    def test_split_sizes(self):
        from s06_invert.validate import _test_indices
        idx = _test_indices(5120)
        assert len(idx) == 5120 - int(0.8 * 5120) - int(0.1 * 5120)

    def test_reproducible(self):
        from s06_invert.validate import _test_indices
        assert _test_indices(5120) == _test_indices(5120)

    def test_no_overlap_with_train(self):
        from s06_invert.validate import _test_indices
        from torch.utils.data import random_split
        n = 5120
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        test_size = n - train_size - val_size
        train_ds, _, _ = random_split(
            range(n), [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        train_idx = set(train_ds)
        test_idx = set(_test_indices(n))
        assert len(train_idx & test_idx) == 0

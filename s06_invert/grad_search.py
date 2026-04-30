"""Multi-start gradient descent through the frozen surrogate.

Treats the parameter vector as a learnable tensor and backprops cosine
distance from a target embedding. Clamps to [0, 1] before every forward
pass to keep inputs in-distribution.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def grad_invert(
    surrogate: torch.nn.Module,
    target_emb: torch.Tensor,
    note: float,
    d_params: int,
    n_starts: int = 16,
    steps: int = 300,
    lr: float = 5e-2,
    device: str = "cuda",
) -> tuple[float, torch.Tensor]:
    """Find params minimising cosine distance to target_emb.

    Returns (best_score, best_params) where best_params is shape [d_params],
    values in [0, 1], on CPU. best_score is cosine distance (lower = better).
    """
    surrogate.eval()
    target = target_emb.to(device).unsqueeze(0)  # [1, d_embed]
    note_t = torch.full((1,), note / 127.0, device=device)  # [1]

    best_score = float("inf")
    best_params: torch.Tensor | None = None

    for _ in range(n_starts):
        params = torch.rand(1, d_params, device=device, requires_grad=True)
        opt = torch.optim.Adam([params], lr=lr)

        for _ in range(steps):
            opt.zero_grad()
            pred = surrogate(params.clamp(0.0, 1.0), note_t)
            loss = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
            loss.backward()
            opt.step()

        final = params.detach().clamp(0.0, 1.0)
        with torch.no_grad():
            pred = surrogate(final, note_t)
            score = (1.0 - F.cosine_similarity(pred, target, dim=-1)).item()

        if score < best_score:
            best_score = score
            best_params = final.squeeze(0).cpu()

    assert best_params is not None
    return best_score, best_params

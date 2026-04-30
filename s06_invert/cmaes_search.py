"""CMA-ES gradient-free refinement through the frozen surrogate.

Handles discontinuous regions (filter self-oscillation thresholds, enum
transitions) that gradient descent struggles with. Seeded from grad_search
output. Enum params are passed as continuous [0,1] values; snap to nearest
category *after* this returns, before sending to the synth.

Requires: pip install cma
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def cmaes_invert(
    surrogate: torch.nn.Module,
    target_emb: torch.Tensor,
    note: float,
    d_params: int,
    x0: torch.Tensor | np.ndarray,
    sigma0: float = 0.15,
    maxiter: int = 200,
    popsize: int | None = None,
    device: str = "cuda",
) -> tuple[float, np.ndarray]:
    """CMA-ES refinement seeded from x0.

    Returns (best_score, best_params) where best_params is np.ndarray shape
    [d_params], values clipped to [0, 1]. best_score is cosine distance.
    """
    try:
        import cma
    except ImportError:
        raise ImportError("cma package required for CMA-ES search: pip install cma")

    surrogate.eval()
    target = target_emb.to(device).unsqueeze(0)  # [1, d_embed]

    x0_list = (x0.cpu().numpy() if isinstance(x0, torch.Tensor) else np.asarray(x0)).tolist()

    inopts: dict = {
        "bounds": [[0.0] * d_params, [1.0] * d_params],
        "maxiter": maxiter,
        "verbose": -9,
    }
    if popsize is not None:
        inopts["popsize"] = popsize

    es = cma.CMAEvolutionStrategy(x0_list, sigma0, inopts)

    while not es.stop():
        xs = es.ask()
        batch = torch.tensor(np.array(xs), dtype=torch.float32, device=device)
        note_t = torch.full((len(xs),), note / 127.0, device=device)
        with torch.no_grad():
            preds = surrogate(batch, note_t)
            scores = (
                1.0 - F.cosine_similarity(preds, target.expand_as(preds), dim=-1)
            )
        es.tell(xs, scores.cpu().numpy().tolist())

    best_x = np.array(es.best.x).clip(0.0, 1.0)
    return float(es.best.f), best_x

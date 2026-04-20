"""
Sampling strategy for Bucket 3 dataset generation.

Three pieces:
- `cold_start_vectors(m, d)`: scrambled Sobol, 2**m points, deterministic.
- `to_synth_value(u, spec)`: apply log-scale transform if the profile flags it.
- `apply_importance(u_vec, modulated, profile, mode)`: two interpretations of
  the `importance` field (filter vs. scale), documented in the Bucket 3 doc.
"""
from __future__ import annotations
import math
from typing import Sequence, Literal
import numpy as np
from scipy.stats.qmc import Sobol


def cold_start_vectors(m: int, d: int, seed: int = 0) -> np.ndarray:
    """Generate 2**m Sobol points in d dimensions via scrambled random_base2.

    Takes the exponent `m` directly instead of a sample count `n` — avoids the
    silent-truncation trap of `sobol.random_base2(m=int(np.log2(n)))` for
    non-power-of-2 n. Returns shape (2**m, d)."""
    if m < 0:
        raise ValueError(f"m must be non-negative, got {m}")
    sobol = Sobol(d=d, scramble=True, seed=seed)
    return sobol.random_base2(m=m)


def to_synth_value(u: float, spec: dict) -> float:
    """Map a Sobol draw u in [0,1] to a normalised synth param value.

    If spec has `log_scale: true`, applies a perceptually-log mapping so that
    low values get finer resolution. Endpoints are preserved:
    u=0 → 0, u=1 → 1."""
    if not spec.get("log_scale"):
        return float(u)
    log_base = float(spec.get("log_base", 50))
    if log_base <= 0:
        raise ValueError(f"log_base must be positive, got {log_base}")
    return float(np.expm1(u * math.log1p(log_base)) / log_base)


def apply_importance(
    u_row: np.ndarray,
    modulated: Sequence[str],
    profile: dict,
    mode: Literal["filter", "scale"] = "filter",
) -> dict[str, float]:
    """Turn one row of Sobol draws into a {param_name: value} dict.

    mode='filter' (Bucket 2 V1 behaviour): importance is already used to pick
    which params are in `modulated`; here we just zip them with the raw values
    and apply any log_scale transform.

    mode='scale' (Bucket 3 extension): importance also scales the range around
    each param's reset value. importance=1.0 → full [0,1] range; importance=0.3
    → a narrow ±0.15 band around reset. The reset is read from profile['reset'].
    """
    if mode not in ("filter", "scale"):
        raise ValueError(f"Unknown mode: {mode}")
    if len(u_row) != len(modulated):
        raise ValueError(
            f"u_row length {len(u_row)} != modulated count {len(modulated)}"
        )

    out: dict[str, float] = {}
    for name, u in zip(modulated, u_row):
        spec = profile["parameters"][name]
        v = to_synth_value(u, spec)
        if mode == "scale":
            importance = float(spec.get("importance", 1.0))
            reset = float(profile.get("reset", {}).get(name, 0.5))
            # Centre the sampled band on reset; width scales with importance.
            v = reset + importance * (v - 0.5)
            v = float(np.clip(v, 0.0, 1.0))
        out[name] = v
    return out

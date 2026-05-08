"""Strategy 1: per-param hill-climbing on real VST renders.

Coordinate descent over a small fixed grid of offsets. Walks the s06b output
toward a lower real-synth cosine distance using *only* real renders — no
surrogate gradients, since those carry the surrogate-to-real bias that the
refinement step is trying to correct.

Algorithm
---------
For each unpinned param `p`:
    for each offset in `offsets`:
        trial_df[p] = clip(df[p] + offset, 0, 1)        # global shift
        score      = score(render(trial_df))
    if best_offset_score < current_score:
        df[p]         = clip(df[p] + best_offset, 0, 1)
        current_score = best_offset_score

Repeat the outer pass until no param improves, or until `n_passes` is hit.

Why a global per-param offset, not per-frame?
    The s06b output already has rich per-frame variation produced by the
    surrogate. We don't want to overwrite that. We want to *correct* a
    systematic bias — e.g. "the surrogate consistently predicts filter
    cutoff 0.05 too low across all frames". A global offset is exactly the
    fix for that bias and preserves frame-to-frame dynamics.

Why not optimize a per-region offset?
    Could help on multi-note targets where regions have different timbres,
    but it scales the search by `n_regions` for each param. Keeping it
    global at the start; can extend later if measurement justifies it.

Pinned params
    The `pinned_cols` set must include every column whose value was held
    fixed during s06b inversion (Osc 1 Pitch, Amp Env Release, LFO 1 to
    Osc 1 Pitch). Hill-climbing skips those columns entirely. Without
    this, the optimizer would happily push Osc 1 Pitch back toward an
    extreme since the embedding-only objective doesn't penalise it.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Iterable

import numpy as np
import pandas as pd
import torch

import numpy as np
from s07_refine.audio_compare import render_and_score


DEFAULT_OFFSETS: tuple[float, ...] = (-0.15, -0.05, 0.05, 0.15)


def hill_climb(
    df: pd.DataFrame,
    note_regions: list[dict],
    param_cols: list[str],
    pinned_cols: Iterable[str],
    profile_path: _Path,
    total_sec: float,
    target_emb_t: torch.Tensor,
    embedder,
    device: str,
    offsets: Iterable[float] = DEFAULT_OFFSETS,
    n_passes: int = 2,
    verbose: bool = True,
    target_mrstft_audio: np.ndarray | None = None,
    target_ap: np.ndarray | None = None,
) -> tuple[pd.DataFrame, float, list[dict]]:
    """Run coordinate-descent hill climbing on per-param global offsets.

    Args:
        df: per-frame parameter trajectory. Modified in place via copy.
        note_regions: from s06b's note segmentation.
        param_cols: ordered `p_<name>` columns; same as the manifest's.
        pinned_cols: subset of `param_cols` to leave alone.
        profile_path: synth profile YAML.
        total_sec: render duration (should match target audio length).
        target_emb_t: torch tensor on `device`, EnCodec embedding of target.
        embedder: `s04_embed.embed.Embedder` instance (for scoring).
        device: torch device string.
        offsets: offset values to try for each param. The empty offset (0)
            is implicitly "current value" so it is omitted from the list.
        n_passes: maximum number of full sweeps over `param_cols`. Sweep
            terminates early if no param improves the score.
        verbose: print per-param progress.

    Returns:
        (refined_df, final_score, change_log) where:
            refined_df is the dataframe after all accepted offsets;
            final_score is the real-synth cosine distance after refinement;
            change_log is a list of `{pass, param, offset, score, delta}`
              dicts, one per accepted move.
    """
    pinned_cols = set(pinned_cols)
    candidates = [c for c in param_cols if c not in pinned_cols]

    if not candidates:
        # Should never happen with the OB-Xf profile, but be defensive.
        if verbose:
            print("  Hill climb: no unpinned params to optimise.")
        return df.copy(), float("inf"), []

    df = df.copy()

    # Initial score: render the current trajectory once. Subsequent best
    # candidates are compared against this.
    current_score, _ = render_and_score(
        df, note_regions, param_cols, profile_path, total_sec,
        target_emb_t, embedder, device,
        target_mrstft_audio=target_mrstft_audio, target_ap=target_ap,
    )
    initial_score = current_score
    if verbose:
        print(f"  Hill climb start: real-synth score = {current_score:.4f}")
        print(f"  Optimising {len(candidates)} unpinned params over {len(list(offsets))} "
              f"offsets, max {n_passes} passes")

    change_log: list[dict] = []

    # offsets may be a generator; materialise once so we can re-iterate.
    offsets = tuple(offsets)

    for pass_i in range(1, n_passes + 1):
        if verbose:
            print(f"\n  Pass {pass_i}/{n_passes}:")
        improved = False

        for col in candidates:
            best_offset = 0.0
            best_score = current_score
            original = df[col].values.copy()

            for off in offsets:
                trial_df = df.copy()
                trial_df[col] = np.clip(original + off, 0.0, 1.0)
                s, _ = render_and_score(
                    trial_df, note_regions, param_cols, profile_path, total_sec,
                    target_emb_t, embedder, device,
                    target_mrstft_audio=target_mrstft_audio, target_ap=target_ap,
                )
                if s < best_score:
                    best_score = s
                    best_offset = off

            if best_offset != 0.0:
                df[col] = np.clip(original + best_offset, 0.0, 1.0)
                delta = best_score - current_score   # negative = improvement
                change_log.append({
                    "pass": pass_i,
                    "param": col,
                    "offset": best_offset,
                    "score": best_score,
                    "delta": delta,
                })
                if verbose:
                    p_name = col.removeprefix("p_")
                    print(f"    {p_name:30s}  off={best_offset:+.2f}  "
                          f"score={best_score:.4f}  Δ={delta:+.4f}")
                current_score = best_score
                improved = True

        if not improved:
            if verbose:
                print(f"  No improvement in pass {pass_i} — stopping.")
            break

    if verbose:
        total_delta = current_score - initial_score
        print(f"\n  Hill climb done: {initial_score:.4f} → {current_score:.4f}  "
              f"(Δ {total_delta:+.4f}, {len(change_log)} moves)")

    return df, current_score, change_log

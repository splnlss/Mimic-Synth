"""Strategy 2: CMA-ES on real VST renders — three modes.

Modes
-----
global (bypass)
    Single CMA-ES run applying a global per-param offset to all frames.
    The original working approach (achieved ~0.029 on crane scream).
    Use this when you want speed or when per-region adds no benefit.
    Fastest: ~120–320 renders.

per-region
    Independent CMA-ES per note region, each scored against that region's
    own target embedding. Conceptually ideal for multi-note targets but
    prone to step-function discontinuities and struggles on short regions
    (< 400ms) where EnCodec embeddings are noisy.
    Slowest: ~1000–3500 renders.

hybrid (default)
    Global CMA-ES first (fast, robust, gets to ~0.029).  Then, for each
    region whose duration >= min_region_sec, run a short per-region
    fine-tune pass seeded from the global result.  Only accept the
    per-region result when it beats the global score by more than
    `per_region_improvement` (5% by default).  Apply linear crossfade
    at region boundaries to avoid abrupt timbral jumps.
    Recommended: combines global robustness with per-region richness.

Expanded parameter space (22 total)
    15 surrogate params (from s06b) + 7 from `cmaes_extra_params` in
    the profile: Filter Env Attack/Decay/Sustain/Release, Osc 2 Volume,
    Amp Env Sustain, Unison Detune.  These 7 are the primary source of
    timbral richness in subtractive synthesis and were frozen at reset
    values in all previous stages.

Research
--------
Hansen (2016) "CMA-ES Tutorial" — sigma guidelines; IPOP restart details.
Yee-King (2011) — osc config scouting outperforms single-waveform assumption.
Reniery et al. (2023) — hybrid global+local search beats pure global.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml

from s07_refine.audio_compare import (
    render_trajectory,
    render_region,
    render_region_and_score,
    score_audio,
    score_audio_composite,
    compute_mrstft_features,
    compute_ap_features,
)


# ── Constants ────────────────────────────────────────────────────────────────

OSC_CONFIGS: dict[str, dict[str, float]] = {
    "saw":       {"Osc 1 Saw Wave": 1.0, "Osc 1 Pulse Wave": 0.0},
    "pulse":     {"Osc 1 Saw Wave": 0.0, "Osc 1 Pulse Wave": 1.0},
    "saw+pulse": {"Osc 1 Saw Wave": 1.0, "Osc 1 Pulse Wave": 1.0},
}

STAGNATION_THRESHOLD = 0.002
RESTART_LIMIT = 2
SCOUT_MAXITER = 5
SCOUT_POPSIZE = 8


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class CMAESResult:
    best_df: pd.DataFrame
    global_score: float
    region_scores: list[float]       # per-region scores (empty for global mode)
    osc_config: str
    mode: str                        # which mode was used
    n_renders: int
    restarts_used: int
    param_deltas: dict[str, float]
    extra_param_names: list[str]
    iteration_log: list[dict] = field(default_factory=list)


# ── Profile helpers ──────────────────────────────────────────────────────────

def load_extra_params(profile_path: _Path) -> dict[str, dict]:
    """Return {param_name: {reset, lo, hi}} for cmaes_extra_params in profile.

    The `bounds` key in each entry is optional; defaults to [0.0, 1.0].
    The `reset` value is the initial CMA-ES starting point for that param.
    """
    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    extras = profile.get("cmaes_extra_params", {})
    reset = profile.get("reset", {})
    result: dict[str, dict] = {}
    for name, cfg in extras.items():
        cfg = cfg or {}
        bounds = cfg.get("bounds", [0.0, 1.0])
        lo, hi = float(bounds[0]), float(bounds[1])
        init = float(reset.get(name, 0.5))
        # Clamp reset to the allowed bounds
        init = float(np.clip(init, lo, hi))
        result[name] = {"reset": init, "lo": lo, "hi": hi}
    return result


def _extend_df(df: pd.DataFrame, extra_params: dict[str, dict]) -> tuple[pd.DataFrame, list[str]]:
    """Add extra param columns (constant at reset value). Returns (df, extra_cols)."""
    df = df.copy()
    extra_cols: list[str] = []
    for name, cfg in extra_params.items():
        col = f"p_{name}"
        if col not in df.columns:
            df[col] = float(cfg["reset"])
        extra_cols.append(col)
    return df, extra_cols


# ── Main entry point ─────────────────────────────────────────────────────────

def cmaes_refine(
    df: pd.DataFrame,
    note_regions: list[dict],
    param_cols: list[str],
    pinned_cols: Iterable[str],
    profile_path: _Path,
    total_sec: float,
    target_wav: _Path,
    embedder,
    device: str,
    # Mode
    mode: str = "hybrid",           # "global" | "per-region" | "hybrid"
    # CMA-ES tuning
    sigma0: float = 0.10,
    popsize: int = 16,
    maxiter: int = 20,
    # Per-region / hybrid settings
    min_region_sec: float = 0.40,   # regions shorter than this skip per-region pass
    per_region_improvement: float = 0.05,  # min relative improvement to accept per-region
    crossfade_sec: float = 0.05,    # linear blend window at region boundaries
    verbose: bool = True,
) -> CMAESResult:
    """CMA-ES refinement with selectable mode.

    Args:
        mode:
            "global"     — single global CMA-ES pass (fastest, robust ~0.029).
            "per-region" — independent per-region CMA-ES (experimental).
            "hybrid"     — global first, then per-region fine-tune where
                           beneficial. Default and recommended.
        min_region_sec: regions shorter than this (ms) skip the per-region
            pass in hybrid mode (EnCodec embeddings are unreliable on very
            short clips).
        per_region_improvement: in hybrid mode, per-region result is only
            accepted when it beats the global score by this fraction.
        crossfade_sec: in per-region / hybrid mode, params are linearly
            interpolated over this window at each region boundary to avoid
            abrupt timbral steps.
    """
    try:
        import cma
    except ImportError:
        raise ImportError(
            "cma package required: conda run -n mimic-synth pip install cma"
        )

    pinned_cols = set(pinned_cols)

    # ── Extend to expanded param space ───────────────────────────────────────
    extra_param_meta = load_extra_params(profile_path)
    df, extra_cols = _extend_df(df, extra_param_meta)
    all_param_cols = param_cols + [c for c in extra_cols if c not in param_cols]
    extra_col_names = [c[2:] for c in extra_cols]

    # Build per-param bounds array (shape [n_free, 2]).  Surrogate params use
    # [0, 1]; extra params use their profile-defined bounds.
    extra_bounds: dict[str, tuple[float, float]] = {
        f"p_{name}": (cfg["lo"], cfg["hi"])
        for name, cfg in extra_param_meta.items()
    }

    if verbose:
        print(f"  Mode: {mode}  |  {len(param_cols)} surrogate + "
              f"{len(extra_cols)} extra = {len(all_param_cols)} total params")

    # ── Load and embed target ─────────────────────────────────────────────────
    target_audio, sr_t = sf.read(str(target_wav), dtype="float32")
    if target_audio.ndim == 2:
        target_audio = target_audio.mean(axis=1)
    full_target_emb_t = torch.tensor(
        embedder.encodec_embed(target_audio, sr_t, pool="mean"),
        dtype=torch.float32, device=device,
    )
    full_target_mrstft = compute_mrstft_features(target_audio)
    full_target_ap = compute_ap_features(target_audio, int(sr_t))  # None if pyworld absent

    active = "EnCodec+MRSTFT" + ("+AP" if full_target_ap is not None else "")
    if verbose:
        print(f"  Scoring: {active} composite distance (renormalised weights)")

    # ── Osc config scouting (once, regardless of mode) ───────────────────────
    best_osc_config, scout_renders = _scout_osc_configs(
        df=df, all_param_cols=all_param_cols, pinned_cols=pinned_cols,
        note_regions=note_regions, profile_path=profile_path, total_sec=total_sec,
        target_emb_t=full_target_emb_t, target_mrstft=full_target_mrstft,
        target_ap=full_target_ap, embedder=embedder, device=device,
        sigma0=sigma0 * 1.5, extra_bounds=extra_bounds, verbose=verbose, cma=cma,
    )
    extra_osc = OSC_CONFIGS[best_osc_config]
    if verbose:
        print(f"\n  Selected oscillator config: {best_osc_config}")

    # ── Dispatch to mode ──────────────────────────────────────────────────────
    shared = dict(
        df=df, all_param_cols=all_param_cols, pinned_cols=pinned_cols,
        note_regions=note_regions, profile_path=profile_path, total_sec=total_sec,
        target_emb_t=full_target_emb_t, target_mrstft=full_target_mrstft,
        target_ap=full_target_ap,
        embedder=embedder, device=device,
        extra_osc=extra_osc, sigma0=sigma0, popsize=popsize, maxiter=maxiter,
        extra_bounds=extra_bounds, verbose=verbose, cma=cma,
    )

    if mode == "global":
        result_df, global_score, n_renders, restarts, log = _run_global(**shared)
        region_scores: list[float] = []

    elif mode == "per-region":
        region_embs = _compute_region_embs(target_audio, sr_t, note_regions, embedder, device, full_target_emb_t)
        region_mrstfts = _compute_region_mrstfts(target_audio, sr_t, note_regions, full_target_mrstft)
        region_aps = _compute_region_aps(target_audio, int(sr_t), note_regions, full_target_ap)
        result_df, region_scores, global_score, n_renders, restarts, log = _run_per_region(
            **shared, region_embs=region_embs, region_mrstfts=region_mrstfts,
            region_aps=region_aps, crossfade_sec=crossfade_sec, sr=sr_t,
        )

    else:  # hybrid
        region_embs = _compute_region_embs(target_audio, sr_t, note_regions, embedder, device, full_target_emb_t)
        region_mrstfts = _compute_region_mrstfts(target_audio, sr_t, note_regions, full_target_mrstft)
        region_aps = _compute_region_aps(target_audio, int(sr_t), note_regions, full_target_ap)
        result_df, region_scores, global_score, n_renders, restarts, log = _run_hybrid(
            **shared, region_embs=region_embs, region_mrstfts=region_mrstfts,
            region_aps=region_aps, min_region_sec=min_region_sec,
            per_region_improvement=per_region_improvement,
            crossfade_sec=crossfade_sec, sr=sr_t,
        )

    total_renders = scout_renders + n_renders

    # ── Param deltas ──────────────────────────────────────────────────────────
    param_deltas: dict[str, float] = {}
    for col in all_param_cols:
        if col in pinned_cols:
            continue
        old = float(df[col].median()) if col in df.columns else 0.5
        new = float(result_df[col].median()) if col in result_df.columns else old
        if abs(new - old) > 0.005:
            param_deltas[col] = new - old

    if verbose:
        print(f"\n  CMA-ES done ({mode} mode):")
        if region_scores:
            print(f"    Region scores: {[f'{s:.4f}' for s in region_scores]}")
        print(f"    Global score:  {global_score:.4f}")
        print(f"    Total renders: {total_renders}  restarts: {restarts}")
        print(f"  Params moved ({len(param_deltas)}):")
        for col, d in sorted(param_deltas.items(), key=lambda x: -abs(x[1])):
            tag = " ★" if col[2:] in extra_col_names else ""
            print(f"    {col[2:]:35s} Δ={d:+.3f}{tag}")
        if extra_col_names:
            print(f"  (★ = new extra param)")

    return CMAESResult(
        best_df=result_df,
        global_score=global_score,
        region_scores=region_scores,
        osc_config=best_osc_config,
        mode=mode,
        n_renders=total_renders,
        restarts_used=restarts,
        param_deltas=param_deltas,
        extra_param_names=extra_col_names,
        iteration_log=log,
    )


# ── Mode 1: Global ────────────────────────────────────────────────────────────

def _make_bounds(free_cols: list[str], extra_bounds: dict[str, tuple[float, float]]):
    """Return (lo_list, hi_list) for the free param columns."""
    lo = [extra_bounds.get(c, (0.0, 1.0))[0] for c in free_cols]
    hi = [extra_bounds.get(c, (0.0, 1.0))[1] for c in free_cols]
    return lo, hi


def _run_global(
    df, all_param_cols, pinned_cols, note_regions, profile_path, total_sec,
    target_emb_t, target_mrstft, target_ap, embedder, device, extra_osc,
    sigma0, popsize, maxiter, extra_bounds, verbose, cma,
) -> tuple[pd.DataFrame, float, int, int, list]:
    """Single CMA-ES over global per-param offsets. The clean bypass path."""
    free_cols = [c for c in all_param_cols if c not in pinned_cols]
    x0_median = {c: float(df[c].median()) if c in df.columns else 0.5
                 for c in all_param_cols}
    lo, hi = _make_bounds(free_cols, extra_bounds)
    x0_free = np.clip([x0_median[c] for c in free_cols], lo, hi).astype(np.float64)

    best_x, best_score, n_renders, restarts, log = _ipop_cmaes(
        objective=lambda x: _score_global(
            x, free_cols, pinned_cols, x0_median, df, all_param_cols,
            note_regions, profile_path, total_sec, target_emb_t, embedder, device,
            extra_osc, target_mrstft, target_ap,
        ),
        x0=x0_free, n_dims=len(x0_free), bounds=(lo, hi),
        sigma0=sigma0, popsize=popsize, maxiter=maxiter,
        score_target=0.04, verbose=verbose, verbose_prefix="  ", cma=cma,
    )

    result_df = _apply_global_offsets(df, best_x, free_cols, x0_median, all_param_cols, pinned_cols)

    audio, sr = render_trajectory(result_df, note_regions, all_param_cols, profile_path, total_sec, extra_osc)
    global_score = score_audio_composite(audio, sr, target_emb_t, embedder, device, target_mrstft, target_ap)
    n_renders += 1

    return result_df, global_score, n_renders, restarts, log


# ── Mode 2: Per-region ────────────────────────────────────────────────────────

def _run_per_region(
    df, all_param_cols, pinned_cols, note_regions, profile_path, total_sec,
    target_emb_t, target_mrstft, target_ap, region_embs, region_mrstfts, region_aps,
    embedder, device, extra_osc, sigma0, popsize, maxiter, extra_bounds,
    crossfade_sec, sr, verbose, cma,
) -> tuple[pd.DataFrame, list[float], float, int, int, list]:
    """Independent CMA-ES per region, stitched with crossfade."""
    free_cols = [c for c in all_param_cols if c not in pinned_cols]
    x0_median = {c: float(df[c].median()) if c in df.columns else 0.5
                 for c in all_param_cols}

    region_best_params: list[dict[str, float]] = []
    region_scores: list[float] = []
    total_renders = 0
    total_restarts = 0
    all_logs: list[dict] = []

    for reg_idx, region in enumerate(note_regions):
        reg_dur = max(0.05, region["offset_sec"] - region["onset_sec"])
        reg_note = region["midi_note"]
        if verbose:
            print(f"\n  Region {reg_idx}: {region['onset_sec']:.2f}–"
                  f"{region['offset_sec']:.2f}s  MIDI={reg_note}  ({reg_dur:.2f}s)")

        mask = ((df["timestamp"] >= region["onset_sec"]) &
                (df["timestamp"] <= region["offset_sec"]))
        region_df = df[mask] if mask.sum() > 0 else df
        x0_reg = {c: float(region_df[c].median()) if c in region_df.columns else x0_median.get(c, 0.5)
                  for c in all_param_cols}
        lo_r, hi_r = _make_bounds(free_cols, extra_bounds)
        x0_free = np.clip([x0_reg.get(c, 0.5) for c in free_cols], lo_r, hi_r).astype(np.float64)
        best_x, best_score, n, restarts, log = _ipop_cmaes(
            objective=lambda x, _note=reg_note, _dur=reg_dur, \
                    _emb=region_embs[reg_idx], _mrstft=region_mrstfts[reg_idx], \
                    _ap=region_aps[reg_idx]: \
                _score_region(x, free_cols, pinned_cols, x0_reg,
                              _note, _dur, profile_path, _emb, embedder, device,
                              extra_osc, _mrstft, _ap),
            x0=x0_free, n_dims=len(x0_free), bounds=(lo_r, hi_r),
            sigma0=sigma0, popsize=popsize, maxiter=maxiter,
            score_target=0.03, verbose=verbose, verbose_prefix="    ", cma=cma,
        )
        total_renders += n
        total_restarts += restarts
        for e in log: e["region"] = reg_idx
        all_logs.extend(log)

        best_params = x0_reg.copy()
        for i, col in enumerate(free_cols):
            best_params[col] = float(np.clip(best_x[i], 0.0, 1.0))
        region_best_params.append(best_params)
        region_scores.append(best_score)
        if verbose:
            print(f"    Region {reg_idx} final: {best_score:.4f}")

    result_df = _stitch_with_crossfade(df, note_regions, region_best_params, all_param_cols, crossfade_sec, sr)

    audio, sr_r = render_trajectory(result_df, note_regions, all_param_cols, profile_path, total_sec, extra_osc)
    global_score = score_audio_composite(audio, sr_r, target_emb_t, embedder, device, target_mrstft, target_ap)
    total_renders += 1

    return result_df, region_scores, global_score, total_renders, total_restarts, all_logs


# ── Mode 3: Hybrid ────────────────────────────────────────────────────────────

def _run_hybrid(
    df, all_param_cols, pinned_cols, note_regions, profile_path, total_sec,
    target_emb_t, target_mrstft, target_ap, region_embs, region_mrstfts, region_aps,
    embedder, device, extra_osc, sigma0, popsize, maxiter, extra_bounds,
    min_region_sec, per_region_improvement, crossfade_sec, sr, verbose, cma,
) -> tuple[pd.DataFrame, list[float], float, int, int, list]:
    """Global first, then per-region fine-tune where beneficial."""
    if verbose:
        print("\n  Phase 1: Global CMA-ES")

    result_df, global_score, n1, restarts1, log1 = _run_global(
        df, all_param_cols, pinned_cols, note_regions, profile_path, total_sec,
        target_emb_t, target_mrstft, target_ap, embedder, device, extra_osc,
        sigma0=sigma0, popsize=popsize, maxiter=maxiter,
        extra_bounds=extra_bounds, verbose=verbose, cma=cma,
    )
    if verbose:
        print(f"  Phase 1 done: global score = {global_score:.4f}")

    free_cols = [c for c in all_param_cols if c not in pinned_cols]
    region_scores: list[float] = []
    n2 = 0
    restarts2 = 0
    log2: list[dict] = []
    per_region_params = []

    if verbose:
        print(f"\n  Phase 2: Per-region fine-tune "
              f"(min_region={min_region_sec:.2f}s, "
              f"min_improvement={per_region_improvement*100:.0f}%)")

    for reg_idx, region in enumerate(note_regions):
        reg_dur = max(0.0, region["offset_sec"] - region["onset_sec"])
        reg_note = region["midi_note"]

        if reg_dur < min_region_sec:
            if verbose:
                print(f"  Region {reg_idx}: {reg_dur:.2f}s < {min_region_sec:.2f}s — skipped")
            per_region_params.append(None)
            region_scores.append(float("nan"))
            continue

        mask = ((result_df["timestamp"] >= region["onset_sec"]) &
                (result_df["timestamp"] <= region["offset_sec"]))
        region_df = result_df[mask] if mask.sum() > 0 else result_df
        x0_reg = {c: float(region_df[c].median()) if c in region_df.columns else 0.5
                  for c in all_param_cols}

        global_params_named = {c[2:]: x0_reg.get(c, 0.5) for c in all_param_cols}
        baseline_region_score = render_region_and_score(
            global_params_named, reg_note, reg_dur, profile_path,
            region_embs[reg_idx], embedder, device, extra_osc,
            target_mrstft=region_mrstfts[reg_idx],
            target_ap=region_aps[reg_idx],
        )
        n2 += 1

        if verbose:
            print(f"\n  Region {reg_idx}: {region['onset_sec']:.2f}–"
                  f"{region['offset_sec']:.2f}s  MIDI={reg_note}  "
                  f"({reg_dur:.2f}s)  baseline={baseline_region_score:.4f}")

        threshold = baseline_region_score * (1.0 - per_region_improvement)
        lo_r, hi_r = _make_bounds(free_cols, extra_bounds)
        x0_free = np.clip([x0_reg.get(c, 0.5) for c in free_cols], lo_r, hi_r).astype(np.float64)

        best_x, best_score, n, restarts, log = _ipop_cmaes(
            objective=lambda x, _note=reg_note, _dur=reg_dur, \
                    _emb=region_embs[reg_idx], _mrstft=region_mrstfts[reg_idx], \
                    _ap=region_aps[reg_idx]: \
                _score_region(x, free_cols, pinned_cols, x0_reg,
                              _note, _dur, profile_path, _emb, embedder, device,
                              extra_osc, _mrstft, _ap),
            x0=x0_free, n_dims=len(x0_free), bounds=(lo_r, hi_r),
            sigma0=sigma0 * 0.7, popsize=max(8, popsize // 2), maxiter=maxiter,
            score_target=0.03, verbose=verbose, verbose_prefix="    ", cma=cma,
        )
        n2 += n
        restarts2 += restarts
        for e in log: e["region"] = reg_idx
        log2.extend(log)

        if best_score < threshold:
            best_params = x0_reg.copy()
            for i, col in enumerate(free_cols):
                best_params[col] = float(np.clip(best_x[i], 0.0, 1.0))
            per_region_params.append(best_params)
            region_scores.append(best_score)
            if verbose:
                print(f"    ✓ Accepted: {baseline_region_score:.4f} → {best_score:.4f} "
                      f"(improvement {(baseline_region_score - best_score)/baseline_region_score*100:.1f}%)")
        else:
            per_region_params.append(None)
            region_scores.append(baseline_region_score)
            if verbose:
                print(f"    ✗ Rejected: {best_score:.4f} did not beat "
                      f"threshold {threshold:.4f} — keeping global params")

    has_any_override = any(p is not None for p in per_region_params)
    if has_any_override:
        filled_params: list[dict[str, float]] = []
        for reg_idx, region in enumerate(note_regions):
            if per_region_params[reg_idx] is not None:
                filled_params.append(per_region_params[reg_idx])
            else:
                mask = ((result_df["timestamp"] >= region["onset_sec"]) &
                        (result_df["timestamp"] <= region["offset_sec"]))
                region_df = result_df[mask] if mask.sum() > 0 else result_df
                filled_params.append({
                    c: float(region_df[c].median()) if c in region_df.columns else 0.5
                    for c in all_param_cols
                })

        result_df = _stitch_with_crossfade(
            result_df, note_regions, filled_params, all_param_cols, crossfade_sec, sr
        )
        audio, sr_r = render_trajectory(
            result_df, note_regions, all_param_cols, profile_path, total_sec, extra_osc
        )
        final_score = score_audio_composite(audio, sr_r, target_emb_t, embedder, device, target_mrstft, target_ap)
        n2 += 1
        if verbose:
            print(f"\n  Hybrid final global score: {final_score:.4f} "
                  f"(vs global-only: {global_score:.4f})")
        global_score = final_score
    else:
        if verbose:
            print("\n  No per-region overrides accepted — global result unchanged.")

    return result_df, region_scores, global_score, n1 + n2, restarts1 + restarts2, log1 + log2


# ── IPOP CMA-ES core ──────────────────────────────────────────────────────────

def _ipop_cmaes(
    objective, x0, n_dims, sigma0, popsize, maxiter, score_target,
    verbose, verbose_prefix, cma,
    bounds: tuple[list, list] | None = None,
) -> tuple[np.ndarray, float, int, int, list]:
    """IPOP-CMA-ES: restart with 1.5× population when stagnating."""
    best_x = x0.copy()
    best_score = float("inf")
    total_renders = 0
    total_restarts = 0
    all_log: list[dict] = []
    current_sigma = sigma0
    current_pop = popsize

    for restart in range(RESTART_LIMIT + 1):
        if restart > 0 and verbose:
            print(f"{verbose_prefix}IPOP restart {restart} "
                  f"(σ={current_sigma:.3f}, pop={current_pop})")

        x, score, n, log = _run_cmaes_once(
            objective=objective,
            x0=best_x.copy(),
            n_dims=n_dims,
            sigma0=current_sigma,
            popsize=current_pop,
            maxiter=maxiter,
            score_target=score_target,
            verbose=verbose,
            verbose_prefix=verbose_prefix,
            cma=cma,
            bounds=bounds,
        )
        total_renders += n
        all_log.extend(log)

        if score < best_score:
            best_score = score
            best_x = x.copy()

        if best_score <= score_target or total_restarts >= RESTART_LIMIT:
            break
        total_restarts += 1
        current_pop = int(current_pop * 1.5)
        current_sigma = max(current_sigma * 0.8, 0.03)

    return best_x, best_score, total_renders, total_restarts, all_log


def _run_cmaes_once(
    objective, x0, n_dims, sigma0, popsize, maxiter, score_target,
    verbose, verbose_prefix, cma,
    bounds: tuple[list, list] | None = None,
) -> tuple[np.ndarray, float, int, list]:
    lo = bounds[0] if bounds else [0.0] * n_dims
    hi = bounds[1] if bounds else [1.0] * n_dims
    es = cma.CMAEvolutionStrategy(
        x0.tolist(), sigma0,
        {"bounds": [lo, hi],
         "maxiter": maxiter, "popsize": popsize, "verbose": -9},
    )
    best_x = x0.copy()
    best_score = float("inf")
    n_renders = 0
    log: list[dict] = []

    while not es.stop():
        xs = es.ask()
        scores = [float(objective(np.array(x))) for x in xs]
        n_renders += len(xs)
        es.tell(xs, scores)

        idx = int(np.argmin(scores))
        if scores[idx] < best_score:
            best_score = scores[idx]
            best_x = np.array(xs[idx]).clip(0.0, 1.0)

        log.append({"iter": len(log), "best": float(best_score), "sigma": float(es.sigma)})
        if verbose:
            print(f"{verbose_prefix}iter {len(log):2d}  best={best_score:.4f}  σ={es.sigma:.4f}")
        if best_score <= score_target:
            break

    return best_x, best_score, n_renders, log


# ── Objective helpers ─────────────────────────────────────────────────────────

def _score_global(
    x_free, free_cols, pinned_cols, x0_median, df, all_param_cols,
    note_regions, profile_path, total_sec, target_emb_t, embedder, device,
    extra_osc, target_mrstft=None, target_ap=None,
) -> float:
    trial_df = _apply_global_offsets(df, x_free, free_cols, x0_median, all_param_cols, pinned_cols)
    audio, sr = render_trajectory(trial_df, note_regions, all_param_cols, profile_path, total_sec, extra_osc)
    return score_audio_composite(audio, sr, target_emb_t, embedder, device, target_mrstft, target_ap)


def _score_region(
    x_free, free_cols, pinned_cols, x0_dict, note, region_dur,
    profile_path, target_emb_t, embedder, device, extra_osc,
    target_mrstft=None, target_ap=None,
) -> float:
    params = x0_dict.copy()
    for i, col in enumerate(free_cols):
        params[col] = float(np.clip(x_free[i], 0.0, 1.0))
    params_named = {col[2:]: v for col, v in params.items()}
    return render_region_and_score(
        params_named, note, region_dur, profile_path,
        target_emb_t, embedder, device, extra_osc,
        target_mrstft=target_mrstft, target_ap=target_ap,
    )


# ── Trajectory helpers ────────────────────────────────────────────────────────

def _apply_global_offsets(
    df, x_free, free_cols, x0_median, all_param_cols, pinned_cols,
) -> pd.DataFrame:
    """Apply global per-param offsets uniformly across all frames."""
    result_df = df.copy()
    for i, col in enumerate(free_cols):
        new_val = float(np.clip(x_free[i], 0.0, 1.0))
        delta = new_val - x0_median.get(col, 0.5)
        if col in result_df.columns:
            result_df[col] = np.clip(result_df[col] + delta, 0.0, 1.0)
        else:
            result_df[col] = new_val
    return result_df


def _stitch_with_crossfade(
    df: pd.DataFrame,
    note_regions: list[dict],
    region_params: list[dict[str, float]],
    all_param_cols: list[str],
    crossfade_sec: float,
    sr: int,
) -> pd.DataFrame:
    """Step-function trajectory with linear crossfade at region boundaries.

    For each region, frames are set to that region's optimal params.
    In the `crossfade_sec` window around each boundary, values are linearly
    interpolated between adjacent region params to avoid abrupt timbral jumps.
    """
    result_df = df.copy()
    for col in all_param_cols:
        if col not in result_df.columns:
            result_df[col] = 0.5

    timestamps = result_df["timestamp"].values

    # First pass: set each frame to its region's params
    for i, (region, best_params) in enumerate(zip(note_regions, region_params)):
        mask = ((result_df["timestamp"] >= region["onset_sec"]) &
                (result_df["timestamp"] <= region["offset_sec"]))
        for col in all_param_cols:
            val = best_params.get(col, float(result_df[col].median()))
            result_df.loc[mask, col] = float(np.clip(val, 0.0, 1.0))

    # Second pass: crossfade at each boundary between adjacent regions
    if crossfade_sec > 0:
        for i in range(len(note_regions) - 1):
            boundary = note_regions[i + 1]["onset_sec"]
            half = crossfade_sec / 2.0
            fade_mask = np.abs(timestamps - boundary) < half

            if fade_mask.sum() == 0:
                continue

            for col in all_param_cols:
                val_before = region_params[i].get(col, 0.5)
                val_after = region_params[i + 1].get(col, 0.5)
                if abs(val_after - val_before) < 1e-4:
                    continue

                fade_times = timestamps[fade_mask]
                alphas = np.clip((fade_times - (boundary - half)) / crossfade_sec, 0.0, 1.0)
                blended = val_before * (1.0 - alphas) + val_after * alphas
                result_df.loc[fade_mask, col] = np.clip(blended, 0.0, 1.0)

    return result_df


# ── Osc config scouting ──────────────────────────────────────────────────────

def _scout_osc_configs(
    df, all_param_cols, pinned_cols, note_regions, profile_path, total_sec,
    target_emb_t, target_mrstft, target_ap, embedder, device, sigma0, extra_bounds, verbose, cma,
) -> tuple[str, int]:
    free_cols = [c for c in all_param_cols if c not in pinned_cols]
    x0_median = {c: float(df[c].median()) if c in df.columns else 0.5 for c in all_param_cols}
    lo_s, hi_s = _make_bounds(free_cols, extra_bounds)
    x0_free = np.clip([x0_median[c] for c in free_cols], lo_s, hi_s).astype(np.float64)

    best_config = "saw"
    best_score = float("inf")
    total_renders = 0

    if verbose:
        print(f"\n=== Osc config scouting ===")

    for config_name, osc_extra in OSC_CONFIGS.items():
        _, score, n, _ = _run_cmaes_once(
            objective=lambda x, _osc=osc_extra: _score_global(
                x, free_cols, pinned_cols, x0_median, df, all_param_cols,
                note_regions, profile_path, total_sec, target_emb_t, embedder, device, _osc,
                target_mrstft=target_mrstft, target_ap=target_ap,
            ),
            x0=x0_free.copy(), n_dims=len(x0_free), bounds=(lo_s, hi_s),
            sigma0=sigma0, popsize=SCOUT_POPSIZE, maxiter=SCOUT_MAXITER,
            score_target=0.02, verbose=False, verbose_prefix="", cma=cma,
        )
        total_renders += n
        if verbose:
            print(f"  {config_name:10s} → {score:.4f}  ({n} renders)")
        if score < best_score:
            best_score = score
            best_config = config_name

    return best_config, total_renders


# ── Region embedding / MRSTFT helpers ────────────────────────────────────────

def _compute_region_mrstfts(
    target_audio: np.ndarray,
    sr_t: int,
    note_regions: list[dict],
    fallback_mrstft: np.ndarray,
    min_samples: int | None = None,
) -> list[np.ndarray]:
    """MRSTFT features for each note region; falls back to full-audio features."""
    if min_samples is None:
        min_samples = int(0.25 * sr_t)  # ~250ms minimum for stable std estimate
    mrstfts = []
    for r in note_regions:
        s = int(r["onset_sec"] * sr_t)
        e = int(r["offset_sec"] * sr_t)
        seg = target_audio[s:e]
        if len(seg) >= min_samples:
            mrstfts.append(compute_mrstft_features(seg))
        else:
            mrstfts.append(fallback_mrstft)
    return mrstfts


def _compute_region_aps(
    target_audio: np.ndarray,
    sr_t: int,
    note_regions: list[dict],
    fallback_ap: np.ndarray | None,
    min_samples: int | None = None,
) -> list[np.ndarray | None]:
    """Aperiodicity features per note region; falls back to full-audio features."""
    if min_samples is None:
        min_samples = int(0.10 * sr_t)  # 100ms minimum for WORLD stability
    aps = []
    for r in note_regions:
        s = int(r["onset_sec"] * sr_t)
        e = int(r["offset_sec"] * sr_t)
        seg = target_audio[s:e]
        if len(seg) >= min_samples:
            aps.append(compute_ap_features(seg, sr_t))
        else:
            aps.append(fallback_ap)
    return aps


def _compute_region_embs(
    target_audio, sr_t, note_regions, embedder, device, fallback_emb_t,
) -> list[torch.Tensor]:
    """Embed each region's audio slice. Falls back to full-target emb if too short."""
    MIN_SAMPLES = int(0.25 * sr_t)  # ~250ms minimum for reliable embedding
    embs = []
    for r in note_regions:
        s = int(r["onset_sec"] * sr_t)
        e = int(r["offset_sec"] * sr_t)
        seg = target_audio[s:e]
        if len(seg) >= MIN_SAMPLES:
            emb = torch.tensor(
                embedder.encodec_embed(seg, sr_t, pool="mean"),
                dtype=torch.float32, device=device,
            )
        else:
            emb = fallback_emb_t  # region too short — use full-target embedding
        embs.append(emb)
    return embs

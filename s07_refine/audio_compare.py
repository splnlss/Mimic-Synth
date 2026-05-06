"""Shared render + score helpers for s07 refinement strategies.

Renders a per-frame parameter trajectory through OB-Xf via DawDreamer and
scores the result against a target embedding. Each render allocates a fresh
DawDreamer engine + plugin so there is no patch bleed between candidates;
this matches the s06b refinement loop's behaviour.

The functions take a `pd.DataFrame` whose columns are timestamp + p_<name>
columns, plus a list of note_regions. This is exactly the shape produced by
`s06b_live.stream_invert.stream_invert()` and persisted to
`stream_params.parquet`, so refinement modules can consume that file directly.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml


def render_trajectory(
    df: pd.DataFrame,
    note_regions: list[dict],
    param_cols: list[str],
    profile_path: _Path,
    total_sec: float,
    extra_params: dict[str, float] | None = None,
) -> tuple[np.ndarray, int]:
    """Render the per-frame parameter trajectory in `df` through the real VST.

    Applies all reset values from the profile first so the synth starts from
    a known state on every render. This is critical for parameter comparisons:
    without it, OB-Xf's internal defaults (e.g. saw wave on/off) may differ
    between render calls.

    Args:
        df: per-frame param trajectory. Must contain a `timestamp` column and
            one `p_<name>` column per param. Frame ordering follows the row
            order in `df`.
        note_regions: list of dicts with keys `onset_sec`, `offset_sec`,
            `midi_note`. Note-off for region N is sent at region N+1's onset
            (or at `offset_sec` for the last region).
        param_cols: ordered list of `p_<name>` columns to apply as VST
            parameter automation. Anything in `df` not in `param_cols` is
            ignored.
        profile_path: profile YAML; supplies `synth.plugin_path_linux`,
            `probe.sample_rate`, and `reset` values.
        total_sec: render duration in seconds. Should match the target audio
            length so embeddings align.
        extra_params: optional dict of {param_name: value} applied AFTER the
            reset and BEFORE the automation. Used by the CMA-ES oscillator
            config loop to override waveform selection (e.g. set Osc 1 Pulse
            Wave to 1.0 for a pulse-wave render pass).

    Returns:
        (audio_mono_float32, sample_rate)
    """
    import dawdreamer as daw

    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    sr = profile.get("probe", {}).get("sample_rate", 48000)
    vst_path = _Path(profile["synth"]["plugin_path_linux"])
    reset_values: dict[str, float] = profile.get("reset", {})

    engine = daw.RenderEngine(sr, 512)
    plugin = engine.make_plugin_processor("synth", str(vst_path))
    engine.load_graph([(plugin, [])])

    num_params = plugin.get_plugin_parameter_size()
    name_to_idx = {plugin.get_parameter_name(i): i for i in range(num_params)}

    # Apply profile reset values to establish a known starting state.
    # Without this, OB-Xf falls back to an undefined internal default for
    # any parameter we don't explicitly set.
    for name, val in reset_values.items():
        if name in name_to_idx:
            plugin.set_parameter(name_to_idx[name], float(val))

    # Apply oscillator-config overrides (e.g. switch from saw to pulse).
    if extra_params:
        for name, val in extra_params.items():
            if name in name_to_idx:
                plugin.set_parameter(name_to_idx[name], float(val))

    # Per-region MIDI note-on/off. Note-off lands at next region's onset (or
    # at this region's offset for the final region). Matches s06b semantics.
    for i, r in enumerate(note_regions):
        note_on = r["onset_sec"]
        if i < len(note_regions) - 1:
            note_off = note_regions[i + 1]["onset_sec"]
        else:
            note_off = r["offset_sec"]
        dur = max(0.0, note_off - note_on)
        if dur > 0:
            plugin.add_midi_note(r["midi_note"], 100, note_on, dur)

    # Parameter automation. Each param column becomes a (timestamp, value)
    # pairs array via DawDreamer's set_automation API.
    timestamps = df["timestamp"].values
    for col in param_cols:
        p_name = col.removeprefix("p_")
        if p_name in name_to_idx:
            data = np.column_stack((timestamps, df[col].values))
            plugin.set_automation(name_to_idx[p_name], data)

    engine.render(total_sec)
    audio = plugin.get_audio().transpose()
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def score_audio(
    audio: np.ndarray,
    sr: int,
    target_emb_t: torch.Tensor,
    embedder,
    device: str,
) -> float:
    """Cosine distance between `audio`'s EnCodec embedding and `target_emb_t`.

    target_emb_t must already be on `device`. Returns a Python float in
    [0, 2] (cosine *distance*, not similarity).
    """
    emb = embedder.encodec_embed(audio, sr, pool="mean")
    emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
    return (
        1.0 - F.cosine_similarity(emb_t.unsqueeze(0), target_emb_t.unsqueeze(0))
    ).item()


def render_and_score(
    df: pd.DataFrame,
    note_regions: list[dict],
    param_cols: list[str],
    profile_path: _Path,
    total_sec: float,
    target_emb_t: torch.Tensor,
    embedder,
    device: str,
    extra_params: dict[str, float] | None = None,
) -> tuple[float, np.ndarray]:
    """Convenience: render `df` and return (score, audio)."""
    audio, sr = render_trajectory(
        df, note_regions, param_cols, profile_path, total_sec, extra_params
    )
    score = score_audio(audio, sr, target_emb_t, embedder, device)
    return score, audio

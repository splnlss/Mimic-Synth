"""Shared render + score helpers for s07 refinement strategies.

Renders a per-frame parameter trajectory through OB-Xf via DawDreamer and
scores the result against a target embedding. Each render allocates a fresh
DawDreamer engine + plugin so there is no patch bleed between candidates;
this matches the s06b refinement loop's behaviour.

The functions take a `pd.DataFrame` whose columns are timestamp + p_<name>
columns, plus a list of note_regions. This is exactly the shape produced by
`s06b_live.stream_invert.stream_invert()` and persisted to
`stream_params.parquet`, so refinement modules can consume that file directly.

Composite scoring (EnCodec + MRSTFT)
-------------------------------------
`score_audio` uses EnCodec cosine distance only (fast, high-level timbre).
`score_audio_composite` blends EnCodec (60%) with MRSTFT (40%).

The MRSTFT term captures what EnCodec misses:
  - Noise texture and inharmonic "air" content
  - Filter movement (temporal spectral variation / std across frames)
  - Attack transients and release character (pitch drop, spectral thickening)

MRSTFT features: at each of 4 FFT sizes (256/512/1024/2048 samples), compute
log-magnitude STFT and take the per-bin mean and std across time frames.
Mean = spectral shape; std = temporal variation (filter sweep, transient energy).
Total feature dim: sum of 2*(n_fft//2+1) for each n_fft = 3848.
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

# Composite scoring weights.
# EnCodec: high-level timbre color and resonance.
# MRSTFT: temporal spectral variation — filter sweeps, attack/release character.
# AP (aperiodicity): noise-vs-harmonic balance — "squaky", breathy, inharmonic content.
# When a term is unavailable (e.g. pyworld not installed), weights are renormalised.
ENCODEC_WEIGHT: float = 0.50
MRSTFT_WEIGHT: float = 0.30
AP_WEIGHT: float = 0.20


def _stft(audio: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Real-valued STFT returning magnitude [freq_bins, frames]."""
    window = np.hanning(n_fft)
    n_frames = 1 + (len(audio) - n_fft) // hop
    if n_frames <= 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float32)
    out = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame = audio[i * hop : i * hop + n_fft] * window
        out[:, i] = np.abs(np.fft.rfft(frame)).astype(np.float32)
    return out


def compute_mrstft_features(audio: np.ndarray, fft_sizes=(256, 512, 1024, 2048)) -> np.ndarray:
    """Multi-resolution STFT: per-bin log-magnitude mean and std across time.

    Mean captures spectral shape (filter cutoff position, harmonic content).
    Std captures temporal variation: filter sweeps, attack transients, noise
    bursts, and pitch-drop release thickening — the exact qualities EnCodec
    embeddings are blind to.

    Returns a float32 1-D array of length sum(2*(n_fft//2+1)) = 3848.
    Falls back to zeros for audio shorter than the smallest FFT window;
    cosine_distance(zeros, zeros) = 1 (neutral, not harmful).
    """
    audio = audio.astype(np.float32)
    feats: list[np.ndarray] = []
    for n_fft in fft_sizes:
        hop = n_fft // 4
        n_bins = n_fft // 2 + 1
        if len(audio) < n_fft:
            feats.append(np.zeros(n_bins, dtype=np.float32))  # mean
            feats.append(np.zeros(n_bins, dtype=np.float32))  # std
            continue
        spec = _stft(audio, n_fft=n_fft, hop=hop)            # [bins, frames]
        log_spec = np.log1p(spec)
        feats.append(log_spec.mean(axis=1))                   # spectral shape
        feats.append(log_spec.std(axis=1))                    # temporal variation
    return np.concatenate(feats).astype(np.float32)


def compute_ap_features(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """Aperiodicity profile via WORLD vocoder (pyworld).

    WORLD decomposes audio into f0 (pitch), sp (spectral envelope), and ap
    (aperiodicity per frequency bin). The AP vector directly quantifies how
    noisy vs harmonic the sound is at each frequency — 0 = purely periodic,
    1 = purely noise. Features:
        ap_mean [fft_bins]: average noise content per band (overall noisiness)
        ap_std  [fft_bins]: temporal variation in noise (burst / transient character)
    Total: 2 × fft_bins (typically 2050 at 48kHz).

    Returns None when pyworld is unavailable or audio is too short (< 100ms).
    Scores are always renormalised when this term is absent.
    """
    try:
        import pyworld as pw
    except ImportError:
        return None

    min_samples = int(sr * 0.10)   # 100ms minimum for stable WORLD analysis
    if len(audio) < min_samples:
        return None

    try:
        f0, sp, ap = pw.wav2world(audio.astype(np.float64), int(sr))
    except Exception:
        return None

    if ap.shape[0] == 0:
        return None

    ap_mean = ap.mean(axis=0).astype(np.float32)   # [fft_bins]
    ap_std = ap.std(axis=0).astype(np.float32)     # temporal noise variation
    return np.concatenate([ap_mean, ap_std]).astype(np.float32)


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2]. Returns 1.0 (neutral) for near-zero vectors."""
    ta = torch.tensor(a, dtype=torch.float32)
    tb = torch.tensor(b, dtype=torch.float32)
    if ta.norm() < 1e-6 or tb.norm() < 1e-6:
        return 1.0
    return float(1.0 - F.cosine_similarity(ta.unsqueeze(0), tb.unsqueeze(0)))


def score_audio_composite(
    audio: np.ndarray,
    sr: int,
    target_emb_t: torch.Tensor,
    embedder,
    device: str,
    target_mrstft: np.ndarray | None = None,
    target_ap: np.ndarray | None = None,
) -> float:
    """Composite score: EnCodec + MRSTFT + AP (aperiodicity), renormalised.

    Active terms and their base weights:
        EnCodec  50%  — high-level timbral color and resonance
        MRSTFT   30%  — filter movement, temporal spectral variation
        AP       20%  — noise-vs-harmonic balance (squaky, breathy quality)

    Inactive terms (None) are dropped and the remaining weights are
    renormalised so the score always has a consistent scale ∈ [0, 2].
    """
    enc_dist = score_audio(audio, sr, target_emb_t, embedder, device)
    terms: list[tuple[float, float]] = [(ENCODEC_WEIGHT, enc_dist)]

    if target_mrstft is not None:
        cand_mrstft = compute_mrstft_features(audio)
        terms.append((MRSTFT_WEIGHT, _cosine_dist(target_mrstft, cand_mrstft)))

    if target_ap is not None:
        cand_ap = compute_ap_features(audio, sr)
        if cand_ap is not None:
            terms.append((AP_WEIGHT, _cosine_dist(target_ap, cand_ap)))

    total_w = sum(w for w, _ in terms)
    return sum(w / total_w * d for w, d in terms)


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


def render_region(
    params: dict[str, float],
    note: int,
    region_dur: float,
    profile_path: _Path,
    release_tail: float = 0.3,
    extra_params: dict[str, float] | None = None,
) -> tuple[np.ndarray, int]:
    """Render a single static patch for one note region.

    Unlike `render_trajectory` which applies per-frame automation across the
    full timeline, this renders a short single-note clip at constant params.
    It is used by per-region CMA-ES where each region gets its own target
    embedding and independent optimisation.

    Args:
        params: {param_name_without_p_prefix: value} for every param to set.
            Any param not listed will take its profile reset value.
        note: MIDI note to play (0–127).
        region_dur: note-on duration in seconds. The render continues for
            `region_dur + release_tail` so the amp/filter release is captured
            in the embedding.
        profile_path: profile YAML.
        release_tail: extra render time beyond note-off to capture decay.
        extra_params: osc-config overrides applied after reset (e.g. waveform
            toggles from OSC_CONFIGS). Applied before `params`.

    Returns:
        (audio_mono_float32, sample_rate)
    """
    import dawdreamer as daw

    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    sr = profile.get("probe", {}).get("sample_rate", 48000)
    vst_path = _Path(profile["synth"]["plugin_path_linux"])
    reset_values: dict[str, float] = profile.get("reset", {})

    total_sec = region_dur + release_tail

    engine = daw.RenderEngine(sr, 512)
    plugin = engine.make_plugin_processor("synth", str(vst_path))
    engine.load_graph([(plugin, [])])

    num_params = plugin.get_plugin_parameter_size()
    name_to_idx = {plugin.get_parameter_name(i): i for i in range(num_params)}

    # Reset → osc config → candidate params (each layer overrides the previous)
    for name, val in reset_values.items():
        if name in name_to_idx:
            plugin.set_parameter(name_to_idx[name], float(val))
    if extra_params:
        for name, val in extra_params.items():
            if name in name_to_idx:
                plugin.set_parameter(name_to_idx[name], float(val))
    for name, val in params.items():
        if name in name_to_idx:
            plugin.set_parameter(name_to_idx[name], float(val))

    plugin.add_midi_note(note, 100, 0.0, region_dur)
    engine.render(total_sec)
    audio = plugin.get_audio().transpose()
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    # Return only the region duration (drop the release tail from the
    # embedding — we want to match the target region, not its silence)
    n_region = int(region_dur * sr)
    return audio[:n_region].astype(np.float32), sr


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


def render_region_and_score(
    params: dict[str, float],
    note: int,
    region_dur: float,
    profile_path: _Path,
    target_emb_t: torch.Tensor,
    embedder,
    device: str,
    extra_params: dict[str, float] | None = None,
    target_mrstft: np.ndarray | None = None,
    target_ap: np.ndarray | None = None,
) -> float:
    """Render a region and return composite distance (EnCodec + MRSTFT + AP)."""
    audio, sr = render_region(params, note, region_dur, profile_path,
                              extra_params=extra_params)
    return score_audio_composite(audio, sr, target_emb_t, embedder, device,
                                 target_mrstft, target_ap)


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
    target_mrstft: np.ndarray | None = None,
    target_ap: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Convenience: render `df` and return (score, audio)."""
    audio, sr = render_trajectory(
        df, note_regions, param_cols, profile_path, total_sec, extra_params
    )
    score = score_audio_composite(audio, sr, target_emb_t, embedder, device,
                                  target_mrstft, target_ap)
    return score, audio

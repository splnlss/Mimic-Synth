"""Shared render + score helpers for s07 refinement strategies.

Renders a per-frame parameter trajectory through the VST via DawDreamer and
scores the result against a target embedding. Each render allocates a fresh
DawDreamer engine + plugin so there is no patch bleed between candidates;
this matches the s06b refinement loop's behaviour.

The functions take a `pd.DataFrame` whose columns are timestamp + p_<name>
columns, plus a list of note_regions. This is exactly the shape produced by
`s06b_live.stream_invert.stream_invert()` and persisted to
`stream_params.parquet`, so refinement modules can consume that file directly.

Composite scoring (EnCodec + MRSTFT + AP)
------------------------------------------
`score_audio` uses EnCodec cosine distance only (fast, high-level timbre).
`score_audio_composite` blends EnCodec (50%), MRSTFT (30%), AP (20%).

The MRSTFT term uses auraloss.freq.MultiResolutionSTFTLoss — a differentiable
multi-scale spectral loss at FFT sizes 256/512/1024/2048. It captures what
EnCodec misses: noise texture, filter movement, attack transients, and release
character. Loss is normalized to approx [0, 2] to match cosine distance scale.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
# sys.path insert removed — installed via pyproject.toml

import auraloss.freq as _auraf
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

_MRSTFT_LOSS = _auraf.MultiResolutionSTFTLoss(
    fft_sizes=[256, 512, 1024, 2048],
    hop_sizes=[64, 128, 256, 512],
    win_lengths=[256, 512, 1024, 2048],
    window="hann_window",
)
_MRSTFT_LOSS.eval()

_TARGET_LUFS: float = -23.0


def _load_calibration() -> dict:
    """Load calibration.npz once and cache it. Reads path from defaults.CAL_PATH."""
    if not hasattr(_load_calibration, "_cache"):
        try:
            import mimic_synth.config as _defs
            cal_path = _defs.CAL_PATH
        except Exception:
            cal_path = _Path(__file__).parent / "calibration.npz"
        if cal_path.exists():
            with np.load(str(cal_path)) as f:
                _load_calibration._cache = {k: f[k] for k in f.files}
        else:
            _load_calibration._cache = {}
    return _load_calibration._cache


def _interp_ms(param_val: float, params_key: str, ms_key: str,
               fallback_fn) -> float:
    """Interpolate a param → ms mapping from the calibration table, or fall back."""
    cal = _load_calibration()
    if params_key in cal and ms_key in cal:
        ms = float(np.interp(float(param_val), cal[params_key], cal[ms_key]))
        return max(0.001, ms / 1000.0)
    return fallback_fn(param_val)


def amp_release_sec(param_val: float) -> float:
    """Amp Env Release param [0,1] → seconds.

    Uses calibration table when available (run calibrate_synth.py --amp-adsr).
    Falls back to a quadratic empirical estimate: 0.2 → ~0.20s, 1.0 → ~5s.
    """
    return _interp_ms(param_val, "amp_release_params", "amp_release_ms",
                      lambda p: max(0.01, 5.0 * float(p) ** 2))


def amp_attack_sec(param_val: float) -> float:
    """Amp Env Attack param [0,1] → seconds. Uses calibration table when available."""
    return _interp_ms(param_val, "amp_attack_params", "amp_attack_ms",
                      lambda p: max(0.001, 3.0 * float(p) ** 2))


def amp_decay_sec(param_val: float) -> float:
    """Amp Env Decay param [0,1] → seconds. Uses calibration table when available."""
    return _interp_ms(param_val, "amp_decay_params", "amp_decay_ms",
                      lambda p: max(0.001, 8.0 * float(p) ** 2))


def filter_attack_sec(param_val: float) -> float:
    """Filter Env Attack param [0,1] → seconds. Uses calibration table when available."""
    return _interp_ms(param_val, "filter_attack_params", "filter_attack_ms",
                      lambda p: max(0.001, 3.0 * float(p) ** 2))


def filter_decay_sec(param_val: float) -> float:
    """Filter Env Decay param [0,1] → seconds. Uses calibration table when available."""
    return _interp_ms(param_val, "filter_decay_params", "filter_decay_ms",
                      lambda p: max(0.001, 8.0 * float(p) ** 2))


def _lufs_normalize(audio: np.ndarray, sr: int) -> np.ndarray:
    """Normalize audio to -23 LUFS (ITU-R BS.1770) via pyloudnorm.

    Returns the input unchanged when pyloudnorm is unavailable, the signal
    is effectively silent (integrated loudness < -70 LUFS), or measurement
    fails (e.g. clip shorter than 400ms gating window).
    """
    try:
        import pyloudnorm as pyln
    except ImportError:
        return audio
    try:
        meter = pyln.Meter(int(sr))
        loudness = meter.integrated_loudness(audio.astype(np.float64))
        if not np.isfinite(loudness) or loudness < -70.0:
            return audio
        return pyln.normalize.loudness(
            audio.astype(np.float64), loudness, _TARGET_LUFS
        ).astype(np.float32)
    except Exception:
        return audio


# Composite scoring weights.
# EnCodec: high-level timbre color and resonance.
# MRSTFT: temporal spectral variation — filter sweeps, attack/release character.
# AP (aperiodicity): noise-vs-harmonic balance — "squaky", breathy, inharmonic content.
# When a term is unavailable (e.g. pyworld not installed), weights are renormalised.
ENCODEC_WEIGHT: float = 0.33  # roadmap §III.A revised weights
MRSTFT_WEIGHT:  float = 0.22
AP_WEIGHT:      float = 0.17
SP_WEIGHT:      float = 0.13  # per-frame spectral envelope (pyworld CheapTrick)
ENV_WEIGHT:     float = 0.15  # amplitude envelope correlation (attack/sustain/release shape)


_MRSTFT_N_TERMS = 2 * len([256, 512, 1024, 2048])  # SpectralConvergence + LogMagnitude per scale = 8


def _mrstft_dist(audio_a: np.ndarray, audio_b: np.ndarray) -> float:
    """auraloss MultiResolutionSTFTLoss normalized to approx [0, 2].

    Returns 1.0 (neutral) when either clip is shorter than the smallest
    FFT window (256 samples). Divides raw loss by number of terms (8) so
    that typical "good match" values land near 0.02–0.2, matching the
    cosine distance scale used by the EnCodec and AP terms.
    """
    if len(audio_a) < 256 or len(audio_b) < 256:
        return 1.0
    ta = torch.from_numpy(audio_a.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    tb = torch.from_numpy(audio_b.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    length = min(ta.shape[-1], tb.shape[-1])
    ta, tb = ta[..., :length], tb[..., :length]
    with torch.no_grad():
        loss = _MRSTFT_LOSS(ta, tb).item()
    return min(2.0, loss / _MRSTFT_N_TERMS)


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


def compute_sp_features(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """Per-frame spectral envelope via pyworld CheapTrick (roadmap §II.A, §III.A).

    Returns float32 array [n_frames, fft//2+1], or None when pyworld is
    unavailable, audio is too short (< 100ms), or analysis fails.
    """
    try:
        import pyworld as pw
    except ImportError:
        return None
    if len(audio) < int(sr * 0.10):
        return None
    try:
        audio64 = audio.astype(np.float64)
        f0, t = pw.dio(audio64, sr)
        f0 = pw.stonemask(audio64, f0, t, sr)
        sp = pw.cheaptrick(audio64, f0, t, sr)   # [n_frames, fft//2+1]
        if sp.shape[0] == 0:
            return None
        return sp.astype(np.float32)
    except Exception:
        return None


def _sp_dist(sp_a: np.ndarray, sp_b: np.ndarray) -> float:
    """Mean per-frame cosine distance between two SP matrices (roadmap §III.A).

    score_sp = mean(cosine_dist(SP_source[t], SP_render[t])).
    Uses the shorter frame count. Returns 1.0 (neutral) for empty inputs.
    """
    n = min(len(sp_a), len(sp_b))
    if n == 0:
        return 1.0
    a = torch.tensor(sp_a[:n], dtype=torch.float32)
    b = torch.tensor(sp_b[:n], dtype=torch.float32)
    norms_a = a.norm(dim=1, keepdim=True).clamp(min=1e-8)
    norms_b = b.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos_sim = ((a / norms_a) * (b / norms_b)).sum(dim=1)
    return float((1.0 - cos_sim).mean())


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2]. Returns 1.0 (neutral) for near-zero vectors."""
    ta = torch.tensor(a, dtype=torch.float32)
    tb = torch.tensor(b, dtype=torch.float32)
    if ta.norm() < 1e-6 or tb.norm() < 1e-6:
        return 1.0
    return float(1.0 - F.cosine_similarity(ta.unsqueeze(0), tb.unsqueeze(0)))


def compute_envelope(audio: np.ndarray, sr: int, hop_ms: float = 5.0) -> np.ndarray:
    """Amplitude envelope via 5ms RMS frames. Returns float32 array [n_frames]."""
    hop = max(1, int(hop_ms / 1000 * sr))
    win = hop * 2
    n_frames = max(1, (len(audio) - win) // hop + 1)
    env = np.array([
        float(np.sqrt(np.mean(audio[i * hop: i * hop + win] ** 2)))
        for i in range(n_frames)
    ], dtype=np.float32)
    peak = env.max()
    return env / (peak + 1e-8)


def _envelope_dist(audio_a: np.ndarray, audio_b: np.ndarray, sr: int) -> float:
    """1 − Pearson correlation of amplitude envelopes, in [0, 2].

    Rewards matching attack shape, sustain plateau, and release curve.
    Returns 1.0 (neutral) for very short clips or silent inputs.
    """
    if len(audio_a) < sr * 0.05 or len(audio_b) < sr * 0.05:
        return 1.0
    env_a = compute_envelope(audio_a, sr)
    env_b = compute_envelope(audio_b, sr)
    n = min(len(env_a), len(env_b))
    if n < 4:
        return 1.0
    # Resample the longer envelope to match the shorter
    if len(env_a) != len(env_b):
        env_b = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(env_b[:n]) if len(env_b) > n else len(env_b)),
            env_b[:n] if len(env_b) > n else env_b,
        ).astype(np.float32)
        env_a = env_a[:n]
    corr = float(np.corrcoef(env_a, env_b)[0, 1])
    if not np.isfinite(corr):
        return 1.0
    return float(np.clip(1.0 - corr, 0.0, 2.0))


def score_audio_composite(
    audio: np.ndarray,
    sr: int,
    target_emb_t: torch.Tensor,
    embedder,
    device: str,
    target_mrstft_audio: np.ndarray | None = None,
    target_ap: np.ndarray | None = None,
    target_sp: np.ndarray | None = None,
    target_env: np.ndarray | None = None,
) -> float:
    """Composite score: EnCodec + MRSTFT + AP + SP + Env, renormalised (roadmap §III.A).

    Active terms and their base weights:
        EnCodec  38%  — high-level timbral color and resonance
        MRSTFT   24%  — filter movement, temporal spectral variation (auraloss)
        AP       19%  — noise-vs-harmonic balance (squaky, breathy quality)
        SP       14%  — per-frame spectral envelope match (pyworld CheapTrick)
        Env      10%  — amplitude envelope correlation (attack/sustain/release shape)

    Inactive terms (None) are dropped and remaining weights are renormalised
    so the score always has a consistent scale ∈ [0, 2].
    Both candidate and target are LUFS-normalised to -23 before comparison.
    """
    audio = _lufs_normalize(audio, sr)
    enc_dist = _embed_dist(audio, sr, target_emb_t, embedder, device)
    terms: list[tuple[float, float]] = [(ENCODEC_WEIGHT, enc_dist)]

    if target_mrstft_audio is not None:
        terms.append((MRSTFT_WEIGHT, _mrstft_dist(audio, target_mrstft_audio)))

    if target_ap is not None:
        cand_ap = compute_ap_features(audio, sr)
        if cand_ap is not None:
            terms.append((AP_WEIGHT, _cosine_dist(target_ap, cand_ap)))

    if target_sp is not None:
        cand_sp = compute_sp_features(audio, sr)
        if cand_sp is not None:
            terms.append((SP_WEIGHT, _sp_dist(target_sp, cand_sp)))

    # Envelope distance: use target_env if provided, else fall back to
    # target_mrstft_audio (which is always the raw target audio).
    _env_ref = target_env if target_env is not None else target_mrstft_audio
    if _env_ref is not None:
        terms.append((ENV_WEIGHT, _envelope_dist(audio, _env_ref, sr)))

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
    without it, the plugin's internal defaults (e.g. saw wave on/off) may differ
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
    # Without this, the plugin falls back to an undefined internal default for
    # any parameter we don't explicitly set.
    for name, val in reset_values.items():
        if name in name_to_idx:
            plugin.set_parameter(name_to_idx[name], float(val))

    # Apply oscillator-config overrides (e.g. switch from saw to pulse).
    if extra_params:
        for name, val in extra_params.items():
            if name in name_to_idx:
                plugin.set_parameter(name_to_idx[name], float(val))

    # Per-region MIDI note-on/off. Note-off is pulled back by amp_release_sec
    # so the release tail *ends* at the intended boundary rather than starting
    # there. Without this the rendered note overshoots every region endpoint.
    _rel_param = float(df["p_Amp Env Release"].median()) if "p_Amp Env Release" in df.columns else 0.2
    _rel_sec   = amp_release_sec(_rel_param)
    for i, r in enumerate(note_regions):
        note_on = r["onset_sec"]
        if i < len(note_regions) - 1:
            raw_off = note_regions[i + 1]["onset_sec"]
        else:
            raw_off = r["offset_sec"]
        note_off = max(note_on + 0.001, raw_off - _rel_sec)
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


def _embed_dist(
    audio: np.ndarray,
    sr: int,
    target_emb_t: torch.Tensor,
    embedder,
    device: str,
) -> float:
    """Inner EnCodec cosine distance — caller must pre-normalise audio."""
    emb = embedder.encodec_embed(audio, sr, pool="mean")
    emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
    return (
        1.0 - F.cosine_similarity(emb_t.unsqueeze(0), target_emb_t.unsqueeze(0))
    ).item()


def score_audio(
    audio: np.ndarray,
    sr: int,
    target_emb_t: torch.Tensor,
    embedder,
    device: str,
) -> float:
    """Cosine distance between `audio`'s EnCodec embedding and `target_emb_t`.

    target_emb_t must already be on `device`. Returns a Python float in
    [0, 2] (cosine *distance*, not similarity). Audio is LUFS-normalised
    before embedding so level differences don't inflate the distance.
    """
    audio = _lufs_normalize(audio, sr)
    return _embed_dist(audio, sr, target_emb_t, embedder, device)


def render_region_and_score(
    params: dict[str, float],
    note: int,
    region_dur: float,
    profile_path: _Path,
    target_emb_t: torch.Tensor,
    embedder,
    device: str,
    extra_params: dict[str, float] | None = None,
    target_mrstft_audio: np.ndarray | None = None,
    target_ap: np.ndarray | None = None,
    target_sp: np.ndarray | None = None,
) -> float:
    """Render a region and return composite distance (EnCodec + MRSTFT + AP + SP)."""
    audio, sr = render_region(params, note, region_dur, profile_path,
                              extra_params=extra_params)
    return score_audio_composite(audio, sr, target_emb_t, embedder, device,
                                 target_mrstft_audio, target_ap, target_sp)


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
    target_mrstft_audio: np.ndarray | None = None,
    target_ap: np.ndarray | None = None,
    target_sp: np.ndarray | None = None,
    target_env: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Convenience: render `df` and return (score, audio)."""
    audio, sr = render_trajectory(
        df, note_regions, param_cols, profile_path, total_sec, extra_params
    )
    score = score_audio_composite(audio, sr, target_emb_t, embedder, device,
                                  target_mrstft_audio, target_ap, target_sp, target_env)
    return score, audio

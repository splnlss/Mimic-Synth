"""Analyze target audio and map features to synth parameter initializations.

This module answers the five design questions before CMA-ES begins so that
the optimizer starts near a sensible region rather than a random one.

Design questions (from build_instructions/07 Refine VST Loop.md):

  Q1 — Oscillator type: saw, square/pulse, noise, or combination?
       → Detected via harmonic structure. Sawtooth has strong odd+even
         harmonics; square/pulse has strong odd harmonics; noise has flat
         spectrum. The profile's continuous params (Osc Pulsewidth,
         Cross Modulation, Osc 2 Detune) are all tunable; wave TYPE is a
         discrete choice (saw vs pulse toggle) handled by OscConfig in
         vst_cmaes.py.

  Q2 — Amp ADSR: slow or fast attack? Long or short decay/release?
       → Derived from the amplitude envelope of the first note region.
         Attack = time from onset to first energy peak. Decay = time from
         peak to steady-state (sustain). These map directly to Amp Env
         Attack and Amp Env Decay parameters.

  Q3 — Filter and resonance: what cutoff, how much resonance?
       → Spectral centroid → filter cutoff. Spectral peakedness (sharpness
         of formant peaks) → resonance. A bright, harmonics-rich sound needs
         high cutoff; a narrow-band tonal sound benefits from resonance.

  Q4 — Mod / filter envelope ADSR: does the timbre evolve over time?
       → Spectral flux (how much the spectrum changes frame-to-frame) maps to
         Filter Env Amount. A static timbre → low env amount. A bright-to-dark
         or dark-to-bright timbre evolution → higher env amount. LFO Rate is
         estimated from cyclic spectral centroid oscillation (vibrato proxy).

  Q5 — Pitch shift: necessary? Does it follow tonal shifts?
       → Tracked by s06b note_regions (per-region MIDI notes). Within a
         region the surrogate already accounts for pitch via note/127 input
         and Osc 1 Pitch is pinned. CMA-ES does NOT change pitch tracking;
         this analysis only detects whether cross-modulation (FM) might help
         achieve unusual timbres (e.g. bell-like inharmonicity).

Research references
-------------------
* Yee-King (2011) "Automatic Programming of VST Sound Synthesizers Using Deep
  Networks and Other Techniques" — parametric initialization significantly
  outperforms random init; spectral features are the most useful predictors
  of filter parameters.
* Engel et al. (2020) "DDSP: Differentiable Digital Signal Processing" —
  spectral centroid and harmonic series analysis for instrument parameter
  estimation.
* Cartwright & Pardo (2015) "SynthAssist" — multi-objective scoring combining
  temporal (ADSR) and spectral (filter) objectives.
* Grey & Gordon (1978) "Perceptual effects of spectral modifications on musical
  timbres" — spectral centroid and envelope shape as primary timbre dimensions.
"""
from __future__ import annotations

import sys as _sys
from dataclasses import dataclass, field
from pathlib import Path as _Path
from typing import Optional

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import scipy.signal as signal


# ── Centroid trajectory (fallback: librosa; override: pyworld SP) ────────────

def _centroid_trajectory(audio: np.ndarray, sr: int, hop_ms: float = 10.0) -> np.ndarray | None:
    """Per-frame spectral centroid in Hz using librosa.

    Returns array of shape [n_frames] or None if librosa is unavailable.
    Used as a fallback when pyworld is absent; pyworld SP centroids are
    more accurate and override this when available.
    """
    try:
        import librosa
    except ImportError:
        return None
    hop = max(1, int(hop_ms / 1000 * sr))
    n_fft = min(2048, max(32, len(audio) // 8 * 2))
    if len(audio) < n_fft:
        return None
    try:
        ct = librosa.feature.spectral_centroid(
            y=audio.astype(np.float32), sr=sr, n_fft=n_fft, hop_length=hop
        )[0]
        return ct.astype(np.float64)
    except Exception:
        return None


def _filter_adsr_from_centroid(
    ct_hz: np.ndarray,
    frame_rate: float,          # frames per second
) -> tuple[float, float, float, float]:
    """Derive Filter Env Attack/Decay/Sustain/Amount from centroid Hz trajectory.

    Reads the shape of the spectral brightness curve:
      - Bright attack then darkening  → short attack, medium decay, high Env Amount
      - Constant bright               → low Env Amount, high Sustain
      - Slow brightening              → long attack, high Env Amount

    Returns (filter_env_attack_est, filter_env_decay_est,
             filter_env_sustain_est, filter_env_amount_est), all in [0, 1].
    """
    if len(ct_hz) < 4:
        return 0.0, 0.3, 0.8, 0.1

    ct = ct_hz / (ct_hz.max() + 1e-8)  # normalise to [0, 1]
    peak_idx = int(np.argmax(ct))
    n_ct = len(ct)

    # Attack: frames from start to centroid peak
    attack_ms = (peak_idx / frame_rate) * 1000.0
    filter_env_attack_est = _ms_to_filter_env_param(attack_ms)

    # Decay: frames from peak to first crossing of steady-state level
    filter_env_decay_est = 0.3
    if peak_idx < n_ct - 1:
        tail = ct[peak_idx:]
        steady = float(np.median(ct))
        decay_frames = int(np.argmin(np.abs(tail - steady)))
        decay_ms = (max(decay_frames, 1) / frame_rate) * 1000.0
        filter_env_decay_est = _ms_to_filter_decay_param(decay_ms)

    # Sustain: median brightness as fraction of peak
    filter_env_sustain_est = float(np.clip(np.median(ct) / max(float(ct.max()), 1e-6), 0.0, 1.0))

    # Env Amount: how much the centroid varies — high variance = more filter movement
    if ct.mean() > 1e-8:
        filter_env_amount_est = float(np.clip(ct.std() / ct.mean() * 0.5, 0.0, 0.5))
    else:
        filter_env_amount_est = 0.1

    return filter_env_attack_est, filter_env_decay_est, filter_env_sustain_est, filter_env_amount_est


# ── Feature dataclass ────────────────────────────────────────────────────────

@dataclass
class TargetAnalysis:
    """All extracted features and the derived parameter suggestions.

    Attributes that are None indicate the analysis could not confidently
    estimate the value (e.g. no detectable pitch → fundamental_hz is None).
    """

    # Q1 — Oscillator character
    spectral_centroid_norm: float     # [0,1] relative to Nyquist; high = bright
    harmonic_ratio: float             # [0,1] fraction of energy in harmonic partials
    inharmonicity: float              # [0,1] degree of inharmonic content
    spectral_flatness: float          # [0,1] flat = noise-like, peaky = tonal
    fundamental_hz: Optional[float]   # detected F0 or None

    # Q2 — Amp envelope
    attack_ms: float                  # time from onset to peak (ms)
    decay_ms: float                   # time from peak to 90% of steady-state (ms)
    sustain_level: float              # [0,1] RMS at steady-state / peak RMS
    release_ms: float                 # estimated silence onset after note-off

    # Q3 — Filter
    filter_cutoff_est: float          # [0,1] cutoff param suggestion
    filter_resonance_est: float       # [0,1] resonance param suggestion
    filter_mode_est: float            # [0,1] 0=LP, 0.33=HP, 0.67=BP, 1=notch

    # Q4 — Modulation
    spectral_flux_norm: float         # [0,1] normalised; high = evolving timbre
    filter_env_amount_est: float      # [0,1] Filter Env Amount suggestion
    lfo_rate_est: float               # [0,1] LFO 1 Rate suggestion
    lfo_to_filter_est: float          # [0,1] LFO 1 to Filter Cutoff

    # Q5 — Pitch / inharmonic character
    pitch_is_stable: bool             # True if F0 is approximately constant
    cross_mod_est: float              # [0,1] Cross Modulation suggestion
    osc2_detune_est: float            # [0,1] Osc 2 Detune suggestion (0.5=unison)

    # Diagnostics
    notes: list[str] = field(default_factory=list)

    # CREPE + DDSP-style harmonic analysis (populated when torchcrepe available)
    noise_volume_est:         float = 0.0    # from d4c AP mean → Noise Volume [0, 0.40]
    filter_env_attack_est:    float = 0.0    # centroid rise time → Filter Env Attack
    filter_env_decay_est:     float = 0.3    # centroid fall time → Filter Env Decay
    filter_env_sustain_est:   float = 0.8    # steady-state brightness → Filter Env Sustain
    harmonic_richness:        float = 0.5    # overtone strength relative to fundamental → Osc 2 Volume
    harmonic_slope_resonance: float = 0.2    # harmonic decay slope → Filter Resonance (replaces flatness)
    f0_source:                str   = "hps"  # "crepe" | "dio" | "hps" — diagnostic
    transient_min_ms:         float = 1000.0  # shortest onset interval (ms); 1000.0 = no short transients detected


# ── Core analysis ────────────────────────────────────────────────────────────

def analyze_target(
    audio: np.ndarray,
    sr: int,
    note_regions: list[dict] | None = None,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> TargetAnalysis:
    """Extract timbral, envelope, and modulation features from `audio`.

    Args:
        audio: mono float32 signal.
        sr: sample rate.
        note_regions: s06b note region dicts (onset_sec, offset_sec, midi_note).
            If provided, analysis is restricted to voiced regions.
        frame_ms: analysis frame length in milliseconds.
        hop_ms: analysis hop in milliseconds.

    Returns:
        TargetAnalysis with all features and parameter suggestions.
    """
    notes: list[str] = []
    audio = audio.astype(np.float64)
    win = int(frame_ms / 1000 * sr)
    hop = int(hop_ms / 1000 * sr)

    # Determine voiced segments from note_regions (if provided)
    if note_regions:
        voiced_samples: list[np.ndarray] = []
        for r in note_regions:
            s = int(r["onset_sec"] * sr)
            e = int(r["offset_sec"] * sr)
            if e > s:
                voiced_samples.append(audio[s:e])
        voiced = np.concatenate(voiced_samples) if voiced_samples else audio
    else:
        voiced = audio

    # ── Q1: Oscillator character ─────────────────────────────────────────────

    # Spectral centroid (mean-frequency centre of gravity, normalised to [0,1])
    freqs, psd = signal.welch(voiced, sr, nperseg=min(len(voiced), 4096))
    total_power = psd.sum()
    if total_power > 0:
        centroid_hz = float((freqs * psd).sum() / total_power)
    else:
        centroid_hz = float(freqs[len(freqs) // 4])
    spectral_centroid_norm = float(np.clip(centroid_hz / (sr / 2), 0.0, 1.0))

    # Spectral flatness (geometric-mean / arithmetic-mean of PSD)
    # 0 = perfectly tonal spike, 1 = flat (white noise)
    eps = 1e-12
    psd_nz = psd[psd > eps]
    if len(psd_nz) >= 2:
        spectral_flatness = float(
            np.exp(np.mean(np.log(psd_nz))) / np.mean(psd_nz)
        )
        spectral_flatness = float(np.clip(spectral_flatness, 0.0, 1.0))
    else:
        spectral_flatness = 0.5
        notes.append("spectral_flatness: could not compute (very short audio)")

    # Fundamental frequency via HPS (harmonic product spectrum)
    fundamental_hz = _detect_fundamental_hps(voiced, sr)

    # Harmonic ratio: fraction of total power in harmonic partial bands
    # (only meaningful when a fundamental is detected)
    if fundamental_hz and fundamental_hz > 0:
        harmonic_ratio = _harmonic_ratio(psd, freqs, fundamental_hz, n_harmonics=8)
        # Inharmonicity: how spread-out the energy is between harmonics
        inharmonicity = float(np.clip(1.0 - harmonic_ratio, 0.0, 1.0))
        pitch_is_stable = _check_pitch_stability(voiced, sr, fundamental_hz)
    else:
        harmonic_ratio = 0.3
        inharmonicity = 0.7
        pitch_is_stable = False
        notes.append("fundamental not detected; treating as noise-like")

    # ── Q2: Amplitude envelope ───────────────────────────────────────────────

    attack_ms, decay_ms, sustain_level, release_ms = _estimate_adsr(voiced, sr)
    transient_min_ms = _detect_transient_min_ms(voiced, sr)

    # ── Q3: Filter parameters ────────────────────────────────────────────────

    # Filter cutoff: map spectral centroid → cutoff parameter.
    # Uses measured calibration table from calibrate_synth.py when available;
    # falls back to heuristic linear approximation otherwise.
    filter_cutoff_est = float(_centroid_hz_to_cutoff(centroid_hz, sr))

    # Resonance: sharp spectral peaks suggest the target has a prominent
    # formant / resonance; match with synth resonance.
    # Proxy: inverse spectral flatness (tonal peaks → higher resonance est.)
    filter_resonance_est = float(np.clip(0.05 + (1.0 - spectral_flatness) * 0.35, 0.05, 0.5))

    # Filter mode: default LP (0.0); if energy is concentrated high
    # (high centroid + low low-freq power), consider bandpass (0.5).
    low_freq_power = float(psd[freqs < 500].sum())
    high_freq_power = float(psd[freqs > 3000].sum())
    if total_power > 0 and low_freq_power / total_power < 0.05 and spectral_centroid_norm > 0.4:
        filter_mode_est = 0.3   # lean toward bandpass
        notes.append("filter_mode: leaning bandpass (low bass content, high centroid)")
    else:
        filter_mode_est = 0.0   # lowpass is almost always correct

    # ── Q4: Modulation / filter envelope ────────────────────────────────────

    spectral_flux_norm, lfo_rate_est = _estimate_spectral_dynamics(voiced, sr, win, hop)

    # Filter Env Amount: when timbre evolves significantly, open the filter
    # envelope to let the synth replicate that brightness sweep.
    # Low flux → static timbre → small env amount.
    # High flux → sweeping timbre → higher env amount.
    filter_env_amount_est = float(np.clip(spectral_flux_norm * 0.6, 0.0, 0.5))

    # LFO to Filter: if the spectral centroid oscillates at a detectable
    # sub-audio rate, some LFO modulation on the filter could help.
    lfo_to_filter_est = float(np.clip(spectral_flux_norm * 0.3, 0.0, 0.25))

    # ── Q5: Pitch / cross-modulation ────────────────────────────────────────

    cross_mod_est = float(np.clip(inharmonicity * 0.25, 0.0, 0.25))
    osc2_detune_est = 0.5 + float(np.clip(inharmonicity * 0.1, -0.1, 0.1))
    osc2_detune_est = float(np.clip(osc2_detune_est, 0.4, 0.6))

    # ── Filter ADSR from spectral centroid trajectory ─────────────────────────
    # Compute unconditionally via librosa; pyworld overrides when available.
    noise_volume_est         = 0.0
    filter_env_attack_est    = 0.0
    filter_env_decay_est     = 0.3
    filter_env_sustain_est   = 0.8
    harmonic_richness        = 0.5
    harmonic_slope_resonance = filter_resonance_est   # start from Welch-based estimate
    f0_source                = "hps"

    _CENTROID_HOP_MS = 10.0
    ct_librosa = _centroid_trajectory(voiced.astype(np.float32), sr, hop_ms=_CENTROID_HOP_MS)
    if ct_librosa is not None and len(ct_librosa) >= 4:
        _frame_rate = 1000.0 / _CENTROID_HOP_MS
        (filter_env_attack_est, filter_env_decay_est,
         filter_env_sustain_est, filter_env_amount_est) = _filter_adsr_from_centroid(
            ct_librosa, _frame_rate
        )
        # Recalculate filter_env_amount_est from trajectory (better than flux)
        if ct_librosa.mean() > 1e-8:
            filter_env_amount_est = float(np.clip(
                ct_librosa.std() / ct_librosa.mean() * 0.5, 0.0, 0.5
            ))
        notes.append("filter_env_adsr: from librosa centroid trajectory")

    # ── CREPE + pyworld harmonic analysis (overrides librosa centroid) ────────
    crepe_result = _crepe_f0(audio.astype(np.float32), sr)
    if crepe_result is not None:
        f0_ext, t_ext = crepe_result
        f0_source = "crepe"
        notes.append(f"f0_source: CREPE ({int((f0_ext > 0).sum())} voiced frames)")
    else:
        f0_ext, t_ext = None, None

    world = _pyworld_analysis(audio.astype(np.float32), sr,
                              f0_external=f0_ext, t_external=t_ext)
    if world is None and crepe_result is None:
        f0_source = "hps"

    if world is not None:
        ct = world["centroid_norm"]   # per voiced frame, normalised [0,1]

        # Override centroid and cutoff with higher-accuracy SP-based estimate
        spectral_centroid_norm = float(np.median(ct))
        filter_cutoff_est = float(np.clip(0.3 + spectral_centroid_norm * 0.6, 0.3, 0.90))

        # Noise Volume from d4c aperiodicity
        noise_volume_est = float(np.clip(world["noise_level"] * 0.8, 0.0, 0.40))

        # Override filter ADSR with higher-accuracy pyworld SP trajectory
        # (pyworld centroids are per voiced frame at 5ms hop, vs librosa at 10ms)
        if len(ct) >= 4:
            _pw_frame_rate = 1000.0 / 5.0   # pyworld uses 5ms hop
            (filter_env_attack_est, filter_env_decay_est,
             filter_env_sustain_est, filter_env_amount_est) = _filter_adsr_from_centroid(
                ct * (ct.max() + 1e-8),    # denormalise to Hz-ish for consistency
                _pw_frame_rate,
            )
            notes.append("filter_env_adsr: overridden with pyworld SP centroid (5ms)")

        # Harmonic richness: low AP = strong harmonics
        harmonic_richness = float(np.clip(1.0 - world["noise_level"] * 2.0, 0.0, 1.0))

        # Harmonic slope resonance from per-harmonic amplitudes
        f0_world = world["f0"]
        harm_amps = _harmonic_amplitudes(
            audio.astype(np.float32), sr, f0_world, world["t"]
        )
        if harm_amps is not None:
            voiced_harms = harm_amps[f0_world > 30.0]
            if len(voiced_harms) > 0:
                h1   = voiced_harms[:, 0].mean()
                h2_4 = voiced_harms[:, 1:4].mean()
                harmonic_richness = float(np.clip(h2_4 / max(h1, 1e-8), 0.0, 1.0))

                mean_amps = voiced_harms.mean(axis=0)
                k_vals    = np.arange(1, mean_amps.shape[0] + 1)
                safe_amps = np.where(mean_amps > 0, mean_amps, 1e-8)
                slope     = np.polyfit(k_vals, np.log(safe_amps), 1)[0]
                # Steeper negative slope → darker/filtered → lower resonance
                harmonic_slope_resonance = float(np.clip(-slope * 2.0, 0.05, 0.30))

    return TargetAnalysis(
        spectral_centroid_norm=spectral_centroid_norm,
        harmonic_ratio=harmonic_ratio,
        inharmonicity=inharmonicity,
        spectral_flatness=spectral_flatness,
        fundamental_hz=fundamental_hz,
        attack_ms=attack_ms,
        decay_ms=decay_ms,
        sustain_level=sustain_level,
        release_ms=release_ms,
        filter_cutoff_est=filter_cutoff_est,
        filter_resonance_est=filter_resonance_est,
        filter_mode_est=filter_mode_est,
        spectral_flux_norm=spectral_flux_norm,
        filter_env_amount_est=filter_env_amount_est,
        lfo_rate_est=lfo_rate_est,
        lfo_to_filter_est=lfo_to_filter_est,
        pitch_is_stable=pitch_is_stable,
        cross_mod_est=cross_mod_est,
        osc2_detune_est=osc2_detune_est,
        notes=notes,
        noise_volume_est=noise_volume_est,
        filter_env_attack_est=filter_env_attack_est,
        filter_env_decay_est=filter_env_decay_est,
        filter_env_sustain_est=filter_env_sustain_est,
        harmonic_richness=harmonic_richness,
        harmonic_slope_resonance=harmonic_slope_resonance,
        f0_source=f0_source,
        transient_min_ms=transient_min_ms,
    )


def suggest_x0(
    analysis: TargetAnalysis,
    param_cols: list[str],
    pinned_cols: set[str],
    current_x0: np.ndarray | None = None,
) -> np.ndarray:
    """Map analysis features to a concrete synth parameter vector.

    Where `current_x0` is provided, the analysis-derived value is blended
    50/50 with the existing value so we don't throw away information from
    the surrogate inversion. This is especially useful for params where the
    surrogate is reliable (e.g. oscillator volumes) vs params where the
    surrogate is biased (e.g. filter cutoff).

    Args:
        analysis: output of analyze_target().
        param_cols: ordered list of `p_<name>` param columns.
        pinned_cols: subset of param_cols whose values must not change.
        current_x0: existing init vector (e.g. from hill-climb output).
            If None, uses analysis values directly with surrogate defaults.

    Returns:
        float32 array, shape (len(param_cols),), all values in [0, 1].
    """
    # Analysis-to-param mapping (param name without `p_` prefix)
    # Each value is the analysis-derived suggestion in [0, 1].
    suggestions: dict[str, float] = {
        # Q3 — filter
        "Filter Cutoff":      analysis.filter_cutoff_est,
        # Use CREPE/DDSP harmonic slope for resonance when available (more physical)
        "Filter Resonance":   analysis.harmonic_slope_resonance,
        "Filter Mode":        analysis.filter_mode_est,
        "Filter Env Amount":  analysis.filter_env_amount_est,
        # Q2 — amp envelope (all four params now calibration-backed)
        "Amp Env Attack":   _attack_ms_to_param(analysis.attack_ms),
        "Amp Env Decay":    _decay_ms_to_param(analysis.decay_ms),
        "Amp Env Sustain":  float(np.clip(analysis.sustain_level, 0.0, 1.0)),
        # Release: provides a good CMA-ES starting point even while pinned in
        # the surrogate stage. If Amp Env Release is removed from pinned_cols
        # it will take effect directly.
        "Amp Env Release":  _release_ms_to_param(analysis.release_ms),
        # Q4 — LFO
        "LFO 1 Rate":         analysis.lfo_rate_est,
        "LFO 1 to Filter Cutoff": analysis.lfo_to_filter_est,
        # Q1/Q5 — oscillator
        "Osc 2 Detune":       analysis.osc2_detune_est,
        "Cross Modulation":   analysis.cross_mod_est,
        "Osc Pulsewidth":     0.5,     # neutral; osc config handles wave type
        "Osc 1 Volume":       0.7,     # safe default; osc 1 is primary
        # CREPE/DDSP-derived extra param estimates (were at reset defaults before)
        "Noise Volume":           analysis.noise_volume_est,
        "Ring Mod Volume":        float(np.clip(analysis.inharmonicity * 0.15, 0.0, 0.20)),
        "Filter Env Attack":      analysis.filter_env_attack_est,
        "Filter Env Decay":       analysis.filter_env_decay_est,
        "Filter Env Sustain":     analysis.filter_env_sustain_est,
        "Osc 2 Volume":           float(np.clip(analysis.harmonic_richness * 0.4, 0.0, 0.50)),
    }

    if current_x0 is not None:
        x0 = current_x0.copy().astype(np.float64)
    else:
        x0 = np.full(len(param_cols), 0.5, dtype=np.float64)

    for i, col in enumerate(param_cols):
        if col in pinned_cols:
            continue
        p_name = col.removeprefix("p_")
        if p_name in suggestions:
            sug = suggestions[p_name]
            if current_x0 is not None:
                # Blend: 50% analysis, 50% existing.  The surrogate is
                # reasonably good; don't override it aggressively.
                x0[i] = 0.5 * sug + 0.5 * float(x0[i])
            else:
                x0[i] = sug

    return np.clip(x0, 0.0, 1.0).astype(np.float32)


# ── Private helpers ──────────────────────────────────────────────────────────

def _crepe_f0(
    audio: np.ndarray,
    sr: int,
    hop_ms: float = 5.0,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    periodicity_threshold: float = 0.21,
) -> tuple[np.ndarray, np.ndarray] | None:
    """CREPE neural pitch detection. Returns (f0_hz, t_sec) at hop_ms intervals.

    Much more accurate than pyworld DIO for non-speech audio (bird calls,
    instruments). Requires audio resampled to 16kHz internally (pitfall 2).

    Returns None when torchcrepe unavailable, audio < 100ms, or inference fails.
    Frames below periodicity_threshold are set to F0=0 (unvoiced, pitfall 3).
    """
    try:
        import torchcrepe
        import torchaudio
        import torch
    except ImportError:
        return None
    if len(audio) < int(sr * 0.10):
        return None
    try:
        audio_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # [1, T]
        audio_16k = torchaudio.functional.resample(audio_t, sr, 16000)
        hop_samples_16k = max(1, int(16000 * hop_ms / 1000.0))  # 5ms → 80 samples

        freq, periodicity = torchcrepe.predict(
            audio_16k, 16000,
            hop_length=hop_samples_16k,
            fmin=fmin, fmax=fmax,
            model="tiny",           # 17MB; fast; cached after first download
            return_periodicity=True,
            device="cpu",           # never call inside scoring loop — only here
            batch_size=512,
        )
        freq = freq.squeeze(0).numpy().astype(np.float64)      # [n_frames]
        periodicity = periodicity.squeeze(0).numpy()
        freq[periodicity < periodicity_threshold] = 0.0         # unvoiced frames
        t = np.arange(len(freq)) * (hop_ms / 1000.0)
        return freq, t
    except Exception:
        return None


def _pyworld_analysis(
    audio: np.ndarray,
    sr: int,
    f0_external: np.ndarray | None = None,
    t_external: np.ndarray | None = None,
) -> dict | None:
    """WORLD vocoder decomposition: SP (spectral envelope) + AP (aperiodicity).

    If f0_external/t_external are provided (e.g. from CREPE), uses them instead
    of pw.dio+stonemask — this is the key CREPE integration point.

    Returns dict with keys: f0, sp, ap, t, voiced, centroid_norm, noise_level.
    Returns None when pyworld unavailable, audio too short, or analysis fails.
    """
    try:
        import pyworld as pw
    except ImportError:
        return None
    if len(audio) < int(sr * 0.10):
        return None
    try:
        audio64 = audio.astype(np.float64)

        if f0_external is not None and t_external is not None:
            f0, t = f0_external, t_external
        else:
            f0, t = pw.dio(audio64, sr)
            f0 = pw.stonemask(audio64, f0, t, sr)

        sp = pw.cheaptrick(audio64, f0, t, sr)   # [n_frames, fft//2+1]
        ap = pw.d4c(audio64, f0, t, sr)           # [n_frames, fft//2+1]

        voiced = f0 > 30.0
        if not voiced.any():
            return None

        freqs = np.linspace(0.0, sr / 2.0, sp.shape[1])
        sp_v = sp[voiced]
        sp_power = sp_v.sum(axis=1)
        safe = np.where(sp_power > 0, sp_power, 1.0)
        centroid_hz = (sp_v * freqs[np.newaxis, :]).sum(axis=1) / safe
        centroid_norm = np.clip(centroid_hz / (sr / 2.0), 0.0, 1.0)  # [n_voiced]

        noise_level = float(ap[voiced].mean())   # AP: 0=harmonic, 1=noise

        return {
            "f0": f0, "sp": sp, "ap": ap, "t": t,
            "voiced": voiced,
            "centroid_norm": centroid_norm,
            "noise_level": noise_level,
        }
    except Exception:
        return None


def _harmonic_amplitudes(
    audio: np.ndarray,
    sr: int,
    f0_hz: np.ndarray,
    t_pw: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 240,      # 5ms at 48kHz
    n_harmonics: int = 16,
    max_harmonic_hz: float = 8000.0,
) -> np.ndarray | None:
    """Per-harmonic amplitude envelopes A_k(t) — the DDSP harmonic synthesizer input.

    For each voiced frame t and harmonic k, reads the STFT magnitude at k*F0(t)
    with sub-bin interpolation (pitfall 7). Caps harmonics above max_harmonic_hz
    where spectral smoothing dominates real partials (pitfall 8).

    Returns float32 array [n_pw_frames, n_harmonics] or None on failure.
    Unvoiced frames have all-zero amplitudes.
    """
    try:
        voiced = f0_hz > 30.0
        if not voiced.any():
            return None

        _, t_stft, Zxx = signal.stft(
            audio.astype(np.float32), fs=sr,
            nperseg=n_fft, noverlap=n_fft - hop_length,
            window='hann',
        )
        mag = np.abs(Zxx)   # [n_fft//2+1, n_stft_frames]

        result = np.zeros((len(f0_hz), n_harmonics), dtype=np.float32)
        freq_per_bin = sr / n_fft

        for i in range(len(f0_hz)):
            f0 = f0_hz[i]
            if f0 <= 0 or not voiced[i]:
                continue
            stft_idx = int(np.argmin(np.abs(t_stft - t_pw[i])))
            stft_idx = min(stft_idx, mag.shape[1] - 1)
            spec = mag[:, stft_idx]

            for k in range(1, n_harmonics + 1):
                harm_hz = k * f0
                if harm_hz > max_harmonic_hz or harm_hz >= sr / 2.0:
                    break
                bin_f = harm_hz / freq_per_bin
                bin_lo = int(bin_f)
                bin_hi = min(bin_lo + 1, spec.shape[0] - 1)
                alpha = bin_f - bin_lo
                result[i, k - 1] = float((1 - alpha) * spec[bin_lo] + alpha * spec[bin_hi])

        return result
    except Exception:
        return None


def _ms_to_filter_env_param(ms: float) -> float:
    """Map envelope time (ms) to Filter Env Attack param [0,1].
    Uses calibration table when available."""
    return _ms_to_param(ms, "filter_attack_params", "filter_attack_ms",
                        fallback_min_ms=1.0, fallback_max_ms=5000.0)


def _ms_to_filter_decay_param(ms: float) -> float:
    """Map envelope time (ms) to Filter Env Decay param [0,1].
    Uses calibration table when available."""
    return _ms_to_param(ms, "filter_decay_params", "filter_decay_ms",
                        fallback_min_ms=1.0, fallback_max_ms=5000.0)


_FILTER_CALIB_TA:  dict | None = None   # module-level cache
_ADSR_CALIB_TA:   dict | None = None   # amp + filter envelope calibration cache


def _load_adsr_calibration() -> dict | None:
    """Load amp/filter ADSR calibration from defaults.CAL_PATH. Cached."""
    global _ADSR_CALIB_TA
    if _ADSR_CALIB_TA is None:
        try:
            import defaults as _defs
            cal_path = _defs.CAL_PATH
        except Exception:
            cal_path = _Path(__file__).parent.parent / "s01_project-profile" / "calibration.npz"
        if cal_path.exists():
            d = np.load(str(cal_path))
            _ADSR_CALIB_TA = {k: d[k].astype(np.float64) for k in d.files}
        else:
            _ADSR_CALIB_TA = {}
    return _ADSR_CALIB_TA if _ADSR_CALIB_TA else None


def _ms_to_param(ms: float, params_key: str, ms_key: str,
                 fallback_min_ms: float = 1.0, fallback_max_ms: float = 10000.0) -> float:
    """Interpolate ms → param [0,1] from calibration table, or log10 fallback."""
    cal = _load_adsr_calibration()
    if cal and params_key in cal and ms_key in cal:
        return float(np.clip(np.interp(ms, cal[ms_key], cal[params_key]), 0.0, 1.0))
    # Fallback: log-scale spanning fallback_min_ms → fallback_max_ms
    ms = max(ms, fallback_min_ms)
    span = np.log10(fallback_max_ms) - np.log10(fallback_min_ms)
    return float(np.clip((np.log10(ms) - np.log10(fallback_min_ms)) / span, 0.0, 1.0))


def _centroid_hz_to_cutoff(centroid_hz: float, sr: int) -> float:
    """Map spectral centroid (Hz) → Filter Cutoff [0, 1].

    Uses measured calibration from calibrate_synth.py when available;
    falls back to heuristic linear approximation otherwise.
    """
    global _FILTER_CALIB_TA
    if _FILTER_CALIB_TA is None:
        cal = _load_adsr_calibration()
        if cal and "filter_cutoff_centroids_hz" in cal:
            _FILTER_CALIB_TA = {
                "hz":   cal["filter_cutoff_centroids_hz"],
                "vals": cal["filter_cutoff_values"],
            }
    if _FILTER_CALIB_TA is not None:
        return float(np.clip(
            np.interp(centroid_hz, _FILTER_CALIB_TA["hz"], _FILTER_CALIB_TA["vals"]),
            0.0, 1.0,
        ))
    centroid_norm = float(np.clip(centroid_hz / (sr / 2), 0.0, 1.0))
    return float(np.clip(0.3 + centroid_norm * 0.6, 0.3, 0.95))


def _detect_fundamental_hps(audio: np.ndarray, sr: int) -> Optional[float]:
    """Harmonic Product Spectrum F0 estimate with sub-harmonic disambiguation.

    HPS works by accumulating spectral magnitude at integer multiples of a
    candidate F0; the candidate with the highest product is returned. Pure
    sines (no overtones) are inherently ambiguous for HPS — the method is
    designed for harmonic-rich signals. We mitigate octave errors by:
      1. Running HPS on the raw spectrum to get a candidate.
      2. Also testing F0/2 (one octave down) using the harmonic fit score.
      3. Returning the candidate with the better harmonic fit.
    """
    n_fft = min(len(audio), 4096)
    if n_fft < 128:
        return None
    spec = np.abs(np.fft.rfft(audio[:n_fft] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    hps = spec.copy()
    for k in range(2, 6):
        decimated = spec[::k]
        hps[:len(decimated)] *= decimated
        hps[len(decimated):] = 0.0

    valid = (freqs >= 50) & (freqs <= 5000)
    if valid.sum() == 0:
        return None

    candidate = float(freqs[valid][np.argmax(hps[valid])])
    if candidate <= 0:
        return None

    # Sub-harmonic check: if candidate/2 is also in the valid range and has
    # a better harmonic fit score, prefer the lower octave.
    sub = candidate / 2.0
    if sub >= 50.0:
        score_candidate = _harmonic_ratio(spec / (spec.max() + 1e-12), freqs, candidate)
        score_sub = _harmonic_ratio(spec / (spec.max() + 1e-12), freqs, sub)
        if score_sub > score_candidate + 0.1:
            candidate = sub

    return candidate


def _harmonic_ratio(
    psd: np.ndarray,
    freqs: np.ndarray,
    f0: float,
    n_harmonics: int = 8,
    band_hz: float = 30.0,
) -> float:
    """Fraction of spectral power contained in harmonic partial bands."""
    harmonic_power = 0.0
    total = float(psd.sum()) + 1e-12
    for k in range(1, n_harmonics + 1):
        center = k * f0
        mask = np.abs(freqs - center) < band_hz
        harmonic_power += float(psd[mask].sum())
    return float(np.clip(harmonic_power / total, 0.0, 1.0))


def _check_pitch_stability(
    audio: np.ndarray,
    sr: int,
    f0: float,
    n_frames: int = 8,
) -> bool:
    """True if F0 stays within ±2 semitones across short analysis frames."""
    frame_n = len(audio) // n_frames
    if frame_n < 256:
        return True
    pitches = []
    for i in range(n_frames):
        seg = audio[i * frame_n:(i + 1) * frame_n]
        p = _detect_fundamental_hps(seg, sr)
        if p:
            pitches.append(p)
    if len(pitches) < 2:
        return True
    ratios = np.array(pitches) / f0
    semitone_deviations = np.abs(12.0 * np.log2(np.clip(ratios, 0.01, 100)))
    return bool(semitone_deviations.max() < 2.0)


def _estimate_adsr(audio: np.ndarray, sr: int) -> tuple[float, float, float, float]:
    """Estimate attack, decay, sustain level, and release in milliseconds."""
    rms_hop = max(1, int(0.005 * sr))   # 5ms hop
    rms_win = max(1, int(0.020 * sr))   # 20ms window
    n_frames = max(1, (len(audio) - rms_win) // rms_hop + 1)
    rms = np.array([
        float(np.sqrt(np.mean(audio[i * rms_hop: i * rms_hop + rms_win] ** 2)))
        for i in range(n_frames)
    ])

    if len(rms) == 0 or rms.max() < 1e-6:
        return 10.0, 100.0, 0.7, 200.0

    peak_idx = int(np.argmax(rms))
    peak_rms = rms[peak_idx]

    # Attack: frames from first above-threshold frame to peak
    threshold = 0.05 * peak_rms
    onset_idx = int(np.argmax(rms > threshold))
    attack_ms = float(max(0, (peak_idx - onset_idx) * rms_hop / sr * 1000))

    # Sustain level: median RMS in the middle third of the voiced region
    mid_start = n_frames // 3
    mid_end = 2 * n_frames // 3
    if mid_end > mid_start:
        sustain_rms = float(np.median(rms[mid_start:mid_end]))
        sustain_level = float(np.clip(sustain_rms / peak_rms, 0.0, 1.0))
    else:
        sustain_level = 0.5

    # Decay: frames from peak to where RMS first reaches sustain_level
    decay_idx = peak_idx
    target_sustain = sustain_level * peak_rms
    for j in range(peak_idx, n_frames):
        if rms[j] <= target_sustain:
            decay_idx = j
            break
    decay_ms = float(max(0, (decay_idx - peak_idx) * rms_hop / sr * 1000))

    # Release: frames from where RMS drops below threshold to end
    below_thresh = rms < threshold
    release_start = n_frames - 1
    for j in range(n_frames - 1, peak_idx, -1):
        if not below_thresh[j]:
            release_start = j
            break
    release_ms = float(max(0, (n_frames - release_start) * rms_hop / sr * 1000))

    return attack_ms, decay_ms, sustain_level, release_ms


def _estimate_spectral_dynamics(
    audio: np.ndarray,
    sr: int,
    win: int,
    hop: int,
) -> tuple[float, float]:
    """Return (spectral_flux_norm, lfo_rate_est).

    spectral_flux_norm: how much the spectrum changes frame-to-frame, in [0,1].
    lfo_rate_est: LFO Rate param suggestion (0=slow, 1=fast).
    """
    n_frames = max(1, (len(audio) - win) // hop + 1)
    if n_frames < 3:
        return 0.1, 0.2

    centroids = []
    for i in range(n_frames):
        seg = audio[i * hop: i * hop + win]
        freqs, psd = signal.welch(seg, sr, nperseg=min(len(seg), win))
        total = psd.sum()
        if total > 1e-10:
            centroids.append(float((freqs * psd).sum() / total))
        elif centroids:
            centroids.append(centroids[-1])
        else:
            centroids.append(sr / 4.0)

    centroids = np.array(centroids)
    # Spectral flux: normalised std of centroid changes
    diffs = np.abs(np.diff(centroids))
    mean_c = centroids.mean()
    if mean_c > 0:
        spectral_flux_norm = float(np.clip(diffs.mean() / mean_c, 0.0, 1.0))
    else:
        spectral_flux_norm = 0.0

    # LFO rate: if centroid oscillates at a sub-audio rate, estimate it.
    # Simple autocorrelation peak detection on centroid sequence.
    lfo_rate_hz = _estimate_lfo_rate(centroids, hop / sr)
    # Map Hz → LFO Rate param [0,1]. Range ≈ 0.1–30 Hz.
    if lfo_rate_hz > 0:
        lfo_rate_est = float(np.clip(np.log10(max(lfo_rate_hz, 0.1)) / np.log10(30.0), 0.0, 1.0))
    else:
        lfo_rate_est = 0.2  # slow default; most targets don't need fast LFO

    return spectral_flux_norm, lfo_rate_est


def _estimate_lfo_rate(centroids: np.ndarray, hop_sec: float) -> float:
    """Detect oscillation rate in centroid sequence. Returns Hz, or 0."""
    if len(centroids) < 8:
        return 0.0
    x = centroids - centroids.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    # Search for autocorr peaks between 0.2 Hz (5s period) and 20 Hz (50ms)
    min_lag = max(1, int(0.05 / hop_sec))   # 50ms → 20 Hz cap
    max_lag = min(len(ac) - 1, int(5.0 / hop_sec))  # 5s → 0.2 Hz floor
    if max_lag <= min_lag:
        return 0.0
    peaks, props = signal.find_peaks(ac[min_lag:max_lag], height=0.1 * ac[0])
    if len(peaks) == 0:
        return 0.0
    best_lag = peaks[np.argmax(props["peak_heights"])] + min_lag
    return float(1.0 / (best_lag * hop_sec))


def _detect_transient_min_ms(audio: np.ndarray, sr: int) -> float:
    """Detect the shortest inter-onset interval in `audio` (milliseconds).

    Returns 1000.0 when librosa is unavailable, audio is too short, or fewer
    than 2 onsets are detected — interpreted as "no short transients detected".

    Used by vst_cmaes.py to dynamically tighten the Amp Env Attack upper bound:
    an attack longer than transient_min_ms * 0.5 would cause the synth to miss
    the onset of each transient burst entirely.
    """
    try:
        import librosa
    except ImportError:
        return 1000.0
    if len(audio) < int(sr * 0.05):
        return 1000.0
    try:
        onsets = librosa.onset.onset_detect(
            y=audio.astype(np.float32), sr=sr,
            hop_length=int(sr * 0.005), backtrack=True, units="samples"
        )
        if len(onsets) < 2:
            return 1000.0
        intervals_ms = np.diff(onsets.astype(np.float64)) / sr * 1000.0
        return float(intervals_ms.min())
    except Exception:
        return 1000.0


def _attack_ms_to_param(attack_ms: float) -> float:
    """Map measured attack time (ms) to Amp Env Attack param [0,1].
    Uses calibration table (42ms–2953ms measured); falls back to log10 heuristic."""
    return _ms_to_param(attack_ms, "amp_attack_params", "amp_attack_ms",
                        fallback_min_ms=1.0, fallback_max_ms=10000.0)


def _decay_ms_to_param(decay_ms: float) -> float:
    """Map measured decay time (ms) to Amp Env Decay param [0,1].
    Uses calibration table (5ms–12000ms measured); falls back to log10 heuristic."""
    return _ms_to_param(decay_ms, "amp_decay_params", "amp_decay_ms",
                        fallback_min_ms=1.0, fallback_max_ms=15000.0)


def _release_ms_to_param(release_ms: float) -> float:
    """Map measured release time (ms) to Amp Env Release param [0,1].
    Uses calibration table (4ms–12000ms measured); falls back to log10 heuristic."""
    return _ms_to_param(release_ms, "amp_release_params", "amp_release_ms",
                        fallback_min_ms=1.0, fallback_max_ms=15000.0)


# ── Pretty-print ─────────────────────────────────────────────────────────────

def print_analysis(a: TargetAnalysis) -> None:
    """Print a human-readable analysis summary."""
    print("\n=== Target Analysis ===")
    print(f"  Q1 Oscillator:")
    print(f"     Spectral centroid   {a.spectral_centroid_norm:.3f}  (0=bass 1=bright)")
    print(f"     Harmonic ratio      {a.harmonic_ratio:.3f}  (0=noise 1=pure tonal)")
    print(f"     Inharmonicity       {a.inharmonicity:.3f}  (0=harmonic 1=inharmonic)")
    print(f"     Spectral flatness   {a.spectral_flatness:.3f}  (0=tonal 1=noise)")
    print(f"     Fundamental         {a.fundamental_hz:.1f}Hz" if a.fundamental_hz else "     Fundamental         none detected")
    print(f"  Q2 Amp ADSR:")
    print(f"     Attack              {a.attack_ms:.1f}ms  → param {_attack_ms_to_param(a.attack_ms):.3f}")
    print(f"     Decay               {a.decay_ms:.1f}ms  → param {_decay_ms_to_param(a.decay_ms):.3f}")
    print(f"     Sustain level       {a.sustain_level:.2f}  → param {a.sustain_level:.3f}")
    print(f"     Release             {a.release_ms:.1f}ms  → param {_release_ms_to_param(a.release_ms):.3f}")
    print(f"  Q3 Filter:")
    print(f"     Cutoff estimate     {a.filter_cutoff_est:.2f}")
    print(f"     Resonance estimate  {a.filter_resonance_est:.2f}")
    print(f"     Mode estimate       {a.filter_mode_est:.2f}  (0=LP 0.33=HP 0.67=BP)")
    print(f"  Q4 Modulation:")
    print(f"     Spectral flux       {a.spectral_flux_norm:.3f}  (0=static 1=sweeping)")
    print(f"     Filter Env Amount   {a.filter_env_amount_est:.2f}")
    print(f"     LFO Rate            {a.lfo_rate_est:.2f}")
    print(f"     LFO→Filter          {a.lfo_to_filter_est:.2f}")
    print(f"  Q5 Pitch:")
    print(f"     Stable pitch        {a.pitch_is_stable}")
    print(f"     Cross Modulation    {a.cross_mod_est:.2f}")
    print(f"     Osc 2 Detune        {a.osc2_detune_est:.2f}  (0.5=unison)")
    print(f"  CREPE/DDSP analysis (f0_source={a.f0_source}):")
    print(f"     Noise Volume est    {a.noise_volume_est:.3f}  → Noise Volume")
    print(f"     Harmonic richness   {a.harmonic_richness:.3f}  → Osc 2 Volume {a.harmonic_richness*0.4:.2f}")
    print(f"     Harmonic slope res  {a.harmonic_slope_resonance:.3f}  → Filter Resonance")
    print(f"     Filter Env Attack   {a.filter_env_attack_est:.3f}")
    print(f"     Filter Env Decay    {a.filter_env_decay_est:.3f}")
    print(f"     Filter Env Sustain  {a.filter_env_sustain_est:.3f}")
    if a.notes:
        print("  Notes:")
        for n in a.notes:
            print(f"    • {n}")

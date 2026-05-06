"""Analyze target audio and map features to OB-Xf parameter initializations.

This module answers the five design questions before CMA-ES begins so that
the optimizer starts near a sensible region rather than a random one.

Design questions (from build_instructions/07 Refine VST Loop.md):

  Q1 — Oscillator type: saw, square/pulse, noise, or combination?
       → Detected via harmonic structure. Sawtooth has strong odd+even
         harmonics; square/pulse has strong odd harmonics; noise has flat
         spectrum. The OB-Xf profile's continuous params (Osc Pulsewidth,
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
    filter_cutoff_est: float          # [0,1] OB-Xf cutoff param suggestion
    filter_resonance_est: float       # [0,1] OB-Xf resonance param suggestion
    filter_mode_est: float            # [0,1] 0=LP, 0.33=HP, 0.67=BP, 1=notch

    # Q4 — Modulation
    spectral_flux_norm: float         # [0,1] normalised; high = evolving timbre
    filter_env_amount_est: float      # [0,1] OB-Xf Filter Env Amount suggestion
    lfo_rate_est: float               # [0,1] OB-Xf LFO 1 Rate suggestion
    lfo_to_filter_est: float          # [0,1] OB-Xf LFO 1 to Filter Cutoff

    # Q5 — Pitch / inharmonic character
    pitch_is_stable: bool             # True if F0 is approximately constant
    cross_mod_est: float              # [0,1] Cross Modulation suggestion
    osc2_detune_est: float            # [0,1] Osc 2 Detune suggestion (0.5=unison)

    # Diagnostics
    notes: list[str] = field(default_factory=list)


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

    # ── Q3: Filter parameters ────────────────────────────────────────────────

    # Filter cutoff: map spectral centroid to OB-Xf cutoff parameter.
    # The OB-Xf lowpass filter at param 0.5 ≈ 1000-2000 Hz depending on
    # tracking. High centroid (bright) → needs high cutoff to pass content.
    # Reference: Yee-King (2011) Eq. 3 - linear mapping with modest scaling.
    filter_cutoff_est = float(np.clip(0.3 + spectral_centroid_norm * 0.6, 0.3, 0.95))

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
        filter_mode_est = 0.0   # lowpass is almost always correct for OB-Xf

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

    # Cross Modulation (FM) is useful for bell-like, metallic, or distinctly
    # inharmonic sounds. High inharmonicity → try some cross-mod.
    # Note: Osc 1 Pitch is pinned; we can't do true pitch tracking here.
    # But we can recommend whether cross-mod is likely helpful.
    cross_mod_est = float(np.clip(inharmonicity * 0.25, 0.0, 0.25))

    # Osc 2 Detune: slight detune adds thickness; more detune adds chorus/beating.
    # For pitched tonal sounds: keep near unison (0.5 ± 0.05).
    # For inharmonic/noise: allow more detune.
    osc2_detune_est = 0.5 + float(np.clip(inharmonicity * 0.1, -0.1, 0.1))
    osc2_detune_est = float(np.clip(osc2_detune_est, 0.4, 0.6))

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
    )


def suggest_x0(
    analysis: TargetAnalysis,
    param_cols: list[str],
    pinned_cols: set[str],
    current_x0: np.ndarray | None = None,
) -> np.ndarray:
    """Map analysis features to a concrete OB-Xf parameter vector.

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
        "Filter Resonance":   analysis.filter_resonance_est,
        "Filter Mode":        analysis.filter_mode_est,
        "Filter Env Amount":  analysis.filter_env_amount_est,
        # Q2 — amp envelope (attack/decay; release is PINNED)
        "Amp Env Attack":     _attack_ms_to_param(analysis.attack_ms),
        "Amp Env Decay":      _decay_ms_to_param(analysis.decay_ms),
        # Q4 — LFO
        "LFO 1 Rate":         analysis.lfo_rate_est,
        "LFO 1 to Filter Cutoff": analysis.lfo_to_filter_est,
        # Q1/Q5 — oscillator
        "Osc 2 Detune":       analysis.osc2_detune_est,
        "Cross Modulation":   analysis.cross_mod_est,
        "Osc Pulsewidth":     0.5,     # neutral default; osc config handles wave type
        "Osc 1 Volume":       0.7,     # safe default; osc 1 is primary
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
    lfo_rate_est: OB-Xf LFO Rate param suggestion (0=slow, 1=fast).
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
    # Map Hz → OB-Xf LFO Rate param [0,1]. OB-Xf LFO range ≈ 0.1–30 Hz.
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


def _attack_ms_to_param(attack_ms: float) -> float:
    """Map measured attack time to OB-Xf Amp Env Attack param [0,1].

    OB-Xf attack range is approximately 1ms (param=0.0) to 10s (param=1.0)
    with roughly logarithmic scaling.
    """
    attack_ms = max(attack_ms, 1.0)
    # log-scale: 1ms→0.0, 10ms→0.15, 100ms→0.4, 1000ms→0.65, 10000ms→1.0
    log_val = np.log10(attack_ms)          # 0 → 4
    param = float(np.clip(log_val / 4.0, 0.0, 1.0))
    return param


def _decay_ms_to_param(decay_ms: float) -> float:
    """Map measured decay time to OB-Xf Amp Env Decay param [0,1]."""
    decay_ms = max(decay_ms, 1.0)
    log_val = np.log10(decay_ms)
    return float(np.clip(log_val / 4.0, 0.0, 1.0))


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
    print(f"     Attack              {a.attack_ms:.1f}ms  → param {_attack_ms_to_param(a.attack_ms):.2f}")
    print(f"     Decay               {a.decay_ms:.1f}ms  → param {_decay_ms_to_param(a.decay_ms):.2f}")
    print(f"     Sustain level       {a.sustain_level:.2f}")
    print(f"     Release             {a.release_ms:.1f}ms  (pinned to 0.2 in render)")
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
    if a.notes:
        print("  Notes:")
        for n in a.notes:
            print(f"    • {n}")

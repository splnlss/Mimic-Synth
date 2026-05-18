"""
One-time calibration script for synth parameter → time/frequency mappings.

Calibrations
------------
--filter-cutoff   Sweep Filter Cutoff 0→1, measure spectral centroid (Hz).
--amp-adsr        Sweep Amp Env Attack / Decay / Release, measure time (ms).
--filter-adsr     Sweep Filter Env Attack / Decay, measure centroid rise/fall time (ms).
--all             Run all calibrations (default when no flag given).

Output
------
s01_project-profile/calibration.npz — new keys merged into existing file:
    filter_cutoff_values           float64[N]  param values
    filter_cutoff_centroids_hz     float64[N]  sorted ascending

    amp_attack_params              float64[N]
    amp_attack_ms                  float64[N]
    amp_decay_params               float64[N]
    amp_decay_ms                   float64[N]
    amp_release_params             float64[N]
    amp_release_ms                 float64[N]

    filter_attack_params           float64[N]
    filter_attack_ms               float64[N]
    filter_decay_params            float64[N]
    filter_decay_ms                float64[N]

Usage
-----
    conda activate mimic-synth
    cd /home/sanss/Mimic-Synth
    python calibrate_synth.py --all          # full run (~25 min)
    python calibrate_synth.py --amp-adsr     # amp envelopes only (~10 min)
    python calibrate_synth.py --filter-adsr  # filter envelopes only (~8 min)
    python calibrate_synth.py --filter-cutoff  # cutoff curve only (~1 min)
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import librosa
import numpy as np
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent   # s01_setup/ → repo root
sys.path.insert(0, str(REPO_ROOT))

# ── Shared constants ──────────────────────────────────────────────────────────

SR          = 48000
BUFFER_SIZE = 512
MIDI_NOTE   = 60    # C4 — mid-range
MIDI_VEL    = 100
N_STEPS     = 32    # sweep resolution for ADSR (64 for cutoff is retained)
OUT_PATH    = REPO_ROOT / "s01_project-profile" / "calibration.npz"   # legacy fallback; pass --out to override

# Envelope measurement settings
RMS_WIN_MS  = 2.0   # RMS window size in ms
RMS_HOP_MS  = 0.5   # RMS hop in ms
SILENCE_DB  = -40.0 # threshold for "silence" in release measurement
ATTACK_FRAC = 0.90  # fraction of peak to call "attack complete"
DECAY_FRAC  = 0.55  # fraction of peak sustain to call "decay complete" (+10% tolerance)


# ── Profile / plugin helpers ──────────────────────────────────────────────────

def _load_profile(path: Path | None = None) -> dict:
    if path is None:
        from defaults import PROFILE_PATH
        path = PROFILE_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def _plugin_path(profile: dict) -> str:
    key = {"Darwin": "plugin_path_macos",
           "Windows": "plugin_path_windows",
           "Linux": "plugin_path_linux"}.get(platform.system())
    path = profile["synth"].get(key)
    if not path:
        raise RuntimeError(f"Profile missing {key}")
    return path


def _setup_synth(profile: dict):
    """Return (engine, plugin, name_to_idx). Applies profile reset values."""
    import dawdreamer as daw
    plugin_path = _plugin_path(profile)
    engine = daw.RenderEngine(SR, BUFFER_SIZE)
    synth_id = profile.get("synth", {}).get("id", "synth")
    plugin = engine.make_plugin_processor(synth_id, plugin_path)
    engine.load_graph([(plugin, [])])
    nidx: dict[str, int] = {
        plugin.get_parameter_name(i): i
        for i in range(plugin.get_plugin_parameter_size())
    }
    for name, val in profile.get("reset", {}).items():
        if name in nidx:
            plugin.set_parameter(nidx[name], float(val))
    return engine, plugin, nidx


def _set(plugin, nidx: dict, **kwargs) -> None:
    """Set named parameters. Unknown names are silently ignored."""
    for name, val in kwargs.items():
        if name in nidx:
            plugin.set_parameter(nidx[name], float(val))


def _render_note(engine, plugin, note_on: float, note_dur: float,
                 total_sec: float) -> np.ndarray:
    """Render a single note, return mono float32 audio."""
    plugin.clear_midi()
    plugin.add_midi_note(MIDI_NOTE, MIDI_VEL, note_on, note_dur)
    engine.render(total_sec)
    audio = plugin.get_audio()
    mono = audio[0] if audio.ndim > 1 else audio
    return mono.astype(np.float32)


# ── Envelope measurement helpers ──────────────────────────────────────────────

def _rms_envelope(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Windowed RMS envelope. Returns (rms_values, time_sec)."""
    win = max(2, int(SR * RMS_WIN_MS / 1000))
    hop = max(1, int(SR * RMS_HOP_MS / 1000))
    rms   = librosa.feature.rms(y=audio, frame_length=win, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=hop)
    return rms.astype(np.float64), times


def _measure_attack_ms(audio: np.ndarray, note_on_sec: float) -> float:
    """Time (ms) from note-on to ATTACK_FRAC × peak in the RMS envelope."""
    rms, times = _rms_envelope(audio)
    note_on_frame = np.searchsorted(times, note_on_sec)
    after = rms[note_on_frame:]
    if np.max(after) < 1e-7:
        return np.nan
    peak    = np.max(after)
    target  = ATTACK_FRAC * peak
    reached = np.where(after >= target)[0]
    if len(reached) == 0:
        return np.nan
    return float((times[note_on_frame + reached[0]] - note_on_sec) * 1000)


def _measure_decay_ms(audio: np.ndarray, note_on_sec: float,
                      db_target: float = -6.0,
                      transient_skip_ms: float = 80.0) -> float:
    """Time (ms) from peak (after transient) to db_target dB below that peak.

    Uses a 20ms RMS window (vs 2ms for attack) for stability, and skips the
    first transient_skip_ms after note-on so onset noise doesn't inflate
    the peak reference. The synth should be set with Sustain=0.0 so the
    signal decays cleanly to silence — giving the same clean measurement
    strategy as release.
    """
    # Larger window + hop for stable envelope on a slowly-changing signal
    win = max(2, int(SR * 20.0 / 1000))
    hop = max(1, int(SR * 5.0  / 1000))
    rms   = librosa.feature.rms(y=audio, frame_length=win, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=hop)

    # Skip past the onset transient before looking for the true peak
    skip_sec    = note_on_sec + transient_skip_ms / 1000.0
    start_frame = np.searchsorted(times, skip_sec)
    after_skip  = rms[start_frame:]

    if np.max(after_skip) < 1e-7:
        return np.nan

    peak_idx  = int(np.argmax(after_skip))
    peak      = after_skip[peak_idx]
    threshold = peak * 10.0 ** (db_target / 20.0)   # -6dB → 0.5 × peak

    after_peak = after_skip[peak_idx:]
    crossed    = np.where(after_peak <= threshold)[0]
    if len(crossed) == 0:
        return np.nan

    t_peak    = times[start_frame + peak_idx]
    t_crossed = times[start_frame + peak_idx + crossed[0]]
    return float((t_crossed - t_peak) * 1000)


def _measure_release_ms(audio: np.ndarray, note_off_sec: float) -> float:
    """Time (ms) from note-off to SILENCE_DB dB below peak RMS."""
    rms, times = _rms_envelope(audio)
    # Peak before note-off
    note_off_frame = np.searchsorted(times, note_off_sec)
    before = rms[:note_off_frame]
    if np.max(before) < 1e-7:
        return np.nan
    peak       = np.max(before)
    threshold  = peak * 10 ** (SILENCE_DB / 20.0)
    after_off  = rms[note_off_frame:]
    silent     = np.where(after_off <= threshold)[0]
    if len(silent) == 0:
        return np.nan
    return float((times[note_off_frame + silent[0]] - note_off_sec) * 1000)


def _measure_centroid_attack_ms(audio: np.ndarray, note_on_sec: float,
                                 sr: int = SR) -> float:
    """Time (ms) from note-on to spectral centroid reaching ATTACK_FRAC × peak."""
    hop = max(1, int(sr * 5.0 / 1000))   # 5ms hop for centroid
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop)[0]
    times    = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=hop)
    note_on_frame = np.searchsorted(times, note_on_sec)
    after = centroid[note_on_frame:]
    if np.max(after) < 10:
        return np.nan
    peak    = np.max(after)
    base    = centroid[:note_on_frame].mean() if note_on_frame > 0 else 0.0
    target  = base + ATTACK_FRAC * (peak - base)
    reached = np.where(after >= target)[0]
    if len(reached) == 0:
        return np.nan
    return float((times[note_on_frame + reached[0]] - note_on_sec) * 1000)


def _measure_centroid_decay_ms(audio: np.ndarray, note_on_sec: float,
                                sr: int = SR) -> float:
    """Time (ms) from centroid peak to centroid falling to 50% of its swing."""
    hop      = max(1, int(sr * 5.0 / 1000))
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop)[0]
    times    = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=hop)
    note_on_frame = np.searchsorted(times, note_on_sec)
    after    = centroid[note_on_frame:]
    if np.max(after) < 10:
        return np.nan
    peak_idx = int(np.argmax(after))
    peak     = after[peak_idx]
    # Sustained centroid = median of last third of the note
    sustained = np.median(after[len(after) * 2 // 3:])
    target    = sustained + 0.5 * (peak - sustained)
    after_peak = after[peak_idx:]
    settled    = np.where(after_peak <= target)[0]
    if len(settled) == 0:
        return np.nan
    abs_frame = note_on_frame + peak_idx + settled[0]
    return float((times[abs_frame] - (times[note_on_frame] + times[note_on_frame + peak_idx]
                                       - times[note_on_frame])) * 1000)


# ── Calibration routines ──────────────────────────────────────────────────────

def calibrate_filter_cutoff(engine, plugin, nidx: dict,
                             profile: dict) -> dict:
    """Sweep Filter Cutoff 0→1, record spectral centroid at each step."""
    N = 64
    SUSTAIN_SEC    = 1.5
    SETTLE_SEC     = 0.5
    ANALYSIS_START = 0.25
    NOTE_ON        = 0.05
    render_sec     = NOTE_ON + SUSTAIN_SEC + SETTLE_SEC

    print("\n=== Filter Cutoff calibration ===")
    _set(plugin, nidx, **{k: 0.0 for k in
         ["Filter Resonance", "Filter Env Amount", "LFO 1 to Filter Cutoff"]})

    for name, val in profile.get("reset", {}).items():
        if name in nidx:
            plugin.set_parameter(nidx[name], float(val))
    _set(plugin, nidx, **{"Filter Resonance": 0.0})

    values      = np.linspace(0.0, 1.0, N)
    centroids   : list[float] = []

    pbar = tqdm(values, desc="  filter cutoff", unit="step")
    for v in pbar:
        _set(plugin, nidx, **{"Filter Cutoff": float(v)})
        audio = _render_note(engine, plugin, NOTE_ON, SUSTAIN_SEC, render_sec)
        start = int(SR * (NOTE_ON + ANALYSIS_START))
        seg   = audio[start:]
        if np.max(np.abs(seg)) < 1e-6:
            c = 0.0
        else:
            c = float(librosa.feature.spectral_centroid(y=seg, sr=SR).mean())
        centroids.append(c)
        pbar.set_postfix(cutoff=f"{v:.3f}", centroid=f"{c:.0f}Hz")

    arr   = np.array(centroids, dtype=np.float64)
    order = np.argsort(arr)
    sv, sc = values[order], arr[order]
    print(f"  ✓ Centroid: {sc[0]:.0f} Hz (cutoff={sv[0]:.3f}) → "
          f"{sc[-1]:.0f} Hz (cutoff={sv[-1]:.3f})")
    return {"filter_cutoff_values": sv, "filter_cutoff_centroids_hz": sc}


def calibrate_amp_adsr(engine, plugin, nidx: dict,
                       profile: dict) -> dict:
    """Sweep Amp Env Attack, Decay, Release; measure time constants in ms."""
    results: dict[str, np.ndarray] = {}

    # ── Shared reset ─────────────────────────────────────────────────────────
    for name, val in profile.get("reset", {}).items():
        if name in nidx:
            plugin.set_parameter(nidx[name], float(val))
    # Bright source so signal is easy to measure
    _set(plugin, nidx, **{
        "Osc 1 Saw Wave": 1.0, "Osc 1 Volume": 0.8,
        "Filter Cutoff": 0.8, "Filter Resonance": 0.0, "Filter Env Amount": 0.0,
    })

    params = np.linspace(0.0, 1.0, N_STEPS)

    # ── Attack ────────────────────────────────────────────────────────────────
    print("\n=== Amp Env Attack calibration ===")
    # Decay=0 (instant), Sustain=1.0 (stays at peak), Release=0 (fast tail)
    _set(plugin, nidx, **{
        "Amp Env Decay": 0.0, "Amp Env Sustain": 1.0, "Amp Env Release": 0.0,
    })
    NOTE_ON   = 0.05
    HOLD_SEC  = 3.0    # long enough for slow attacks
    TAIL_SEC  = 0.3
    render_sec = NOTE_ON + HOLD_SEC + TAIL_SEC
    attack_ms: list[float] = []

    pbar = tqdm(params, desc="  amp attack", unit="step")
    for v in pbar:
        _set(plugin, nidx, **{"Amp Env Attack": float(v)})
        audio = _render_note(engine, plugin, NOTE_ON, HOLD_SEC, render_sec)
        ms    = _measure_attack_ms(audio, NOTE_ON)
        attack_ms.append(ms if not np.isnan(ms) else 3000.0)
        pbar.set_postfix(param=f"{v:.3f}", attack_ms=f"{attack_ms[-1]:.0f}")

    results["amp_attack_params"] = params.copy()
    results["amp_attack_ms"]     = np.array(attack_ms, dtype=np.float64)
    print(f"  ✓ Attack range: {min(attack_ms):.1f} ms → {max(attack_ms):.1f} ms")

    # ── Decay ─────────────────────────────────────────────────────────────────
    print("\n=== Amp Env Decay calibration ===")
    # Sustain=0.0 so decay goes all the way to silence — same clean measurement
    # as release (clear signal drop). Attack=0.0 is fine because _measure_decay_ms
    # skips 80ms of transient before looking for the peak.
    _set(plugin, nidx, **{
        "Amp Env Attack": 0.0, "Amp Env Sustain": 0.0, "Amp Env Release": 0.0,
    })
    HOLD_SEC   = 20.0   # slow decays can be many seconds
    TAIL_SEC   = 0.5
    render_sec = NOTE_ON + HOLD_SEC + TAIL_SEC
    decay_ms: list[float] = []

    pbar = tqdm(params, desc="  amp decay", unit="step")
    for v in pbar:
        _set(plugin, nidx, **{"Amp Env Decay": float(v)})
        audio = _render_note(engine, plugin, NOTE_ON, HOLD_SEC, render_sec)
        ms    = _measure_decay_ms(audio, NOTE_ON)
        decay_ms.append(ms if not np.isnan(ms) else 20000.0)
        pbar.set_postfix(param=f"{v:.3f}", decay_ms=f"{decay_ms[-1]:.0f}")

    results["amp_decay_params"] = params.copy()
    results["amp_decay_ms"]     = np.array(decay_ms, dtype=np.float64)
    print(f"  ✓ Decay range: {min(decay_ms):.1f} ms → {max(decay_ms):.1f} ms")

    # ── Release ───────────────────────────────────────────────────────────────
    print("\n=== Amp Env Release calibration ===")
    # Attack=0, Decay=0, Sustain=1.0 — so we're at full amplitude at note-off
    _set(plugin, nidx, **{
        "Amp Env Attack": 0.0, "Amp Env Decay": 0.0, "Amp Env Sustain": 1.0,
    })
    HOLD_SEC   = 1.0    # hold in sustain
    TAIL_SEC   = 12.0   # enough for very slow releases
    render_sec = NOTE_ON + HOLD_SEC + TAIL_SEC
    NOTE_OFF   = NOTE_ON + HOLD_SEC
    release_ms: list[float] = []

    pbar = tqdm(params, desc="  amp release", unit="step")
    for v in pbar:
        _set(plugin, nidx, **{"Amp Env Release": float(v)})
        audio = _render_note(engine, plugin, NOTE_ON, HOLD_SEC, render_sec)
        ms    = _measure_release_ms(audio, NOTE_OFF)
        release_ms.append(ms if not np.isnan(ms) else 12000.0)
        pbar.set_postfix(param=f"{v:.3f}", release_ms=f"{release_ms[-1]:.0f}")

    results["amp_release_params"] = params.copy()
    results["amp_release_ms"]     = np.array(release_ms, dtype=np.float64)
    print(f"  ✓ Release range: {min(release_ms):.1f} ms → {max(release_ms):.1f} ms")

    return results


def calibrate_filter_adsr(engine, plugin, nidx: dict,
                           profile: dict) -> dict:
    """Sweep Filter Env Attack and Decay; measure spectral centroid rise/fall time (ms)."""
    results: dict[str, np.ndarray] = {}

    for name, val in profile.get("reset", {}).items():
        if name in nidx:
            plugin.set_parameter(nidx[name], float(val))

    # Amp ADSR neutral (instant attack/decay, full sustain, fast release)
    # so amp doesn't interfere with the filter measurement
    _set(plugin, nidx, **{
        "Osc 1 Saw Wave": 1.0, "Osc 1 Volume": 0.8,
        "Amp Env Attack": 0.0, "Amp Env Decay": 0.0,
        "Amp Env Sustain": 1.0, "Amp Env Release": 0.0,
        # Filter: low cutoff so filter opens clearly; high Env Amount for measurable swing
        "Filter Cutoff": 0.1,
        "Filter Resonance": 0.2,
        "Filter Env Amount": 0.8,
    })

    params = np.linspace(0.0, 1.0, N_STEPS)

    # ── Filter Attack ─────────────────────────────────────────────────────────
    print("\n=== Filter Env Attack calibration ===")
    # Decay=0, Sustain=1 (filter stays open after attack)
    _set(plugin, nidx, **{
        "Filter Env Decay": 0.0, "Filter Env Sustain": 1.0,
        "Filter Env Release": 0.0,
    })
    NOTE_ON    = 0.05
    HOLD_SEC   = 3.0
    TAIL_SEC   = 0.3
    render_sec = NOTE_ON + HOLD_SEC + TAIL_SEC
    f_attack_ms: list[float] = []

    pbar = tqdm(params, desc="  filter attack", unit="step")
    for v in pbar:
        _set(plugin, nidx, **{"Filter Env Attack": float(v)})
        audio = _render_note(engine, plugin, NOTE_ON, HOLD_SEC, render_sec)
        ms    = _measure_centroid_attack_ms(audio, NOTE_ON)
        f_attack_ms.append(ms if not np.isnan(ms) else 3000.0)
        pbar.set_postfix(param=f"{v:.3f}", attack_ms=f"{f_attack_ms[-1]:.0f}")

    results["filter_attack_params"] = params.copy()
    results["filter_attack_ms"]     = np.array(f_attack_ms, dtype=np.float64)
    print(f"  ✓ Filter attack range: {min(f_attack_ms):.1f} ms → {max(f_attack_ms):.1f} ms")

    # ── Filter Decay ──────────────────────────────────────────────────────────
    print("\n=== Filter Env Decay calibration ===")
    # Attack=0 (instant open), Sustain=0 (filter closes fully after decay)
    _set(plugin, nidx, **{
        "Filter Env Attack": 0.0, "Filter Env Sustain": 0.0,
        "Filter Env Release": 0.0,
    })
    HOLD_SEC   = 8.0
    render_sec = NOTE_ON + HOLD_SEC + TAIL_SEC
    f_decay_ms: list[float] = []

    pbar = tqdm(params, desc="  filter decay", unit="step")
    for v in pbar:
        _set(plugin, nidx, **{"Filter Env Decay": float(v)})
        audio = _render_note(engine, plugin, NOTE_ON, HOLD_SEC, render_sec)
        ms    = _measure_centroid_decay_ms(audio, NOTE_ON)
        f_decay_ms.append(ms if not np.isnan(ms) else 8000.0)
        pbar.set_postfix(param=f"{v:.3f}", decay_ms=f"{f_decay_ms[-1]:.0f}")

    results["filter_decay_params"] = params.copy()
    results["filter_decay_ms"]     = np.array(f_decay_ms, dtype=np.float64)
    print(f"  ✓ Filter decay range: {min(f_decay_ms):.1f} ms → {max(f_decay_ms):.1f} ms")

    return results


# ── NPZ merge + save ──────────────────────────────────────────────────────────

def _save(new_data: dict) -> None:
    """Merge new calibration data into the existing NPZ file."""
    existing: dict = {}
    if OUT_PATH.exists():
        with np.load(str(OUT_PATH)) as f:
            existing = {k: f[k] for k in f.files}
    existing.update(new_data)
    np.savez(str(OUT_PATH), **existing)
    print(f"\n  ✓ Saved {OUT_PATH}  ({', '.join(sorted(existing.keys()))})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Synth calibration sweeps")
    ap.add_argument("--profile", default=None,
                    help="Path to profile YAML (default: defaults.PROFILE_PATH)")
    ap.add_argument("--out", default=None,
                    help="Path for output .npz (default: s01_project-profile/calibration.npz)")
    ap.add_argument("--filter-cutoff", action="store_true",
                    help="Sweep Filter Cutoff → spectral centroid Hz")
    ap.add_argument("--amp-adsr",      action="store_true",
                    help="Sweep Amp Env Attack / Decay / Release → ms")
    ap.add_argument("--filter-adsr",   action="store_true",
                    help="Sweep Filter Env Attack / Decay → ms")
    ap.add_argument("--all",           action="store_true",
                    help="Run all calibrations (default if no flag given)")
    args = ap.parse_args()

    run_all        = args.all or not any([args.filter_cutoff,
                                          args.amp_adsr, args.filter_adsr])
    do_cutoff      = run_all or args.filter_cutoff
    do_amp_adsr    = run_all or args.amp_adsr
    do_filter_adsr = run_all or args.filter_adsr

    # Resolve profile path
    if args.profile:
        profile_path = Path(args.profile)
    else:
        from defaults import PROFILE_PATH
        profile_path = PROFILE_PATH

    # Resolve output path
    global OUT_PATH
    if args.out:
        OUT_PATH = Path(args.out)
    elif args.profile:
        OUT_PATH = Path(args.profile).parent / "calibration.npz"

    try:
        import dawdreamer  # noqa: F401
    except ImportError:
        print("dawdreamer not available — run under conda env mimic-synth")
        sys.exit(1)

    profile = _load_profile(profile_path)
    print(f"Profile : {profile_path}")
    print(f"Plugin  : {_plugin_path(profile)}")
    print(f"Output  : {OUT_PATH}")
    print(f"Steps   : {N_STEPS} (ADSR)  64 (filter cutoff)")

    engine, plugin, nidx = _setup_synth(profile)
    all_data: dict = {}

    if do_cutoff:
        all_data.update(calibrate_filter_cutoff(engine, plugin, nidx, profile))
        _save(all_data)

    if do_amp_adsr:
        all_data.update(calibrate_amp_adsr(engine, plugin, nidx, profile))
        _save(all_data)

    if do_filter_adsr:
        all_data.update(calibrate_filter_adsr(engine, plugin, nidx, profile))
        _save(all_data)

    print("\n=== Calibration complete ===")


if __name__ == "__main__":
    main()

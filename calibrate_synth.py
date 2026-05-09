"""
One-time calibration script: sweep OB-Xf Filter Cutoff 0→1 and measure
the spectral centroid of the output at each step.

Usage:
    conda run -n mimic-synth python calibrate_synth.py

Output:
    s07_refine/obxf_calibration.npz
        filter_cutoff_values      : float64[N_STEPS] in [0, 1]
        filter_cutoff_centroids_hz: float64[N_STEPS], sorted ascending

The calibration table is used by _centroid_hz_to_filter_cutoff in
s06b_live/stream_invert.py and target_analysis.py to convert a measured
spectral centroid (Hz) to the corresponding Filter Cutoff parameter value.
"""

import platform
import sys
from pathlib import Path

import librosa
import numpy as np
import yaml

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
from defaults import PROFILE_PATH

# ── Constants ────────────────────────────────────────────────────────────────
SR = 48000
BUFFER_SIZE = 512
N_STEPS = 64           # number of cutoff values to sweep (0/63, 1/63, …, 63/63)
MIDI_NOTE = 60         # C4 — mid-range, avoids interaction with Osc 2 Pitch limits
MIDI_VEL = 100
SUSTAIN_SEC = 1.5      # hold note this long
SETTLE_SEC = 0.5       # silence after note-off for tail to decay
ANALYSIS_START = 0.25  # skip first N sec (attack transient) when measuring centroid
OUT_PATH = REPO_ROOT / "s07_refine" / "obxf_calibration.npz"


def load_profile(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_plugin_path(profile: dict) -> str:
    sys_name = platform.system()
    key = {"Darwin": "plugin_path_macos",
           "Windows": "plugin_path_windows",
           "Linux": "plugin_path_linux"}.get(sys_name)
    if key is None:
        raise RuntimeError(f"Unknown platform: {sys_name}")
    path = profile["synth"].get(key)
    if path is None:
        raise RuntimeError(f"Profile missing {key}")
    return path


def run_calibration() -> None:
    try:
        import dawdreamer as daw
    except ImportError:
        print("dawdreamer not available — run under conda env mimic-synth")
        sys.exit(1)

    profile = load_profile(PROFILE_PATH)
    plugin_path = resolve_plugin_path(profile)
    reset_params: dict = profile.get("reset", {})

    print(f"Plugin : {plugin_path}")
    print(f"Steps  : {N_STEPS}  (cutoff 0.000 → 1.000)")
    print(f"Note   : MIDI {MIDI_NOTE}, {SUSTAIN_SEC}s sustain + {SETTLE_SEC}s settle\n")

    engine = daw.RenderEngine(SR, BUFFER_SIZE)
    synth = engine.make_plugin_processor("obxf", plugin_path)
    engine.load_graph([(synth, [])])

    name_to_idx: dict[str, int] = {
        synth.get_parameter_name(i): i
        for i in range(synth.get_plugin_parameter_size())
    }

    if "Filter Cutoff" not in name_to_idx:
        print("ERROR: 'Filter Cutoff' parameter not found in plugin.")
        print("Available params:", list(name_to_idx.keys())[:20])
        sys.exit(1)

    cutoff_idx = name_to_idx["Filter Cutoff"]
    resonance_idx = name_to_idx.get("Filter Resonance")

    # Apply neutral reset patch (avoids interaction with other params)
    for param_name, val in reset_params.items():
        if param_name in name_to_idx:
            synth.set_parameter(name_to_idx[param_name], float(val))

    # Ensure resonance=0 so centroid reflects cutoff cleanly (no resonant peak)
    if resonance_idx is not None:
        synth.set_parameter(resonance_idx, 0.0)

    values = np.linspace(0.0, 1.0, N_STEPS)
    centroids_hz: list[float] = []
    render_sec = SUSTAIN_SEC + SETTLE_SEC

    for i, v in enumerate(values):
        synth.set_parameter(cutoff_idx, float(v))
        synth.clear_midi()
        synth.add_midi_note(MIDI_NOTE, MIDI_VEL, 0.05, SUSTAIN_SEC)
        engine.render(render_sec)
        audio = synth.get_audio()
        # Use left channel, mono
        mono = audio[0] if audio.ndim > 1 else audio
        # Skip attack transient
        start = int(SR * ANALYSIS_START)
        analysis = mono[start:]
        if np.max(np.abs(analysis)) < 1e-6:
            # Silent (e.g. very low cutoff with saw oscillator produces near-silence)
            centroid = 0.0
        else:
            centroid = float(librosa.feature.spectral_centroid(y=analysis, sr=SR).mean())
        centroids_hz.append(centroid)
        print(f"  [{i+1:2d}/{N_STEPS}]  cutoff={v:.4f}  centroid={centroid:7.0f} Hz")

    centroids_arr = np.array(centroids_hz, dtype=np.float64)

    # Sort by centroid for monotone np.interp inversion
    order = np.argsort(centroids_arr)
    sorted_vals = values[order]
    sorted_cents = centroids_arr[order]

    np.savez(str(OUT_PATH),
             filter_cutoff_values=sorted_vals,
             filter_cutoff_centroids_hz=sorted_cents)
    print(f"\nSaved {OUT_PATH}")
    print(f"  Centroid range: {sorted_cents[0]:.0f} Hz (cutoff={sorted_vals[0]:.3f}) "
          f"→ {sorted_cents[-1]:.0f} Hz (cutoff={sorted_vals[-1]:.3f})")

    # Quick sanity: centroid should be broadly monotone with cutoff
    lo_cent = np.median(centroids_arr[:N_STEPS // 4])
    hi_cent = np.median(centroids_arr[3 * N_STEPS // 4:])
    if hi_cent > lo_cent:
        print("  ✓ Centroid is higher at high cutoff (monotone OK)")
    else:
        print("  ⚠ Centroid not monotone — check plugin output (resonant self-oscillation?)")


if __name__ == "__main__":
    run_calibration()

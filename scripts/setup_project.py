"""
setup_project.py — Project setup wizard for MimicSynth.

Steps (in execution order)
---------------------------
1. Project info      — synth name, version, project name, data root
2. Folder structure  — create all stage directories, project.yaml, PROJECT_STATUS.md
3. VST discovery     — scan OS VST3 paths (guided by synth name from step 1)
4. MIDI CC mapping   — import/enter CC→parameter assignments; enriches profile
5. Parameter enum    — load VST, enumerate all params, generate profile.yaml
6. MIDI port detect  — list MIDI input ports (hardware routing info)
7. Validation render — test render a single note; confirm audio output
8. Calibration       — run calibrate_synth.py --all (~20 min)

Usage
-----
    conda activate mimic-synth
    cd /home/sanss/Mimic-Synth
    python setup_project.py
    python setup_project.py --project-name Moog_Model_D --data-root /mnt/d/Mimic-Synth-Data
    python setup_project.py --vst ~/.vst3/MyPlugin.vst3 --no-interactive
"""
from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ → repo root
# sys.path.insert removed — installed via pyproject.toml

# ── Folder layout ──────────────────────────────────────────────────────────────

_PROFILE_DIR = "s01_project-profile"   # all project config lives here

_STAGE_DIRS = [
    "s01_project-profile",             # profile, project.yaml, calibration, status
    "s02_capture/wav",
    "s03_dataset",
    "s04_embed",
    "s05_surrogate/runs",
    "s06_invert/patches",
    "targets",
    "logs",
]

# ── VST3 search paths ──────────────────────────────────────────────────────────

_VST3_SEARCH = {
    "Linux":   [Path.home() / ".vst3",
                Path("/usr/lib/vst3"),
                Path("/usr/local/lib/vst3"),
                Path("/usr/lib/x86_64-linux-gnu/vst3")],
    "Darwin":  [Path.home() / "Library/Audio/Plug-Ins/VST3",
                Path("/Library/Audio/Plug-Ins/VST3")],
    "Windows": [Path("C:/Program Files/Common Files/VST3"),
                Path("C:/Program Files (x86)/Common Files/VST3")],
}

# ── Parameter heuristics ───────────────────────────────────────────────────────
# (keywords, importance, log_scale, continuous)

_RULES: list[tuple[list[str], float, bool, bool]] = [
    (["high quality", "quality mode", "voice count",
      "master tune", "master volume", "panic", "midi channel",
      "midi cc", "cc |", "| cc"],                                0.0,  False, False),
    (["filter cutoff", "cutoff freq"],                           1.00, True,  True),
    (["filter resonance", "resonance", "filter q"],              0.90, False, True),
    (["filter env amount", "env amt"],                           0.85, False, True),
    (["filter env attack"],                                      0.90, False, True),
    (["filter env decay"],                                       0.85, False, True),
    (["filter env sustain"],                                     0.80, False, True),
    (["filter env release"],                                     0.60, False, True),
    (["filter mode", "filter type", "filter slope"],             0.55, False, False),
    (["filter keytrack"],                                        0.40, False, True),
    (["amp env attack", "amp attack"],                           0.70, False, True),
    (["amp env decay",  "amp decay"],                            0.70, False, True),
    (["amp env sustain","amp sustain"],                          0.75, False, True),
    (["amp env release","amp release"],                          0.60, False, True),
    (["saw wave", "sawtooth"],                                   0.70, False, True),
    (["pulse wave", "square wave"],                              0.70, False, True),
    (["pulsewidth", "pulse width", "pw"],                        0.75, False, True),
    (["osc volume", "osc level", "osc mix"],                     0.80, False, True),
    (["osc pitch", "osc tune", "osc coarse"],                    0.70, False, True),
    (["osc detune", "fine tune", "finetune"],                    0.70, False, True),
    (["osc sync", "hard sync", "sync"],                          0.40, False, False),
    (["cross mod", "cross modulation"],                          0.50, False, True),
    (["ring mod", "ring modulation"],                            0.50, False, True),
    (["noise volume", "noise level", "noise mix"],               0.40, False, True),
    (["noise color", "noise type"],                              0.35, False, True),
    (["sub osc", "sub volume"],                                  0.45, False, True),
    (["lfo rate", "lfo speed", "lfo freq"],                      0.60, True,  True),
    (["lfo depth", "lfo amount", "lfo to"],                      0.50, False, True),
    (["lfo shape", "lfo wave"],                                  0.35, False, False),
    (["chorus", "ensemble", "width"],                            0.40, False, True),
    (["reverb", "delay", "echo"],                                0.30, False, True),
    (["drive", "distortion", "saturation"],                      0.60, False, True),
    (["unison", "voice"],                                        0.40, False, True),
    (["portamento", "glide"],                                    0.30, False, True),
    (["velocity", "vel"],                                        0.40, False, True),
    (["volume", "level", "gain", "mix", "amount"],               0.60, False, True),
    (["frequency", "freq"],                                      0.60, True,  True),
    (["attack"],                                                 0.65, False, True),
    (["decay"],                                                  0.65, False, True),
    (["sustain"],                                                 0.65, False, True),
    (["release"],                                                0.55, False, True),
]


def _classify(name: str) -> tuple[float, bool, bool]:
    n = name.lower()
    for keywords, imp, log, cont in _RULES:
        if any(kw in n for kw in keywords):
            return imp, log, cont
    return 0.5, False, True


# ── Interactive helpers ────────────────────────────────────────────────────────

def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val    = input(f"  {msg}{suffix}: ").strip()
    return val if val else default


def _choose(msg: str, options: list[str], default: int = 0) -> int:
    for i, opt in enumerate(options):
        marker = " ← default" if i == default else ""
        print(f"    {i + 1}. {opt}{marker}")
    while True:
        raw = input(f"  {msg} (1–{len(options)}) [{default + 1}]: ").strip()
        if not raw:
            return default
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass


# ── Step implementations ───────────────────────────────────────────────────────

def step1_project_info(args) -> dict:
    """Collect synth name, version, project name, and data root."""
    interactive = not args.no_interactive

    raw_name = args.project_name or (
        _prompt("Synth name (e.g. 'OB-Xf', 'Minimoog Model D')", "")
        if interactive else ""
    )
    if not raw_name:
        print("  ERROR: --project-name required in non-interactive mode")
        sys.exit(1)

    synth_id      = raw_name.lower().replace(" ", "_").replace("-", "_")
    synth_version = _prompt("Synth version", "1.0") if interactive else "1.0"
    project_name  = raw_name.replace(" ", "_")

    raw_root = args.data_root or (
        _prompt("Data root directory", "/mnt/d/Mimic-Synth-Data")
        if interactive else "/mnt/d/Mimic-Synth-Data"
    )

    data_root   = Path(raw_root)
    project_dir = data_root / project_name

    print(f"\n  Synth       : {raw_name} v{synth_version}")
    print(f"  Project dir : {project_dir}")

    return {
        "synth_name":    raw_name,
        "synth_id":      synth_id,
        "synth_version": synth_version,
        "project_name":  project_name,
        "project_dir":   project_dir,
        "data_root":     data_root,
    }


def step2_folder_structure(info: dict) -> None:
    """Create all stage directories, project.yaml, and PROJECT_STATUS.md."""
    project_dir = info["project_dir"]
    project_dir.mkdir(parents=True, exist_ok=True)

    for d in _STAGE_DIRS:
        (project_dir / d).mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created {len(_STAGE_DIRS)} stage directories")

    # s01_project-profile/project.yaml — single source of truth for stage paths
    s01_dir = project_dir / _PROFILE_DIR
    stage_dirs = [s.split("/")[0] for s in _STAGE_DIRS if s != _PROFILE_DIR]
    project_yaml = {
        "synth":   info["synth_name"],
        "version": info["synth_version"],
        "profile": f"{_PROFILE_DIR}/profile.yaml",
        "calibration": f"{_PROFILE_DIR}/calibration.npz",
        "capture": {"m": 10, "sample_rate": 48000},
        "stages": {s: f"{s}/" for s in stage_dirs},
    }
    out = s01_dir / "project.yaml"
    with open(out, "w") as f:
        yaml.dump(project_yaml, f, default_flow_style=False, sort_keys=True)
    print(f"  ✓ Written {_PROFILE_DIR}/project.yaml")

    # s01_project-profile/PROJECT_STATUS.md
    status = f"""# Project Status — {project_dir.name}

> Generated by setup_project.py. Update this file as each stage completes.

## Synth

| Field | Value |
|-------|-------|
| Name | {info['synth_name']} |
| Version | {info['synth_version']} |

## Stage Status

| Stage | Status | Notes |
|-------|--------|-------|
| S01 Profile | 🔄 In progress | setup_project.py running |
| Calibration | 🔲 Pending | Runs at end of setup |
| S02 Capture | 🔲 Not started | |
| S03 Dataset | 🔲 Not started | |
| S04 Embed   | 🔲 Not started | |
| S05 Surrogate | 🔲 Not started | |
| S06b Inversion | 🔲 Not started | |
| S07 Refinement | 🔲 Not started | |

## Next Steps

1. Review and tune `profile.yaml` importance weights
2. Update `defaults.py`:
   ```python
   DATA_ROOT    = Path("{info['data_root']}")
   PROJECT_NAME = "{info['project_name']}"
   ```
3. Run capture: `python s02_capture/capture_v1_2.py`
"""
    (s01_dir / "PROJECT_STATUS.md").write_text(status)
    print(f"  ✓ Written {_PROFILE_DIR}/PROJECT_STATUS.md")


def step3_vst_discovery(info: dict, args) -> Path:
    """Find the VST3 plugin on disk, guided by the synth name."""
    interactive = not args.no_interactive

    if args.vst:
        p = Path(args.vst)
        print(f"  Using provided path: {p}")
        if not p.exists():
            print(f"  ERROR: {p} does not exist")
            sys.exit(1)
        return p

    system = platform.system()
    search = _VST3_SEARCH.get(system, _VST3_SEARCH["Linux"])
    all_vsts: list[Path] = []
    for d in search:
        if d.exists():
            all_vsts.extend(sorted(d.glob("**/*.vst3")))

    if not all_vsts:
        print("  No VST3 plugins found in standard paths.")
        if interactive:
            raw = _prompt("Enter full path to .vst3 file")
            p   = Path(raw)
            if not p.exists():
                print(f"  ERROR: {p} does not exist")
                sys.exit(1)
            return p
        print("  ERROR: no VST found. Use --vst to specify path.")
        sys.exit(1)

    # Try to pre-select by matching synth name keywords
    name_lower  = info["synth_name"].lower().replace(" ", "")
    ranked      = sorted(all_vsts,
                         key=lambda p: -int(name_lower[:4] in p.stem.lower()))

    print(f"  Found {len(ranked)} VST3 plugin(s):")
    if interactive and len(ranked) > 1:
        idx = _choose("Select plugin", [str(p) for p in ranked], default=0)
        chosen = ranked[idx]
    else:
        chosen = ranked[0]

    print(f"  ✓ Selected: {chosen}")
    return chosen


def step4_midi_cc_mapping(info: dict, args) -> dict[str, int]:
    """Build a CC-number → param-name mapping.

    Accepts three input methods:
      a) CSV file with columns: CC, Parameter
      b) Inline text (paste a CC chart directly)
      c) Skip (returns empty dict)

    The mapping is stored in profile.yaml as midi_cc: <int> per parameter
    and used to annotate which params are hardware-CC-controllable.
    """
    interactive = not args.no_interactive
    cc_map: dict[str, int] = {}   # param_name → CC number

    if not interactive:
        print("  Skipped (non-interactive mode).")
        return cc_map

    print(f"""
  MIDI CC mapping enriches the profile with hardware-controller assignments.
  Sources accepted:
    1. CSV file path  (columns: CC,Parameter or Parameter,CC)
    2. Paste inline   (same CSV format, blank line to finish)
    3. Skip
""")
    choice = _choose("Input method", ["CSV file", "Paste inline", "Skip"], default=2)

    raw_lines: list[str] = []

    if choice == 0:
        csv_path = _prompt("CSV file path")
        try:
            raw_lines = Path(csv_path).read_text().splitlines()
        except Exception as e:
            print(f"  ⚠ Could not read {csv_path}: {e}")
            return cc_map

    elif choice == 1:
        print("  Paste CSV lines (CC,Parameter or Parameter,CC). Blank line to finish:")
        while True:
            line = input("    ").strip()
            if not line:
                break
            raw_lines.append(line)

    else:
        print("  Skipped.")
        return cc_map

    # Parse: detect column order from header or first data row
    if not raw_lines:
        return cc_map

    reader = csv.reader(StringIO("\n".join(raw_lines)))
    for row in reader:
        if len(row) < 2:
            continue
        a, b = row[0].strip(), row[1].strip()
        # Detect which column is CC (numeric)
        if a.isdigit():
            cc_num, param = int(a), b
        elif b.isdigit():
            cc_num, param = int(b), a
        else:
            continue    # header row or unparseable — skip
        if 0 <= cc_num <= 127 and param:
            cc_map[param] = cc_num

    if cc_map:
        print(f"  ✓ Loaded {len(cc_map)} CC assignments:")
        for param, cc in sorted(cc_map.items(), key=lambda x: x[1]):
            print(f"    CC {cc:3d}  →  {param}")
    else:
        print("  ⚠ No CC assignments parsed — check CSV format.")

    return cc_map


def step5_enumerate_params(info: dict, vst_path: Path,
                            cc_map: dict[str, int]) -> tuple[list[dict], dict]:
    """Load the VST, enumerate all parameters, generate and write profile.yaml."""
    try:
        import dawdreamer as daw
    except ImportError:
        raise RuntimeError("dawdreamer not available — run under conda env mimic-synth")

    print(f"  Loading {vst_path.name} …")
    sr     = 48000
    engine = daw.RenderEngine(sr, 512)
    plugin = engine.make_plugin_processor("probe", str(vst_path))
    engine.load_graph([(plugin, [])])

    n_total = plugin.get_plugin_parameter_size()
    params : list[dict]         = []
    reset  : dict[str, float]   = {}

    for i in range(n_total):
        name  = plugin.get_parameter_name(i)
        value = plugin.get_parameter(i)
        imp, log, cont = _classify(name)
        entry = {"name": name, "importance": imp,
                 "log_scale": log, "continuous": cont}
        if name in cc_map:
            entry["midi_cc"] = cc_map[name]
        params.append(entry)
        reset[name] = round(float(value), 6)

    sampled  = [p for p in params if p["importance"] > 0.0]
    excluded = [p for p in params if p["importance"] == 0.0]
    cc_count = sum(1 for p in params if "midi_cc" in p)

    print(f"  ✓ {n_total} total params  →  {len(sampled)} sampled  "
          f"|  {len(excluded)} excluded  |  {cc_count} with CC assignment")
    print(f"\n  Sampled parameters:")
    for p in sampled:
        tags = ""
        if p["log_scale"]:   tags += " [log]"
        if not p["continuous"]: tags += " [discrete]"
        if "midi_cc" in p:   tags += f" [CC{p['midi_cc']}]"
        print(f"    {p['name']:42s}  imp={p['importance']:.2f}{tags}")

    # Determine cross-platform plugin paths
    sys_name = platform.system()
    stem     = vst_path.name
    if sys_name == "Linux":
        p_linux = str(vst_path)
        p_mac   = f"/Library/Audio/Plug-Ins/VST3/{stem}"
        p_win   = f"C:/Program Files/Common Files/VST3/{stem}"
    elif sys_name == "Darwin":
        p_mac   = str(vst_path)
        p_linux = f"{Path.home()}/.vst3/{stem}"
        p_win   = f"C:/Program Files/Common Files/VST3/{stem}"
    else:
        p_win   = str(vst_path)
        p_linux = f"{Path.home()}/.vst3/{stem}"
        p_mac   = f"/Library/Audio/Plug-Ins/VST3/{stem}"

    # Build parameters block (sampled only)
    parameters: dict = {}
    for p in params:
        if p["importance"] <= 0.0:
            continue
        entry: dict = {
            "encoding":   "vst",
            "range":      [0.0, 1.0],
            "continuous": p["continuous"],
            "importance": p["importance"],
        }
        if p["log_scale"]:
            entry["log_scale"] = True
        if "midi_cc" in p:
            entry["midi_cc"] = p["midi_cc"]
        parameters[p["name"]] = entry

    # CMA-ES extra params: common tonal-richness params frozen in surrogate
    cmaes_extra: dict = {}
    for p in params:
        n = p["name"].lower()
        if any(kw in n for kw in ["filter env attack", "filter env decay",
                                   "filter env sustain", "filter env release",
                                   "amp env attack", "amp env sustain",
                                   "unison detune", "osc 2 volume",
                                   "ring mod", "cross mod"]):
            if p["name"] in parameters:
                cmaes_extra[p["name"]] = {"importance": p["importance"],
                                           "bounds": [0.0, 1.0]}

    profile = {
        "synth": {
            "id":                  info["synth_id"],
            "name":                info["synth_name"],
            "version":             info["synth_version"],
            "parameter_encoding":  "vst_automation",
            "transport":           "vst_host",
            "plugin_path_linux":   p_linux,
            "plugin_path_macos":   p_mac,
            "plugin_path_windows": p_win,
        },
        # Measured/confirmed properties of this synth's parameters.
        # Fill these in after consulting the synth manual and running calibration.
        "behavior": {
            "osc_pitch_range_semitones": 0,
            "osc_pitch_param_formula":   "param = 0.5 + semitones / (2 * range)",
            "pitch_bend_semitones_per_param": 0.0,
            "lfo_rate_hz_min": 0.0,
            "lfo_rate_hz_max": 0.0,
            "max_release_ring_sec": 0.0,
            "notes": [
                "TODO: fill in osc_pitch_range_semitones from synth manual",
                "TODO: measure pitch_bend_semitones_per_param (see calibrate_synth.py)",
                "TODO: measure lfo_rate_hz range",
                "TODO: set max_release_ring_sec (used by settle loop ceiling)",
                "TODO: document any parameters that must be pinned during inversion",
            ],
        },
        "parameters": parameters,
        "probe": {
            "notes":        list(range(24, 97, 3)),   # C1–C7 every 3 semitones
            "velocity":     100,
            "hold_sec":     1.5,
            "release_sec":  4.5,
            "pre_roll_sec": 0.2,
            "render_sec":   6.1,
            "sample_rate":  48000,
        },
        "reset":              reset,
        "cmaes_extra_params": cmaes_extra,
    }

    profile_path = info["project_dir"] / _PROFILE_DIR / "profile.yaml"
    with open(profile_path, "w") as f:
        yaml.dump(profile, f, default_flow_style=False, sort_keys=False)
    print(f"\n  ✓ Written {profile_path}")

    return params, reset


def step6_midi_ports() -> None:
    """Detect and display available MIDI input ports."""
    try:
        import rtmidi
        midi_in = rtmidi.MidiIn()
        ports   = midi_in.get_ports()
        if ports:
            print(f"  Found {len(ports)} MIDI port(s):")
            for p in ports:
                print(f"    • {p}")
        else:
            print("  No MIDI ports active.")
    except ImportError:
        print("  python-rtmidi not installed — install with: pip install python-rtmidi")
        print("  For VST synthesis this is optional; MIDI is generated programmatically.")
    except Exception as e:
        print(f"  MIDI detection error: {e}")


def step7_validate(info: dict) -> bool:
    """Test-render a single note; confirm non-silent output."""
    profile_path = info["project_dir"] / _PROFILE_DIR / "profile.yaml"
    if not profile_path.exists():
        print("  ⚠ profile.yaml not found — skipping")
        return False

    try:
        import dawdreamer as daw
    except ImportError:
        print("  ⚠ dawdreamer not available — skipping")
        return False

    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    sys_name = platform.system()
    key      = {"Darwin": "plugin_path_macos",
                "Windows": "plugin_path_windows",
                "Linux": "plugin_path_linux"}.get(sys_name, "plugin_path_linux")
    vst_path = profile["synth"].get(key, "")

    if not Path(vst_path).exists():
        print(f"  ⚠ VST not found at {vst_path}")
        return False

    try:
        sr     = 48000
        engine = daw.RenderEngine(sr, 512)
        plugin = engine.make_plugin_processor("val", vst_path)
        engine.load_graph([(plugin, [])])

        nidx = {plugin.get_parameter_name(i): i
                for i in range(plugin.get_plugin_parameter_size())}
        for name, val in profile.get("reset", {}).items():
            if name in nidx:
                plugin.set_parameter(nidx[name], float(val))

        plugin.clear_midi()
        plugin.add_midi_note(60, 100, 0.05, 1.5)
        engine.render(2.5)
        audio = plugin.get_audio()
        mono  = audio[0] if audio.ndim > 1 else audio
        rms   = float(np.sqrt(np.mean(mono ** 2)))
        peak  = float(np.max(np.abs(mono)))

        if peak < 1e-4:
            print(f"  ✗ FAILED — output is silent (peak={peak:.6f})")
            print("    Check plugin path and reset values in profile.yaml")
            return False

        print(f"  ✓ Passed — RMS={rms:.4f}  peak={peak:.4f}  (C4, 1.5s hold)")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def step8_calibration(info: dict, args) -> None:
    """Run calibrate_synth.py --all for this project's profile."""
    interactive  = not args.no_interactive
    profile_path = info["project_dir"] / _PROFILE_DIR / "profile.yaml"
    cal_path     = info["project_dir"] / _PROFILE_DIR / "calibration.npz"

    if interactive:
        raw = _prompt("Run full calibration now? (~20 min)", "y")
        if raw.lower() != "y":
            _print_cal_cmd(profile_path, cal_path)
            return

    print(f"  Profile : {profile_path}")
    print(f"  Output  : {cal_path}\n")
    cmd = [sys.executable, str(REPO_ROOT / "s01_setup" / "calibrate_synth.py"),
           "--all", "--profile", str(profile_path), "--out", str(cal_path)]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print("  ⚠ Calibration exited with errors — re-run manually when ready")
        _print_cal_cmd(profile_path, cal_path)
    else:
        _update_project_status(info["project_dir"], "Calibration", "✅ Complete")
        print(f"  ✓ Calibration complete → {cal_path}")


def _print_cal_cmd(profile_path: Path, cal_path: Path) -> None:
    print(f"\n  To run calibration later:")
    print(f"    conda activate mimic-synth")
    print(f"    python calibrate_synth.py --all \\")
    print(f"        --profile {profile_path} \\")
    print(f"        --out {cal_path}")


def _update_project_status(project_dir: Path, stage: str, status: str) -> None:
    p = project_dir / _PROFILE_DIR / "PROJECT_STATUS.md"
    if not p.exists():
        return
    text = p.read_text()
    text = text.replace(f"| {stage} | 🔲 Pending |",
                        f"| {stage} | {status} |")
    text = text.replace(f"| {stage} | 🔄 In progress |",
                        f"| {stage} | {status} |")
    p.write_text(text)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="MimicSynth project setup wizard")
    ap.add_argument("--project-name",  default=None,
                    help="Synth / project name (e.g. 'OB-Xf' or 'Minimoog_Model_D')")
    ap.add_argument("--data-root",     default=None,
                    help="Root dir for project data (default: /mnt/d/Mimic-Synth-Data)")
    ap.add_argument("--vst",           default=None,
                    help="Path to .vst3 file (skips auto-discovery)")
    ap.add_argument("--no-interactive", action="store_true",
                    help="Non-interactive: use defaults and CLI args only")
    ap.add_argument("--no-calibrate",  action="store_true",
                    help="Skip calibration step (run calibrate_synth.py later)")
    ap.add_argument("--no-validate",   action="store_true",
                    help="Skip test render")
    args = ap.parse_args()

    print("=" * 62)
    print("  MimicSynth — Project Setup Wizard")
    print("=" * 62)

    # ── Step 1: Project info ──────────────────────────────────────────────────
    print("\n=== Step 1: Project Information ===")
    info = step1_project_info(args)

    # ── Step 2: Folder structure ──────────────────────────────────────────────
    print("\n=== Step 2: Folder Structure ===")
    step2_folder_structure(info)

    # ── Step 3: VST discovery ─────────────────────────────────────────────────
    print("\n=== Step 3: VST3 Discovery ===")
    vst_path = step3_vst_discovery(info, args)

    # ── Step 4: MIDI CC mapping ───────────────────────────────────────────────
    print("\n=== Step 4: MIDI CC Mapping ===")
    cc_map = step4_midi_cc_mapping(info, args)

    # ── Step 5: Parameter enumeration + profile.yaml ─────────────────────────
    print("\n=== Step 5: Parameter Enumeration & Profile ===")
    params, reset = step5_enumerate_params(info, vst_path, cc_map)
    _update_project_status(info["project_dir"], "S01 Profile", "✅ Complete")

    # ── Step 6: MIDI port detection ───────────────────────────────────────────
    print("\n=== Step 6: MIDI Port Detection ===")
    step6_midi_ports()

    # ── Step 7: Validation render ─────────────────────────────────────────────
    if not args.no_validate:
        print("\n=== Step 7: Validation Render ===")
        step7_validate(info)
    else:
        print("\n=== Step 7: Validation (skipped) ===")

    # ── Step 8: Calibration ───────────────────────────────────────────────────
    if not args.no_calibrate:
        print("\n=== Step 8: Calibration (~20 min) ===")
        step8_calibration(info, args)
    else:
        print("\n=== Step 8: Calibration (skipped) ===")
        _print_cal_cmd(info["project_dir"] / _PROFILE_DIR / "profile.yaml",
                       info["project_dir"] / _PROFILE_DIR / "calibration.npz")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Setup complete!")
    print("=" * 62)
    n_sampled = sum(1 for p in params if p["importance"] > 0)
    n_cc      = sum(1 for p in params if "midi_cc" in p)
    print(f"""
  Project     : {info['project_dir']}
  Profile     : {info['project_dir'] / _PROFILE_DIR / 'profile.yaml'}
                ({n_sampled} sampled params, {n_cc} with CC assignments)
  Calibration : {info['project_dir'] / _PROFILE_DIR / 'calibration.npz'}

  Next steps:
    1. Review profile.yaml — tune importance weights, verify log_scale flags
    2. Update defaults.py:
         DATA_ROOT    = Path("{info['data_root']}")
         PROJECT_NAME = "{info['project_name']}"
    3. Start capture:
         conda activate mimic-synth
         python s02_capture/capture_v1_2.py
""")


if __name__ == "__main__":
    main()

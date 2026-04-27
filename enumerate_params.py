"""
Enumerate all parameters exposed by OB-Xf at runtime.
Run this once to reconcile s01_profiles/obxf.yaml against the installed plugin build.

Usage:
    python enumerate_params.py
    python enumerate_params.py --filter cut     # case-insensitive substring filter
"""
import argparse
import platform
from pathlib import Path
import dawdreamer as daw
from defaults import SAMPLE_RATE, BUFFER_SIZE

PLUGIN_PATHS = {
    "Darwin":  "/Library/Audio/Plug-Ins/VST3/OB-Xf.vst3",
    "Windows": "C:/Program Files/Common Files/VST3/OB-Xf.vst3",
    "Linux":   "~/.vst3/OB-Xf.vst3",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default="", help="Substring filter for param names")
    args = parser.parse_args()

    sys = platform.system()
    plugin_path = Path(PLUGIN_PATHS[sys]).expanduser()
    if not plugin_path.exists():
        raise FileNotFoundError(f"OB-Xf not found at {plugin_path}")

    engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
    synth = engine.make_plugin_processor("obxf", str(plugin_path))
    n = synth.get_plugin_parameter_size()

    print(f"OB-Xf exposes {n} parameters at {plugin_path}\n")
    print(f"{'idx':>4}  {'name':<50}  {'value':>8}")
    print("-" * 68)

    filt = args.filter.lower()
    for i in range(n):
        name = synth.get_parameter_name(i)
        value = synth.get_parameter(i)
        if filt and filt not in name.lower():
            continue
        print(f"{i:>4}  {name:<50}  {value:>8.4f}")

if __name__ == "__main__":
    main()

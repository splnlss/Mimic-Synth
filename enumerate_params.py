"""
Enumerate all parameters exposed by the VST plugin at runtime.
Run this once to reconcile the profile against the installed plugin build.

Usage:
    python enumerate_params.py
    python enumerate_params.py --filter cut     # case-insensitive substring filter
    python enumerate_params.py --profile path/to/profile.yaml
"""
import argparse
import platform
import yaml
from pathlib import Path
import dawdreamer as daw
import sys

sys.path.insert(0, str(Path(__file__).parent))
from defaults import SAMPLE_RATE, BUFFER_SIZE, PROFILE_PATH


def _resolve_plugin_path(profile: dict) -> Path:
    key = {
        "Darwin":  "plugin_path_macos",
        "Windows": "plugin_path_windows",
        "Linux":   "plugin_path_linux",
    }.get(platform.system())
    path_str = profile.get("synth", {}).get(key)
    if not path_str:
        raise RuntimeError(f"Profile missing plugin path for {platform.system()}")
    return Path(path_str).expanduser()


def main():
    parser = argparse.ArgumentParser(description="List all VST parameters from the active profile")
    parser.add_argument("--filter", default="", help="Substring filter for param names")
    parser.add_argument("--profile", default=None, help="Path to profile YAML")
    args = parser.parse_args()

    profile_path = Path(args.profile) if args.profile else PROFILE_PATH
    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    plugin_path = _resolve_plugin_path(profile)
    if not plugin_path.exists():
        synth_name = profile.get("synth", {}).get("name", "VST plugin")
        raise FileNotFoundError(f"{synth_name} not found at {plugin_path}")

    synth_id = profile.get("synth", {}).get("id", "synth")
    synth_name = profile.get("synth", {}).get("name", plugin_path.name)

    engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
    synth = engine.make_plugin_processor(synth_id, str(plugin_path))
    n = synth.get_plugin_parameter_size()

    print(f"{synth_name} exposes {n} parameters at {plugin_path}\n")
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

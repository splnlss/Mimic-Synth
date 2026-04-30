#!/usr/bin/env python3
"""Render the crane bird scream inverted patch."""

import yaml
import platform
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import dawdreamer as daw
from defaults import S06_PATCHES_DIR, PROFILE_PATH

def main():
    # Paths
    patch_path = S06_PATCHES_DIR / "816426_crane-bird-scream_mono" / "best_patch.yaml"
    profile_path = PROFILE_PATH
    out_path = S06_PATCHES_DIR / "816426_crane-bird-scream_mono" / "rendered.wav"
    
    # Load patch and profile
    with open(patch_path) as f:
        patch = yaml.safe_load(f)
    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    
    print(f"Rendering patch for note {patch['note']}")
    print(f"Score: {patch['score']}")
    
    # Get plugin path with absolute path
    plat = platform.system()
    key = {"Darwin": "plugin_path_macos", "Windows": "plugin_path_windows", "Linux": "plugin_path_linux"}[plat]
    plugin_path_str = profile["synth"][key]
    
    # Expand ~ to actual home directory
    if plugin_path_str.startswith("~"):
        # Use actual user home, not HERMES home
        actual_home = Path("/home/sanss").expanduser().resolve()
        plugin_path = Path(str(plugin_path_str).replace("~", str(actual_home)))
    else:
        plugin_path = Path(plugin_path_str).expanduser()
    
    print(f"Plugin path: {plugin_path}")
    print(f"Plugin exists: {plugin_path.exists()}")
    
    # Setup DawDreamer
    SR = int(profile["probe"]["sample_rate"])
    BUFFER = 512
    pre_roll = float(profile["probe"]["pre_roll_sec"])
    hold = float(profile["probe"]["hold_sec"])
    release = float(profile["probe"]["release_sec"])
    render_sec = pre_roll + hold + release
    note = int(patch["note"])
    velocity = int(profile["probe"]["velocity"])
    
    print(f"Sample rate: {SR}, Buffer: {BUFFER}")
    print(f"Render time: {render_sec}s (pre: {pre_roll}, hold: {hold}, release: {release})")
    
    try:
        engine = daw.RenderEngine(SR, BUFFER)
        print(f"Created engine, loading plugin...")
        synth = engine.make_plugin_processor("obxf", str(plugin_path))
        print(f"Plugin loaded successfully")
        
        n = synth.get_plugin_parameter_size()
        print(f"Plugin has {n} parameters")
        
        # Build name index
        name_idx = {synth.get_parameter_name(i): i for i in range(n)}
        print(f"Built name index with {len(name_idx)} parameters")
        
        # Apply reset values first
        reset_vals = profile.get("reset", {})
        for name, val in reset_vals.items():
            if name in name_idx:
                synth.set_parameter(name_idx[name], float(val))
        
        # Apply patch parameters
        params = patch["params"]
        for name, val in params.items():
            if name in name_idx:
                synth.set_parameter(name_idx[name], float(val))
                print(f"  Set {name} = {val}")
            else:
                print(f"  Warning: Parameter {name} not found in plugin")
        
        # Load graph
        engine.load_graph([(synth, [])])
        
        # Render
        synth.clear_midi()
        synth.add_midi_note(note, velocity, pre_roll, hold)
        engine.set_bpm(120)
        print(f"Rendering note {note}...")
        engine.render(render_sec)
        
        # Get audio
        audio = engine.get_audio()  # (channels, samples)
        if audio.ndim == 2:
            audio = audio.mean(axis=0)  # mono
        audio = audio.astype(np.float32)
        
        print(f"Audio shape: {audio.shape}, max: {np.abs(audio).max():.6f}, RMS: {np.sqrt(np.mean(audio**2)):.6f}")
        
        # Save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, audio, SR)
        print(f"Saved to {out_path}")
        
        # Also save normalized version
        target_path = Path(patch["target"])
        if target_path.exists():
            target_audio, sr = sf.read(target_path, dtype="float32", always_2d=False)
            if target_audio.ndim == 2:
                target_audio = target_audio.mean(axis=1)
            target_rms = np.sqrt(np.mean(target_audio**2))
            rendered_rms = np.sqrt(np.mean(audio**2))
            if rendered_rms > 0 and target_rms > 0:
                normalized = audio * (target_rms / rendered_rms)
                norm_path = out_path.parent / "rendered_normalized.wav"
                sf.write(norm_path, normalized, SR)
                print(f"Saved normalized audio to {norm_path}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
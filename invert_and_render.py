#!/usr/bin/env python3
"""Invert and render a target audio file, handling stereo-to-mono conversion."""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import yaml
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from defaults import TARGETS_DIR, S05_RUNS_DIR, S06_PATCHES_DIR, PROFILE_PATH

from s04_embed.embed import Embedder
from s05_surrogate.model import Surrogate
from s06_invert.grad_search import grad_invert
from s06_invert.cmaes_search import cmaes_invert

def embed_target_mono(wav_path: Path, device: str) -> np.ndarray:
    """Embed target audio, converting stereo to mono if needed."""
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # average across channels
    embedder = Embedder(device=device)
    return embedder.encodec_embed(audio, sr, pool="mean")  # [128]

def load_surrogate(checkpoint_path: Path, device: str):
    import json
    manifest_path = checkpoint_path.parent / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    model = Surrogate(input_dim=manifest["input_dim"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model, manifest

def invert_and_render(target_wav_path, surrogate_path, profile_path, out_dir, note=None):
    """Run inversion and render the best patch."""
    target_wav_path = Path(target_wav_path)
    surrogate_path = Path(surrogate_path)
    profile_path = Path(profile_path)
    out_dir = Path(out_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load profile
    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    
    # Load surrogate
    surrogate, manifest = load_surrogate(surrogate_path, device)
    param_cols = manifest["param_cols"]
    d_params = len(param_cols)
    
    # Embed target (with stereo-to-mono fix)
    print(f"Embedding target: {target_wav_path.name}")
    target_emb_np = embed_target_mono(target_wav_path, device)
    target_emb = torch.tensor(target_emb_np, dtype=torch.float32)
    
    # Candidate notes
    candidate_notes = [note] if note is not None else profile["probe"]["notes"]
    print(f"Candidate notes: {candidate_notes}")
    
    # Inversion loop
    candidates = []
    for n in candidate_notes:
        print(f"  Note {n}: gradient descent...")
        grad_score, grad_params = grad_invert(
            surrogate, target_emb, n, d_params,
            n_starts=16, steps=300, device=device,
        )
        print(f"    grad score: {grad_score:.6f}")
        print(f"    CMA-ES refinement...")
        cmaes_score, cmaes_params_np = cmaes_invert(
            surrogate, target_emb, n, d_params, grad_params,
            maxiter=200, device=device,
        )
        print(f"    cmaes score: {cmaes_score:.6f}")
        candidates.append({
            "note": n,
            "method": "grad",
            "score": grad_score,
            "params": grad_params.numpy(),
        })
        candidates.append({
            "note": n,
            "method": "cmaes",
            "score": cmaes_score,
            "params": cmaes_params_np,
        })
    
    # Pick best
    candidates.sort(key=lambda c: c["score"])
    best = candidates[0]
    print(f"\nBest score: {best['score']:.6f} (note {best['note']}, method {best['method']})")
    
    # Create output directory
    target_stem = target_wav_path.stem
    run_dir = out_dir / target_stem
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save target embedding
    np.save(str(run_dir / "target_embedding.npy"), target_emb_np)
    
    # Save candidates
    rows = []
    for c in candidates:
        row = {"note": c["note"], "method": c["method"], "score": c["score"]}
        for col, val in zip(param_cols, c["params"]):
            row[col] = float(val)
        rows.append(row)
    pd.DataFrame(rows).to_parquet(run_dir / "candidates.parquet")
    
    # Save best patch
    best_patch = {
        "target": str(target_wav_path),
        "note": int(best["note"]),
        "method": best["method"],
        "score": float(best["score"]),
        "params": {col.removeprefix("p_"): float(v)
                   for col, v in zip(param_cols, best["params"])},
    }
    with open(run_dir / "best_patch.yaml", "w") as f:
        yaml.dump(best_patch, f, default_flow_style=False, sort_keys=False)
    
    print(f"Saved results to {run_dir}")
    
    # Render the best patch
    render_best_patch(best_patch, profile, run_dir)
    
    return best_patch, run_dir

def render_best_patch(best_patch, profile, run_dir):
    """Render the best patch using DawDreamer."""
    import dawdreamer as daw
    import platform
    
    # Get plugin path
    plat = platform.system()
    key = {"Darwin": "plugin_path_macos", "Windows": "plugin_path_windows", "Linux": "plugin_path_linux"}[plat]
    plugin_path_str = profile["synth"][key]
    plugin_path = Path(plugin_path_str).expanduser()
    
    # Setup DawDreamer
    SR = int(profile["probe"]["sample_rate"])
    BUFFER = 512
    pre_roll = float(profile["probe"]["pre_roll_sec"])
    hold = float(profile["probe"]["hold_sec"])
    release = float(profile["probe"]["release_sec"])
    render_sec = pre_roll + hold + release
    note = int(best_patch["note"])
    velocity = int(profile["probe"]["velocity"])
    
    print(f"\nRendering patch (note {note})...")
    print(f"  Sample rate: {SR}, buffer: {BUFFER}")
    print(f"  Render time: {render_sec}s")
    
    try:
        engine = daw.RenderEngine(SR, BUFFER)
        synth = engine.make_plugin_processor("obxf", str(plugin_path))
        n = synth.get_plugin_parameter_size()
        name_idx = {synth.get_parameter_name(i): i for i in range(n)}
        
        # Apply reset values
        reset_vals = profile.get("reset", {})
        for name, val in reset_vals.items():
            if name in name_idx:
                synth.set_parameter(name_idx[name], float(val))
        
        # Apply patch parameters
        params = best_patch["params"]
        for name, val in params.items():
            if name in name_idx:
                synth.set_parameter(name_idx[name], float(val))
            else:
                print(f"  Warning: Parameter '{name}' not found in plugin")
        
        # Load graph and render
        engine.load_graph([(synth, [])])
        synth.clear_midi()
        synth.add_midi_note(note, velocity, pre_roll, hold)
        engine.set_bpm(120)
        engine.render(render_sec)
        
        # Get audio
        audio = engine.get_audio()  # (channels, samples)
        if audio.ndim == 2:
            audio = audio.mean(axis=0)  # mono
        audio = audio.astype(np.float32)
        
        # Save
        out_path = run_dir / "rendered.wav"
        sf.write(out_path, audio, SR)
        print(f"  Saved rendered audio to {out_path}")
        
        # Normalize to match target RMS
        target_audio, sr = sf.read(best_patch["target"], dtype="float32", always_2d=False)
        if target_audio.ndim == 2:
            target_audio = target_audio.mean(axis=1)
        target_rms = np.sqrt(np.mean(target_audio**2))
        rendered_rms = np.sqrt(np.mean(audio**2))
        if rendered_rms > 0 and target_rms > 0:
            normalized = audio * (target_rms / rendered_rms)
            norm_path = run_dir / "rendered_normalized.wav"
            sf.write(norm_path, normalized, SR)
            print(f"  Saved normalized audio to {norm_path}")
        
        return out_path
        
    except Exception as e:
        print(f"  Error rendering: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    target = str(TARGETS_DIR / "816426_crane-bird-scream.wav")
    surrogate = str(S05_RUNS_DIR / "run_20260429_145056" / "state_dict.pt")
    profile = str(PROFILE_PATH)
    out_dir = str(S06_PATCHES_DIR)
    
    print(f"Target: {target}")
    print(f"Surrogate: {surrogate}")
    print(f"Profile: {profile}")
    print(f"Output dir: {out_dir}")
    print("-" * 50)
    
    best_patch, run_dir = invert_and_render(target, surrogate, profile, out_dir)
    
    print("\n✅ Done!")
    print(f"Best patch saved to {run_dir}/best_patch.yaml")
    print(f"Rendered audio in {run_dir}/")
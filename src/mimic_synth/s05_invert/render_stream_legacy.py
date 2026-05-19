import argparse
import pandas as pd
import numpy as np
import soundfile as sf
import dawdreamer as daw
from pathlib import Path
import yaml

def render_stream(
    parquet_path: Path,
    profile_path: Path,
    vst_path: Path,
    out_path: Path
):
    df = pd.DataFrame(pd.read_parquet(parquet_path))
    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    
    sr = profile.get("probe", {}).get("sample_rate", 48000)
    # Determine total duration from last timestamp + hop
    hop_sec = df["timestamp"].iloc[1] - df["timestamp"].iloc[0] if len(df) > 1 else 0.02
    total_sec = df["timestamp"].iloc[-1] + hop_sec
    
    engine = daw.RenderEngine(sr, 512)
    plugin = engine.make_plugin_processor("synth", str(vst_path))
    engine.load_graph([
        (plugin, [])
    ])
    
    # Use corrected DawDreamer API (from verified s06 render experience)
    num_params = plugin.get_plugin_parameter_size()
    param_name_to_index = {plugin.get_parameter_name(i): i for i in range(num_params)}
    
    # We'll automate parameters frame-by-frame
    # Parquet columns are like "p_Filter Cutoff"
    param_cols = [c for c in df.columns if c.startswith("p_")]
    
    # Set fixed MIDI note from profile or detected best (assuming note 84 was best)
    plugin.add_midi_note(84, 100, 0.0, total_sec)
    
    # We collect automation data for each parameter and apply it as an array
    automation_data = {p_idx: [] for p_idx in param_name_to_index.values()}
    timestamps = df["timestamp"].values

    for _, row in df.iterrows():
        for col in param_cols:
            p_name = col.removeprefix("p_")
            if p_name in param_name_to_index:
                p_idx = param_name_to_index[p_name]
                automation_data[p_idx].append(row[col])

    for p_idx, values in automation_data.items():
        if values:
            # DawDreamer set_automation expects a 2D array: [[timestamp, value], ...]
            data = np.column_stack((timestamps, values))
            plugin.set_automation(p_idx, data)
                
    print(f"Rendering stream: {total_sec:.2f}s...")
    engine.render(total_sec)
    audio = plugin.get_audio()
    
    # Save results
    sf.write(out_path, audio.transpose(), sr)
    
    # Normalization
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        norm_audio = audio / max_val * 0.5
        sf.write(str(out_path).replace(".wav", "_normalized.wav"), norm_audio.transpose(), sr)
    
    print(f"Stream render complete: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--vst", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    
    render_stream(Path(args.parquet), Path(args.profile), Path(args.vst), Path(args.out))

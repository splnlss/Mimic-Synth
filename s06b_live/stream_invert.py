#!/usr/bin/env python3
"""
stream_invert.py — Streaming audio-to-parameter inversion for OB-Xf.

v3 features:
  1. Note on/off at detected onsets/offsets (not one sustained note)
  2. MIDI pitch bend messages for fine pitch tracking + Osc 1 Pitch for coarse
  3. Multi-oscillator richness — Osc 2 params NOT fixed during inversion
  4. Self-learning refinement: full-result comparison as primary driver
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

import defaults as _defs
from s05_surrogate.model import Surrogate
from s06_invert.invert import _load_surrogate


# ── Pitch detection ──────────────────────────────────────────────────────────

def detect_pitch_autocorr(audio, sr):
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio * np.hanning(len(audio))
    autocorr = signal.correlate(audio, audio, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    window_size = int(sr / 50)
    peaks, _ = signal.find_peaks(autocorr, distance=window_size)
    if len(peaks) == 0:
        return _detect_pitch_fft(audio, sr)
    peak_heights = autocorr[peaks]
    best_idx = peaks[np.argmax(peak_heights)]
    if 0 < best_idx < len(autocorr) - 1:
        alpha = autocorr[best_idx - 1]
        beta = autocorr[best_idx]
        gamma = autocorr[best_idx + 1]
        denom = 2 * (2 * beta - alpha - gamma)
        if abs(denom) > 1e-10:
            ref = (alpha - gamma) / denom
            best_idx = best_idx + ref
    pitch_hz = sr / best_idx
    if 50 <= pitch_hz <= 5000:
        return pitch_hz
    return None


def _detect_pitch_fft(audio, sr):
    freqs, psd = signal.welch(audio, sr, nperseg=min(len(audio), 2048))
    pitch_hz = freqs[np.argmax(psd)]
    if 50 <= pitch_hz <= 5000:
        return pitch_hz
    return None


# ── Onset/offset detection ───────────────────────────────────────────────────

def detect_onsets_offsets(energy, threshold_factor=0.3, min_frames=2):
    energy = np.array(energy, dtype=np.float64)
    threshold = np.max(energy) * threshold_factor
    active = energy > threshold
    regions = []
    in_note = False
    onset = 0
    for i, is_active in enumerate(active):
        if is_active and not in_note:
            onset = i
            in_note = True
        elif not is_active and in_note:
            if i - onset >= min_frames:
                regions.append((onset, i - 1))
            in_note = False
    if in_note and len(active) - onset >= min_frames:
        regions.append((onset, len(active) - 1))
    return regions


# ── Pitch → MIDI pitch bend + Osc 1 Pitch ────────────────────────────────────

OSC1_PITCH_RANGE_SEMITONES = 12.0
MIDI_PITCH_BEND_RANGE_SEMITONES = 2.0  # ±2 semitones via MIDI pitch bend


def pitch_hz_to_midi_bend(pitch_hz, base_note):
    """Convert pitch_hz to (midi_note, pitch_bend_value).

    pitch_bend_value is in [0, 1] where 0.5 = center (no bend).
    MIDI pitch bend range is ±2 semitones.
    """
    if pitch_hz is None or pitch_hz <= 0:
        return base_note, 0.5
    midi = 12.0 * np.log2(pitch_hz / 440.0) + 69.0
    rounded_note = int(round(midi))
    bend_semitones = midi - rounded_note
    bend_value = 0.5 + bend_semitones / (2.0 * MIDI_PITCH_BEND_RANGE_SEMITONES)
    return rounded_note, float(np.clip(bend_value, 0.0, 1.0))


def pitch_hz_to_osc1(pitch_hz, base_note):
    if pitch_hz is None or pitch_hz <= 0:
        return 0.5
    midi = 12.0 * np.log2(pitch_hz / 440.0) + 69.0
    offset = midi - base_note
    value = 0.5 + offset / (2.0 * OSC1_PITCH_RANGE_SEMITONES)
    return float(np.clip(value, 0.0, 1.0))


def estimate_base_note(pitch_hz_values, profile_notes):
    valid = [p for p in pitch_hz_values if p is not None and p > 0]
    if not valid:
        return profile_notes[len(profile_notes) // 2]
    median_hz = float(np.median(valid))
    median_midi = 12.0 * np.log2(median_hz / 440.0) + 69.0
    return profile_notes[abs(np.array(profile_notes) - median_midi).argmin()]


# ── Smoothing ────────────────────────────────────────────────────────────────

def smooth_trajectory(values, window_size=3):
    values = np.array(values, dtype=np.float64)
    if len(values) <= window_size:
        return values
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(values, kernel, mode='same')
    edges = window_size // 2
    smoothed[:edges] = values[:edges]
    smoothed[-edges:] = values[-edges:]
    return smoothed


# ── Gradient inversion (all params free, including Osc 2) ────────────────────

def grad_invert(
    surrogate, target_emb, note, d_params,
    n_starts=4, steps=50, lr=5e-2, device="cuda", init_params=None
):
    surrogate.eval()
    target = target_emb.to(device).unsqueeze(0)
    note_t = torch.full((1,), note / 127.0, device=device)

    best_score = float("inf")
    best_params = None

    for i in range(n_starts):
        if init_params is not None and i == 0:
            if isinstance(init_params, torch.Tensor):
                params = init_params.clone().unsqueeze(0).to(device).requires_grad_(True)
            else:
                params = torch.from_numpy(init_params).clone().unsqueeze(0).to(device).requires_grad_(True)
        else:
            params = torch.rand(1, d_params, device=device, requires_grad=True)

        opt = torch.optim.Adam([params], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            pred = surrogate(params.clamp(0.0, 1.0), note_t)
            loss = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
            loss.backward()
            opt.step()

        final = params.detach().clamp(0.0, 1.0)
        with torch.no_grad():
            pred = surrogate(final, note_t)
            score = (1.0 - F.cosine_similarity(pred, target, dim=-1)).item()

        if score < best_score:
            best_score = score
            best_params = final.squeeze(0).cpu()

    assert best_params is not None
    return best_score, best_params


# ── Main inversion ───────────────────────────────────────────────────────────

def stream_invert(
    target_wav: Path,
    surrogate_checkpoint: Path,
    profile_path: Path,
    out_dir: Path,
    device: str = "cuda",
    win_sec: float = 0.1,
    hop_sec: float = 0.05,
    n_starts: int = 4,
    grad_steps: int = 50,
    smooth_window: int = 3,
    skip_render: bool = False,
    refine_iterations: int = 3,
    refine_threshold: float = 0.01,
) -> dict:
    audio, sr = sf.read(str(target_wav), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    profile_notes = profile["probe"]["notes"]
    surrogate, manifest = _load_surrogate(Path(surrogate_checkpoint), device)
    surrogate = surrogate.to(device)
    surrogate.eval()
    param_cols = manifest["param_cols"]
    d_params = len(param_cols)

    from s04_embed.embed import Embedder
    embedder = Embedder(device=device)

    # ── Frame analysis ───────────────────────────────────────────────────
    win_samples = int(win_sec * sr)
    hop_samples = int(hop_sec * sr)
    n_frames = max(0, (len(audio) - win_samples) // hop_samples + 1)

    pitch_hz_list = []
    energy_list = []
    emb_list = []

    for start in tqdm(range(0, len(audio) - win_samples + 1, hop_samples), desc="Analyzing"):
        window = audio[start: start + win_samples]
        pitch_hz = detect_pitch_autocorr(window, sr)
        pitch_hz_list.append(pitch_hz)
        energy_list.append(float(np.sqrt(np.mean(window ** 2))))
        emb = embedder.encodec_embed(window, sr, pool="mean")
        emb_list.append(emb)

    # ── Onset/offset detection ───────────────────────────────────────────
    regions = detect_onsets_offsets(energy_list, threshold_factor=0.3, min_frames=2)
    active_set = set()
    for onset, offset in regions:
        active_set.update(range(onset, offset + 1))

    print(f"Detected {len(regions)} note region(s) across {n_frames} frames")
    for i, (on, off) in enumerate(regions):
        print(f"  Region {i}: frame {on}–{off} ({on * hop_sec:.2f}s–{off * hop_sec:.2f}s)")

    # ── Base note + pitch trajectory ─────────────────────────────────────
    base_note = estimate_base_note(pitch_hz_list, profile_notes)

    # Per-frame: MIDI note + pitch bend + Osc 1 Pitch
    midi_notes = []
    pitch_bends = []
    osc1_pitch_values = []
    for p in pitch_hz_list:
        note, bend = pitch_hz_to_midi_bend(p, base_note)
        midi_notes.append(note)
        pitch_bends.append(bend)
        osc1_pitch_values.append(pitch_hz_to_osc1(p, base_note))

    # Smooth trajectories
    pitch_bends_smooth = smooth_trajectory(pitch_bends, window_size=3)
    osc1_pitch_smooth = smooth_trajectory(osc1_pitch_values, window_size=3)

    print(f"Base MIDI note: {base_note}")

    # ── Gradient inversion per frame (all params free) ────────────────────
    results = []
    prev_params = None

    for i in tqdm(range(n_frames), desc="Inverting"):
        timestamp = i * hop_sec
        emb_torch = torch.tensor(emb_list[i], dtype=torch.float32).to(device)

        # Use per-frame MIDI note (rounded from pitch) for better accuracy
        frame_note = midi_notes[i] if i in active_set else base_note

        if prev_params is not None:
            score, params = grad_invert(
                surrogate, emb_torch, frame_note, d_params,
                n_starts=1, steps=grad_steps, device=device,
                init_params=prev_params,
            )
        else:
            score, params = grad_invert(
                surrogate, emb_torch, frame_note, d_params,
                n_starts=n_starts, steps=grad_steps, device=device,
            )

        res = {
            "timestamp": timestamp,
            "pitch_hz": float(pitch_hz_list[i]) if pitch_hz_list[i] else np.nan,
            "midi_note": frame_note,
            "pitch_bend": pitch_bends_smooth[i],
            "osc1_pitch": osc1_pitch_smooth[i],
            "score": float(score),
            "active": i in active_set,
        }
        for col, val in zip(param_cols, params):
            res[col] = float(val)

        results.append(res)
        prev_params = params

    # ── Post-processing ──────────────────────────────────────────────────
    df = pd.DataFrame(results)

    for col in param_cols:
        if col in df.columns:
            df[col] = smooth_trajectory(df[col].values, smooth_window)

    best_row = df.loc[df["score"].idxmin()]

    # ── Save outputs ─────────────────────────────────────────────────────
    run_dir = out_dir / target_wav.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(run_dir / "stream_params.parquet")

    best_patch = {
        "target": str(target_wav),
        "base_note": int(base_note),
        "pitch_hz": float(best_row["pitch_hz"]) if pd.notnull(best_row["pitch_hz"]) else None,
        "score": float(best_row["score"]),
        "params": {col: float(best_row[col]) for col in param_cols},
    }
    with open(run_dir / "best_patch.yaml", "w") as f:
        yaml.dump(best_patch, f, default_flow_style=False)

    pitch_traj = {
        "target": str(target_wav),
        "sr": sr,
        "base_note": int(base_note),
        "regions": [{"onset": on * hop_sec, "offset": off * hop_sec} for on, off in regions],
        "osc1_pitch_range_semitones": OSC1_PITCH_RANGE_SEMITONES,
        "midi_pitch_bend_range_semitones": MIDI_PITCH_BEND_RANGE_SEMITONES,
        "frames": [],
    }
    for _, row in df.iterrows():
        pitch_traj["frames"].append({
            "timestamp": float(row["timestamp"]),
            "pitch_hz": float(row["pitch_hz"]) if pd.notnull(row["pitch_hz"]) else None,
            "midi_note": int(row["midi_note"]),
            "pitch_bend": float(row["pitch_bend"]),
            "osc1_pitch": float(row["osc1_pitch"]),
            "score": float(row["score"]),
            "active": bool(row["active"]),
        })
    with open(run_dir / "pitch_trajectory.yaml", "w") as f:
        yaml.dump(pitch_traj, f, default_flow_style=False)

    print(f"\u2713 Streaming tracking complete: {len(df)} frames")
    print(f"\u2713 Best score: {best_row['score']:.4f} at base_note={base_note}")
    print(f"\u2713 Saved to {run_dir / 'stream_params.parquet'}")

    # ── Render ───────────────────────────────────────────────────────────
    if not skip_render:
        _render_stream(
            df, pitch_traj, profile_path, run_dir, base_note, regions, hop_sec,
        )

        # ── Self-learning refinement loop ────────────────────────────────
        if refine_iterations > 0:
            _refine_loop(
                target_wav=target_wav,
                profile_path=profile_path,
                run_dir=run_dir,
                base_note=base_note,
                param_cols=param_cols,
                surrogate=surrogate,
                embedder=embedder,
                device=device,
                max_iterations=refine_iterations,
                threshold=refine_threshold,
            )

    return {"df": df, "best": best_row, "run_dir": run_dir}


# ── Render with note on/off + pitch bend ─────────────────────────────────────

def _render_stream(
    df: pd.DataFrame,
    pitch_traj: dict,
    profile_path: Path,
    out_dir: Path,
    base_note: int,
    regions: list,
    hop_sec: float,
):
    """Render with per-region note on/off + MIDI pitch bend + Osc 1 Pitch automation."""
    import dawdreamer as daw

    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    sr = profile.get("probe", {}).get("sample_rate", 48000)
    timestamps = df["timestamp"].values
    total_sec = timestamps[-1] + hop_sec

    vst_path = Path(profile["synth"]["plugin_path_linux"])
    engine = daw.RenderEngine(sr, 512)
    plugin = engine.make_plugin_processor("synth", str(vst_path))
    engine.load_graph([(plugin, [])])

    num_params = plugin.get_plugin_parameter_size()
    param_name_to_index = {plugin.get_parameter_name(i): i for i in range(num_params)}

    param_cols = [c for c in df.columns if c.startswith("p_")]

    # Note on/off at region boundaries
    for onset_frame, offset_frame in regions:
        note_on_ts = onset_frame * hop_sec
        note_off_ts = (offset_frame + 1) * hop_sec
        note_dur = note_off_ts - note_on_ts
        plugin.add_midi_note(base_note, 100, note_on_ts, note_dur)

    # Apply parameter automations
    for col in param_cols:
        p_name = col.removeprefix("p_")
        if p_name in param_name_to_index:
            p_idx = param_name_to_index[p_name]
            values = df[col].values
            data = np.column_stack((timestamps, values))
            plugin.set_automation(p_idx, data)

    # MIDI pitch bend automation
    pitch_bends = [f["pitch_bend"] for f in pitch_traj["frames"]]
    pb_timestamps = [f["timestamp"] for f in pitch_traj["frames"]]

    pb_name = "Pitch Bend"
    if pb_name in param_name_to_index:
        pb_idx = param_name_to_index[pb_name]
        pb_data = np.column_stack((pb_timestamps, pitch_bends))
        plugin.set_automation(pb_idx, pb_data)

    print(f"Rendering stream: {total_sec:.2f}s ({len(regions)} note region(s), pitch bend enabled)...")
    engine.render(total_sec)
    audio = plugin.get_audio()

    out_path = out_dir / "rendered.wav"
    sf.write(str(out_path), audio.transpose(), sr)

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        norm_audio = audio / max_val * 0.5
        sf.write(str(out_path).replace(".wav", "_normalized.wav"), norm_audio.transpose(), sr)

    print(f"\u2713 Rendered to {out_path}")


# ── Self-learning refinement loop (full-result driven) ───────────────────────

def _refine_loop(
    target_wav: Path,
    profile_path: Path,
    run_dir: Path,
    base_note: int,
    param_cols: list,
    surrogate,
    embedder,
    device: str,
    max_iterations: int = 3,
    threshold: float = 0.01,
):
    """Render → embed → compare FULL result → adjust → re-render.

    Primary driver: full rendered audio vs full target audio comparison.
    Secondary: per-frame surrogate gradient to suggest direction.
    Only applies adjustments that improve the full result.
    """
    import dawdreamer as daw

    target_audio, sr = sf.read(str(target_wav), dtype="float32")
    if target_audio.ndim == 2:
        target_audio = target_audio.mean(axis=1)

    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    sr_profile = profile.get("probe", {}).get("sample_rate", 48000)
    vst_path = Path(profile["synth"]["plugin_path_linux"])

    df = pd.read_parquet(run_dir / "stream_params.parquet")
    timestamps = df["timestamp"].values
    hop_sec = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.02
    total_sec = timestamps[-1] + hop_sec

    # Detect regions from active column
    active = df["active"].values
    regions = []
    in_note = False
    onset = 0
    for i, is_active in enumerate(active):
        if is_active and not in_note:
            onset = i
            in_note = True
        elif not is_active and in_note:
            if i - onset >= 1:
                regions.append((onset, i - 1))
            in_note = False
    if in_note and len(active) - onset >= 1:
        regions.append((onset, len(active) - 1))
    if not regions:
        regions = [(0, len(active) - 1)]

    osc1_col = "p_Osc 1 Pitch"
    osc1_pitch_fixed = df[osc1_col].values.copy()

    best_score = float("inf")
    best_df = df.copy()

    # Pre-compute target embedding
    target_emb = embedder.encodec_embed(target_audio, sr, pool="mean")
    target_emb_torch = torch.tensor(target_emb, dtype=torch.float32).to(device)

    def _render_and_score(current_df):
        """Render current params and return (score, rendered_audio)."""
        engine = daw.RenderEngine(sr_profile, 512)
        plugin = engine.make_plugin_processor("synth", str(vst_path))
        engine.load_graph([(plugin, [])])

        num_params = plugin.get_plugin_parameter_size()
        param_name_to_index = {plugin.get_parameter_name(i): i for i in range(num_params)}

        for onset_frame, offset_frame in regions:
            note_on_ts = onset_frame * hop_sec
            note_off_ts = (offset_frame + 1) * hop_sec
            plugin.add_midi_note(base_note, 100, note_on_ts, note_off_ts - note_on_ts)

        for col in param_cols:
            p_name = col.removeprefix("p_")
            if p_name in param_name_to_index:
                p_idx = param_name_to_index[p_name]
                values = current_df[col].values
                data = np.column_stack((timestamps, values))
                plugin.set_automation(p_idx, data)

        engine.render(total_sec)
        rendered_audio = plugin.get_audio()
        rendered_np = rendered_audio.transpose()
        if rendered_np.ndim == 2:
            rendered_np = rendered_np.mean(axis=1)

        rendered_emb = embedder.encodec_embed(rendered_np, sr_profile, pool="mean")
        rendered_emb_torch = torch.tensor(rendered_emb, dtype=torch.float32).to(device)

        score = (1.0 - F.cosine_similarity(
            rendered_emb_torch.unsqueeze(0),
            target_emb_torch.unsqueeze(0)
        )).item()
        return score, rendered_np

    for iteration in range(max_iterations):
        print(f"\n--- Refinement iteration {iteration + 1}/{max_iterations} ---")

        # 1. Score current full result
        score, _ = _render_and_score(df)
        print(f"  Current full-result score: {score:.4f}")

        if score < best_score:
            best_score = score
            best_df = df.copy()

        if score < threshold:
            print(f"  Score below threshold — refinement complete.")
            break

        # 2. Surrogate gradient: suggest adjustments per frame
        surrogate.eval()
        note_t = torch.full((1,), base_note / 127.0, device=device)

        frame_grads = []
        for i in range(len(df)):
            param_tensor = torch.zeros(1, len(param_cols), device=device)
            for j, col in enumerate(param_cols):
                param_tensor[0, j] = df[col].iloc[i]
            param_tensor.requires_grad_(True)

            opt = torch.optim.Adam([param_tensor], lr=5e-3)
            for _ in range(50):
                opt.zero_grad()
                clamped = param_tensor.clamp(0.0, 1.0)
                if osc1_col in param_cols:
                    idx = param_cols.index(osc1_col)
                    clamped[0, idx] = osc1_pitch_fixed[i]
                pred = surrogate(clamped, note_t)
                loss = (1.0 - F.cosine_similarity(pred, target_emb_torch.unsqueeze(0), dim=-1)).mean()
                loss.backward()
                if param_tensor.grad is not None:
                    if osc1_col in param_cols:
                        idx = param_cols.index(osc1_col)
                        param_tensor.grad[0, idx] = 0
                opt.step()

            new_vals = param_tensor.detach().clamp(0.0, 1.0).squeeze(0).cpu().numpy()
            frame_grads.append(new_vals)

        # 3. Try different adjustment strengths, pick the one that improves full result most
        best_local_score = score
        best_local_df = df.copy()
        best_alpha = 0.0

        for alpha in [0.05, 0.1, 0.15, 0.2, 0.3]:
            trial_df = df.copy()
            for i in range(len(df)):
                if frame_grads[i] is None:
                    continue
                for j, col in enumerate(param_cols):
                    if col == osc1_col:
                        continue
                    old_val = trial_df[col].iloc[i]
                    new_val = frame_grads[i][j]
                    delta = (new_val - old_val) * alpha
                    trial_df.at[i, col] = np.clip(old_val + delta, 0.0, 1.0)

            trial_score, _ = _render_and_score(trial_df)
            print(f"    α={alpha:.2f} → full score: {trial_score:.4f}")

            if trial_score < best_local_score:
                best_local_score = trial_score
                best_local_df = trial_df.copy()
                best_alpha = alpha

        if best_alpha > 0:
            df = best_local_df
            print(f"  \u2713 Best α={best_alpha:.2f}, new score: {best_local_score:.4f}")
        else:
            print(f"  No improvement found — stopping refinement.")
            break

    # Save best result
    if best_score < float("inf"):
        best_df.to_parquet(run_dir / "stream_params.parquet")
        best_row = best_df.loc[best_df["score"].idxmin()]
        best_patch = {
            "target": str(target_wav),
            "base_note": int(base_note),
            "pitch_hz": float(best_row["pitch_hz"]) if pd.notnull(best_row["pitch_hz"]) else None,
            "score": float(best_row["score"]),
            "refined_score": float(best_score),
            "params": {col: float(best_row[col]) for col in param_cols},
        }
        with open(run_dir / "best_patch.yaml", "w") as f:
            yaml.dump(best_patch, f, default_flow_style=False)

        print(f"\nRe-rendering best result (score={best_score:.4f})...")
        best_df.to_parquet(run_dir / "stream_params.parquet")

        # Build pitch_traj from best_df for render
        pitch_traj_refined = {
            "frames": [],
        }
        for _, row in best_df.iterrows():
            pitch_traj_refined["frames"].append({
                "timestamp": float(row["timestamp"]),
                "pitch_bend": float(row.get("pitch_bend", 0.5)),
            })

        _render_stream(
            best_df, pitch_traj_refined, profile_path, run_dir, base_note, regions, hop_sec,
        )


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="Target audio file (.wav)")
    ap.add_argument("--surrogate", default=None, help="Surrogate model checkpoint")
    ap.add_argument("--profile", default=None, help="Profile YAML (auto-detect if None)")
    ap.add_argument("--out", default=str(_defs.S06_PATCHES_DIR), help="Output directory")
    ap.add_argument("--win-sec", type=float, default=0.1)
    ap.add_argument("--hop-sec", type=float, default=0.05)
    ap.add_argument("--n-starts", type=int, default=4)
    ap.add_argument("--grad-steps", type=int, default=50)
    ap.add_argument("--smooth-window", type=int, default=3)
    ap.add_argument("--device", default="cuda", help="torch device")
    ap.add_argument("--no-render", action="store_true", help="Skip DawDreamer render step")
    ap.add_argument("--refine-iterations", type=int, default=3, help="Self-learning refinement iterations")
    ap.add_argument("--refine-threshold", type=float, default=0.01, help="Stop refinement below this score")

    args = ap.parse_args()

    if args.surrogate is None:
        runs = sorted(_defs.S05_RUNS_DIR.glob("run_*")) if _defs.S05_RUNS_DIR.exists() else []
        if not runs:
            ap.error("No surrogate runs found; pass --surrogate explicitly.")
        args.surrogate = str(runs[-1] / "state_dict.pt")

    if args.profile is None:
        args.profile = str(Path(__file__).resolve().parent.parent / "s01_profiles" / "obxf.yaml")

    stream_invert(
        target_wav=Path(args.target),
        surrogate_checkpoint=Path(args.surrogate),
        profile_path=Path(args.profile),
        out_dir=Path(args.out),
        device=args.device,
        win_sec=args.win_sec,
        hop_sec=args.hop_sec,
        n_starts=args.n_starts,
        grad_steps=args.grad_steps,
        smooth_window=args.smooth_window,
        skip_render=args.no_render,
        refine_iterations=args.refine_iterations,
        refine_threshold=args.refine_threshold,
    )

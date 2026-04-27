"""
Bucket 3 addendum — time-aligned param-trajectory + audio sequences.

Bucket 6b (live streaming inverse) needs training data with temporal coherence:
short param sweeps rendered with a synchronised audio track, so a model can
learn frame-to-frame dynamics. Single-frame (vec, note, audio) captures from
`build_dataset.py` won't teach that.

This module produces the sequence variant:
    (param_trajectory[T, d_params], audio[T * hop])

Strategy (from Bucket 6b § "Training the inverse"):
1. Draw pairs of Sobol endpoints.
2. Linearly interpolate over T frames at a control rate of 50-100 Hz.
3. Drive the synth via DawDreamer's per-parameter automation API while
   rendering one sustained note covering the trajectory.

Layout written to disk:
    out_dir/
      sequences.parquet          # metadata, one row per sequence
      wav/<seq_hash>.wav         # mono audio, length = T / control_hz seconds
      params/<seq_hash>.npy      # float32 [T, d_params] trajectory
"""
from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

from .sampling import cold_start_vectors, to_synth_value
from .manifest import Manifest, Phase, Counts, new_manifest, write_manifest, MANIFEST_FILENAME


SEQ_MANIFEST_SAMPLER = "sobol_interpolated"


def interpolated_trajectory(
    u_start: np.ndarray,
    u_end: np.ndarray,
    n_frames: int,
) -> np.ndarray:
    """Linear interpolation between two param vectors. Returns [n_frames, d]."""
    if u_start.shape != u_end.shape:
        raise ValueError(f"shape mismatch: {u_start.shape} vs {u_end.shape}")
    if n_frames < 2:
        raise ValueError(f"n_frames must be >= 2, got {n_frames}")
    a = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)[:, None]
    start = u_start.astype(np.float32)[None, :]
    end = u_end.astype(np.float32)[None, :]
    return ((1.0 - a) * start + a * end).astype(np.float32)


def endpoint_pairs(m: int, d: int, seed: int = 0) -> np.ndarray:
    """Generate 2**m Sobol points then pair consecutive ones. Returns [n_pairs, 2, d]."""
    pts = cold_start_vectors(m=m, d=d, seed=seed)
    # Pair (0,1), (2,3), ... — this halves the usable count but keeps low-discrepancy.
    pts = pts[: (len(pts) // 2) * 2]
    return pts.reshape(-1, 2, d)


def apply_log_scale(trajectory: np.ndarray, specs: Sequence[dict]) -> np.ndarray:
    """Apply per-param log-scale transform to a [T, d] trajectory."""
    out = trajectory.copy()
    for i, spec in enumerate(specs):
        if spec.get("log_scale"):
            col = out[:, i]
            # Vectorised to_synth_value — same math as sampling.to_synth_value
            log_base = float(spec.get("log_base", 50))
            out[:, i] = (np.expm1(col * np.log1p(log_base)) / log_base).astype(np.float32)
    return out


def sequence_hash(trajectory: np.ndarray, note: int) -> str:
    """Stable 12-hex hash of the trajectory + note."""
    return hashlib.md5(trajectory.tobytes() + bytes([note])).hexdigest()[:12]


def render_sequence(
    engine,
    synth,
    trajectory: np.ndarray,
    note: int,
    profile: dict,
    modulated: Sequence[str],
    name_idx: dict[str, int],
    control_hz: float,
    sample_rate: int,
) -> np.ndarray:
    """Render one param-trajectory as a single mono audio clip.

    Uses DawDreamer's `set_automation` so parameter changes are interpolated
    across the render timeline, producing time-aligned audio.
    """
    T, d = trajectory.shape
    if d != len(modulated):
        raise ValueError(f"trajectory d={d} != modulated params {len(modulated)}")

    total_sec = T / float(control_hz)
    synth.clear_midi()
    vel = int(profile["probe"]["velocity"])
    pre_roll = float(profile["probe"].get("pre_roll_sec", 0.0))
    # Hold the note slightly shorter than the render so the final release is captured.
    hold = max(0.05, total_sec - pre_roll - 0.2)
    synth.add_midi_note(note, vel, pre_roll, hold)

    for i, name in enumerate(modulated):
        values = trajectory[:, i].astype(np.float32)
        # DawDreamer stretches the array across the render duration.
        # ppqn here is "points per quarter note" — with bpm=120, 1 quarter = 0.5 s,
        # so points_per_sec = ppqn * 2. We pin ppqn so points_per_sec == control_hz.
        ppqn = control_hz / 2.0
        synth.set_automation(name_idx[name], values, ppqn=ppqn)

    engine.set_bpm(120)
    engine.render(total_sec)
    audio = engine.get_audio()
    return audio.mean(axis=0).astype(np.float32)


def build_sequence_dataset(
    profile_path: Path | str,
    out_dir: Path | str,
    m: int,
    seconds: float = 5.0,
    control_hz: float = 100.0,
    note: int | None = None,
    seed: int = 0,
) -> Manifest:
    """Render 2**(m-1) sequences into out_dir. Writes sequences.parquet + wav + params."""
    import dawdreamer as daw                              # noqa: local import
    from s02_capture.capture_v1_2 import (                 # noqa: local import
        resolve_plugin_path, build_name_index, reset,
        SAMPLE_RATE, BUFFER_SIZE,
    )

    out_dir = Path(out_dir)
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    (out_dir / "params").mkdir(parents=True, exist_ok=True)

    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    modulated = [n for n, s in profile["parameters"].items() if s.get("importance", 0) > 0]
    specs = [profile["parameters"][n] for n in modulated]
    if note is None:
        note = int(profile["probe"]["notes"][len(profile["probe"]["notes"]) // 2])
    n_frames = int(round(seconds * control_hz))

    engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
    synth = engine.make_plugin_processor("obxf", resolve_plugin_path(profile))
    name_idx = build_name_index(synth)
    engine.load_graph([(synth, [])])

    pairs = endpoint_pairs(m=m, d=len(modulated), seed=seed)
    manifest = new_manifest(
        seed=seed, profile=profile,
        importance_mode="filter",
        log_scale_applied=any(s.get("log_scale") for s in specs),
        sampler=SEQ_MANIFEST_SAMPLER,
    )
    manifest.phases.append(Phase(
        name="interpolated_sobol",
        n=len(pairs), seed=seed,
        extra={"seconds": seconds, "control_hz": control_hz, "note": note, "frames": n_frames},
    ))
    counts = Counts()
    rows = []

    for (u_start, u_end) in tqdm(pairs, desc="sequences"):
        reset(synth, profile, name_idx)
        traj = interpolated_trajectory(u_start, u_end, n_frames)
        traj = apply_log_scale(traj, specs)

        audio = render_sequence(
            engine, synth, traj, note, profile, modulated, name_idx,
            control_hz=control_hz, sample_rate=SAMPLE_RATE,
        )
        counts.rendered += 1
        if np.max(np.abs(audio)) < 1e-4:
            counts.silent += 1
            continue
        counts.valid += 1

        h = sequence_hash(traj, note)
        sf.write(out_dir / "wav" / f"{h}.wav", audio, SAMPLE_RATE)
        np.save(out_dir / "params" / f"{h}.npy", traj)
        rows.append({
            "seq_hash": h, "note": note, "frames": n_frames,
            "control_hz": control_hz, "seconds": seconds,
            "wav": f"wav/{h}.wav", "params": f"params/{h}.npy",
            "n_params": len(modulated),
        })

    pd.DataFrame(rows).to_parquet(out_dir / "sequences.parquet")
    manifest.counts = counts
    write_manifest(out_dir / MANIFEST_FILENAME, manifest)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", type=int, required=True,
                    help="Sobol exponent — generates 2**(m-1) sequence pairs")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--control-hz", type=float, default=100.0)
    ap.add_argument("--note", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    manifest = build_sequence_dataset(
        args.profile, args.out, m=args.m,
        seconds=args.seconds, control_hz=args.control_hz,
        note=args.note, seed=args.seed,
    )
    print(f"Done. Sequences rendered={manifest.counts.rendered} valid={manifest.counts.valid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

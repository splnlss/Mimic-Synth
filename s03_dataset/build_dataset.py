"""
Bucket 3 dataset builder.

Two modes:
1. Live capture (--m): delegates to capture_v1_2.capture_vector() with
   importance weighting, quality gates, and manifest.
2. Post-hoc (--from-capture): reads an existing capture directory
   (samples.parquet + WAVs from capture_v1_2.py), applies quality gates,
   copies valid WAVs, and writes a clean dataset with manifest. No
   DawDreamer required.

Usage:
    # Live capture
    python -m s03_dataset.build_dataset --profile s01_profiles/obxf.yaml --m 10 --out data/

    # Post-hoc from existing capture
    python -m s03_dataset.build_dataset --profile s01_profiles/obxf.yaml --from-capture s02_capture/data/ --out s03_dataset/data/
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

from .sampling import cold_start_vectors, apply_importance
from .quality import analyse
from .manifest import Manifest, Phase, Counts, new_manifest, write_manifest, MANIFEST_FILENAME

import defaults as _defs


def _list_modulated(profile: dict) -> list[str]:
    return [n for n, s in profile["parameters"].items() if s.get("importance", 0) > 0]


def build_dataset(
    profile_path: Path | str,
    out_dir: Path | str,
    m: int,
    seed: int = 0,
    importance_mode: str = "filter",
) -> Manifest:
    """Render 2**m * len(notes) captures into out_dir and write manifest.yaml."""
    import dawdreamer as daw                              # noqa: local import
    from s02_capture.capture_v1_2 import (                # noqa: local import
        resolve_plugin_path, build_name_index, capture_vector,
        SAMPLE_RATE, BUFFER_SIZE,
    )

    out_dir = Path(out_dir)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    modulated = _list_modulated(profile)
    notes = profile["probe"]["notes"]

    engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
    synth = engine.make_plugin_processor("obxf", resolve_plugin_path(profile))
    name_idx = build_name_index(synth)
    engine.load_graph([(synth, [])])

    vectors = cold_start_vectors(m=m, d=len(modulated), seed=seed)
    manifest = new_manifest(
        seed=seed, profile=profile,
        importance_mode=importance_mode,
        log_scale_applied=any(profile["parameters"][n].get("log_scale") for n in modulated),
    )
    manifest.phases.append(Phase(name="cold_start", n=len(vectors), seed=seed))
    counts = Counts()
    rows = []

    played_notes_prev: list[int] = []

    for u_row in tqdm(vectors, desc="capturing"):
        # Apply importance weighting to the uniform Sobol row
        params = apply_importance(u_row, modulated, profile, mode=importance_mode)
        vec = np.array([params[k] for k in modulated], dtype=np.float64)

        results, _ = capture_vector(
            engine, synth, vec, notes, profile, name_idx,
            modulated, played_notes_prev=played_notes_prev,
        )
        played_notes_prev = [r["note"] for r in results]

        for r in results:
            counts.rendered += 1
            audio = r["audio"]
            stats = analyse(
                audio, sample_rate=SAMPLE_RATE,
                hold_sec=float(profile["probe"]["hold_sec"]),
                release_sec=float(profile["probe"]["release_sec"]),
                pre_roll_sec=float(profile["probe"].get("pre_roll_sec", 0.0)),
                self_noise=r["self_noise"],
            )
            if stats.silent:     counts.silent += 1
            if stats.clipped:    counts.clipped += 1
            if stats.stuck:      counts.stuck += 1
            if stats.prev_bleed: counts.prev_bleed += 1
            if not stats.is_valid():
                continue
            counts.valid += 1

            wav_path = wav_dir / f"{r['hash']}_n{r['note']}.wav"
            sf.write(wav_path, audio, SAMPLE_RATE)
            rows.append({
                "hash": r["hash"],
                "note": r["note"],
                "wav": str(wav_path.relative_to(out_dir)),
                "self_noise": r["self_noise"],
                **{f"p_{k}": v for k, v in r["params_dict"].items()},
            })

    pd.DataFrame(rows).to_parquet(out_dir / "samples.parquet")
    manifest.counts = counts
    write_manifest(out_dir / MANIFEST_FILENAME, manifest)
    return manifest


def build_from_capture(
    capture_dir: Path | str,
    profile_path: Path | str,
    out_dir: Path | str,
) -> Manifest:
    """Build a quality-gated dataset from an existing capture directory.

    Reads samples.parquet + WAVs produced by capture_v1_2.py, runs quality
    gates on each WAV, copies valid captures to out_dir, and writes a
    manifest. No DawDreamer required.
    """
    capture_dir = Path(capture_dir)
    out_dir = Path(out_dir)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    parquet_path = capture_dir / "samples.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No samples.parquet in {capture_dir}")
    df = pd.read_parquet(parquet_path)

    # WAV paths may be relative to capture_dir or its parent depending on
    # how capture_v1_2 stored them. Probe the first row to find the root.
    wav_root = capture_dir
    if not df.empty:
        sample_wav = df["wav"].iloc[0]
        if not (capture_dir / sample_wav).exists() and (capture_dir.parent / sample_wav).exists():
            wav_root = capture_dir.parent

    sample_rate = int(profile["probe"]["sample_rate"])
    hold_sec = float(profile["probe"]["hold_sec"])
    release_sec = float(profile["probe"]["release_sec"])
    pre_roll_sec = float(profile["probe"].get("pre_roll_sec", 0.0))

    manifest = new_manifest(
        seed=0, profile=profile,
        capture_rig="v1.2",
        importance_mode="from_capture",
        log_scale_applied=False,
    )
    manifest.phases.append(Phase(
        name="post_hoc",
        n=len(df),
        extra={"source": str(capture_dir)},
    ))
    counts = Counts()
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="quality gates"):
        counts.rendered += 1
        wav_path = Path(row["wav"])
        if not wav_path.is_absolute():
            wav_path = wav_root / wav_path
        if not wav_path.exists():
            continue

        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if sr != sample_rate:
            continue

        self_noise = float(row.get("self_noise", 0.0))
        stats = analyse(
            audio, sample_rate=sample_rate,
            hold_sec=hold_sec,
            release_sec=release_sec,
            pre_roll_sec=pre_roll_sec,
            self_noise=self_noise,
        )
        if stats.silent:     counts.silent += 1
        if stats.clipped:    counts.clipped += 1
        if stats.stuck:      counts.stuck += 1
        if stats.prev_bleed: counts.prev_bleed += 1
        if not stats.is_valid():
            continue
        counts.valid += 1

        h = str(row["hash"])
        note = int(row["note"])
        out_wav = wav_dir / f"{h}_n{note}.wav"
        if not out_wav.exists():
            shutil.copy2(wav_path, out_wav)

        row_dict = {"hash": h, "note": note, "wav": f"wav/{h}_n{note}.wav", "self_noise": self_noise}
        for col in df.columns:
            if col.startswith("p_"):
                row_dict[col] = float(row[col])
        rows.append(row_dict)

    pd.DataFrame(rows).to_parquet(out_dir / "samples.parquet")
    manifest.counts = counts
    write_manifest(out_dir / MANIFEST_FILENAME, manifest)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(_defs.PROFILE_PATH))
    ap.add_argument("--out", default=str(_defs.S03_DIR))

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--m", type=int, help="Live capture: Sobol exponent — generates 2**m vectors")
    mode.add_argument("--from-capture", default=None,
                      help=f"Post-hoc: path to existing capture dir (default: {_defs.S02_DIR})")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--importance-mode", choices=["filter", "scale"], default="filter")
    args = ap.parse_args()

    if args.from_capture:
        manifest = build_from_capture(args.from_capture, args.profile, args.out)
    else:
        manifest = build_dataset(
            args.profile, args.out, m=args.m, seed=args.seed,
            importance_mode=args.importance_mode,
        )
    print(f"Done. Rendered={manifest.counts.rendered} valid={manifest.counts.valid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Bucket 3 dataset builder.

Delegates capture to capture_v1_2.capture_vector() so all settle, self-noise,
and hard_reset logic lives in one place. This module adds:
- Scrambled Sobol cold-start sampling with importance weighting
- Per-capture quality gates (drops invalid captures)
- Reproducibility manifest

Usage:
    python -m s03_dataset.build_dataset --profile s01_profiles/obxf.yaml --m 10 --out data/
    # 2**10 = 1024 vectors * len(notes) captures
"""
from __future__ import annotations
import argparse
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
        vec = np.array([v for _, v in params], dtype=np.float64)

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", type=int, required=True, help="Sobol exponent — generates 2**m vectors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--importance-mode", choices=["filter", "scale"], default="filter")
    args = ap.parse_args()
    manifest = build_dataset(
        args.profile, args.out, m=args.m, seed=args.seed,
        importance_mode=args.importance_mode,
    )
    print(f"Done. Rendered={manifest.counts.rendered} valid={manifest.counts.valid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

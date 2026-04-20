"""
Post-hoc dataset verifier.

Load samples.parquet + every WAV, run each through quality.analyse, and report:
- silence/clip/stuck/bleed rates (all must be < 1% per s03_dataset doc)
- WAV-to-parquet consistency (filename hash, referenced WAVs exist)
- duration check (matches profile's render_sec)
- parameter-coverage stats (mean, std, min, max per column)

Usage:
    python -m s03_dataset.verify_dataset --dataset data/ --profile s01_profiles/obxf.yaml
    python -m s03_dataset.verify_dataset --dataset data/ --profile s01_profiles/obxf.yaml --fail-threshold 0.01
"""
from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from .quality import analyse, CaptureStats


REQUIRED_COLUMNS = {"hash", "note", "wav"}


@dataclass
class Report:
    total_rows: int = 0
    wavs_ok: int = 0
    wavs_missing: int = 0
    wavs_unreadable: int = 0
    wrong_duration: int = 0
    hash_mismatch: int = 0
    silent: int = 0
    clipped: int = 0
    stuck: int = 0
    prev_bleed: int = 0
    valid: int = 0
    param_coverage: dict[str, dict[str, float]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    failure_rows: list[dict] = field(default_factory=list)

    def rate(self, field_name: str) -> float:
        if self.total_rows == 0:
            return 0.0
        return getattr(self, field_name) / self.total_rows


def _check_required_columns(df: pd.DataFrame) -> list[str]:
    missing = REQUIRED_COLUMNS - set(df.columns)
    return [f"Parquet missing required columns: {sorted(missing)}"] if missing else []


def _check_param_columns(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    param_cols = [c for c in df.columns if c.startswith("p_")]
    stats: dict[str, dict[str, float]] = {}
    for col in param_cols:
        s = df[col].astype(float)
        stats[col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    return stats


def _resolve_wav_root(df: pd.DataFrame, parquet_dir: Path) -> Path:
    """Return the directory that makes the first WAV path in the parquet resolvable.

    Tries parquet_dir first, then parquet_dir.parent — handles the case where
    the parquet lives inside a subdirectory but paths were stored relative to
    the parent (e.g. parquet at data/samples.parquet, paths as data/wav/...)."""
    if df.empty:
        return parquet_dir
    sample = df["wav"].iloc[0]
    if (parquet_dir / sample).exists():
        return parquet_dir
    if (parquet_dir.parent / sample).exists():
        return parquet_dir.parent
    return parquet_dir


def verify_row(row: pd.Series, dataset_dir: Path, profile: dict) -> tuple[CaptureStats | None, list[str]]:
    """Load one row's WAV and analyse it. Returns (stats, issue_list)."""
    issues: list[str] = []
    wav_path = Path(row["wav"])
    if not wav_path.is_absolute():
        wav_path = dataset_dir / wav_path
    if not wav_path.exists():
        issues.append(f"missing_wav:{wav_path}")
        return None, issues
    try:
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    except Exception as e:
        issues.append(f"unreadable_wav:{wav_path}:{e!r}")
        return None, issues

    expected_sr = int(profile["probe"]["sample_rate"])
    if sr != expected_sr:
        issues.append(f"wrong_sample_rate:{wav_path}:{sr}!={expected_sr}")

    expected_samples = int(profile["probe"]["render_sec"] * expected_sr)
    if abs(len(audio) - expected_samples) > expected_sr * 0.05:  # 50ms tolerance
        issues.append(f"wrong_duration:{wav_path}:{len(audio)}!={expected_samples}")

    if "_n" in wav_path.name and str(row["hash"]) not in wav_path.name:
        issues.append(f"hash_filename_mismatch:{wav_path}:expected={row['hash']}")

    stats = analyse(
        audio,
        sample_rate=expected_sr,
        hold_sec=float(profile["probe"]["hold_sec"]),
        release_sec=float(profile["probe"]["release_sec"]),
    )
    return stats, issues


def verify_dataset(
    dataset_dir: Path | str,
    profile_path: Path | str,
    max_errors_shown: int = 10,
) -> Report:
    dataset_dir = Path(dataset_dir)
    parquet_path = dataset_dir / "samples.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No samples.parquet found in {dataset_dir}")

    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    df = pd.read_parquet(parquet_path)
    report = Report(total_rows=len(df))
    report.errors.extend(_check_required_columns(df))
    report.param_coverage = _check_param_columns(df)

    # WAV paths in the parquet are relative to where the parquet was written,
    # which may differ from dataset_dir (e.g. parquet in data/ but paths start with data/wav/).
    wav_root = _resolve_wav_root(df, parquet_path.parent)
    for _, row in df.iterrows():
        stats, issues = verify_row(row, wav_root, profile)
        for issue in issues:
            code = issue.split(":", 1)[0]
            if code == "missing_wav":
                report.wavs_missing += 1
            elif code == "unreadable_wav":
                report.wavs_unreadable += 1
            elif code == "wrong_duration":
                report.wrong_duration += 1
            elif code == "hash_filename_mismatch":
                report.hash_mismatch += 1
            if len(report.errors) < max_errors_shown:
                report.errors.append(issue)

        if stats is None:
            continue
        report.wavs_ok += 1
        if stats.silent:
            report.silent += 1
        if stats.clipped:
            report.clipped += 1
        if stats.stuck:
            report.stuck += 1
        if stats.prev_bleed:
            report.prev_bleed += 1
        if stats.is_valid():
            report.valid += 1
        elif not stats.is_valid():
            report.failure_rows.append({
                "wav": row["wav"],
                "hash": row.get("hash", ""),
                "note": row.get("note", ""),
                "silent": stats.silent,
                "clipped": stats.clipped,
                "stuck": stats.stuck,
                "prev_bleed": stats.prev_bleed,
                "peak": round(stats.peak, 6),
                "rms": round(stats.rms, 6),
            })
    return report


def print_report(report: Report, fail_threshold: float) -> bool:
    """Print a human-readable summary. Returns True if the dataset passes."""
    print(f"Total rows:         {report.total_rows}")
    print(f"WAVs readable:      {report.wavs_ok}")
    print(f"WAVs missing:       {report.wavs_missing}  ({report.rate('wavs_missing'):.2%})")
    print(f"WAVs unreadable:    {report.wavs_unreadable}  ({report.rate('wavs_unreadable'):.2%})")
    print(f"Wrong duration:     {report.wrong_duration}  ({report.rate('wrong_duration'):.2%})")
    print(f"Hash/filename diff: {report.hash_mismatch}  ({report.rate('hash_mismatch'):.2%})")
    print(f"Silent:             {report.silent}  ({report.rate('silent'):.2%})")
    print(f"Clipped:            {report.clipped}  ({report.rate('clipped'):.2%})")
    print(f"Stuck notes:        {report.stuck}  ({report.rate('stuck'):.2%})")
    print(f"Prev-note bleed:    {report.prev_bleed}  ({report.rate('prev_bleed'):.2%})")
    print(f"Valid:              {report.valid}  ({report.rate('valid'):.2%})")
    if report.param_coverage:
        print("\nParameter coverage:")
        for name, s in sorted(report.param_coverage.items()):
            print(f"  {name:<30} mean={s['mean']:.3f} std={s['std']:.3f} "
                  f"min={s['min']:.3f} max={s['max']:.3f}")
    if report.errors:
        print(f"\nFirst {len(report.errors)} issues:")
        for e in report.errors:
            print(f"  - {e}")

    failures = {
        "silent": report.rate("silent"),
        "clipped": report.rate("clipped"),
        "stuck": report.rate("stuck"),
        "prev_bleed": report.rate("prev_bleed"),
        "wavs_missing": report.rate("wavs_missing"),
        "wavs_unreadable": report.rate("wavs_unreadable"),
    }
    breach = {k: v for k, v in failures.items() if v > fail_threshold}
    if breach:
        print(f"\nFAIL: rates above threshold {fail_threshold:.2%}: {breach}")
        return False
    print(f"\nPASS: all failure rates ≤ {fail_threshold:.2%}")
    return True


def _write_failure_report(report: Report, dataset_arg: str) -> None:
    import csv
    from datetime import datetime
    dataset_name = Path(dataset_arg).resolve().parent.name
    timestamp = datetime.now().strftime("%H%M%S")
    reports_dir = Path(__file__).resolve().parent.parent / "tests" / "reports" / "verify_dataset"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"data_{dataset_name}_{timestamp}.csv"
    if not report.failure_rows:
        print("\nNo failures to write.")
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report.failure_rows[0].keys())
        writer.writeheader()
        writer.writerows(report.failure_rows)
    print(f"\nFailure report written to {out_path} ({len(report.failure_rows)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--profile", required=True, help="Path to profile YAML")
    parser.add_argument(
        "--fail-threshold", type=float, default=0.01,
        help="Max acceptable rate for silent/clipped/stuck/bleed/missing (default 1%%)",
    )
    parser.add_argument(
        "--dump-failures", action="store_true",
        help="Write failed rows to tests/reports/verify_dataset/data_<name>_<HHMMSS>.csv",
    )
    args = parser.parse_args()
    report = verify_dataset(args.dataset, args.profile)
    ok = print_report(report, args.fail_threshold)
    if args.dump_failures:
        _write_failure_report(report, args.dataset)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

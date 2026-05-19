"""
Dataset manifest: reproducibility metadata that travels with every dataset.

Commit one manifest.yaml per dataset directory alongside samples.parquet.
The manifest pins: seed, profile content hash, sampler version, phase history,
capture counts, and the git sha of the capture rig.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml

SAMPLER_VERSION = "1.0.0"
MANIFEST_FILENAME = "manifest.yaml"


def profile_hash(profile: dict[str, Any]) -> str:
    """Stable SHA-256 of the profile YAML.

    Normalises to sorted-key JSON so formatting changes (whitespace, key order)
    don't invalidate the hash. Returns 'sha256:<64 hex>'."""
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{h}"


def git_sha(repo_root: Path | str | None = None) -> str | None:
    """Current git commit SHA of the capture-rig repo, or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root or "."), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


@dataclass
class Phase:
    name: str
    n: int
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Counts:
    rendered: int = 0
    silent: int = 0
    clipped: int = 0
    stuck: int = 0
    prev_bleed: int = 0
    valid: int = 0


@dataclass
class Manifest:
    created: str                            # ISO-8601 UTC
    seed: int
    sampler: str
    sampler_version: str
    profile_hash: str
    capture_rig: str                        # 'v1' or 'v2'
    capture_rig_git_sha: str | None
    importance_mode: str                    # 'filter' or 'scale'
    log_scale_applied: bool
    phases: list[Phase] = field(default_factory=list)
    counts: Counts = field(default_factory=Counts)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {"dataset": d}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        body = data.get("dataset", data)
        phases = [Phase(**p) for p in body.pop("phases", [])]
        counts_raw = body.pop("counts", {}) or {}
        counts = Counts(**counts_raw)
        return cls(phases=phases, counts=counts, **body)


def new_manifest(
    seed: int,
    profile: dict[str, Any],
    capture_rig: str = "v1",
    importance_mode: str = "filter",
    log_scale_applied: bool = False,
    sampler: str = "sobol_scrambled",
) -> Manifest:
    return Manifest(
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seed=seed,
        sampler=sampler,
        sampler_version=SAMPLER_VERSION,
        profile_hash=profile_hash(profile),
        capture_rig=capture_rig,
        capture_rig_git_sha=git_sha(),
        importance_mode=importance_mode,
        log_scale_applied=log_scale_applied,
    )


def write_manifest(path: Path | str, manifest: Manifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(manifest.to_dict(), f, sort_keys=False)


def read_manifest(path: Path | str) -> Manifest:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Manifest.from_dict(data)


def assert_profile_matches(manifest: Manifest, profile: dict[str, Any]) -> None:
    """Raise if the profile has changed since the dataset was built."""
    current = profile_hash(profile)
    if current != manifest.profile_hash:
        raise ValueError(
            f"Profile hash mismatch — dataset was built with {manifest.profile_hash}, "
            f"current profile hashes to {current}. Rebuild the dataset or restore "
            "the original profile."
        )

#!/usr/bin/env python3
"""Create an immutable evaluation-run directory from a manifest and baseline config."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        help="Optional frozen manifest that must be byte-identical to --manifest (for matched baselines).",
    )
    options = parser.parse_args()

    if options.run_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing run directory: {options.run_dir}")
    if not options.manifest.is_file() or not options.baseline_config.is_file():
        raise SystemExit("Manifest and baseline config must both exist")
    if options.reference_manifest is not None:
        if not options.reference_manifest.is_file():
            raise SystemExit("Reference manifest must exist")
        if sha256(options.manifest) != sha256(options.reference_manifest):
            raise SystemExit("Manifest does not match the required reference manifest")

    manifest_rows = [line for line in options.manifest.read_text(encoding="utf-8").splitlines() if line]
    if not manifest_rows:
        raise SystemExit("Manifest is empty")
    options.run_dir.mkdir(parents=True)
    for relative in ("config", "raw_predictions", "predictions", "runtime", "metrics", "evaluator"):
        (options.run_dir / relative).mkdir()
    shutil.copy2(options.manifest, options.run_dir / "manifest.jsonl")
    shutil.copy2(options.baseline_config, options.run_dir / "config" / "baseline.toml")
    metadata = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "harness_git_commit": git_commit(),
        "manifest_rows": len(manifest_rows),
        "manifest_sha256": sha256(options.manifest),
        "baseline_config_sha256": sha256(options.baseline_config),
        "reference_manifest": str(options.reference_manifest) if options.reference_manifest else None,
        "reference_manifest_sha256": (
            sha256(options.reference_manifest) if options.reference_manifest else None
        ),
    }
    (options.run_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

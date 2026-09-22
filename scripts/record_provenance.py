#!/usr/bin/env python3
"""Capture the actual host/runtime facts needed to reproduce an initialized run."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def command(*argv: str) -> dict[str, object]:
    try:
        result = subprocess.run(argv, text=True, capture_output=True, check=False)
    except OSError as error:
        return {"argv": list(argv), "returncode": None, "stdout": "", "stderr": str(error)}
    return {"argv": list(argv), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Provenance path; defaults to RUN_DIR/provenance.json. Use a separate cloud path after local setup.",
    )
    options = parser.parse_args()
    config = options.run_dir / "config" / "baseline.toml"
    manifest = options.run_dir / "manifest.jsonl"
    output = options.output or options.run_dir / "provenance.json"
    if not output.is_absolute():
        output = options.run_dir / output
    if output.exists():
        raise SystemExit(f"Refusing to overwrite provenance: {output}")
    if not config.is_file() or not manifest.is_file():
        raise SystemExit("Expected initialized run directory")
    record = {
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "baseline_config_sha256": sha256(config),
        "manifest_sha256": sha256(manifest),
        "commands": {
            "pip_freeze": command(sys.executable, "-m", "pip", "freeze"),
            "nvidia_smi": command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
            "docker_version": command("docker", "version", "--format", "{{json .}}"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

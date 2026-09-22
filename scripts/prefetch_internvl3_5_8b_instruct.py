#!/usr/bin/env python3
"""Materialize and hash the pinned InternVL3.5-8B-Instruct checkpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError as error:  # pragma: no cover - cloud runtime guard
    raise SystemExit("This harness requires Python 3.11+ for TOML parsing") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(root).parts:
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if not entries:
        raise RuntimeError(f"No checkpoint files were materialized under {root}")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--model-cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if not options.baseline_config.is_file():
        raise SystemExit(f"Missing baseline config: {options.baseline_config}")
    config = tomllib.loads(options.baseline_config.read_text(encoding="utf-8"))
    model = config["model"]
    cache_root = (options.model_cache_root or Path(config["environment"]["model_cache_root"])).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    from huggingface_hub import snapshot_download

    snapshot_path = Path(
        snapshot_download(
            repo_id=str(model["huggingface_model"]),
            revision=str(model["huggingface_revision"]),
            cache_dir=str(cache_root / "huggingface"),
        )
    ).resolve()
    if snapshot_path.name != str(model["huggingface_revision"]):
        raise RuntimeError(
            f"Expected snapshot {model['huggingface_revision']}, got {snapshot_path.name}"
        )
    output = options.output or cache_root / "pinned-models.json"
    if not output.is_absolute():
        output = cache_root / output
    record = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline_config": str(options.baseline_config.resolve()),
        "baseline_config_sha256": sha256(options.baseline_config),
        "model_cache_root": str(cache_root),
        "snapshots": [
            {
                "role": "internvl_instruct",
                "repository": model["huggingface_model"],
                "revision": model["huggingface_revision"],
                "snapshot_path": str(snapshot_path),
                "files": file_manifest(snapshot_path),
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"Pinned InternVL checkpoint manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

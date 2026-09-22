#!/usr/bin/env python3
"""Materialize and hash the pinned HunyuanOCR-1.5 checkpoint."""

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
except ModuleNotFoundError:  # Python 3.10 is the official vLLM AR runtime.
    import tomli as tomllib


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".cache" not in path.relative_to(root).parts:
            files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    if not files:
        raise RuntimeError(f"No checkpoint files found under {root}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--model-cache-root", type=Path)
    options = parser.parse_args()
    config_path = options.baseline_config.resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    model, environment = config["model"], config["environment"]
    cache_root = (options.model_cache_root or Path(environment["model_cache_root"])).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        repo_id=str(model["huggingface_model"]), revision=str(model["huggingface_revision"]),
        cache_dir=str(cache_root / "huggingface"), ignore_patterns=list(model.get("download_ignore_patterns", [])),
    )).resolve()
    if snapshot.name != model["huggingface_revision"]:
        raise RuntimeError(f"Expected snapshot {model['huggingface_revision']}, got {snapshot.name}")
    result = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline_config": str(config_path), "baseline_config_sha256": sha256(config_path),
        "model_cache_root": str(cache_root),
        "snapshots": [{"role": "hunyuanocr_1_5", "repository": model["huggingface_model"], "revision": model["huggingface_revision"], "ignore_patterns": model.get("download_ignore_patterns", []), "snapshot_path": str(snapshot), "files": file_manifest(snapshot)}],
    }
    output = cache_root / "pinned-models.json"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"Pinned HunyuanOCR checkpoint manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

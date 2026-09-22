#!/usr/bin/env python3
"""Download and hash the exact PaddleOCR-VL-1.6 native-pipeline model snapshots.

PaddleOCR/PaddleX otherwise resolves models from a moving upstream default
branch. This tool materializes the two model snapshots named in the frozen
baseline TOML under a persistent cache directory and writes a content manifest
that the inference runner copies into each immutable run directory.
"""

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
    """Hash materialized model files, excluding Hugging Face bookkeeping files."""
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
        raise RuntimeError(f"No model files were materialized under {root}")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument(
        "--model-cache-root",
        type=Path,
        help="Defaults to environment.model_cache_root in the frozen config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to MODEL_CACHE_ROOT/pinned-models.json.",
    )
    options = parser.parse_args()
    if not options.baseline_config.is_file():
        raise SystemExit(f"Missing baseline config: {options.baseline_config}")
    config = tomllib.loads(options.baseline_config.read_text(encoding="utf-8"))
    model = config["model"]
    cache_root = options.model_cache_root or Path(config["environment"]["model_cache_root"])
    cache_root = cache_root.resolve()
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface-hub"))
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_root / "paddlex"))
    from huggingface_hub import snapshot_download

    snapshots = [
        {
            "role": "vl_recognition",
            "repository": model["vl_model_repository"].removeprefix("https://huggingface.co/"),
            "revision": model["vl_model_revision"],
            "destination": cache_root / "official_models" / "PaddleOCR-VL-1.6",
        },
        {
            "role": "layout_detection",
            "repository": model["layout_model_repository"].removeprefix("https://huggingface.co/"),
            "revision": model["layout_model_revision"],
            "destination": cache_root / "official_models" / "PP-DocLayoutV3",
        },
    ]
    for snapshot in snapshots:
        snapshot["destination"].mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=snapshot["repository"],
            revision=snapshot["revision"],
            local_dir=str(snapshot["destination"]),
        )
        snapshot["files"] = file_manifest(snapshot["destination"])
        snapshot["destination"] = str(snapshot["destination"])

    output = options.output or cache_root / "pinned-models.json"
    if not output.is_absolute():
        output = cache_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline_config": str(options.baseline_config.resolve()),
        "baseline_config_sha256": sha256(options.baseline_config),
        "model_cache_root": str(cache_root),
        "snapshots": snapshots,
    }
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"Pinned PaddleOCR-VL model manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

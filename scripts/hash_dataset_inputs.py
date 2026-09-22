#!/usr/bin/env python3
"""Hash the exact official annotation file and every image referenced by it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise SystemExit(f"Refusing to overwrite existing checksum manifest: {options.output}")
    annotations = json.loads(options.annotations.read_text(encoding="utf-8"))
    records = []
    for sample in annotations:
        relative_path = sample["page_info"]["image_path"]
        path = options.image_root / relative_path
        if not path.is_file():
            raise SystemExit(f"Missing dataset image: {path}")
        records.append({"image_path": relative_path, "sha256": sha256(path), "bytes": path.stat().st_size})
    records.sort(key=lambda record: record["image_path"])
    encoded_records = json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    report = {
        "dataset_revision": options.dataset_revision,
        "annotations_path": str(options.annotations),
        "annotations_sha256": sha256(options.annotations),
        "image_root": str(options.image_root),
        "images": records,
        "images_sha256_manifest": hashlib.sha256(encoded_records).hexdigest(),
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

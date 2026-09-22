#!/usr/bin/env python3
"""Add pinned per-image digests to a frozen OmniDocBench page manifest.

The canonical smoke and full manifests intentionally describe the benchmark
selection only.  Some runners (including HunyuanOCR) additionally require a
digest on each page to verify that a prediction was produced from the exact
image that was selected.  This utility derives such a manifest solely from the
already-pinned dataset checksum registry; it never reorders or otherwise
changes the selected pages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"Invalid JSON in {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise SystemExit(f"Manifest row {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise SystemExit(f"Manifest is empty: {path}")
    return rows


def valid_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen source page manifest.")
    parser.add_argument("--checksum-manifest", type=Path, required=True, help="Pinned dataset checksum registry.")
    parser.add_argument("--expected-images-sha256-manifest", required=True, help="Expected registry identifier from the baseline TOML.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, help="Defaults beside --output.")
    options = parser.parse_args()

    if not options.manifest.is_file() or not options.checksum_manifest.is_file():
        raise SystemExit("Both --manifest and --checksum-manifest must be files")
    provenance = options.provenance_output or options.output.with_suffix(".provenance.json")
    if options.output.exists() or provenance.exists():
        raise SystemExit("Refusing to overwrite an existing derived manifest or provenance record")

    registry = json.loads(options.checksum_manifest.read_text(encoding="utf-8"))
    if registry.get("images_sha256_manifest") != options.expected_images_sha256_manifest:
        raise SystemExit(
            "Checksum registry identifier does not match the expected baseline value: "
            f"{registry.get('images_sha256_manifest')!r}"
        )
    checksums: dict[str, dict[str, Any]] = {}
    for entry in registry.get("images", []):
        if not isinstance(entry, dict):
            raise SystemExit("Checksum registry contains a non-object image entry")
        image_path = entry.get("image_path")
        if not isinstance(image_path, str) or not valid_digest(entry.get("sha256")) or not isinstance(entry.get("bytes"), int):
            raise SystemExit(f"Malformed checksum entry: {entry!r}")
        if image_path in checksums:
            raise SystemExit(f"Duplicate checksum entry: {image_path}")
        checksums[image_path] = entry

    rows = load_jsonl(options.manifest)
    seen_page_ids: set[str] = set()
    materialized: list[dict[str, Any]] = []
    for row in rows:
        page_id = row.get("page_id")
        image_path = row.get("image_path")
        prediction_filename = row.get("prediction_filename")
        if not all(isinstance(value, str) and value for value in (page_id, image_path, prediction_filename)):
            raise SystemExit(f"Manifest row lacks a non-empty page_id, image_path, or prediction_filename: {row!r}")
        if page_id in seen_page_ids:
            raise SystemExit(f"Duplicate page_id: {page_id}")
        seen_page_ids.add(page_id)
        checksum = checksums.get(image_path)
        if checksum is None:
            raise SystemExit(f"Selected image is missing from the checksum registry: {image_path}")
        derived = dict(row)
        derived["input_sha256"] = checksum["sha256"]
        derived["input_bytes"] = checksum["bytes"]
        materialized.append(derived)

    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("x", encoding="utf-8") as handle:
        for row in materialized:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(
        json.dumps(
            {
                "checksum_manifest": str(options.checksum_manifest),
                "checksum_manifest_sha256": sha256(options.checksum_manifest),
                "images_sha256_manifest": registry["images_sha256_manifest"],
                "output_manifest": str(options.output),
                "output_manifest_sha256": sha256(options.output),
                "pages": len(materialized),
                "source_manifest": str(options.manifest),
                "source_manifest_sha256": sha256(options.manifest),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Materialized {len(materialized)} hash-addressed pages: {options.output}")
    print(f"Provenance: {provenance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

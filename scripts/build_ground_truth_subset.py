#!/usr/bin/env python3
"""Create the exact ground-truth JSON subset corresponding to a run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise SystemExit(f"Refusing to overwrite existing subset: {options.output}")
    annotations = json.loads(options.annotations.read_text(encoding="utf-8"))
    manifest = [json.loads(line) for line in options.manifest.read_text(encoding="utf-8").splitlines() if line]
    by_index = {int(row["source_index"]): row for row in manifest}
    if len(by_index) != len(manifest):
        raise SystemExit("Manifest has duplicate source_index values")
    subset = []
    for index, sample in enumerate(annotations):
        row = by_index.get(index)
        if row is None:
            continue
        image_path = Path(sample["page_info"]["image_path"]).name
        if image_path != row["page_id"]:
            raise SystemExit(f"Manifest/annotation mismatch at index {index}: {image_path} != {row['page_id']}")
        subset.append(sample)
    if len(subset) != len(manifest):
        raise SystemExit(f"Expected {len(manifest)} pages but selected {len(subset)}")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

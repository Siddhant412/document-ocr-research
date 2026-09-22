#!/usr/bin/env python3
"""Create a one-record-per-page manifest for the complete pinned benchmark."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from build_smoke_manifest import image_name, label, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path, required=True)
    parser.add_argument("--language", help="Restrict manifest to one exact OmniDocBench page language label.")
    options = parser.parse_args()
    if options.output.exists() or options.coverage_output.exists():
        raise SystemExit("Refusing to overwrite an existing full manifest or coverage report")
    annotations = json.loads(options.annotations.read_text(encoding="utf-8"))
    pages = []
    seen = set()
    for source_index, sample in enumerate(annotations):
        if options.language is not None and label(sample, "language") != options.language:
            continue
        page_id = image_name(sample)
        if page_id in seen:
            raise SystemExit(f"Duplicate image basename is not supported: {page_id}")
        seen.add(page_id)
        pages.append(
            {
                "sequence": source_index,
                "source_index": source_index,
                "page_id": page_id,
                "image_path": sample["page_info"]["image_path"],
                "prediction_filename": f"{Path(page_id).stem}.md",
                "data_source": label(sample, "data_source"),
                "language": label(sample, "language"),
                "layout": label(sample, "layout"),
                "page_attributes": sample.get("page_info", {}).get("page_attribute", {}),
            }
        )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("x", encoding="utf-8") as handle:
        for page in pages:
            handle.write(json.dumps(page, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(
        options.coverage_output,
        {
            "selection": {"pages": len(pages), "kind": "full_dataset", "language_filter": options.language},
            "actual": {
                "data_source": dict(collections.Counter(page["data_source"] for page in pages)),
                "language": dict(collections.Counter(page["language"] for page in pages)),
                "layout": dict(collections.Counter(page["layout"] for page in pages)),
            },
            "uncovered_nonempty_source_language_layout": [],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

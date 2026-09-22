#!/usr/bin/env python3
"""Create an exact, non-reporting manifest subset from a frozen manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--page-id", action="append", help="Select these exact page ids")
    selection.add_argument("--first", type=int, help="Select the first N rows in frozen manifest order")
    options = parser.parse_args()

    if not options.manifest.is_file():
        raise SystemExit(f"Manifest does not exist: {options.manifest}")
    if options.output.exists():
        raise SystemExit(f"Refusing to overwrite an existing subset: {options.output}")
    rows = [json.loads(line) for line in options.manifest.read_text(encoding="utf-8").splitlines() if line]
    if options.first is not None:
        if options.first < 1:
            raise SystemExit("--first must be positive")
        if options.first > len(rows):
            raise SystemExit(f"--first requested {options.first} rows but manifest contains {len(rows)}")
        selected = rows[: options.first]
    else:
        assert options.page_id is not None
        wanted = set(options.page_id)
        if len(wanted) != len(options.page_id):
            raise SystemExit("Each --page-id must be distinct")
        selected = [row for row in rows if row.get("page_id") in wanted]
        selected_ids = {row["page_id"] for row in selected}
        missing = sorted(wanted - selected_ids)
        if missing:
            raise SystemExit(f"Requested page ids not present in manifest: {missing}")

    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(selected)} page(s) to {options.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

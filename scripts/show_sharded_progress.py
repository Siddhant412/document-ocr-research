#!/usr/bin/env python3
"""Show durable, combined progress for a parent run created by init_sharded_run.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = frozenset({"ok", "empty_output", "error"})


def latest_terminal_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") in TERMINAL_STATUSES and isinstance(record.get("page_id"), str):
            records[record["page_id"]] = record
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    options = parser.parse_args()
    assignment_path = options.run_dir / "sharding" / "assignment.json"
    if not assignment_path.is_file():
        raise SystemExit("Expected a sharded parent run")
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    shards = assignment.get("shards")
    if not isinstance(shards, list):
        raise SystemExit("Malformed shard assignment")
    total = sum(int(shard["manifest_rows"]) for shard in shards)
    total_processed = total_ok = total_empty = total_errors = 0
    details = []
    for shard in shards:
        name = shard["name"]
        expected = int(shard["manifest_rows"])
        records = latest_terminal_records(options.run_dir / "shards" / name / "runtime" / "pages.jsonl")
        processed = len(records)
        ok = sum(record.get("status") == "ok" for record in records.values())
        empty = sum(record.get("status") == "empty_output" for record in records.values())
        errors = sum(record.get("status") == "error" for record in records.values())
        total_processed += processed
        total_ok += ok
        total_empty += empty
        total_errors += errors
        details.append((name, processed, expected, ok, empty, errors))
    print(
        f"Processed: {total_processed}/{total} ({total_processed / total:.2%}) | "
        f"Remaining: {total - total_processed}"
    )
    print(f"Successful: {total_ok} | Empty output: {total_empty} | Inference errors: {total_errors}")
    for name, processed, expected, ok, empty, errors in details:
        print(f"{name}: {processed}/{expected} | ok={ok} | empty={empty} | error={errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

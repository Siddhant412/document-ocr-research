#!/usr/bin/env python3
"""Report durable inference progress from a run's append-only page ledger."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = frozenset({"ok", "empty_output", "error"})


def terminal_records(runtime_path: Path) -> dict[str, dict[str, Any]]:
    """Read the latest complete terminal record for each page, ignoring a torn final line."""
    if not runtime_path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for line in runtime_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        page_id = record.get("page_id")
        if isinstance(page_id, str) and record.get("status") in TERMINAL_STATUSES:
            result[page_id] = record
    return result


def progress_report(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    expected_pages = {page["page_id"] for page in manifest}
    records = terminal_records(run_dir / "runtime" / "pages.jsonl")
    completed = {page_id: record for page_id, record in records.items() if page_id in expected_pages}
    statuses = Counter(record["status"] for record in completed.values())
    total = len(expected_pages)
    processed = len(completed)
    last_record = max(completed.values(), key=lambda record: record.get("started_at_utc", ""), default=None)
    return {
        "total_pages": total,
        "processed_pages": processed,
        "remaining_pages": total - processed,
        "percent_complete": round(100 * processed / total, 2) if total else 0.0,
        "successful_pages": statuses["ok"],
        "empty_model_outputs": statuses["empty_output"],
        "inference_errors": statuses["error"],
        "last_finished_page_id": last_record.get("page_id") if last_record else None,
        "last_finished_started_at_utc": last_record.get("started_at_utc") if last_record else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    options = parser.parse_args()
    report = progress_report(options.run_dir)
    if options.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"Processed: {report['processed_pages']}/{report['total_pages']} "
            f"({report['percent_complete']:.2f}%) | Remaining: {report['remaining_pages']}"
        )
        print(
            f"Successful: {report['successful_pages']} | Empty output: {report['empty_model_outputs']} "
            f"| Inference errors: {report['inference_errors']}"
        )
        if report["last_finished_page_id"]:
            print(
                f"Last finished: {report['last_finished_page_id']} "
                f"(started {report['last_finished_started_at_utc']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

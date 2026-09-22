#!/usr/bin/env python3
"""Summarize HunyuanOCR streaming termination evidence from a completed run."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    options = parser.parse_args()
    ledger = options.run_dir / "runtime" / "pages.jsonl"
    if not ledger.is_file():
        raise SystemExit(f"No merged or standalone ledger: {ledger}")
    records: dict[str, dict] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record.get("page_id"), str) and record.get("status") in {"ok", "empty_output", "error"}:
            records[record["page_id"]] = record
    cap = collections.Counter(str(record.get("generation_cap_status", "not_recorded")) for record in records.values())
    finish = collections.Counter(str(record.get("server_finish_reason", "not_recorded")) for record in records.values())
    early = sum(bool(record.get("client_early_stopped_for_tail_repetition")) for record in records.values())
    report = {"terminal_pages": len(records), "generation_cap_status": dict(sorted(cap.items())), "server_finish_reason": dict(sorted(finish.items())), "client_early_stopped_for_tail_repetition": early}
    output = options.run_dir / "runtime" / "generation_termination_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

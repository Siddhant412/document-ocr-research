#!/usr/bin/env python3
"""Check that every selected page has an auditable prediction before scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    options = parser.parse_args()
    manifest_path = options.run_dir / "manifest.jsonl"
    predictions_dir = options.run_dir / "predictions"
    expected = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    expected_names = {row["prediction_filename"] for row in expected}
    actual_names = {path.name for path in predictions_dir.glob("*.md")}
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    empty = sorted(path.name for path in predictions_dir.glob("*.md") if path.stat().st_size == 0)
    report = {
        "expected_prediction_files": len(expected_names),
        "actual_prediction_files": len(actual_names),
        "missing": missing,
        "unexpected": unexpected,
        "zero_byte_predictions": empty,
        "complete": not missing and not unexpected and len(expected_names) == len(expected),
    }
    output = options.run_dir / "metrics" / "coverage_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

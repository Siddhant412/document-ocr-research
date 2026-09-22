#!/usr/bin/env python3
"""Verify and assemble a complete parent prediction set from child shards.

The command refuses to merge incomplete, overlapping, unexpected, or
hash-mismatched shard outputs. It does not delete or modify raw shard evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = frozenset({"ok", "empty_output", "error"})


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid manifest: {path}") from error
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows


def latest_terminal_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"Shard runtime ledger is absent: {path}")
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") in TERMINAL_STATUSES and isinstance(record.get("page_id"), str):
            records[record["page_id"]] = record
    return records


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.merge-tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def write_if_absent_or_identical(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Refusing to replace existing non-identical assembled artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.merge-tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    options = parser.parse_args()
    run_dir = options.run_dir
    parent_manifest_path = run_dir / "manifest.jsonl"
    assignment_path = run_dir / "sharding" / "assignment.json"
    if not parent_manifest_path.is_file() or not assignment_path.is_file():
        raise SystemExit("Expected a parent run created by init_sharded_run.py")
    parent_rows = load_manifest(parent_manifest_path)
    parent_hash = sha256(parent_manifest_path)
    assignment = load_json(assignment_path)
    if assignment.get("parent_manifest_sha256") != parent_hash:
        raise SystemExit("Shard assignment was not created from this parent manifest")
    assignment_shards = assignment.get("shards")
    assignment_rows = assignment.get("assignments")
    if not isinstance(assignment_shards, list) or not isinstance(assignment_rows, list):
        raise SystemExit("Shard assignment is malformed")

    parent_by_id = {row["page_id"]: row for row in parent_rows}
    parent_by_filename = {row["prediction_filename"]: row for row in parent_rows}
    expected_assignment = {entry.get("page_id"): entry.get("shard") for entry in assignment_rows}
    if set(expected_assignment) != set(parent_by_id):
        raise SystemExit("Shard assignment does not cover parent manifest exactly once")
    if len(assignment_rows) != len(parent_rows):
        raise SystemExit("Shard assignment contains duplicate page ids")

    combined_records: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    merge_inputs: list[dict[str, Any]] = []
    for shard in assignment_shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("name"), str):
            raise SystemExit("Shard assignment contains malformed shard metadata")
        name = shard["name"]
        shard_dir = run_dir / "shards" / name
        manifest_path = shard_dir / "manifest.jsonl"
        prediction_dir = shard_dir / "predictions"
        runtime_path = shard_dir / "runtime" / "pages.jsonl"
        if not manifest_path.is_file() or not prediction_dir.is_dir():
            raise SystemExit(f"Shard is missing expected files: {name}")
        shard_rows = load_manifest(manifest_path)
        if len(shard_rows) != shard.get("manifest_rows") or sha256(manifest_path) != shard.get("manifest_sha256"):
            raise SystemExit(f"Shard manifest has changed: {name}")
        expected_ids = {row["page_id"] for row in shard_rows}
        if any(expected_assignment.get(page_id) != name for page_id in expected_ids):
            raise SystemExit(f"Shard contents disagree with assignment: {name}")
        records = latest_terminal_records(runtime_path)
        if set(records) != expected_ids:
            missing = sorted(expected_ids - set(records))
            extra = sorted(set(records) - expected_ids)
            raise SystemExit(f"Shard {name} has incomplete terminal ledger records; missing={missing}, extra={extra}")
        actual_files = {path.name: path for path in prediction_dir.glob("*.md")}
        expected_files = {row["prediction_filename"] for row in shard_rows}
        if set(actual_files) != expected_files:
            missing = sorted(expected_files - set(actual_files))
            extra = sorted(set(actual_files) - expected_files)
            raise SystemExit(f"Shard {name} prediction coverage differs; missing={missing}, extra={extra}")
        for row in shard_rows:
            page_id = row["page_id"]
            record = records[page_id]
            source = actual_files[row["prediction_filename"]]
            if record.get("prediction_filename") != source.name:
                raise SystemExit(f"Shard {name} ledger filename mismatch for {page_id}")
            if source.stat().st_size != record.get("prediction_bytes") or sha256(source) != record.get("prediction_sha256"):
                raise SystemExit(f"Shard {name} prediction does not match ledger for {page_id}")
            if page_id in combined_records:
                raise SystemExit(f"Page exists in multiple shards: {page_id}")
            combined_records[page_id] = record
        failures_path = shard_dir / "failures.jsonl"
        if failures_path.is_file():
            for line in failures_path.read_text(encoding="utf-8").splitlines():
                if line:
                    failures.append(json.loads(line))
        merge_inputs.append(
            {
                "name": name,
                "manifest_sha256": sha256(manifest_path),
                "runtime_pages_sha256": sha256(runtime_path),
                "failure_ledger_sha256": sha256(failures_path) if failures_path.is_file() else None,
            }
        )

    if set(combined_records) != set(parent_by_id):
        raise SystemExit("Combined shard records do not cover the parent manifest exactly")
    parent_predictions = run_dir / "predictions"
    existing_parent_files = {path.name: path for path in parent_predictions.glob("*.md")}
    if set(existing_parent_files) - set(parent_by_filename):
        raise SystemExit("Parent predictions contain unexpected files")
    for page in parent_rows:
        page_id = page["page_id"]
        source = run_dir / "shards" / expected_assignment[page_id] / "predictions" / page["prediction_filename"]
        destination = parent_predictions / page["prediction_filename"]
        if destination.exists():
            if destination.stat().st_size != source.stat().st_size or sha256(destination) != sha256(source):
                raise SystemExit(f"Parent prediction conflicts with shard output: {destination.name}")
        else:
            atomic_copy(source, destination)

    pages_content = "".join(
        json.dumps(combined_records[row["page_id"]], sort_keys=True, ensure_ascii=False) + "\n" for row in parent_rows
    )
    write_if_absent_or_identical(run_dir / "runtime" / "pages.jsonl", pages_content)
    failures_content = "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in failures)
    write_if_absent_or_identical(run_dir / "failures.jsonl", failures_content)
    statuses = {status: sum(record.get("status") == status for record in combined_records.values()) for status in TERMINAL_STATUSES}
    summary = {
        "pages_requested": len(parent_rows),
        "pages_merged_from_shards": len(combined_records),
        "shard_count": len(assignment_shards),
        "statuses": statuses,
        "failures_or_empty_outputs": statuses["error"] + statuses["empty_output"],
    }
    write_if_absent_or_identical(
        run_dir / "runtime" / "inference_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    merge_record_path = run_dir / "sharding" / "merge.json"
    if not merge_record_path.exists():
        write_if_absent_or_identical(
            merge_record_path,
            json.dumps(
                {
                    "merged_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "parent_manifest_sha256": parent_hash,
                    "assignment_sha256": sha256(assignment_path),
                    "inputs": merge_inputs,
                    "summary": summary,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

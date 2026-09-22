#!/usr/bin/env python3
"""Create a deterministic multi-GPU parent run and independent shard runs.

Each child shard is a normal, restart-safe inference run.  The parent does not
receive predictions until ``merge_sharded_run.py`` has proved that the child
outputs are a complete, disjoint, hash-verified partition of its manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("Manifest is empty")
    page_ids = [row.get("page_id") for row in rows]
    prediction_names = [row.get("prediction_filename") for row in rows]
    if any(not isinstance(value, str) or not value for value in page_ids + prediction_names):
        raise ValueError("Every manifest row must have non-empty page_id and prediction_filename")
    if len(set(page_ids)) != len(page_ids):
        raise ValueError("Manifest page_id values must be unique")
    if len(set(prediction_names)) != len(prediction_names):
        raise ValueError("Manifest prediction_filename values must be unique")
    return rows


def partition_rows(rows: list[dict[str, Any]], shard_count: int) -> list[list[dict[str, Any]]]:
    """Assign pages by SHA-256 rank, then retain original order within each shard.

    Sorting by a stable digest before round-robin assignment gives equal-sized,
    content-independent shards without making benchmark page order depend on a
    process-local Python hash seed.
    """
    ranked_indices = sorted(
        range(len(rows)),
        key=lambda index: (hashlib.sha256(rows[index]["page_id"].encode("utf-8")).hexdigest(), index),
    )
    owner_by_index = {index: rank % shard_count for rank, index in enumerate(ranked_indices)}
    return [[row for index, row in enumerate(rows) if owner_by_index[index] == shard] for shard in range(shard_count)]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def create_run_layout(run_dir: Path) -> None:
    for relative in ("config", "raw_predictions", "predictions", "runtime", "metrics", "evaluator"):
        (run_dir / relative).mkdir(parents=True, exist_ok=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--reference-manifest", type=Path)
    options = parser.parse_args()

    if options.shards < 1:
        raise SystemExit("--shards must be at least 1")
    if options.run_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing run directory: {options.run_dir}")
    if not options.manifest.is_file() or not options.baseline_config.is_file():
        raise SystemExit("Manifest and baseline config must both exist")
    if options.reference_manifest is not None:
        if not options.reference_manifest.is_file():
            raise SystemExit("Reference manifest must exist")
        if sha256(options.manifest) != sha256(options.reference_manifest):
            raise SystemExit("Manifest does not match the required reference manifest")
    try:
        rows = load_manifest(options.manifest)
    except (ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if options.shards > len(rows):
        raise SystemExit("Cannot create more shards than manifest rows")

    shards = partition_rows(rows, options.shards)
    options.run_dir.mkdir(parents=True)
    create_run_layout(options.run_dir)
    (options.run_dir / "sharding").mkdir()
    (options.run_dir / "shards").mkdir()
    shutil.copy2(options.manifest, options.run_dir / "manifest.jsonl")
    shutil.copy2(options.baseline_config, options.run_dir / "config" / "baseline.toml")

    assignment_shards = []
    assignments = []
    for shard_number, shard_rows in enumerate(shards):
        shard_name = f"shard-{shard_number:03d}"
        shard_dir = options.run_dir / "shards" / shard_name
        shard_dir.mkdir(parents=True)
        create_run_layout(shard_dir)
        write_manifest(shard_dir / "manifest.jsonl", shard_rows)
        shutil.copy2(options.baseline_config, shard_dir / "config" / "baseline.toml")
        shard_hash = sha256(shard_dir / "manifest.jsonl")
        write_json(
            shard_dir / "run.json",
            {
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "run_kind": "shard",
                "parent_run_dir": str(options.run_dir),
                "shard_name": shard_name,
                "shard_number": shard_number,
                "shard_count": options.shards,
                "manifest_rows": len(shard_rows),
                "manifest_sha256": shard_hash,
                "parent_manifest_sha256": sha256(options.manifest),
                "baseline_config_sha256": sha256(options.baseline_config),
            },
        )
        assignment_shards.append(
            {"name": shard_name, "manifest_rows": len(shard_rows), "manifest_sha256": shard_hash}
        )
        assignments.extend(
            {
                "page_id": row["page_id"],
                "prediction_filename": row["prediction_filename"],
                "shard": shard_name,
            }
            for row in shard_rows
        )

    assignment_path = options.run_dir / "sharding" / "assignment.json"
    write_json(
        assignment_path,
        {
            "algorithm": "sha256-page-id-rank-round-robin-v1",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "parent_manifest_sha256": sha256(options.manifest),
            "shard_count": options.shards,
            "shards": assignment_shards,
            "assignments": sorted(assignments, key=lambda assignment: assignment["page_id"]),
        },
    )
    write_json(
        options.run_dir / "run.json",
        {
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_kind": "sharded_parent",
            "harness_git_commit": git_commit(),
            "manifest_rows": len(rows),
            "manifest_sha256": sha256(options.manifest),
            "baseline_config_sha256": sha256(options.baseline_config),
            "reference_manifest": str(options.reference_manifest) if options.reference_manifest else None,
            "reference_manifest_sha256": sha256(options.reference_manifest) if options.reference_manifest else None,
            "shard_assignment": "sharding/assignment.json",
            "shard_assignment_sha256": sha256(assignment_path),
        },
    )
    print(
        f"Created {options.shards} deterministic shards for {len(rows)} pages: "
        + ", ".join(f"{item['name']}={item['manifest_rows']}" for item in assignment_shards)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

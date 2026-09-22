#!/usr/bin/env python3
"""Run the exact documented DeepSeek-OCR Transformers call for a manifest.

This script deliberately does not clean, normalize, or repair model output.
For an inference exception it creates a zero-byte prediction file and records
the exception in failures.jsonl, so every benchmark page remains represented.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError as error:  # pragma: no cover - cloud runtime guard
    raise SystemExit("This harness requires Python 3.11+ for TOML parsing") from error


TERMINAL_STATUSES = frozenset({"ok", "empty_output", "error"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Only process the first N manifest rows (preflight only)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an interrupted run. Completed pages with a matching terminal ledger record "
            "are retained; recorded inference failures remain terminal and are not retried."
        ),
    )
    return parser.parse_args()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_completed_pages(runtime_path: Path, predictions_dir: Path) -> dict[str, dict[str, Any]]:
    """Return pages whose current prediction is verified by a terminal ledger record.

    A file alone is never sufficient to skip work: a disconnect can leave a partial
    file behind. Conversely, a recorded zero-byte error is intentionally terminal so
    the page remains in the denominator without turning a resume into a silent retry.
    """
    if not runtime_path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in runtime_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") in TERMINAL_STATUSES and isinstance(record.get("page_id"), str):
            records[record["page_id"]] = record

    completed: dict[str, dict[str, Any]] = {}
    for page_id, record in records.items():
        filename = record.get("prediction_filename")
        if not isinstance(filename, str):
            continue
        prediction_path = predictions_dir / filename
        if not prediction_path.is_file():
            continue
        if prediction_path.stat().st_size != record.get("prediction_bytes"):
            continue
        if digest(prediction_path) != record.get("prediction_sha256"):
            continue
        completed[page_id] = record
    return completed


def plan_pages(
    manifest: list[dict[str, Any]], predictions_dir: Path, runtime_path: Path, resume: bool
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Choose pending pages and return verified completed records keyed by page id."""
    completed = load_completed_pages(runtime_path, predictions_dir)
    existing_predictions = [path for path in predictions_dir.glob("*.md")]
    if not resume and existing_predictions:
        raise SystemExit(
            "Predictions already exist. Use --resume only after an interruption; it verifies "
            "the runtime ledger before retaining any page."
        )
    if not resume:
        return manifest, completed
    return [page for page in manifest if page["page_id"] not in completed], completed


def next_attempt_dir(raw_dir: Path, prediction_filename: str) -> tuple[Path, int]:
    page_raw_dir = raw_dir / prediction_filename.removesuffix(".md")
    attempts_dir = page_raw_dir / "attempts"
    attempt_numbers = []
    if attempts_dir.is_dir():
        for child in attempts_dir.iterdir():
            if child.is_dir() and child.name.startswith("attempt-"):
                try:
                    attempt_numbers.append(int(child.name.removeprefix("attempt-")))
                except ValueError:
                    pass
    attempt = max(attempt_numbers, default=0) + 1
    return attempts_dir / f"attempt-{attempt:04d}", attempt


def preserve_unverified_prediction(prediction_path: Path, page_raw_dir: Path) -> Path | None:
    """Move a possibly partial prediction aside before retrying an interrupted page."""
    if not prediction_path.exists():
        return None
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = page_raw_dir / "interrupted_predictions" / timestamp / prediction_path.name
    destination.parent.mkdir(parents=True, exist_ok=False)
    shutil.move(str(prediction_path), str(destination))
    return destination


def main() -> int:
    options = parse_args()
    config_path = options.run_dir / "config" / "baseline.toml"
    manifest_path = options.run_dir / "manifest.jsonl"
    if not config_path.is_file() or not manifest_path.is_file():
        raise SystemExit("Run directory must have config/baseline.toml and manifest.jsonl from init_run.py")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    if options.limit is not None:
        manifest = manifest[: options.limit]
    if not manifest:
        raise SystemExit("No manifest rows selected")

    predictions_dir = options.run_dir / "predictions"
    raw_dir = options.run_dir / "raw_predictions"
    runtime_path = options.run_dir / "runtime" / "pages.jsonl"
    failures_path = options.run_dir / "failures.jsonl"
    pending, completed = plan_pages(manifest, predictions_dir, runtime_path, options.resume)

    seed = int(config["inference"]["seed"])
    if not pending:
        summary = {
            "pages_requested": len(manifest),
            "pages_attempted_this_invocation": 0,
            "pages_retained_from_verified_terminal_records": len(completed),
            "failures_or_empty_outputs_this_invocation": 0,
            "resume": options.resume,
        }
        (options.run_dir / "runtime" / "inference_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    model_id = config["model"]["huggingface_model"]
    model_revision = config["model"]["huggingface_revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        revision=model_revision,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)

    failures = 0
    for processed_this_invocation, page in enumerate(pending, start=1):
        page_id = page["page_id"]
        image_path = options.image_root / page["image_path"]
        prediction_path = predictions_dir / page["prediction_filename"]
        page_raw_dir = raw_dir / page["prediction_filename"].removesuffix(".md")
        attempt_dir, attempt = next_attempt_dir(raw_dir, page["prediction_filename"])
        official_output_dir = attempt_dir / "official_output"
        preserved_prediction = preserve_unverified_prediction(prediction_path, page_raw_dir) if options.resume else None
        started = time.perf_counter()
        record: dict[str, Any] = {
            "page_id": page_id,
            "prediction_filename": page["prediction_filename"],
            "image_path": str(image_path),
            "input_sha256": digest(image_path) if image_path.is_file() else None,
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "seed": seed,
            "attempt": attempt,
            "resume": options.resume,
            "resumed_from_unverified_prediction": str(preserved_prediction) if preserved_prediction else None,
        }
        stream = io.StringIO()
        try:
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest image does not exist: {image_path}")
            official_output_dir.mkdir(parents=True, exist_ok=False)
            # This is the documented native call, intentionally kept argument-for-argument.
            with contextlib.redirect_stdout(stream):
                model.infer(
                    tokenizer,
                    prompt=config["inference"]["prompt"],
                    image_file=str(image_path),
                    output_path=str(official_output_dir),
                    base_size=int(config["inference"]["base_size"]),
                    image_size=int(config["inference"]["image_size"]),
                    crop_mode=bool(config["inference"]["crop_mode"]),
                    save_results=bool(config["inference"]["save_results"]),
                    test_compress=bool(config["inference"]["test_compress"]),
                )
            (attempt_dir / "stdout.txt").write_text(stream.getvalue(), encoding="utf-8")
            official_markdown = official_output_dir / "result.mmd"
            if not official_markdown.is_file():
                raise RuntimeError("Official model call completed without writing result.mmd")
            temporary_prediction = attempt_dir / f"{prediction_path.name}.tmp"
            shutil.copyfile(official_markdown, temporary_prediction)
            os.replace(temporary_prediction, prediction_path)
            record["status"] = "ok" if prediction_path.stat().st_size else "empty_output"
            record["prediction_sha256"] = digest(prediction_path)
            record["prediction_bytes"] = prediction_path.stat().st_size
            if record["status"] != "ok":
                failures += 1
                append_jsonl(
                    failures_path,
                    {**record, "failure_kind": "empty_model_output", "traceback": None},
                )
        except Exception as error:  # Keep the benchmark denominator intact.
            failures += 1
            attempt_dir.mkdir(parents=True, exist_ok=True)
            (attempt_dir / "stdout.txt").write_text(stream.getvalue(), encoding="utf-8")
            if prediction_path.exists():
                unexpected_output = attempt_dir / "prediction_present_at_error.md"
                shutil.move(str(prediction_path), str(unexpected_output))
            prediction_path.touch()  # Required zero-byte prediction for an inference error.
            record.update({"status": "error", "prediction_sha256": digest(prediction_path), "prediction_bytes": 0})
            append_jsonl(
                failures_path,
                {
                    **record,
                    "failure_kind": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        finally:
            record["elapsed_seconds"] = round(time.perf_counter() - started, 6)
            if torch.cuda.is_available():
                record["gpu_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            append_jsonl(runtime_path, record)
            completed_total = len(completed) + processed_this_invocation
            print(
                f"Progress: {completed_total}/{len(manifest)} pages "
                f"({completed_total / len(manifest):.1%}); "
                f"latest={page_id} status={record['status']} "
                f"elapsed={record['elapsed_seconds']:.1f}s",
                flush=True,
            )

    summary = {
        "pages_requested": len(manifest),
        "pages_attempted_this_invocation": len(pending),
        "pages_retained_from_verified_terminal_records": len(completed),
        "failures_or_empty_outputs_this_invocation": failures,
        "resume": options.resume,
    }
    (options.run_dir / "runtime" / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

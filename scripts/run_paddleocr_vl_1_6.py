#!/usr/bin/env python3
"""Run the full native PaddleOCR-VL-1.6 pipeline for a frozen manifest.

The runner calls PaddleOCRVL's documented layout-analysis plus VLM-recognition
pipeline, then scores its exact `save_to_markdown` bytes without cleanup. It
saves native JSON/Markdown artifacts for every attempt. An exception produces
an empty prediction file and a failure ledger entry, so no selected page is
silently omitted. `--resume` retains only hash-verified terminal records.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.resources
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
    parser.add_argument("--limit", type=int, help="Only process the first N manifest rows (mechanical preflight only)")
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
    """Return pages whose current prediction matches a terminal ledger record."""
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
    attempt_numbers: list[int] = []
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
    if not prediction_path.exists():
        return None
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = page_raw_dir / "interrupted_predictions" / timestamp / prediction_path.name
    destination.parent.mkdir(parents=True, exist_ok=False)
    shutil.move(str(prediction_path), str(destination))
    return destination


def write_paddlex_default_config(run_dir: Path) -> None:
    """Preserve the package-owned defaults that the native pipeline uses."""
    destination = run_dir / "runtime" / "paddlex_paddleocr_vl_1_6_defaults.yaml"
    if destination.exists():
        return
    source = importlib.resources.files("paddlex").joinpath("configs/pipelines/PaddleOCR-VL-1.6.yaml")
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def copy_runtime_artifacts(run_dir: Path, config: dict[str, Any]) -> str:
    cache_root = Path(config["environment"]["model_cache_root"])
    artifact_manifest = cache_root / "pinned-models.json"
    environment_freeze = cache_root / "pinned-environment-freeze.txt"
    if not artifact_manifest.is_file():
        raise FileNotFoundError(
            f"Pinned model manifest is absent: {artifact_manifest}. Run setup_paddleocr_vl_1_6_environment.sh first."
        )
    if not environment_freeze.is_file():
        raise FileNotFoundError(
            f"Pinned environment freeze is absent: {environment_freeze}. Run setup_paddleocr_vl_1_6_environment.sh first."
        )
    runtime_dir = run_dir / "runtime"
    shutil.copy2(artifact_manifest, runtime_dir / "pinned_model_artifacts.json")
    shutil.copy2(environment_freeze, runtime_dir / "pinned_environment_freeze.txt")
    return digest(artifact_manifest)


def get_paddle_memory(paddle_module: Any, device: str) -> int | None:
    """Return a best-effort Paddle CUDA peak without making it a run dependency.

    PaddleX can temporarily switch back to CPU after native inference. Passing
    the explicit benchmark device is therefore required; telemetry must never
    turn a completed page into a failed run.
    """
    try:
        method = paddle_module.device.cuda.max_memory_allocated
        return int(method(device))
    except Exception:  # Telemetry is optional and must not interrupt inference.
        return None


def reset_paddle_memory(paddle_module: Any, device: str) -> None:
    try:
        paddle_module.device.cuda.reset_max_memory_allocated(device)
    except Exception:  # See get_paddle_memory: do not fail a benchmark for stats.
        pass


def main() -> int:
    options = parse_args()
    config_path = options.run_dir / "config" / "baseline.toml"
    manifest_path = options.run_dir / "manifest.jsonl"
    if not config_path.is_file() or not manifest_path.is_file():
        raise SystemExit("Run directory must have config/baseline.toml and manifest.jsonl from init_run.py")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    if options.limit is not None:
        if options.limit < 1:
            raise SystemExit("--limit must be positive")
        manifest = manifest[: options.limit]
    if not manifest:
        raise SystemExit("No manifest rows selected")

    predictions_dir = options.run_dir / "predictions"
    raw_dir = options.run_dir / "raw_predictions"
    runtime_path = options.run_dir / "runtime" / "pages.jsonl"
    failures_path = options.run_dir / "failures.jsonl"
    pending, completed = plan_pages(manifest, predictions_dir, runtime_path, options.resume)
    inference = config["inference"]
    seed = int(inference["seed"])
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
    os.environ.setdefault("FLAGS_cudnn_deterministic", "1")
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", config["environment"]["model_source"])
    cache_root = Path(config["environment"]["model_cache_root"])
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_root / "paddlex"))
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface-hub"))

    paddle_module: Any | None = None
    pipeline: Any | None = None
    initialization_error: Exception | None = None
    initialization_traceback: str | None = None
    model_artifact_manifest_sha256: str | None = None
    try:
        model_artifact_manifest_sha256 = copy_runtime_artifacts(options.run_dir, config)
        import paddle
        from paddleocr import PaddleOCRVL

        paddle_module = paddle
        if not paddle.is_compiled_with_cuda():
            raise RuntimeError("PaddlePaddle is not CUDA-enabled")
        paddle.seed(seed)
        paddle.set_device(str(inference["device"]))
        write_paddlex_default_config(options.run_dir)
        pipeline = PaddleOCRVL(
            pipeline_version=config["model"]["pipeline_version"],
            device=str(inference["device"]),
            engine=config["model"]["inference_engine"],
            use_tensorrt=False,
            layout_detection_model_dir=str(cache_root / "official_models" / "PP-DocLayoutV3"),
            vl_rec_model_dir=str(cache_root / "official_models" / "PaddleOCR-VL-1.6"),
            vl_rec_backend="native",
            use_doc_orientation_classify=bool(inference["use_doc_orientation_classify"]),
            use_doc_unwarping=bool(inference["use_doc_unwarping"]),
            use_layout_detection=bool(inference["use_layout_detection"]),
            use_chart_recognition=bool(inference["use_chart_recognition"]),
            use_seal_recognition=bool(inference["use_seal_recognition"]),
            use_ocr_for_image_block=bool(inference["use_ocr_for_image_block"]),
            format_block_content=bool(inference["format_block_content"]),
            merge_layout_blocks=bool(inference["merge_layout_blocks"]),
            use_queues=bool(inference["use_queues"]),
            markdown_ignore_labels=list(inference["markdown_ignore_labels"]),
        )
        (options.run_dir / "runtime" / "pipeline_initialization.json").write_text(
            json.dumps(
                {
                    "initialized_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "device": paddle.get_device(),
                    "paddle_version": paddle.__version__,
                    "paddle_cuda_version": paddle.version.cuda(),
                    "paddleocr_version": __import__("paddleocr").__version__,
                    "paddlex_version": __import__("paddlex").__version__,
                    "model_artifact_manifest_sha256": model_artifact_manifest_sha256,
                    "status": "ok",
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
    except Exception as error:  # Record all selected pages even if model initialization failed.
        initialization_error = error
        initialization_traceback = traceback.format_exc()
        (options.run_dir / "runtime" / "pipeline_initialization.json").write_text(
            json.dumps(
                {
                    "initialized_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "model_artifact_manifest_sha256": model_artifact_manifest_sha256,
                    "status": "error",
                    "error_kind": type(error).__name__,
                    "error": str(error),
                    "traceback": initialization_traceback,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

    failures = 0
    for processed_this_invocation, page in enumerate(pending, start=1):
        page_id = page["page_id"]
        image_path = options.image_root / page["image_path"]
        prediction_path = predictions_dir / page["prediction_filename"]
        page_raw_dir = raw_dir / page["prediction_filename"].removesuffix(".md")
        attempt_dir, attempt = next_attempt_dir(raw_dir, page["prediction_filename"])
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
            "native_pipeline": {
                "pipeline_version": config["model"]["pipeline_version"],
                "engine": config["model"]["inference_engine"],
                "device": inference["device"],
                "decoding": inference["decoding"],
                "max_new_tokens": int(inference["max_new_tokens"]),
                "model_artifact_manifest_sha256": model_artifact_manifest_sha256,
            },
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest image does not exist: {image_path}")
            attempt_dir.mkdir(parents=True, exist_ok=False)
            if initialization_error is not None or pipeline is None:
                (attempt_dir / "pipeline_initialization_error.txt").write_text(
                    initialization_traceback or repr(initialization_error), encoding="utf-8"
                )
                raise RuntimeError("PaddleOCR-VL pipeline initialization failed") from initialization_error
            native_dir = attempt_dir / "native_output"
            native_dir.mkdir()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                results = list(
                    pipeline.predict_iter(
                        str(image_path),
                        layout_shape_mode=str(inference["layout_shape_mode"]),
                        use_queues=bool(inference["use_queues"]),
                        max_new_tokens=int(inference["max_new_tokens"]),
                    )
                )
                if len(results) != 1:
                    raise RuntimeError(f"Expected exactly one PaddleOCR-VL result for an image, got {len(results)}")
                native_result = results[0]
                native_json = native_dir / "result.json"
                native_markdown = native_dir / "result.md"
                native_result.save_to_json(save_path=native_json)
                native_result.save_to_markdown(
                    save_path=native_markdown,
                    pretty=bool(inference["markdown_pretty"]),
                    show_formula_number=bool(inference["markdown_show_formula_number"]),
                )
            (attempt_dir / "stdout.txt").write_text(stdout.getvalue(), encoding="utf-8")
            (attempt_dir / "stderr.txt").write_text(stderr.getvalue(), encoding="utf-8")
            if not native_markdown.is_file():
                raise RuntimeError("Native save_to_markdown completed without writing result.md")
            temporary_prediction = attempt_dir / f"{prediction_path.name}.tmp"
            shutil.copyfile(native_markdown, temporary_prediction)
            os.replace(temporary_prediction, prediction_path)
            record["status"] = "ok" if prediction_path.stat().st_size else "empty_output"
            record["prediction_sha256"] = digest(prediction_path)
            record["prediction_bytes"] = prediction_path.stat().st_size
            if record["status"] != "ok":
                failures += 1
                append_jsonl(
                    failures_path,
                    {**record, "failure_kind": "empty_native_markdown", "traceback": None},
                )
        except Exception as error:  # Keep the benchmark denominator intact.
            failures += 1
            attempt_dir.mkdir(parents=True, exist_ok=True)
            (attempt_dir / "stdout.txt").write_text(stdout.getvalue(), encoding="utf-8")
            (attempt_dir / "stderr.txt").write_text(stderr.getvalue(), encoding="utf-8")
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
            if paddle_module is not None:
                record["gpu_max_memory_allocated_bytes"] = get_paddle_memory(
                    paddle_module, str(inference["device"])
                )
                reset_paddle_memory(paddle_module, str(inference["device"]))
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

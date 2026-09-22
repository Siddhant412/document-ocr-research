#!/usr/bin/env python3
"""Run the pinned HunyuanOCR-1.5 AR baseline against a manifest.

This is an *official-client-compatible* harness for HunyuanOCR's vLLM AR
server. It uses the fixed upstream ``doc_parse`` task prompt, request shape,
sampling values, streaming tail-repetition stop, tail cleanup, and doc_parse
normalization. In addition to the final official processed Markdown, it saves
the text received from the streaming client before cleanup and compact server
stream events. That makes client-side stopping and max-token termination
auditable without storing the input image's base64 payload.

One instance processes one normal run or one shard. For a multi-GPU run use
``launch_hunyuanocr_shards.sh``; it starts one official AR server per shard.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 is the official vLLM AR runtime.
    import tomli as tomllib


TERMINAL_STATUSES = frozenset({"ok", "empty_output", "error"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--limit", type=int, help="Only process first N manifest rows (preflight only)")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")
    required = {"page_id", "image_path", "prediction_filename", "input_sha256"}
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(f"Manifest row has no {missing}: {row}")
        output = Path(str(row["prediction_filename"]))
        if output.is_absolute() or ".." in output.parts or output.suffix != ".md":
            raise RuntimeError(f"Unsafe prediction filename: {row['prediction_filename']}")
    return rows


def load_completed_pages(ledger: Path, predictions: Path) -> dict[str, dict[str, Any]]:
    if not ledger.is_file():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a torn last append after an interrupted pod
        if record.get("status") in TERMINAL_STATUSES and isinstance(record.get("page_id"), str):
            latest[record["page_id"]] = record
    retained: dict[str, dict[str, Any]] = {}
    for page_id, record in latest.items():
        filename = record.get("prediction_filename")
        if not isinstance(filename, str):
            continue
        prediction = predictions / filename
        if prediction.is_file() and prediction.stat().st_size == record.get("prediction_bytes") and sha256(prediction) == record.get("prediction_sha256"):
            retained[page_id] = record
    return retained


def next_attempt(raw_root: Path, filename: str) -> tuple[Path, int]:
    page_root = raw_root / Path(filename).with_suffix("")
    attempts = page_root / "attempts"
    existing: list[int] = []
    if attempts.is_dir():
        for child in attempts.iterdir():
            if child.is_dir() and child.name.startswith("attempt-"):
                try:
                    existing.append(int(child.name.removeprefix("attempt-")))
                except ValueError:
                    pass
    number = max(existing, default=0) + 1
    return attempts / f"attempt-{number:04d}", number


def preserve_unverified_prediction(prediction: Path, raw_root: Path, filename: str) -> str | None:
    if not prediction.exists():
        return None
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = raw_root / Path(filename).with_suffix("") / "interrupted_predictions" / stamp / prediction.name
    destination.parent.mkdir(parents=True, exist_ok=False)
    shutil.move(str(prediction), str(destination))
    return str(destination)


def load_official_modules(toolkit_root: Path) -> tuple[Any, Any]:
    """Load upstream source directly from the pinned checkout, not a reimplementation."""
    utils_dir = toolkit_root / "inference" / "utils"
    task_path = utils_dir / "tasks.py"
    output_path = utils_dir / "output_utils.py"
    if not task_path.is_file() or not output_path.is_file():
        raise FileNotFoundError(f"Pinned HunyuanOCR utilities missing below {utils_dir}")

    def load(name: str, source: Path) -> Any:
        spec = importlib.util.spec_from_file_location(name, source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import official source: {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load("hunyuan_tasks", task_path), load("hunyuan_output_utils", output_path)


def server_is_ready(host: str, port: int, expected_model: str) -> None:
    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url=f"http://{host}:{port}/v1", timeout=20.0)
    models = client.models.list()
    names = {item.id for item in models.data}
    if expected_model not in names:
        raise RuntimeError(f"vLLM server is reachable but does not expose {expected_model!r}: {sorted(names)}")


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return jsonable(value.model_dump(mode="json"))
        except TypeError:
            return jsonable(value.model_dump())
    return str(value)


def instrumented_stream(client: Any, common_kwargs: dict[str, Any], output_path: Path, config: dict[str, Any], output_utils: Any) -> tuple[str, bool, str | None, dict[str, Any] | None]:
    """The official streaming algorithm, augmented only with auditable event capture."""
    stream = client.chat.completions.create(stream=True, **common_kwargs)
    parts: list[str] = []
    acc_len = 0
    next_check_at = int(config["tail_repeat_min_start_chars"])
    check_step = int(config["tail_repeat_check_interval_chars"])
    threshold = int(config["tail_repeat_threshold"])
    early_stopped = False
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    with output_path.open("x", encoding="utf-8") as events:
        for sequence, event in enumerate(stream):
            event_record = {"sequence": sequence, "event": jsonable(event)}
            events.write(json.dumps(event_record, ensure_ascii=False, sort_keys=True) + "\n")
            choices = getattr(event, "choices", None) or []
            if choices:
                maybe_finish = getattr(choices[0], "finish_reason", None)
                if maybe_finish is not None:
                    finish_reason = str(maybe_finish)
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta is not None else None
                if piece:
                    parts.append(piece)
                    acc_len += len(piece)
                    if acc_len >= next_check_at:
                        next_check_at = acc_len + check_step
                        # Same upstream scope and predicate: the most recent 8k chars.
                        if output_utils.has_tail_repetition("".join(parts)[-8000:], min_repeats=threshold):
                            early_stopped = True
                            try:
                                stream.close()
                            except Exception:
                                pass
                            break
            maybe_usage = getattr(event, "usage", None)
            if maybe_usage is not None:
                usage = jsonable(maybe_usage)
    return "".join(parts), early_stopped, finish_reason, usage


def copy_runtime_evidence(run_dir: Path, config: dict[str, Any]) -> None:
    environment = config["environment"]
    model = config["model"]
    runtime = run_dir / "runtime"
    cache_root = Path(environment["model_cache_root"])
    toolkit_root = Path(environment["toolkit_root"])
    for source, destination in (
        (cache_root / "pinned-models.json", runtime / "pinned_model_artifacts.json"),
        (cache_root / "pinned-environment-freeze.txt", runtime / "pinned_environment_freeze.txt"),
        (cache_root / "pinned-toolkit.json", runtime / "pinned_toolkit.json"),
    ):
        if not source.is_file():
            raise FileNotFoundError(f"Required setup evidence is absent: {source}")
        shutil.copy2(source, destination)
    commit = subprocess.run(["git", "-C", str(toolkit_root), "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    actual_commit = commit.stdout.strip()
    if actual_commit != model["toolkit_commit"]:
        raise RuntimeError(f"Pinned HunyuanOCR checkout mismatch: expected {model['toolkit_commit']}, got {actual_commit or 'none'}")
    toolkit_files = [
        toolkit_root / "inference" / "utils" / "tasks.py",
        toolkit_root / "inference" / "utils" / "output_utils.py",
        toolkit_root / "inference" / "vLLM" / "infer_vllm_client.py",
        toolkit_root / "inference" / "vLLM" / "serve.sh",
    ]
    write_json(runtime / "official_client_source.json", {
        "toolkit_commit": actual_commit,
        "files": [{"path": str(path), "sha256": sha256(path)} for path in toolkit_files],
        "harness_policy": "official task/prompt/request/streaming/cleanup/normalization semantics with additive event capture",
    })


def main() -> int:
    options = parse_args()
    run_dir = options.run_dir.resolve()
    config_path = run_dir / "config" / "baseline.toml"
    manifest_path = run_dir / "manifest.jsonl"
    if not config_path.is_file() or not manifest_path.is_file():
        raise SystemExit("Run must be initialized with init_run.py or init_sharded_run.py")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    model, inference, environment = config["model"], config["inference"], config["environment"]
    if model.get("execution_mode") != "vllm_autoregressive_only":
        raise SystemExit("This runner refuses a non-AR Hunyuan configuration")
    if inference.get("stream") is not True:
        raise SystemExit("This baseline is pinned to the official streaming-client path")
    rows = read_manifest(manifest_path)
    if options.limit is not None:
        if options.limit < 1:
            raise SystemExit("--limit must be positive")
        rows = rows[: options.limit]
    predictions = run_dir / "predictions"
    raw_predictions = run_dir / "raw_predictions"
    runtime = run_dir / "runtime"
    if not predictions.is_dir() or not raw_predictions.is_dir() or not runtime.is_dir():
        raise SystemExit("Run directory has an invalid layout")
    ledger = runtime / "pages.jsonl"
    retained = load_completed_pages(ledger, predictions)
    if not options.resume and any(predictions.rglob("*.md")):
        raise SystemExit("Predictions already exist. Use --resume only after an interruption.")
    planned = [row for row in rows if row["page_id"] not in retained] if options.resume else rows
    if not options.image_root.is_dir():
        raise SystemExit(f"Image root does not exist: {options.image_root}")

    copy_runtime_evidence(run_dir, config)
    tasks, output_utils = load_official_modules(Path(environment["toolkit_root"]))
    task_type = str(inference["task_type"])
    official_prompt = tasks.get_prompt(task_type)
    if official_prompt != inference["prompt"]:
        raise RuntimeError("baseline.toml prompt differs from the pinned official task prompt")
    server_is_ready(options.host, options.port, str(config["server"]["served_model_name"]))
    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url=f"http://{options.host}:{options.port}/v1", timeout=3600.0)
    invocation = {
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": options.host,
        "port": options.port,
        "execution_mode": model["execution_mode"],
        "task_type": task_type,
        "prompt": official_prompt,
        "resume": options.resume,
        "pages_requested": len(rows),
        "pages_attempted_this_invocation": len(planned),
        "pages_retained_from_verified_terminal_records": len(rows) - len(planned),
    }
    write_json(runtime / "inference_invocation.json", invocation)
    failures_path = runtime / "failures.jsonl"

    failures_or_empty = 0
    for index, row in enumerate(planned, start=1):
        filename = str(row["prediction_filename"])
        prediction = predictions / filename
        prediction.parent.mkdir(parents=True, exist_ok=True)
        preserved = preserve_unverified_prediction(prediction, raw_predictions, filename)
        attempt_dir, attempt = next_attempt(raw_predictions, filename)
        attempt_dir.mkdir(parents=True, exist_ok=False)
        image = options.image_root / str(row["image_path"])
        started = dt.datetime.now(dt.timezone.utc)
        started_monotonic = time.monotonic()
        record: dict[str, Any] = {
            "page_id": row["page_id"], "image_path": row["image_path"], "input_sha256": row["input_sha256"],
            "prediction_filename": filename, "attempt": attempt, "resume": options.resume,
            "resumed_from_unverified_prediction": preserved, "started_at_utc": started.isoformat(),
            "task_type": task_type, "prompt": official_prompt, "generation": {
                "temperature": inference["temperature"], "top_p": inference["top_p"], "top_k": inference["top_k"],
                "repetition_penalty": inference["repetition_penalty"], "max_tokens": inference["max_tokens"], "stream": inference["stream"],
            },
        }
        try:
            if not image.is_file():
                raise FileNotFoundError(f"Image does not exist: {image}")
            actual_input = sha256(image)
            if actual_input != row["input_sha256"]:
                raise RuntimeError(f"Input hash mismatch: expected {row['input_sha256']}, got {actual_input}")
            request_metadata = {
                "server": f"http://{options.host}:{options.port}/v1", "model": config["server"]["served_model_name"],
                "messages": [{"role": "system", "content": ""}, {"role": "user", "content": [
                    {"type": "image_url", "image_url": "data:image/jpeg;base64:<omitted; image path and SHA-256 recorded>"},
                    {"type": "text", "text": official_prompt},
                ]}], "generation": record["generation"],
            }
            write_json(attempt_dir / "request_metadata.json", request_metadata)
            image_url = output_utils.encode_image_as_data_url(str(image))
            common_kwargs = {
                "model": config["server"]["served_model_name"],
                "messages": [{"role": "system", "content": ""}, {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": official_prompt},
                ]}], "max_tokens": int(inference["max_tokens"]), "temperature": float(inference["temperature"]),
                "top_p": float(inference["top_p"]), "extra_body": {"top_k": int(inference["top_k"]), "repetition_penalty": float(inference["repetition_penalty"]), "skip_special_tokens": True},
            }
            raw_text, early_stopped, finish_reason, usage = instrumented_stream(
                client, common_kwargs, attempt_dir / "server_stream_events.jsonl", inference, output_utils
            )
            (attempt_dir / "client_received_before_cleanup.md").write_text(raw_text, encoding="utf-8")
            cleaned = output_utils.clean_repeated_substrings(raw_text, min_repeats=int(inference["cleanup_min_repeats"]))
            (attempt_dir / "after_tail_cleanup.md").write_text(cleaned, encoding="utf-8")
            processed, postprocess_stats = output_utils.normalize_doc_parse_markdown(cleaned)
            (attempt_dir / "official_client_output.md").write_text(processed, encoding="utf-8")
            prediction.write_text(processed, encoding="utf-8")
            record.update({
                "status": "ok" if processed.strip() else "empty_output", "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
                "prediction_bytes": prediction.stat().st_size, "prediction_sha256": sha256(prediction),
                "raw_client_received_bytes": len(raw_text.encode("utf-8")), "raw_client_received_sha256": sha256(attempt_dir / "client_received_before_cleanup.md"),
                "tail_cleanup_changed_output": raw_text != cleaned, "doc_parse_postprocess_stats": postprocess_stats,
                "client_early_stopped_for_tail_repetition": early_stopped, "server_finish_reason": finish_reason,
                "generation_cap_status": "unknown_client_early_stop" if early_stopped else ("reached" if finish_reason == "length" else ("not_reached" if finish_reason else "unknown")),
                "server_reported_usage": usage,
            })
            if record["status"] == "empty_output":
                failures_or_empty += 1
        except Exception as error:
            prediction.write_bytes(b"")
            record.update({
                "status": "error", "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
                "prediction_bytes": 0, "prediction_sha256": sha256(prediction),
                "error_kind": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            })
            write_json(attempt_dir / "error.json", record)
            append_jsonl(failures_path, record)
            failures_or_empty += 1
        append_jsonl(ledger, record)
        print(f"Progress: {index}/{len(planned)} pages; latest={row['page_id']} status={record['status']} elapsed={record['elapsed_seconds']:.1f}s", flush=True)
    write_json(runtime / "inference_summary.json", {
        "pages_requested": len(rows), "pages_attempted_this_invocation": len(planned),
        "pages_retained_from_verified_terminal_records": len(rows) - len(planned),
        "failures_or_empty_outputs_this_invocation": failures_or_empty, "resume": options.resume,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

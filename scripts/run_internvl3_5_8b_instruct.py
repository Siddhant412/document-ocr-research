#!/usr/bin/env python3
"""Run the pinned native InternVL3.5-8B-Instruct Markdown baseline.

The runner uses InternVL's documented ``AutoModel.chat`` image path, including
its dynamic tiling preprocessor.  It intentionally does *not* enable the
model's R1/thinking system prompt.  Decoded text is saved without cleanup.
Every selected page receives one Markdown file: exceptions create a zero-byte
file plus a full failure record, so the evaluator denominator is unchanged.
``--resume`` retains only hash-verified terminal ledger records.
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
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
    """Return pages whose current prediction matches a terminal ledger record."""
    if not runtime_path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in runtime_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # A torn append must not prevent a restart.
        if record.get("status") in TERMINAL_STATUSES and isinstance(record.get("page_id"), str):
            records[record["page_id"]] = record

    completed: dict[str, dict[str, Any]] = {}
    for page_id, record in records.items():
        filename = record.get("prediction_filename")
        if not isinstance(filename, str):
            continue
        prediction_path = predictions_dir / filename
        if (
            prediction_path.is_file()
            and prediction_path.stat().st_size == record.get("prediction_bytes")
            and digest(prediction_path) == record.get("prediction_sha256")
        ):
            completed[page_id] = record
    return completed


def plan_pages(
    manifest: list[dict[str, Any]], predictions_dir: Path, runtime_path: Path, resume: bool
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    completed = load_completed_pages(runtime_path, predictions_dir)
    if not resume and list(predictions_dir.glob("*.md")):
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
    numbers: list[int] = []
    if attempts_dir.is_dir():
        for child in attempts_dir.iterdir():
            if child.is_dir() and child.name.startswith("attempt-"):
                try:
                    numbers.append(int(child.name.removeprefix("attempt-")))
                except ValueError:
                    pass
    attempt = max(numbers, default=0) + 1
    return attempts_dir / f"attempt-{attempt:04d}", attempt


def preserve_unverified_prediction(prediction_path: Path, page_raw_dir: Path) -> Path | None:
    if not prediction_path.exists():
        return None
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = page_raw_dir / "interrupted_predictions" / timestamp / prediction_path.name
    destination.parent.mkdir(parents=True, exist_ok=False)
    shutil.move(str(prediction_path), str(destination))
    return destination


def copy_runtime_artifacts(run_dir: Path, config: dict[str, Any]) -> str:
    """Copy immutable cache evidence into the run before model loading."""
    cache_root = Path(config["environment"]["model_cache_root"])
    artifact_manifest = cache_root / "pinned-models.json"
    environment_freeze = cache_root / "pinned-environment-freeze.txt"
    if not artifact_manifest.is_file() or not environment_freeze.is_file():
        raise FileNotFoundError(
            "Pinned InternVL cache evidence is absent. Run "
            "setup_internvl3_5_8b_instruct_environment.sh first."
        )
    runtime_dir = run_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact_manifest, runtime_dir / "pinned_model_artifacts.json")
    shutil.copy2(environment_freeze, runtime_dir / "pinned_environment_freeze.txt")
    return digest(artifact_manifest)


def pinned_snapshot_path(config: dict[str, Any]) -> Path:
    cache_root = Path(config["environment"]["model_cache_root"])
    manifest_path = cache_root / "pinned-models.json"
    artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshots = artifact_manifest.get("snapshots", [])
    if len(snapshots) != 1:
        raise RuntimeError(f"Expected one InternVL snapshot in {manifest_path}, found {len(snapshots)}")
    snapshot = snapshots[0]
    expected_model = config["model"]["huggingface_model"]
    expected_revision = config["model"]["huggingface_revision"]
    if snapshot.get("repository") != expected_model or snapshot.get("revision") != expected_revision:
        raise RuntimeError("Pinned checkpoint repository or revision does not match baseline.toml")
    path = Path(snapshot["snapshot_path"])
    if not path.is_dir():
        raise FileNotFoundError(f"Pinned model snapshot no longer exists: {path}")
    return path


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Official InternVL dynamic-tiling ratio selection, kept deterministic."""
    best_ratio = (1, 1)
    best_diff = float("inf")
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        difference = abs(aspect_ratio - target_aspect_ratio)
        if difference < best_diff:
            best_ratio, best_diff = ratio, difference
        elif difference == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def build_transform(image_size: int) -> Any:
    """Build the normalization transform in the official model-card example."""
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import InterpolationMode

    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def dynamic_preprocess(
    image: Any,
    min_num: int,
    max_num: int,
    image_size: int,
    use_thumbnail: bool,
) -> list[Any]:
    """Split an image as prescribed by InternVL's official single-image example."""
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    target_ratios = sorted({
        (i, j)
        for total_tiles in range(min_num, max_num + 1)
        for i in range(1, total_tiles + 1)
        for j in range(1, total_tiles + 1)
        if min_num <= i * j <= max_num
    }, key=lambda ratio: ratio[0] * ratio[1])
    target_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, original_width, original_height, image_size
    )
    target_width, target_height = image_size * target_ratio[0], image_size * target_ratio[1]
    blocks = target_ratio[0] * target_ratio[1]
    resized_image = image.resize((target_width, target_height))
    processed_images = []
    for block_index in range(blocks):
        left = (block_index % (target_width // image_size)) * image_size
        top = (block_index // (target_width // image_size)) * image_size
        processed_images.append(resized_image.crop((left, top, left + image_size, top + image_size)))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(
    image_path: Path, image_size: int, min_num_tiles: int, max_num_tiles: int, use_thumbnail: bool
) -> tuple[Any, int]:
    from PIL import Image
    import torch

    with Image.open(image_path) as image:
        images = dynamic_preprocess(
            image.convert("RGB"), min_num_tiles, max_num_tiles, image_size, use_thumbnail
        )
    transform = build_transform(image_size)
    return torch.stack([transform(tile) for tile in images]), len(images)


def cuda_peak_memory(torch_module: Any) -> int | None:
    try:
        if torch_module.cuda.is_available():
            return int(torch_module.cuda.max_memory_allocated())
    except Exception:
        pass
    return None


def reset_cuda_peak_memory(torch_module: Any) -> None:
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def output_token_count(tokenizer: Any, text: str) -> int | None:
    try:
        return int(len(tokenizer.encode(text, add_special_tokens=False)))
    except Exception:
        return None


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
    inference = config["inference"]
    if inference.get("mode") != "normal_instruct" or bool(inference.get("thinking_system_prompt")):
        raise SystemExit("This benchmark is defined only for InternVL normal instruct mode without a thinking prompt")
    if bool(inference["do_sample"]) or int(inference["num_beams"]) != 1:
        raise SystemExit("This benchmark is defined only for greedy decoding (do_sample=false, num_beams=1)")

    predictions_dir = options.run_dir / "predictions"
    raw_dir = options.run_dir / "raw_predictions"
    runtime_path = options.run_dir / "runtime" / "pages.jsonl"
    failures_path = options.run_dir / "failures.jsonl"
    pending, completed = plan_pages(manifest, predictions_dir, runtime_path, options.resume)
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
    os.environ.setdefault("HF_HOME", str(Path(config["environment"]["model_cache_root"]) / "huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch_module: Any | None = None
    tokenizer: Any | None = None
    model: Any | None = None
    initialization_error: Exception | None = None
    initialization_traceback: str | None = None
    model_artifact_manifest_sha256: str | None = None
    snapshot: Path | None = None
    try:
        model_artifact_manifest_sha256 = copy_runtime_artifacts(options.run_dir, config)
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch_module = torch
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot see a CUDA GPU; do not run this benchmark on CPU")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        snapshot = pinned_snapshot_path(config)
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, trust_remote_code=True, use_fast=False, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            snapshot,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=bool(inference["use_flash_attn"]),
            trust_remote_code=True,
            local_files_only=True,
        ).eval().cuda()
        (options.run_dir / "runtime" / "model_initialization.json").write_text(
            json.dumps(
                {
                    "initialized_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "model_snapshot_path": str(snapshot),
                    "model_artifact_manifest_sha256": model_artifact_manifest_sha256,
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                    "normal_instruct_mode": True,
                    "thinking_system_prompt_set": False,
                    "status": "ok",
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
    except Exception as error:  # Record every page rather than skipping on a setup failure.
        initialization_error = error
        initialization_traceback = traceback.format_exc()
        (options.run_dir / "runtime" / "model_initialization.json").write_text(
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
        preserved = preserve_unverified_prediction(prediction_path, page_raw_dir) if options.resume else None
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
            "resumed_from_unverified_prediction": str(preserved) if preserved else None,
            "native_inference": {
                "interface": "AutoModel.chat",
                "mode": "normal_instruct",
                "thinking_system_prompt_set": False,
                "model_artifact_manifest_sha256": model_artifact_manifest_sha256,
                "image_size": int(inference["image_size"]),
                "min_num_tiles": int(inference["min_num_tiles"]),
                "max_num_tiles": int(inference["max_num_tiles"]),
                "use_thumbnail": bool(inference["use_thumbnail"]),
            },
            "generation": {
                "do_sample": False,
                "temperature": float(inference["temperature"]),
                "temperature_passed_to_generate": False,
                "num_beams": 1,
                "max_new_tokens": int(inference["max_new_tokens"]),
            },
        }
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest image does not exist: {image_path}")
            attempt_dir.mkdir(parents=True, exist_ok=False)
            if initialization_error is not None or model is None or tokenizer is None or torch_module is None:
                (attempt_dir / "model_initialization_error.txt").write_text(
                    initialization_traceback or repr(initialization_error), encoding="utf-8"
                )
                raise RuntimeError("InternVL model initialization failed") from initialization_error
            pixel_values, tile_count = load_image(
                image_path,
                int(inference["image_size"]),
                int(inference["min_num_tiles"]),
                int(inference["max_num_tiles"]),
                bool(inference["use_thumbnail"]),
            )
            question = "<image>\n" + str(inference["prompt"])
            request = {
                "question": question,
                "generation_config": {
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": int(inference["max_new_tokens"]),
                },
                "temperature_note": "Greedy decoding; temperature=0.0 is not passed when do_sample=false.",
                "vision_tile_count": tile_count,
            }
            (attempt_dir / "request.json").write_text(
                json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            record["vision_tile_count"] = tile_count
            pixel_values = pixel_values.to(dtype=torch_module.bfloat16, device="cuda")
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), torch_module.inference_mode():
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    {"do_sample": False, "num_beams": 1, "max_new_tokens": int(inference["max_new_tokens"])},
                )
            output_text = str(response)
            (attempt_dir / "stdout.txt").write_text(stdout.getvalue(), encoding="utf-8")
            (attempt_dir / "stderr.txt").write_text(stderr.getvalue(), encoding="utf-8")
            raw_output = attempt_dir / "model_output.md"
            raw_output.write_text(output_text, encoding="utf-8")
            temporary_prediction = attempt_dir / f"{prediction_path.name}.tmp"
            shutil.copyfile(raw_output, temporary_prediction)
            os.replace(temporary_prediction, prediction_path)
            record["decoded_output_token_count"] = output_token_count(tokenizer, output_text)
            record["generation_reached_cap"] = (
                record["decoded_output_token_count"] is not None
                and record["decoded_output_token_count"] >= int(inference["max_new_tokens"])
            )
            record["status"] = "ok" if prediction_path.stat().st_size else "empty_output"
            record["prediction_sha256"] = digest(prediction_path)
            record["prediction_bytes"] = prediction_path.stat().st_size
            if record["status"] != "ok":
                failures += 1
                append_jsonl(failures_path, {**record, "failure_kind": "empty_model_output", "traceback": None})
        except Exception as error:  # Keep every benchmark page in the denominator.
            failures += 1
            attempt_dir.mkdir(parents=True, exist_ok=True)
            (attempt_dir / "stdout.txt").write_text(stdout.getvalue(), encoding="utf-8")
            (attempt_dir / "stderr.txt").write_text(stderr.getvalue(), encoding="utf-8")
            if prediction_path.exists():
                shutil.move(str(prediction_path), str(attempt_dir / "prediction_present_at_error.md"))
            prediction_path.touch()  # Required empty Markdown file for a failed inference.
            record.update({"status": "error", "prediction_sha256": digest(prediction_path), "prediction_bytes": 0})
            append_jsonl(
                failures_path,
                {**record, "failure_kind": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()},
            )
        finally:
            record["elapsed_seconds"] = round(time.perf_counter() - started, 6)
            if torch_module is not None:
                record["gpu_max_memory_allocated_bytes"] = cuda_peak_memory(torch_module)
                reset_cuda_peak_memory(torch_module)
            append_jsonl(runtime_path, record)
            completed_total = len(completed) + processed_this_invocation
            print(
                f"Progress: {completed_total}/{len(manifest)} pages ({completed_total / len(manifest):.1%}); "
                f"latest={page_id} status={record['status']} elapsed={record['elapsed_seconds']:.1f}s",
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

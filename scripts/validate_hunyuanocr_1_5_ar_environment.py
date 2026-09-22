#!/usr/bin/env python3
"""Fail fast unless the pinned HunyuanOCR CUDA-13 vLLM AR stack is usable.

The required vLLM model registration is incompatible with Transformers 5.13.
This guard enforces the verified vLLM 0.25.1 / Transformers 5.5.3 pairing,
image-only dependency set, and eager launch support before any benchmark page
is submitted.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 runtime
    import tomli as tomllib


def exact_version(requirement: str) -> str:
    prefix, separator, version = requirement.partition("==")
    if not separator or not prefix or not version:
        raise ValueError(f"Expected an exact requirement, got {requirement!r}")
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    options = parser.parse_args()
    config = tomllib.loads(options.baseline_config.read_text(encoding="utf-8"))
    environment = config["environment"]
    model = config["model"]
    server = config["server"]

    expected_environment_root = Path(str(environment["environment_root"])).resolve()
    # A venv's executable is commonly a symlink to the system interpreter;
    # sys.prefix, unlike sys.executable.resolve(), identifies the active venv.
    if Path(sys.prefix).resolve() != expected_environment_root:
        raise SystemExit(
            f"Expected active environment {expected_environment_root}, got {Path(sys.prefix).resolve()}"
        )
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"Expected verified CUDA-13 AR Python 3.12, got {sys.version}")

    import torch
    import transformers
    import vllm
    from vllm.model_executor.models.registry import ModelRegistry

    expected_vllm = exact_version(str(environment["vllm_requirement"]))
    expected_transformers = exact_version(str(environment["transformers_requirement"]))
    expected_cuda_components = (
        ("nvidia-cuda-cccl", exact_version(str(environment["cuda_cccl_requirement"]))),
        ("nvidia-cuda-crt", exact_version(str(environment["cuda_crt_requirement"]))),
        ("nvidia-cuda-nvcc", exact_version(str(environment["cuda_nvcc_requirement"]))),
        ("nvidia-cuda-nvrtc", exact_version(str(environment["cuda_nvrtc_requirement"]))),
        ("nvidia-cuda-runtime", exact_version(str(environment["cuda_runtime_requirement"]))),
        ("nvidia-nvvm", exact_version(str(environment["cuda_nvvm_requirement"]))),
    )
    expected_tokenizers = exact_version(str(environment["tokenizers_requirement"]))
    expected_torch = exact_version(str(environment["torch_requirement"]))
    expected_flash_attention = exact_version(str(environment["flash_attention_requirement"]))
    expected_flashinfer = exact_version(str(environment["flashinfer_requirement"]))
    expected_runai = exact_version(str(environment["runai_model_streamer_requirement"]))
    if vllm.__version__ != expected_vllm:
        raise SystemExit(f"Expected vLLM {expected_vllm}, got {vllm.__version__}")
    if transformers.__version__ != expected_transformers:
        raise SystemExit(f"Expected Transformers {expected_transformers}, got {transformers.__version__}")
    if torch.__version__ != expected_torch:
        raise SystemExit(f"Expected Torch {expected_torch}, got {torch.__version__}")
    for distribution, expected in expected_cuda_components:
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            installed = None
        if installed != expected:
            raise SystemExit(
                f"Expected CUDA component {distribution} {expected}, got {installed or 'not installed'}"
            )
    for distribution, expected in (
        ("tokenizers", expected_tokenizers),
        ("flash-attn", expected_flash_attention),
        ("flashinfer-python", expected_flashinfer),
        ("runai-model-streamer", expected_runai),
    ):
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            installed = None
        if installed != expected:
            raise SystemExit(f"Expected {distribution} {expected}, got {installed or 'not installed'}")
    try:
        torchcodec = metadata.version("torchcodec")
    except metadata.PackageNotFoundError:
        torchcodec = None
    if torchcodec is not None:
        raise SystemExit(f"torchcodec must be absent for Hunyuan's image-only vLLM path, got {torchcodec}")
    try:
        import flash_attn  # noqa: F401
    except Exception as error:
        raise SystemExit(f"flash-attn cannot be imported: {error}") from error
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot see an NVIDIA GPU; refusing CPU fallback")
    if not str(torch.version.cuda).startswith(str(environment["cuda_family"]) + "."):
        raise SystemExit(f"Expected a CUDA {environment['cuda_family']}.x PyTorch wheel, got {torch.version.cuda}")
    nvidia_smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True, capture_output=True, check=False,
    )
    if nvidia_smi.returncode != 0:
        raise SystemExit(f"Could not query NVIDIA driver: {nvidia_smi.stderr.strip()}")
    driver = nvidia_smi.stdout.strip().splitlines()[0]
    try:
        driver_major = int(driver.split(".", 1)[0])
    except ValueError as error:
        raise SystemExit(f"Could not parse NVIDIA driver version: {driver!r}") from error
    if driver_major < int(environment["minimum_nvidia_driver_major"]):
        raise SystemExit(
            f"Expected NVIDIA driver >= {environment['minimum_nvidia_driver_major']} for CUDA 13, got {driver}"
        )
    architecture = "HunYuanVLForConditionalGeneration"
    if architecture not in ModelRegistry.get_supported_archs():
        raise SystemExit(f"vLLM {vllm.__version__} does not register {architecture}")
    if server.get("enforce_eager") is not True:
        raise SystemExit("Hunyuan xDRoPE AR baseline requires server.enforce_eager=true")

    cli = subprocess.run(["vllm", "serve", "--help=all"], text=True, capture_output=True, check=False)
    if cli.returncode != 0:
        raise SystemExit(f"Could not inspect the installed vLLM serve CLI: {cli.stderr.strip()}")
    required_flags = (
        "--served-model-name", "--limit-mm-per-prompt", "--trust-remote-code",
        "--port", "--gpu-memory-utilization", "--max-model-len",
        "--max-num-batched-tokens", "--enforce-eager",
    )
    missing_flags = [flag for flag in required_flags if flag not in cli.stdout]
    if missing_flags:
        raise SystemExit(f"vLLM {vllm.__version__} lacks required official serve flags: {missing_flags}")

    commit = subprocess.run(
        ["git", "-C", str(options.toolkit_root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    if commit != model["toolkit_commit"]:
        raise SystemExit(f"Pinned Hunyuan toolkit mismatch: expected {model['toolkit_commit']}, got {commit or 'none'}")
    for relative in ("inference/vLLM/serve.sh", "inference/vLLM/infer_vllm_client.py", "inference/utils/tasks.py", "inference/utils/output_utils.py"):
        if not (options.toolkit_root / relative).is_file():
            raise SystemExit(f"Official Hunyuan source is missing: {options.toolkit_root / relative}")

    print(json.dumps({
        "status": "ok",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_build_components": dict(expected_cuda_components),
        "nvidia_driver": driver,
        "transformers": transformers.__version__,
        "tokenizers": expected_tokenizers,
        "vllm": vllm.__version__,
        "flash_attention": expected_flash_attention,
        "flashinfer": expected_flashinfer,
        "torchcodec": "absent",
        "architecture": architecture,
        "enforce_eager": True,
        "required_serve_flags": "ok",
        "gpu": torch.cuda.get_device_name(0),
        "toolkit_commit": commit,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

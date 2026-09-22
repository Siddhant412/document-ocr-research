#!/usr/bin/env bash
# Create the pinned CUDA 12.4 Qwen2.5-VL-7B-Instruct environment on a RunPod Pod.
# Run this from the copied repository at /workspace/document-ocr-research.

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
HARNESS_DIR="${HARNESS_DIR:-${WORKSPACE_ROOT}/document-ocr-research}"
ENV_DIR="${ENV_DIR:-${WORKSPACE_ROOT}/venvs/qwen2.5-vl-7b-instruct}"

if [[ ! -f "${HARNESS_DIR}/configs/qwen2.5-vl-7b-instruct.toml" ]]; then
  echo "Harness directory is missing Qwen config: ${HARNESS_DIR}" >&2
  exit 2
fi

mkdir -p "$(dirname "${ENV_DIR}")"
python3 -m venv "${ENV_DIR}"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install transformers==4.49.0 accelerate qwen-vl-utils==0.0.8
python -m pip install ninja packaging
MAX_JOBS="${MAX_JOBS:-8}" python -m pip install flash-attn==2.7.3 --no-build-isolation

python - <<'PY'
import flash_attn
import torch
import transformers
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

assert torch.cuda.is_available(), "PyTorch cannot see the NVIDIA GPU"
assert torch.version.cuda == "12.4", f"Expected CUDA 12.4 PyTorch wheel, got {torch.version.cuda}"
print(
    "Environment verified:",
    f"torch={torch.__version__}",
    f"cuda={torch.version.cuda}",
    f"transformers={transformers.__version__}",
    f"flash_attn={flash_attn.__version__}",
    f"gpu={torch.cuda.get_device_name(0)}",
)
PY

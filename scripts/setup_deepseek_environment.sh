#!/usr/bin/env bash
# Create the pinned CUDA 12.4 DeepSeek-OCR Transformers environment on a RunPod Pod.
# Run this from the repository root after it has been copied to /workspace.

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
HARNESS_DIR="${HARNESS_DIR:-${WORKSPACE_ROOT}/document-ocr-research}"
ENV_DIR="${ENV_DIR:-${WORKSPACE_ROOT}/venvs/deepseek-ocr}"
MODEL_SOURCE_DIR="${MODEL_SOURCE_DIR:-${WORKSPACE_ROOT}/sources/DeepSeek-OCR}"
MODEL_REPOSITORY="https://github.com/deepseek-ai/DeepSeek-OCR.git"
MODEL_COMMIT="09eaf526153e7a01ed16c9dea8c96282aaea29c0"

if [[ ! -f "${HARNESS_DIR}/configs/baseline.toml" ]]; then
  echo "Harness directory is missing configs/baseline.toml: ${HARNESS_DIR}" >&2
  exit 2
fi

mkdir -p "$(dirname "${ENV_DIR}")" "$(dirname "${MODEL_SOURCE_DIR}")"
python3 -m venv "${ENV_DIR}"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

if [[ ! -d "${MODEL_SOURCE_DIR}/.git" ]]; then
  git clone "${MODEL_REPOSITORY}" "${MODEL_SOURCE_DIR}"
fi
if ! git -C "${MODEL_SOURCE_DIR}" cat-file -e "${MODEL_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${MODEL_SOURCE_DIR}" fetch --quiet origin
fi
git -C "${MODEL_SOURCE_DIR}" checkout --quiet --detach "${MODEL_COMMIT}"

python -m pip install -r "${MODEL_SOURCE_DIR}/requirements.txt"
python -m pip install ninja packaging
MAX_JOBS="${MAX_JOBS:-8}" python -m pip install flash-attn==2.7.3 --no-build-isolation

python - <<'PY'
import flash_attn
import torch
import transformers

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

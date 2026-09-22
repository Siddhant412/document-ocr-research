#!/usr/bin/env bash
# Create the pinned InternVL3.5-8B-Instruct native environment on a CUDA-12.8 NVIDIA Pod.

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
HARNESS_DIR="${HARNESS_DIR:-${WORKSPACE_ROOT}/document-ocr-research}"
ENV_DIR="${ENV_DIR:-${WORKSPACE_ROOT}/venvs/internvl3.5-8b-instruct}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-${WORKSPACE_ROOT}/internvl3.5-8b-instruct-cache}"
CONFIG_PATH="${HARNESS_DIR}/configs/internvl3.5-8b-instruct.toml"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Harness directory is missing InternVL config: ${CONFIG_PATH}" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "No NVIDIA GPU runtime is visible. Do not run InternVL inference on CPU." >&2
  exit 2
fi

mkdir -p "$(dirname "${ENV_DIR}")" "${MODEL_CACHE_ROOT}"
python3 -m venv "${ENV_DIR}"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

export HF_HOME="${MODEL_CACHE_ROOT}/huggingface"
export PYTHONUTF8=1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==4.52.1 accelerate==1.6.0 einops==0.8.1 timm==1.0.15 pillow==11.2.1 huggingface_hub==0.30.2
python -m pip install ninja packaging
MAX_JOBS="${MAX_JOBS:-8}" python -m pip install flash-attn==2.7.4.post1 --no-build-isolation

python - <<'PY'
import flash_attn
import torch
import transformers

assert torch.cuda.is_available(), "PyTorch cannot see the NVIDIA GPU"
assert torch.version.cuda == "12.8", f"Expected CUDA 12.8 PyTorch wheel, got {torch.version.cuda}"
print(
    "Environment verified:",
    f"torch={torch.__version__}",
    f"cuda={torch.version.cuda}",
    f"transformers={transformers.__version__}",
    f"flash_attn={flash_attn.__version__}",
    f"gpu={torch.cuda.get_device_name(0)}",
)
PY

python "${HARNESS_DIR}/scripts/prefetch_internvl3_5_8b_instruct.py" \
  --baseline-config "${CONFIG_PATH}" \
  --model-cache-root "${MODEL_CACHE_ROOT}"
python -m pip freeze | sort > "${MODEL_CACHE_ROOT}/pinned-environment-freeze.txt"
printf '%s\n' "Pinned InternVL environment ready: ${ENV_DIR}"

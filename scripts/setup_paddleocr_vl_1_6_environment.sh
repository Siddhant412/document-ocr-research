#!/usr/bin/env bash
# Create the isolated PaddleOCR-VL-1.6 native-pipeline environment on a RunPod Pod.
# Run this from /workspace/document-ocr-research after copying the harness there.

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
HARNESS_DIR="${HARNESS_DIR:-${WORKSPACE_ROOT}/document-ocr-research}"
ENV_DIR="${ENV_DIR:-${WORKSPACE_ROOT}/venvs/paddleocr-vl-1.6}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-${WORKSPACE_ROOT}/paddleocr-vl-1.6-cache}"
CONFIG_PATH="${HARNESS_DIR}/configs/paddleocr-vl-1.6.toml"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Harness directory is missing PaddleOCR-VL config: ${CONFIG_PATH}" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "No NVIDIA GPU runtime is visible. Do not run PaddleOCR-VL inference on CPU." >&2
  exit 2
fi

mkdir -p "$(dirname "${ENV_DIR}")" "${MODEL_CACHE_ROOT}"
python3 -m venv "${ENV_DIR}"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

export HF_HOME="${MODEL_CACHE_ROOT}/huggingface-hub"
export PADDLE_PDX_CACHE_HOME="${MODEL_CACHE_ROOT}/paddlex"
export PADDLE_PDX_MODEL_SOURCE="huggingface"
export PYTHONUTF8=1

python -m pip install --upgrade pip setuptools wheel
# CUDA 11.8 is deliberately selected: it is supported by PaddleOCR-VL and by
# the CUDA 12.4-capable RTX 4090 driver used in the earlier baselines.
python -m pip install paddlepaddle-gpu==3.2.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
python -m pip install "paddleocr[doc-parser]==3.6.0" "paddlex[genai-client,ocr]==3.6.0"

python - <<'PY'
import paddle
import paddleocr
import paddlex

assert paddle.is_compiled_with_cuda(), "Installed PaddlePaddle is not CUDA-enabled"
paddle.set_device("gpu:0")
print(
    "Environment verified:",
    f"paddle={paddle.__version__}",
    f"paddle_cuda={paddle.version.cuda()}",
    f"paddleocr={paddleocr.__version__}",
    f"paddlex={paddlex.__version__}",
    f"device={paddle.get_device()}",
)
PY

python "${HARNESS_DIR}/scripts/prefetch_paddleocr_vl_models.py" \
  --baseline-config "${CONFIG_PATH}" \
  --model-cache-root "${MODEL_CACHE_ROOT}"

python -m pip freeze | sort > "${MODEL_CACHE_ROOT}/pinned-environment-freeze.txt"
printf '%s\n' "Pinned PaddleOCR-VL environment ready: ${ENV_DIR}"

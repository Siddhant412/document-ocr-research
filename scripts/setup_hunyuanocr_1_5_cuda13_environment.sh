#!/usr/bin/env bash
# Set up the pinned HunyuanOCR-1.5 CUDA-13 vLLM AR environment.
#
# This intentionally runs only the official vLLM autoregressive server. It
# never starts DFlash. The pins below were verified by a real server startup:
# vLLM 0.25.1 + Transformers 5.5.3, with torchcodec removed and vLLM launched
# in eager mode by the sharded harness.

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
HARNESS_DIR="${HARNESS_DIR:-${WORKSPACE_ROOT}/document-ocr-research}"
CONFIG_PATH="${HARNESS_DIR}/configs/hunyuanocr-1.5.toml"

[[ -f "$CONFIG_PATH" ]] || { echo "Missing config: $CONFIG_PATH" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "No NVIDIA GPU is visible; refusing CPU fallback." >&2; exit 2; }

readarray -t HUNYUAN_VALUES < <(python3 - "$CONFIG_PATH" <<'PY'
import sys, tomllib
c = tomllib.loads(open(sys.argv[1], encoding="utf-8").read())
e, m = c["environment"], c["model"]
for value in (
    e["environment_root"], e["model_cache_root"], e["toolkit_root"],
    m["toolkit_repository"], m["toolkit_commit"], e["python"],
    e["vllm_requirement"], e["transformers_requirement"],
    e["cuda_cccl_requirement"], e["cuda_crt_requirement"],
    e["cuda_nvcc_requirement"], e["cuda_nvrtc_requirement"],
    e["cuda_runtime_requirement"], e["cuda_nvvm_requirement"],
    e["tokenizers_requirement"], e["flash_attention_requirement"],
    e["flashinfer_requirement"], e["runai_model_streamer_requirement"],
    e["openai_requirement"], e["pillow_requirement"],
):
    print(value)
PY
)
ENV_DIR="${ENV_DIR:-${HUNYUAN_VALUES[0]}}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-${HUNYUAN_VALUES[1]}}"
WHEEL_DIR="${HUNYUAN_WHEEL_DIR:-${MODEL_CACHE_ROOT}/wheels}"
TOOLKIT_ROOT="${HUNYUAN_VALUES[2]}"
TOOLKIT_REPOSITORY="${HUNYUAN_VALUES[3]}"
TOOLKIT_COMMIT="${HUNYUAN_VALUES[4]}"
PYTHON_VERSION="${HUNYUAN_VALUES[5]}"
VLLM_REQUIREMENT="${HUNYUAN_VALUES[6]}"
TRANSFORMERS_REQUIREMENT="${HUNYUAN_VALUES[7]}"
CUDA_CCCL_REQUIREMENT="${HUNYUAN_VALUES[8]}"
CUDA_CRT_REQUIREMENT="${HUNYUAN_VALUES[9]}"
CUDA_NVCC_REQUIREMENT="${HUNYUAN_VALUES[10]}"
CUDA_NVRTC_REQUIREMENT="${HUNYUAN_VALUES[11]}"
CUDA_RUNTIME_REQUIREMENT="${HUNYUAN_VALUES[12]}"
CUDA_NVVM_REQUIREMENT="${HUNYUAN_VALUES[13]}"
TOKENIZERS_REQUIREMENT="${HUNYUAN_VALUES[14]}"
FLASH_ATTENTION_REQUIREMENT="${HUNYUAN_VALUES[15]}"
FLASHINFER_REQUIREMENT="${HUNYUAN_VALUES[16]}"
RUNAI_REQUIREMENT="${HUNYUAN_VALUES[17]}"
OPENAI_REQUIREMENT="${HUNYUAN_VALUES[18]}"
PILLOW_REQUIREMENT="${HUNYUAN_VALUES[19]}"

[[ "$PYTHON_VERSION" == "3.12" ]] || { echo "This setup requires Python 3.12, config requested $PYTHON_VERSION" >&2; exit 2; }

if [[ "${HUNYUAN_SETUP_DRY_RUN:-0}" == "1" ]]; then
  [[ -x "$ENV_DIR/bin/python" ]] || { echo "Environment is absent: $ENV_DIR" >&2; exit 2; }
  [[ -d "$TOOLKIT_ROOT/.git" ]] || { echo "Pinned Hunyuan toolkit is absent: $TOOLKIT_ROOT" >&2; exit 2; }
  ACTUAL_COMMIT="$(git -C "$TOOLKIT_ROOT" rev-parse HEAD)"
  [[ "$ACTUAL_COMMIT" == "$TOOLKIT_COMMIT" ]] || { echo "Toolkit checkout mismatch: expected $TOOLKIT_COMMIT, got $ACTUAL_COMMIT" >&2; exit 2; }
  [[ -f "$MODEL_CACHE_ROOT/pinned-models.json" ]] || { echo "Pinned checkpoint manifest is absent: $MODEL_CACHE_ROOT/pinned-models.json" >&2; exit 2; }
  "$ENV_DIR/bin/python" "$HARNESS_DIR/scripts/validate_hunyuanocr_1_5_ar_environment.py" \
    --baseline-config "$CONFIG_PATH" --toolkit-root "$TOOLKIT_ROOT"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  exit 0
fi

mkdir -p "$(dirname "$ENV_DIR")" "$MODEL_CACHE_ROOT" "$WHEEL_DIR"
# The network volume has enough space for the reusable FlashAttention wheel,
# but not another full CUDA/vLLM runtime. Opt in explicitly if a larger volume
# is attached and retaining uv's full package cache is desired.
if [[ "${HUNYUAN_PERSIST_UV_CACHE:-0}" == "1" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-${MODEL_CACHE_ROOT}/uv-cache}"
fi
BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-${WORKSPACE_ROOT}/venvs/hunyuanocr-bootstrap}"
if [[ ! -x "$BOOTSTRAP_ENV/bin/python" ]]; then
  python3 -m venv "$BOOTSTRAP_ENV"
fi
CONTROL_PYTHON="$BOOTSTRAP_ENV/bin/python"
"$CONTROL_PYTHON" -m pip install --upgrade pip uv
UV="$CONTROL_PYTHON -m uv"

if [[ ! -d "$TOOLKIT_ROOT/.git" ]]; then
  git clone "$TOOLKIT_REPOSITORY" "$TOOLKIT_ROOT"
fi
git -C "$TOOLKIT_ROOT" fetch --tags origin
git -C "$TOOLKIT_ROOT" checkout --detach "$TOOLKIT_COMMIT"
ACTUAL_COMMIT="$(git -C "$TOOLKIT_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$TOOLKIT_COMMIT" ]] || { echo "Toolkit checkout mismatch" >&2; exit 2; }

if [[ ! -d "$ENV_DIR" ]]; then
  PYTHON312="${HUNYUAN_PYTHON312:-python3.12}"
  command -v "$PYTHON312" >/dev/null || { echo "Python 3.12 is unavailable: $PYTHON312" >&2; exit 2; }
  $UV venv --python "$PYTHON312" "$ENV_DIR"
fi

$UV pip install --python "$ENV_DIR/bin/python" --torch-backend=cu130 \
  "$VLLM_REQUIREMENT" "$RUNAI_REQUIREMENT" "$OPENAI_REQUIREMENT" \
  "$PILLOW_REQUIREMENT" "huggingface_hub[cli]"
# vLLM's dependencies permit a newer CUDA compiler even though the Torch
# cu130 wheel supplies CUDA 13.0 headers.  Enforce the complete 13.0 compiler
# subset without asking the resolver to change the already-pinned runtime.
$UV pip install --python "$ENV_DIR/bin/python" --reinstall --no-deps \
  "$CUDA_CCCL_REQUIREMENT" "$CUDA_CRT_REQUIREMENT" "$CUDA_NVCC_REQUIREMENT" \
  "$CUDA_NVRTC_REQUIREMENT" "$CUDA_RUNTIME_REQUIREMENT" "$CUDA_NVVM_REQUIREMENT"
# Hunyuan's vLLM model registration is incompatible with Transformers 5.13.
# Pin its upper-bounded tokenizers dependency as well: vLLM may otherwise
# install a newer incompatible release first.
$UV pip install --python "$ENV_DIR/bin/python" "$TRANSFORMERS_REQUIREMENT" "$TOKENIZERS_REQUIREMENT" --no-deps
# HunyuanOCR is image-only. Removing torchcodec prevents its optional FFmpeg
# import from killing vLLM startup on minimal Pod images.
$UV pip uninstall --python "$ENV_DIR/bin/python" torchcodec || true

FLASH_STATUS="$($ENV_DIR/bin/python - "$FLASH_ATTENTION_REQUIREMENT" <<'PY'
import importlib.metadata as m
import sys
name, expected = sys.argv[1].split("==", 1)
try:
    installed = m.version(name)
except m.PackageNotFoundError:
    installed = None
print("ok" if installed == expected else "missing")
PY
)"
if [[ "$FLASH_STATUS" != "ok" ]]; then
  FLASH_VERSION="${FLASH_ATTENTION_REQUIREMENT#*==}"
  mapfile -t FLASH_WHEELS < <(find "$WHEEL_DIR" -maxdepth 1 -type f -name "flash_attn-${FLASH_VERSION}-*.whl" -print | sort)
  if [[ "${#FLASH_WHEELS[@]}" -gt 1 ]]; then
    printf 'Ambiguous cached FlashAttention wheels for %s:\n%s\n' "$FLASH_ATTENTION_REQUIREMENT" "${FLASH_WHEELS[*]}" >&2
    exit 2
  fi
  if [[ "${#FLASH_WHEELS[@]}" -eq 1 ]]; then
    printf 'Installing persisted FlashAttention wheel: %s\n' "${FLASH_WHEELS[0]}"
    $UV pip install --python "$ENV_DIR/bin/python" "${FLASH_WHEELS[0]}"
  elif [[ "${HUNYUAN_ALLOW_FLASHATTN_BUILD:-0}" != "1" ]]; then
    cat >&2 <<EOF
FlashAttention $FLASH_ATTENTION_REQUIREMENT is missing. Its CUDA source build
can take about an hour on a 4090, so this script refuses to start it silently.
No compatible wheel exists in the persistent cache: $WHEEL_DIR
Review the Pod's CUDA 13 toolchain, then rerun explicitly with
HUNYUAN_ALLOW_FLASHATTN_BUILD=1. A successful build will be retained there.
EOF
    exit 3
  else
    CUDA_HOME="${CUDA_HOME:-$ENV_DIR/lib/python3.12/site-packages/nvidia/cu13}"
    export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    [[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "CUDA 13 compiler missing: $CUDA_HOME/bin/nvcc" >&2; exit 2; }
    # Fast, definitive guard for mismatched pip CUDA compiler/header packages.
    # This costs seconds and must pass before a FlashAttention source build.
    CUDA_PROBE_SOURCE="$(mktemp --suffix=.cu)"
    CUDA_PROBE_OBJECT="${CUDA_PROBE_SOURCE%.cu}.o"
    trap 'rm -f "$CUDA_PROBE_SOURCE" "$CUDA_PROBE_OBJECT"' EXIT
    printf '%s\n' '#include <cuda_runtime.h>' 'int main() { return 0; }' > "$CUDA_PROBE_SOURCE"
    "$CUDA_HOME/bin/nvcc" -c "$CUDA_PROBE_SOURCE" -o "$CUDA_PROBE_OBJECT" || {
      echo "CUDA compiler/header compatibility probe failed; refusing FlashAttention build." >&2
      exit 2
    }
    rm -f "$CUDA_PROBE_SOURCE" "$CUDA_PROBE_OBJECT"
    trap - EXIT
    # uv-created environments do not necessarily include pip. Install this
    # build-only tool explicitly before invoking `python -m pip wheel`.
    $UV pip install --python "$ENV_DIR/bin/python" "pip>=24.0"
    MAX_JOBS="${MAX_JOBS:-6}" NVCC_THREADS="${NVCC_THREADS:-1}" \
      "$ENV_DIR/bin/python" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$WHEEL_DIR" "$FLASH_ATTENTION_REQUIREMENT"
    mapfile -t FLASH_WHEELS < <(find "$WHEEL_DIR" -maxdepth 1 -type f -name "flash_attn-${FLASH_VERSION}-*.whl" -print | sort)
    [[ "${#FLASH_WHEELS[@]}" -eq 1 ]] || { echo "FlashAttention build did not create one wheel in $WHEEL_DIR" >&2; exit 2; }
    $UV pip install --python "$ENV_DIR/bin/python" "${FLASH_WHEELS[0]}"
  fi
fi

export HF_HOME="$MODEL_CACHE_ROOT/huggingface"
export PYTHONUTF8=1
"$ENV_DIR/bin/python" "$HARNESS_DIR/scripts/prefetch_hunyuanocr_1_5.py" \
  --baseline-config "$CONFIG_PATH" --model-cache-root "$MODEL_CACHE_ROOT"
"$ENV_DIR/bin/python" "$HARNESS_DIR/scripts/validate_hunyuanocr_1_5_ar_environment.py" \
  --baseline-config "$CONFIG_PATH" --toolkit-root "$TOOLKIT_ROOT"
$UV pip freeze --python "$ENV_DIR/bin/python" | sort > "$MODEL_CACHE_ROOT/pinned-environment-freeze.txt"

"$ENV_DIR/bin/python" - "$TOOLKIT_ROOT" "$MODEL_CACHE_ROOT/pinned-toolkit.json" <<'PY'
import datetime as dt, hashlib, json, subprocess, sys
root, output = sys.argv[1:]
def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
files = ["inference/utils/tasks.py", "inference/utils/output_utils.py", "inference/vLLM/infer_vllm_client.py", "inference/vLLM/serve.sh"]
data = {
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "toolkit_root": root,
    "commit": subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip(),
    "files": [{"path": path, "sha256": sha(f"{root}/{path}")} for path in files],
}
open(output, "w", encoding="utf-8").write(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
printf '%s\n' "Pinned HunyuanOCR-1.5 CUDA-13 AR environment ready: $ENV_DIR"

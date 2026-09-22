#!/usr/bin/env bash
# Set up the official HunyuanOCR-1.5 AR-only vLLM environment.
#
# This follows docs/inference/archive/vLLM.md from the pinned toolkit:
# Python 3.10 + vLLM 0.18.1 + Transformers 4.57.6.  It deliberately does not
# combine the native-Transformers 5.13 environment with vLLM, use DFlash, or
# compile FlashAttention from source.

# Compatibility entrypoint: the first implementation used the archived CUDA-12
# recipe, which is incompatible with the current checkpoint/vLLM integration.
# Delegate immediately so older command notes cannot recreate that environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/setup_hunyuanocr_1_5_cuda13_environment.sh" "$@"

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
HARNESS_DIR="${HARNESS_DIR:-${WORKSPACE_ROOT}/document-ocr-research}"
CONFIG_PATH="${HARNESS_DIR}/configs/hunyuanocr-1.5.toml"

[[ -f "$CONFIG_PATH" ]] || { echo "Missing config: $CONFIG_PATH" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "No NVIDIA GPU is visible; refusing CPU fallback." >&2; exit 2; }

# Read the configuration before activating the Python-3.10 environment.  The
# Pod's base Python provides tomllib; the target environment installs tomli.
readarray -t HUNYUAN_PATHS < <(python3 - "$CONFIG_PATH" <<'PY'
import sys, tomllib
c = tomllib.loads(open(sys.argv[1], encoding="utf-8").read())
e, m = c["environment"], c["model"]
for value in (
    e["environment_root"], e["model_cache_root"], e["toolkit_root"],
    m["toolkit_repository"], m["toolkit_commit"], e["python"],
    e["vllm_requirement"], e["transformers_requirement"],
    e["openai_requirement"], e["pillow_requirement"], e["tomli_requirement"],
):
    print(value)
PY
)
ENV_DIR="${ENV_DIR:-${HUNYUAN_PATHS[0]}}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-${HUNYUAN_PATHS[1]}}"
TOOLKIT_ROOT="${TOOLKIT_ROOT:-${HUNYUAN_PATHS[2]}}"
TOOLKIT_REPOSITORY="${HUNYUAN_PATHS[3]}"
TOOLKIT_COMMIT="${HUNYUAN_PATHS[4]}"
PYTHON_VERSION="${HUNYUAN_PATHS[5]}"
VLLM_REQUIREMENT="${HUNYUAN_PATHS[6]}"
TRANSFORMERS_REQUIREMENT="${HUNYUAN_PATHS[7]}"
OPENAI_REQUIREMENT="${HUNYUAN_PATHS[8]}"
PILLOW_REQUIREMENT="${HUNYUAN_PATHS[9]}"
TOMLI_REQUIREMENT="${HUNYUAN_PATHS[10]}"

[[ "$PYTHON_VERSION" == "3.10" ]] || { echo "This AR setup requires Python 3.10, config requested $PYTHON_VERSION" >&2; exit 2; }
PYTHON310="${HUNYUAN_PYTHON310:-/usr/bin/python3.10}"
[[ -x "$PYTHON310" ]] || { echo "Official AR Python is missing: $PYTHON310" >&2; exit 2; }

if [[ "${HUNYUAN_SETUP_DRY_RUN:-0}" == "1" ]]; then
  [[ -d "$TOOLKIT_ROOT/.git" ]] || { echo "Pinned Hunyuan toolkit is absent: $TOOLKIT_ROOT" >&2; exit 2; }
  ACTUAL_COMMIT="$(git -C "$TOOLKIT_ROOT" rev-parse HEAD)"
  [[ "$ACTUAL_COMMIT" == "$TOOLKIT_COMMIT" ]] || { echo "Toolkit checkout mismatch: expected $TOOLKIT_COMMIT, got $ACTUAL_COMMIT" >&2; exit 2; }
  [[ -f "$MODEL_CACHE_ROOT/pinned-models.json" ]] || { echo "Pinned checkpoint manifest is absent: $MODEL_CACHE_ROOT/pinned-models.json" >&2; exit 2; }
  python3 - "$ENV_DIR" "$MODEL_CACHE_ROOT" <<'PY'
import json, shutil, sys
from pathlib import Path
env, cache = map(Path, sys.argv[1:])
manifest = json.loads((cache / "pinned-models.json").read_text(encoding="utf-8"))
snapshots = manifest.get("snapshots", [])
if len(snapshots) != 1 or not Path(snapshots[0].get("snapshot_path", "")).is_dir():
    raise SystemExit("Pinned checkpoint manifest does not describe one available snapshot")
disk = shutil.disk_usage(env.parent)
print(json.dumps({
    "status": "dry-run-ok",
    "environment_root": str(env),
    "model_cache_root": str(cache),
    "free_disk_gib": round(disk.free / 1024**3, 2),
    "checkpoint_snapshot": snapshots[0]["snapshot_path"],
}, sort_keys=True))
PY
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  exit 0
fi

mkdir -p "$(dirname "$ENV_DIR")" "$MODEL_CACHE_ROOT"
BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-${WORKSPACE_ROOT}/venvs/hunyuanocr-bootstrap}"
if [[ ! -x "$BOOTSTRAP_ENV/bin/python" ]]; then
  python3 -m venv "$BOOTSTRAP_ENV"
fi
CONTROL_PYTHON="$BOOTSTRAP_ENV/bin/python"
"$CONTROL_PYTHON" -m pip install --upgrade pip uv

if [[ ! -d "$ENV_DIR" ]]; then
  "$CONTROL_PYTHON" -m uv venv --python "$PYTHON310" "$ENV_DIR"
fi
# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Expected Python 3.10 for official Hunyuan AR vLLM, got {sys.version}")
PY

if [[ ! -d "$TOOLKIT_ROOT/.git" ]]; then
  git clone "$TOOLKIT_REPOSITORY" "$TOOLKIT_ROOT"
fi
git -C "$TOOLKIT_ROOT" fetch --tags origin
git -C "$TOOLKIT_ROOT" checkout --detach "$TOOLKIT_COMMIT"
ACTUAL_COMMIT="$(git -C "$TOOLKIT_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$TOOLKIT_COMMIT" ]] || { echo "Toolkit checkout mismatch" >&2; exit 2; }

export HF_HOME="$MODEL_CACHE_ROOT/huggingface"
export PYTHONUTF8=1

# These are prebuilt packages on the officially supported RTX 4090 / CUDA
# driver path.  Do not add flash-attn or CUDA compiler packages here.
"$CONTROL_PYTHON" -m uv pip install --python "$ENV_DIR/bin/python" \
  "$VLLM_REQUIREMENT" "$TRANSFORMERS_REQUIREMENT" "$OPENAI_REQUIREMENT" \
  "$PILLOW_REQUIREMENT" "$TOMLI_REQUIREMENT" "huggingface_hub[cli]"

"$ENV_DIR/bin/python" "$HARNESS_DIR/scripts/validate_hunyuanocr_1_5_ar_environment.py" \
  --baseline-config "$CONFIG_PATH" --toolkit-root "$TOOLKIT_ROOT"
"$ENV_DIR/bin/python" "$HARNESS_DIR/scripts/prefetch_hunyuanocr_1_5.py" \
  --baseline-config "$CONFIG_PATH" --model-cache-root "$MODEL_CACHE_ROOT"
"$CONTROL_PYTHON" -m uv pip freeze --python "$ENV_DIR/bin/python" | sort > "$MODEL_CACHE_ROOT/pinned-environment-freeze.txt"

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
printf '%s\n' "Pinned HunyuanOCR-1.5 AR-only vLLM environment ready: $ENV_DIR"

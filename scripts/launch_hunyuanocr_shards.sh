#!/usr/bin/env bash
# Launch one official HunyuanOCR AR vLLM server and one runner per shard/GPU.
# It deliberately requires a sharded run, which keeps the one-GPU and two-GPU
# procedures identical apart from --shards and --gpu-ids. A one-shard parent
# is valid and is still merged/verified by the same strict path.

set -euo pipefail

usage() {
  echo "Usage: $0 --run-dir RUN_DIR --image-root IMAGE_ROOT --gpu-ids GPU0,GPU1 [--resume]" >&2
  exit 2
}

RUN_DIR=""
IMAGE_ROOT=""
GPU_IDS=""
RESUME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="${2:-}"; shift 2 ;;
    --image-root) IMAGE_ROOT="${2:-}"; shift 2 ;;
    --gpu-ids) GPU_IDS="${2:-}"; shift 2 ;;
    --resume) RESUME="--resume"; shift ;;
    *) usage ;;
  esac
done
[[ -n "$RUN_DIR" && -n "$IMAGE_ROOT" && -n "$GPU_IDS" ]] || usage
[[ -f "$RUN_DIR/sharding/assignment.json" ]] || { echo "Not an initialized sharded run: $RUN_DIR" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable; refusing CPU fallback." >&2; exit 2; }
command -v curl >/dev/null || { echo "curl is required for vLLM readiness checks." >&2; exit 2; }

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
SHARD_COUNT="$(python3 - "$RUN_DIR/sharding/assignment.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["shard_count"])
PY
)"
[[ "${#GPU_ARRAY[@]}" -eq "$SHARD_COUNT" ]] || { echo "Expected $SHARD_COUNT GPU ids, got ${#GPU_ARRAY[@]}: $GPU_IDS" >&2; exit 2; }
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU id: $gpu" >&2; exit 2; }
  nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader >/dev/null
done

readarray -t CONFIG_VALUES < <(python3 - "$RUN_DIR/config/baseline.toml" <<'PY'
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
c = tomllib.loads(open(sys.argv[1], encoding="utf-8").read())
e, s = c["environment"], c["server"]
print(e["toolkit_root"])
print(e["model_cache_root"])
print(s["base_port"])
print(s["gpu_memory_utilization"])
print(s["max_model_len"])
print(s["served_model_name"])
print(s["startup_timeout_seconds"])
print(str(s["enforce_eager"]).lower())
PY
)
TOOLKIT_ROOT="${CONFIG_VALUES[0]}"
MODEL_CACHE_ROOT="${CONFIG_VALUES[1]}"
BASE_PORT="${CONFIG_VALUES[2]}"
GPU_MEM_UTIL="${CONFIG_VALUES[3]}"
MAX_MODEL_LEN="${CONFIG_VALUES[4]}"
SERVED_NAME="${CONFIG_VALUES[5]}"
STARTUP_TIMEOUT="${CONFIG_VALUES[6]}"
ENFORCE_EAGER="${CONFIG_VALUES[7]}"
SERVE_SCRIPT="$TOOLKIT_ROOT/inference/vLLM/serve.sh"
[[ -x "$SERVE_SCRIPT" || -f "$SERVE_SCRIPT" ]] || { echo "Official AR serve script not found: $SERVE_SCRIPT" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Fail before a costly vLLM startup if the manifest cannot satisfy the
# runner's input-integrity contract.  Import the runner's own parser so this
# preflight cannot drift from the execution-time validation.
python3 - "$SCRIPT_DIR" "$RUN_DIR/manifest.jsonl" <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
from run_hunyuanocr_1_5 import read_manifest

rows = read_manifest(Path(sys.argv[2]))
print(f"Hunyuan manifest preflight passed: {len(rows)} hash-addressed page(s)")
PY
python3 "$SCRIPT_DIR/validate_hunyuanocr_1_5_ar_environment.py" \
  --baseline-config "$RUN_DIR/config/baseline.toml" \
  --toolkit-root "$TOOLKIT_ROOT"

MODEL_PATH="$(python3 - "$MODEL_CACHE_ROOT/pinned-models.json" <<'PY'
import json, sys
data=json.load(open(sys.argv[1], encoding="utf-8"))
snapshots=data.get("snapshots", [])
if len(snapshots) != 1: raise SystemExit("Expected exactly one pinned HunyuanOCR snapshot")
print(snapshots[0]["snapshot_path"])
PY
)"
[[ -d "$MODEL_PATH" ]] || { echo "Pinned model snapshot missing: $MODEL_PATH" >&2; exit 2; }

SERVER_PIDS=()
cleanup() {
  local status=$?
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  exit "$status"
}
trap cleanup EXIT INT TERM

for ((index=0; index<SHARD_COUNT; index++)); do
  shard_name="$(printf 'shard-%03d' "$index")"
  shard_dir="$RUN_DIR/shards/$shard_name"
  [[ -d "$shard_dir" ]] || { echo "Missing $shard_dir" >&2; exit 2; }
  port=$((BASE_PORT + index))
  if python3 - "$port" <<'PY'
import socket, sys
sock = socket.socket()
try:
    if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0:
        raise SystemExit(0)
finally:
    sock.close()
raise SystemExit(1)
PY
  then
    echo "Refusing to reuse occupied vLLM port ${port}; stop the existing server first." >&2
    exit 2
  fi
  server_log="$shard_dir/runtime/vllm-server.log"
  mkdir -p "$shard_dir/runtime/launcher_attempts"
  if [[ "$ENFORCE_EAGER" == "true" ]]; then
    # Keep the pinned upstream serve.sh unmodified.  It invokes `nohup vllm`,
    # which bypasses a shell-function wrapper.  Place a one-command executable
    # at the front of PATH instead, so nohup resolves the shim and the real
    # vLLM binary always receives its documented eager-only switch.
    shim_dir="$shard_dir/runtime/eager-vllm-bin"
    mkdir -p "$shim_dir"
    vllm_binary="$(command -v vllm)"
    [[ -x "$vllm_binary" ]] || { echo "Could not resolve executable vllm binary" >&2; exit 2; }
    cat > "$shim_dir/vllm" <<EOF
#!/usr/bin/env bash
exec "$vllm_binary" "\$@" --enforce-eager
EOF
    chmod 0755 "$shim_dir/vllm"
    start_output="$(PATH="$shim_dir:$PATH" MODEL_PATH="$MODEL_PATH" GPU="${GPU_ARRAY[$index]}" PORT="$port" GPU_MEM_UTIL="$GPU_MEM_UTIL" MAX_MODEL_LEN="$MAX_MODEL_LEN" SERVED_NAME="$SERVED_NAME" LOG="$server_log" bash "$SERVE_SCRIPT")"
  else
    echo "HunyuanOCR AR baseline refuses compiled execution; set server.enforce_eager=true." >&2
    exit 2
  fi
  printf '%s\n' "$start_output" | tee "$shard_dir/runtime/vllm-server-launch.log"
  pid="$(printf '%s\n' "$start_output" | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  [[ -n "$pid" ]] || { echo "Could not identify vLLM PID for $shard_name" >&2; exit 2; }
  SERVER_PIDS+=("$pid")
  python3 - "$shard_dir/runtime/vllm-server.json" "$pid" "${GPU_ARRAY[$index]}" "$port" "$MODEL_PATH" <<'PY'
import json, sys
path, pid, gpu, port, model = sys.argv[1:]
open(path, "w", encoding="utf-8").write(json.dumps({"pid": int(pid), "physical_gpu": int(gpu), "port": int(port), "model_snapshot": model, "mode": "official_vllm_ar", "enforce_eager": True}, indent=2, sort_keys=True)+"\n")
PY
done

for ((index=0; index<SHARD_COUNT; index++)); do
  port=$((BASE_PORT + index))
  deadline=$((SECONDS + STARTUP_TIMEOUT))
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null; do
    pid="${SERVER_PIDS[$index]}"
    state="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    if [[ -z "$state" || "$state" == Z* ]]; then
      echo "vLLM exited before readiness for shard-$(printf '%03d' "$index"). See $RUN_DIR/shards/$(printf 'shard-%03d' "$index")/runtime/vllm-server.log" >&2
      tail -n 80 "$RUN_DIR/shards/$(printf 'shard-%03d' "$index")/runtime/vllm-server.log" >&2 || true
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for shard-$(printf '%03d' "$index") vLLM server. See its runtime/vllm-server.log" >&2
      exit 1
    fi
    sleep 5
  done
  echo "vLLM ready: shard-$(printf '%03d' "$index") on GPU ${GPU_ARRAY[$index]}, port $port"
done

RUNNER_PIDS=()
for ((index=0; index<SHARD_COUNT; index++)); do
  shard_name="$(printf 'shard-%03d' "$index")"
  shard_dir="$RUN_DIR/shards/$shard_name"
  attempt=1
  while [[ -e "$(printf '%s/runtime/launcher_attempts/attempt-%04d.log' "$shard_dir" "$attempt")" ]]; do
    ((attempt++))
  done
  log="$(printf '%s/runtime/launcher_attempts/attempt-%04d.log' "$shard_dir" "$attempt")"
  port=$((BASE_PORT + index))
  (
    exec python3 "$SCRIPT_DIR/run_hunyuanocr_1_5.py" --run-dir "$shard_dir" --image-root "$IMAGE_ROOT" --host 127.0.0.1 --port "$port" ${RESUME:+$RESUME}
  ) >"$log" 2>&1 &
  RUNNER_PIDS+=("$!")
  echo "Launched $shard_name on physical GPU ${GPU_ARRAY[$index]}; runner log=$log"
done

STATUS=0
for pid in "${RUNNER_PIDS[@]}"; do
  wait "$pid" || STATUS=1
done
exit "$STATUS"

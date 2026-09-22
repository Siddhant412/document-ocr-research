#!/usr/bin/env bash
# Launch one native InternVL inference process per deterministic shard/GPU.
# Run this inside tmux after activating the pinned InternVL virtual environment.

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

if [[ ! -f "$RUN_DIR/sharding/assignment.json" ]]; then
  echo "Not an initialized sharded run: $RUN_DIR" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi is unavailable; refusing CPU fallback." >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
SHARD_COUNT="$(python3 - "$RUN_DIR/sharding/assignment.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding='utf-8') as handle:
    print(json.load(handle)['shard_count'])
PY
)"
if [[ "${#GPU_ARRAY[@]}" -ne "$SHARD_COUNT" ]]; then
  echo "Expected $SHARD_COUNT GPU ids, received ${#GPU_ARRAY[@]}: $GPU_IDS" >&2
  exit 2
fi
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU id: $gpu" >&2; exit 2; }
  nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader >/dev/null
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS=()
for ((index=0; index<SHARD_COUNT; index++)); do
  shard_name="$(printf 'shard-%03d' "$index")"
  shard_dir="$RUN_DIR/shards/$shard_name"
  [[ -d "$shard_dir" ]] || { echo "Missing $shard_name" >&2; exit 2; }
  log_dir="$shard_dir/runtime/launcher_attempts"
  mkdir -p "$log_dir"
  attempt=1
  while [[ -e "$(printf '%s/attempt-%04d.log' "$log_dir" "$attempt")" ]]; do
    ((attempt++))
  done
  log="$(printf '%s/attempt-%04d.log' "$log_dir" "$attempt")"
  (
    export CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$index]}"
    exec python3 "$SCRIPT_DIR/run_internvl3_5_8b_instruct.py" \
      --run-dir "$shard_dir" \
      --image-root "$IMAGE_ROOT" \
      ${RESUME:+$RESUME}
  ) >"$log" 2>&1 &
  PIDS+=("$!")
  printf 'Launched %s on physical GPU %s; log=%s\n' "$shard_name" "${GPU_ARRAY[$index]}" "$log"
done

set +e
STATUS=0
for index in "${!PIDS[@]}"; do
  wait "${PIDS[$index]}" || STATUS=1
done
set -e
exit "$STATUS"

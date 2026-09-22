#!/usr/bin/env bash
# Run the pinned official OmniDocBench evaluator against an initialized run.
set -euo pipefail

usage() {
  printf '%s\n' "Usage: $0 --run-dir RUN_DIR --gt-json OMNIDOCBENCH.json [--image IMAGE] [--config CONFIG] [--platform PLATFORM]"
}

run_dir=""
gt_json=""
image="ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204"
platform=""
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_path="${script_dir}/../configs/omnidocbench_end2end_v1.7.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) run_dir="$2"; shift 2 ;;
    --gt-json) gt_json="$2"; shift 2 ;;
    --image) image="$2"; shift 2 ;;
    --config) config_path="$2"; shift 2 ;;
    --platform) platform="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

[[ -n "$run_dir" && -n "$gt_json" ]] || { usage; exit 2; }
[[ -d "$run_dir" && -f "$gt_json" && -f "$config_path" ]] || {
  printf '%s\n' "Run directory, ground-truth JSON, and config must exist." >&2
  exit 2
}

python3 "${script_dir}/verify_run.py" --run-dir "$run_dir"
mkdir -p "${run_dir}/evaluator" "${run_dir}/metrics/official"
if [[ -f "${run_dir}/evaluator/evaluator_stdout.log" ]] || \
  find "${run_dir}/metrics/official" -mindepth 1 -print -quit | grep -q .; then
  attempt_dir="${run_dir}/evaluator/attempts/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "${attempt_dir}/partial_metrics"
  [[ ! -f "${run_dir}/evaluator/evaluator_stdout.log" ]] || \
    cp "${run_dir}/evaluator/evaluator_stdout.log" "${attempt_dir}/evaluator_stdout.log"
  cp -R "${run_dir}/metrics/official/." "${attempt_dir}/partial_metrics/"
fi
printf '%s\n' \
  "The pinned evaluator image currently contains evaluate==0.4.3 with huggingface_hub==1.9.1." \
  "evaluate==0.4.3 imports the removed HfFolder API, so this run applies huggingface_hub==0.25.2" \
  "inside the disposable evaluator container before scoring." \
  > "${run_dir}/evaluator/compatibility_override.txt"
pull_args=()
run_args=()
if [[ -n "$platform" ]]; then
  pull_args+=(--platform "$platform")
  run_args+=(--platform "$platform")
fi
docker pull "${pull_args[@]}" "$image"
docker image inspect "$image" --format '{{index .RepoDigests 0}}' > "${run_dir}/evaluator/image_digest.txt"
cp "$config_path" "${run_dir}/evaluator/end2end_config.yaml"

docker run --rm "${run_args[@]}" \
  --entrypoint bash \
  -v "$(cd "$(dirname "$gt_json")" && pwd)/$(basename "$gt_json"):/workspace/gt/OmniDocBench.json:ro" \
  -v "$(cd "${run_dir}/predictions" && pwd):/workspace/data_md/predictions:ro" \
  -v "$(cd "${run_dir}/metrics/official" && pwd):/workspace/result" \
  -v "$(cd "${run_dir}/evaluator" && pwd):/workspace/evaluator" \
  -v "$(cd "$(dirname "$config_path")" && pwd)/$(basename "$config_path"):/workspace/configs/deepseek_ocr_end2end.yaml:ro" \
  "$image" \
  -lc 'python -m pip install --quiet huggingface_hub==0.25.2 && python -c "import huggingface_hub; print(huggingface_hub.__version__)" | tee /workspace/evaluator/compatibility_runtime.txt && python pdf_validation.py --config configs/deepseek_ocr_end2end.yaml' \
  | tee "${run_dir}/evaluator/evaluator_stdout.log"

# OCR/VLM baselines on OmniDocBench: English-only reproducible harness

This repository contains completed DeepSeek-OCR artifacts and a matched
Qwen2.5-VL-7B-Instruct runner for the **English subset of OmniDocBench v1.7**.
Each model uses a two-stage, failure-inclusive evaluation:

1. a deterministic 180-page source-first smoke test; then
2. all 755 English pages with unchanged inputs and configuration.

It is designed for a Linux NVIDIA GPU machine. The intended first machine is a
RunPod RTX 4090 pod. An Apple-silicon Mac is suitable for manifest creation and
artifact review, but not for the official CUDA/FlashAttention inference setup.

## Baseline rules

- The model call is the documented native Transformers call, including the
  document prompt, `base_size=1024`, `image_size=640`, `crop_mode=True`,
  `save_results=True`, and `test_compress=True`.
- No custom output repair or normalization is applied by this harness.
- Every selected page receives one prediction `.md`. An inference exception
  produces a **zero-byte** file and an entry in `failures.jsonl`; it is not
  omitted from evaluation.
- DeepSeek's official `save_results=True` implementation writes `result.mmd`
  after its own reference handling. The harness scores that exact official file
  and separately stores captured emitted stdout under `raw_predictions/`.
- Smoke sampling fixes primary `data_source` quotas first and targets layout
  marginals. Every selected page is English, so there is no multilingual quota.
- The smoke evaluator receives a ground-truth subset with exactly the same 180
  pages. The full run receives the full v1.7 JSON.

## What is pinned

[`configs/baseline.toml`](configs/baseline.toml) records the current upstream
commits for the evaluator and model source, the Hugging Face model revision,
the exact inference settings, and a v1.7 requirement. Before a real run, fill
the three `RECORD_...` dataset/image fields with the downloaded snapshot's
revision and checksums. After smoke inference starts, copy the file only; do
not edit it in place.

The evaluator config uses official `end2end`, `quick_match`, all end-to-end
metrics, and `filter: language: english`.

## Local preparation

This harness requires Python 3.11+ and SciPy for globally constrained manifest
selection. Download the official v1.7 dataset snapshot to `data/` (ignored by Git)
and record its immutable Hugging Face revision. For example, after installing
the Hugging Face CLI:

```bash
hf download opendatalab/OmniDocBench --repo-type dataset --revision <v1.7-revision> --local-dir data/omnidocbench-v1.7
```

Locate the official full annotation JSON and the directory that contains the
paths named by each `page_info.image_path`. Hash those exact inputs before
building the smoke manifest:

```bash
python3 scripts/hash_dataset_inputs.py \
  --annotations data/omnidocbench-v1.7/<OmniDocBench.json> \
  --image-root data/omnidocbench-v1.7/images \
  --dataset-revision <v1.7-revision> \
  --output data/omnidocbench-v1.7-input-checksums.json
```

Copy the revision, annotation hash, and image-manifest hash into
`configs/baseline.toml`, then build the smoke manifest:

```bash
python3 scripts/build_smoke_manifest.py \
  --annotations data/omnidocbench-v1.7/<OmniDocBench.json> \
  --output data/smoke-180-manifest.jsonl \
  --coverage-output data/smoke-180-coverage.json \
  --pages 180 --seed 20260906 --minimum-per-source 5 --language english

python3 scripts/build_ground_truth_subset.py \
  --annotations data/omnidocbench-v1.7/<OmniDocBench.json> \
  --manifest data/smoke-180-manifest.jsonl \
  --output data/smoke-180-ground-truth.json
```

The manifest builder exits nonzero if its exact language/layout marginal targets
cannot be met. Inspect `smoke-180-coverage.json`; only pass
`--allow-margin-residuals` after explicitly accepting the reported residuals.

Create an immutable run directory:

```bash
python3 scripts/init_run.py \
  --run-dir runs/deepseek-ocr-omnidocbench-v1.7-english-smoke-180 \
  --manifest data/smoke-180-manifest.jsonl \
  --baseline-config configs/baseline.toml
```

## Cloud inference

For the currently offered RTX 4090 capacity, use a CUDA 12.4-compatible image,
such as RunPod's official
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`; do not use a CUDA
11.8 image if the Pod creation page reports it as incompatible.

From a second terminal on the Mac, transfer the entire prepared checkout to
the persistent Pod volume. Replace the host and port with the values shown in
RunPod's **SSH over exposed TCP** panel:

```bash
cd /Users/siddhantraje/Documents/ResearchAIDoc/document-ocr-research
rsync -a --partial --info=progress2 \
  -e 'ssh -i ~/.ssh/id_ed25519 -p <pod-port>' \
  ./ root@<pod-ip>:/workspace/document-ocr-research/
```

Back on the Pod, create the pinned environment. It uses Torch 2.6 CUDA 12.4,
the DeepSeek source commit captured in `baseline.toml`, its official
requirements, and FlashAttention 2.7.3:

```bash
cd /workspace/document-ocr-research
bash scripts/setup_deepseek_environment.sh
source /workspace/venvs/deepseek-ocr/bin/activate
```

Before inference, capture actual runtime state:

```bash
python3 scripts/record_provenance.py \
  --run-dir runs/deepseek-ocr-omnidocbench-v1.7-english-smoke-180 \
  --output runtime/cloud_provenance.json
```

Run model inference. `--image-root` is joined with `image_path` values from the
official JSON; it is normally the dataset snapshot root.

```bash
python3 scripts/run_deepseek_ocr.py \
  --run-dir runs/deepseek-ocr-omnidocbench-v1.7-english-smoke-180 \
  --image-root data/omnidocbench-v1.7/images

python3 scripts/verify_run.py \
  --run-dir runs/deepseek-ocr-omnidocbench-v1.7-english-smoke-180
```

### Safe interruption and resume

Run inference inside `tmux` so an SSH/Wi-Fi disconnect does not stop the
process:

```bash
tmux new -s deepseek-ocr
# run the inference command above, then detach with Ctrl-b followed by d
tmux attach -t deepseek-ocr
```

While it runs, open a second SSH session (or a second `tmux` window) and use:

```bash
python3 scripts/show_progress.py \
  --run-dir runs/deepseek-ocr-omnidocbench-v1.7-english-smoke-180
```

It reports processed/total, remaining pages, successful outputs, empty model
outputs, and inference errors. The runner also prints a progress line after
each page. For an automatically refreshing view on the Linux pod:

```bash
watch -n 5 python3 scripts/show_progress.py \
  --run-dir runs/deepseek-ocr-omnidocbench-v1.7-english-smoke-180
```

If the process or pod stops, start it again with `--resume`:

```bash
python3 scripts/run_deepseek_ocr.py \
  --run-dir runs/deepseek-ocr-omnidocbench-v1.7-english-smoke-180 \
  --image-root data/omnidocbench-v1.7/images \
  --resume
```

Resume retains a page only when its prediction file exactly matches a terminal
record in `runtime/pages.jsonl`. A partial/orphan prediction is moved under
`raw_predictions/<page>/interrupted_predictions/` and rerun in a new numbered
attempt. Recorded `error` and `empty_output` pages are terminal by default: the
zero-byte prediction remains included in scoring and is not silently retried.
The official evaluator can be rerun any time from the completed saved
`predictions/` directory; it does not rerun the model.

For a non-reporting repeatability preflight, initialize separate fresh run
directories from a short manifest and compare the SHA-256 values in each
`runtime/pages.jsonl`. Do not put preflight outputs into the reported smoke run.

## Official evaluation

Run the evaluator only after `verify_run.py` reports complete coverage. The
wrapper pulls the official evaluator image, records its resolved digest, copies
the exact config used, and saves evaluator stdout and results under the run.

```bash
bash scripts/run_official_evaluator.sh \
  --run-dir runs/deepseek-ocr-omnidocbench-v1.7-english-smoke-180 \
  --gt-json data/smoke-180-ground-truth.json
```

On an Apple-silicon Mac with Docker Desktop, append `--platform linux/amd64`.
The official evaluator image is large and runs under x86 emulation, so reserve
at least 30 GB of Docker disk space and expect the CPU evaluation to take
longer than it would on a native x86 Linux machine.

The pinned evaluator image currently has an upstream `evaluate` /
`huggingface_hub` compatibility break. The wrapper installs
`huggingface_hub==0.25.2` inside its disposable container before scoring and
records that compatibility override under `evaluator/`; the pinned image digest
and all evaluator inputs remain recorded unchanged.

For the complete evaluation, create the full manifest, then initialize a new
run directory and pass the **full** v1.7 annotation JSON to the evaluator:

```bash
python3 scripts/build_full_manifest.py \
  --annotations data/omnidocbench-v1.7/<OmniDocBench.json> \
  --output data/full-english-manifest.jsonl \
  --coverage-output data/full-coverage.json \
  --language english
```

Do not alter `baseline.toml`, inference settings, or evaluator configuration
between the completed smoke test and the full run.

## Matched Qwen2.5-VL-7B-Instruct baseline

The Qwen baseline is deliberately separate from the completed DeepSeek run.
It uses the same benchmark snapshot, English-only ground truth, evaluator, and
the exact frozen DeepSeek smoke/full manifests. The initialization guard rejects
a manifest whose SHA-256 does not match that reference, preventing accidental
resampling.

[`configs/qwen2.5-vl-7b-instruct.toml`](configs/qwen2.5-vl-7b-instruct.toml)
pins the model revision, CUDA environment, prompt, pixel budget, and greedy
generation settings. Qwen does not publish a single native OCR-to-Markdown
prompt equivalent to DeepSeek's document prompt, so the configuration records
the transparent task-specific Markdown instruction as part of this baseline.
It uses Qwen's official Transformers chat-template and processor flow with
`qwen-vl-utils`; raw decoded output is scored without cleanup.

The Qwen runner passes a literal local image path to `qwen-vl-utils` rather
than a `file://` URI. This is necessary for OmniDocBench filenames containing
spaces, percent signs, or Unicode characters: Qwen's utility accepts local
paths directly but does not URL-decode a `file://` path before opening it.

### RunPod setup

Use the same persistent RTX 4090 Pod/dataset volume as the DeepSeek run. Copy
the updated checkout to `/workspace/document-ocr-research`, then run:

```bash
cd /workspace/document-ocr-research
bash scripts/setup_qwen2_5_vl_environment.sh
source /workspace/venvs/qwen2.5-vl-7b-instruct/bin/activate
```

The setup uses a fresh environment, so it does not modify the verified
DeepSeek environment. It pins CUDA 12.4 Torch 2.6.0, Transformers 4.49.0,
`qwen-vl-utils==0.0.8`, and FlashAttention 2.7.3. The model is loaded in BF16
with Flash Attention 2. The resolution bounds are Qwen's documented
`256*28*28` to `1280*28*28` pixel range.

### Mechanical preflight

This is an installation/OOM/output-path check only. It does not produce a
reportable score and must not be used to tune the prompt or settings.

```bash
python3 scripts/init_run.py \
  --run-dir runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-preflight-3 \
  --manifest data/smoke-180-manifest.jsonl \
  --reference-manifest runs/deepseek-ocr-omnidocbench-v1.7-english-smoke-180/manifest.jsonl \
  --baseline-config configs/qwen2.5-vl-7b-instruct.toml

python3 scripts/record_provenance.py \
  --run-dir runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-preflight-3 \
  --output runtime/cloud_provenance.json

python3 scripts/run_qwen2_5_vl.py \
  --run-dir runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-preflight-3 \
  --image-root data/omnidocbench-v1.7/images \
  --limit 3
```

Review `runtime/pages.jsonl` for three terminal records, model download/load
success, page times, and `generation_reached_cap`. Delete neither the
preflight nor its failure records; simply keep it separate from the smoke run.

For a targeted non-reporting path or image preflight, create a one-page subset
from the frozen smoke manifest with `scripts/build_manifest_subset.py`; it
refuses missing or duplicate page identifiers and never overwrites a subset.

### Qwen smoke and full runs

Initialize the reportable smoke run from the same frozen source manifest, then
run it in `tmux` as described above:

```bash
python3 scripts/init_run.py \
  --run-dir runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-smoke-180 \
  --manifest data/smoke-180-manifest.jsonl \
  --reference-manifest runs/deepseek-ocr-omnidocbench-v1.7-english-smoke-180/manifest.jsonl \
  --baseline-config configs/qwen2.5-vl-7b-instruct.toml

python3 scripts/record_provenance.py \
  --run-dir runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-smoke-180 \
  --output runtime/cloud_provenance.json

python3 scripts/run_qwen2_5_vl.py \
  --run-dir runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-smoke-180 \
  --image-root data/omnidocbench-v1.7/images
```

Verify and evaluate that run using the existing ground-truth subset:

```bash
python3 scripts/verify_run.py \
  --run-dir runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-smoke-180

# On the Mac, after copying the completed run back:
bash scripts/run_official_evaluator.sh \
  --run-dir runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-smoke-180 \
  --gt-json data/smoke-180-ground-truth.json \
  --platform linux/amd64
```

After recording and accepting the smoke timing/metrics, do not alter the Qwen
configuration. Initialize and run the full matched set:

```bash
python3 scripts/init_run.py \
  --run-dir runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-full-755 \
  --manifest data/full-english-manifest.jsonl \
  --reference-manifest runs/deepseek-ocr-omnidocbench-v1.7-english-full-755/manifest.jsonl \
  --baseline-config configs/qwen2.5-vl-7b-instruct.toml

python3 scripts/record_provenance.py \
  --run-dir runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-full-755 \
  --output runtime/cloud_provenance.json

tmux new -s qwen-omnidocbench
python3 scripts/run_qwen2_5_vl.py \
  --run-dir runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-full-755 \
  --image-root data/omnidocbench-v1.7/images
```

For a disconnection or Pod restart, re-run the same command with `--resume`.
Use `scripts/show_progress.py` and `scripts/verify_run.py` exactly as for
DeepSeek. The runner records `generation_reached_cap` per page so a capped
Qwen completion is visible in the runtime ledger rather than silently hidden.

After `verify_run.py` reports 755/755 coverage, copy the Qwen run directory
back to the Mac and run the unchanged official full evaluation:

```bash
python3 scripts/verify_run.py \
  --run-dir runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-full-755

bash scripts/run_official_evaluator.sh \
  --run-dir runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-full-755 \
  --gt-json data/omnidocbench-v1.7/OmniDocBench.json \
  --platform linux/amd64
```

Then create `basline-reports/qwen2.5-vl-7b-instruct/baseline-report.md` from
the saved coverage audit, runtime ledger, evaluator metrics, stage-execution
record, configuration, and provenance. Do not merge Qwen and DeepSeek results
or replace either model's raw predictions.

## Verification

Run the local tests before copying to cloud:

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/*.py
```

## Matched PaddleOCR-VL-1.6 baseline

The Paddle baseline uses the official **complete** `PaddleOCRVL` v1.6
pipeline: layout analysis followed by native local VLM recognition. It does not
call the VLM component by itself or use a generic Transformers prompt. This is
important because PaddleOCR documents those paths as non-equivalent.

[`configs/paddleocr-vl-1.6.toml`](configs/paddleocr-vl-1.6.toml) pins the
PaddleOCR and PaddleX package versions, PaddleOCR source tag, CUDA wheel
family, exact VLM and layout-model Hugging Face revisions, native pipeline
settings, and the unchanged OmniDocBench/evaluator inputs. The setup script
materializes both model revisions under `/workspace/paddleocr-vl-1.6-cache`,
hashes every downloaded model file, and records a full `pip freeze`. The
inference runner copies those records into each run directory.

PaddleOCR-VL's local native engine exposes no sampling switch: in the pinned
PaddleX 3.6.0 implementation, `temperature`, `top_p`, and
`repetition_penalty` are unsupported and ignored. The harness therefore uses
the native engine's deterministic no-sampling `generate` path, pins its 8,192
new-token ceiling, and records that fact in each page ledger record rather than
passing ineffective decoding settings.

### RunPod setup

Use the same RTX 4090 persistent volume that contains the dataset. The pinned
Paddle CUDA 11.8 wheel is intentionally compatible with the CUDA-12.4-capable
driver used by the earlier RTX 4090 Pods; the official pipeline requires CUDA
11.8 or later. Do not run PaddleOCR-VL inference on the Mac.

Copy only the updated harness from the **Mac terminal**, preserving the large
dataset and prior runs already on the Pod:

```bash
cd /Users/siddhantraje/Documents/ResearchAIDoc/document-ocr-research
rsync -rltp --partial --progress --no-owner --no-group \
  --exclude 'data/' \
  --exclude 'runs/' \
  --exclude '.git/' \
  --exclude '**pycache**/' \
  -e 'ssh -i ~/.ssh/id_ed25519 -p <pod-port>' \
  ./ root@<pod-ip>:/workspace/document-ocr-research/
```

Then, in the **Pod SSH terminal**:

```bash
cd /workspace/document-ocr-research
bash scripts/setup_paddleocr_vl_1_6_environment.sh
source /workspace/venvs/paddleocr-vl-1.6/bin/activate
```

Run `record_provenance.py` after each `init_run.py` command below, because its
target run directory must already exist. The setup deliberately creates a
fresh virtual environment and persistent model cache; it does not alter the
verified DeepSeek or Qwen environments.

### Mechanical preflight

This non-reporting check validates the environment, native output path, and
literal handling of a percent sign, Unicode, and spaces in OmniDocBench file
names. It is not evaluated and must not be used to tune settings.

```bash
python3 scripts/build_manifest_subset.py \
  --manifest data/smoke-180-manifest.jsonl \
  --output data/paddleocr-vl-1.6-preflight-3-manifest.jsonl \
  --page-id 'scihub_jee%2Ftoaa103.pdf_2.jpg' \
  --page-id 'book_en_搬书匠-3473-Reactive Programming with RxJS-2015-英文版_page_021.png' \
  --page-id 'newspaper_The Globe and Mail - 2025-1-8@magazinesclubnew_page_027.png'

python3 scripts/init_run.py \
  --run-dir runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-preflight-3 \
  --manifest data/paddleocr-vl-1.6-preflight-3-manifest.jsonl \
  --baseline-config configs/paddleocr-vl-1.6.toml

python3 scripts/record_provenance.py \
  --run-dir runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-preflight-3 \
  --output runtime/cloud_provenance.json

python3 scripts/run_paddleocr_vl_1_6.py \
  --run-dir runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-preflight-3 \
  --image-root data/omnidocbench-v1.7/images

python3 scripts/verify_run.py \
  --run-dir runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-preflight-3
```

Review `runtime/pipeline_initialization.json`,
`runtime/pinned_model_artifacts.json`, `runtime/pinned_environment_freeze.txt`,
and each `raw_predictions/<page>/attempts/attempt-0001/native_output/`
directory. Keep this preflight separate from the reportable smoke run.

### Smoke and full runs

The reportable runs use byte-identical DeepSeek manifests. `init_run.py` aborts
if either Paddle manifest differs from its frozen reference.

```bash
# 180-page reportable smoke run, in the Pod SSH terminal.
python3 scripts/init_run.py \
  --run-dir runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-smoke-180 \
  --manifest data/smoke-180-manifest.jsonl \
  --reference-manifest runs/deepseek-ocr-omnidocbench-v1.7-english-smoke-180/manifest.jsonl \
  --baseline-config configs/paddleocr-vl-1.6.toml

python3 scripts/record_provenance.py \
  --run-dir runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-smoke-180 \
  --output runtime/cloud_provenance.json

tmux new -s paddleocr-vl-smoke
python3 scripts/run_paddleocr_vl_1_6.py \
  --run-dir runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-smoke-180 \
  --image-root data/omnidocbench-v1.7/images
```

In a second Pod SSH terminal, observe durable progress with:

```bash
source /workspace/venvs/paddleocr-vl-1.6/bin/activate
cd /workspace/document-ocr-research
python3 scripts/show_progress.py \
  --run-dir runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-smoke-180
```

Detach the `tmux` session with `Ctrl-b`, then `d`. If inference is interrupted,
run the same inference command with `--resume`. Verified terminal pages,
including explicit zero-byte error predictions, are retained; unverified files
are preserved under `raw_predictions/` and rerun.

After `verify_run.py` reports 180/180 coverage, copy the completed smoke run to
the Mac and run the unchanged official evaluator there:

```bash
bash scripts/run_official_evaluator.sh \
  --run-dir runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-smoke-180 \
  --gt-json data/smoke-180-ground-truth.json \
  --platform linux/amd64
```

Accept the smoke timing and metrics before starting the 755-page run; do not
alter the frozen configuration after that point.

```bash
# 755-page reportable full run, in the Pod SSH terminal.
python3 scripts/init_run.py \
  --run-dir runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755 \
  --manifest data/full-english-manifest.jsonl \
  --reference-manifest runs/deepseek-ocr-omnidocbench-v1.7-english-full-755/manifest.jsonl \
  --baseline-config configs/paddleocr-vl-1.6.toml

python3 scripts/record_provenance.py \
  --run-dir runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755 \
  --output runtime/cloud_provenance.json

tmux new -s paddleocr-vl-full
python3 scripts/run_paddleocr_vl_1_6.py \
  --run-dir runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755 \
  --image-root data/omnidocbench-v1.7/images
```

Once verified 755/755, copy the run back to the Mac and evaluate it exactly as
the other full baselines:

```bash
bash scripts/run_official_evaluator.sh \
  --run-dir runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755 \
  --gt-json data/omnidocbench-v1.7/OmniDocBench.json \
  --platform linux/amd64
```

Create `basline-reports/paddleocr-vl-1.6/baseline-report.md` only from the
saved full-run coverage audit, inference ledger, exact raw predictions,
provenance, frozen configuration, and official evaluator outputs.

## Matched InternVL3.5-8B-Instruct baseline

[`configs/internvl3.5-8b-instruct.toml`](configs/internvl3.5-8b-instruct.toml)
pins `OpenGVLab/InternVL3_5-8B-Instruct` to revision
`61e3b61d08dac94508888b06f5e54d3b44353d28`, its package environment, the
exact Markdown instruction, vision preprocessing, and the unchanged benchmark
inputs. `scripts/setup_internvl3_5_8b_instruct_environment.sh` materializes
the checkpoint in `/workspace/internvl3.5-8b-instruct-cache`, hashes every
downloaded model file, and writes the package freeze. The runner copies those
records into every run directory before it loads the model.

Inference uses the model card's native custom-code `AutoModel.chat` path. It
uses 448-pixel ImageNet-normalized dynamic tiles (one through 12, plus a
thumbnail for multi-tile pages), then writes the model response as-is. The
baseline is deliberately **normal instruct mode**: it never sets an R1 or
thinking system prompt. Decoding is greedy (`do_sample=false`, one beam, 8,192
new tokens). The config records temperature 0.0, but the runner does not pass
it to generation because sampling is disabled.

### Pod requirements

For a one-GPU run, use an H100 SXM. For the faster sharded path, use either two
RTX 4090s or two L4s, with 24 GB VRAM per GPU, on a CUDA 12.8 RunPod PyTorch
template. Each worker loads one complete InternVL model; the observed smoke-run
peak was 17.60 GiB per worker, so no model-parallel split is used. A network
volume is data-center-specific, so a new Pod can attach it only when the chosen
GPU configuration is available in that volume's data center. The volume holds
the checkout, data, cache, environments, and artifacts; the container disk is
disposable.

### Pod setup after deployment

Run the following in the **Pod SSH terminal** after the complete repository
(including `data/`) has been copied to `/workspace/document-ocr-research`:

```bash
cd /workspace/document-ocr-research
nvidia-smi
apt-get update && apt-get install -y tmux
bash scripts/setup_internvl3_5_8b_instruct_environment.sh
source /workspace/venvs/internvl3.5-8b-instruct/bin/activate
```

The setup fails rather than silently using CPU when CUDA is unavailable or the
installed Torch wheel is not CUDA 12.8.

### Mandatory non-reporting preflights

Run a three-path installation preflight first. It deliberately covers percent
signs, Unicode, and spaces in filenames; it is not reportable or evaluated.

```bash
python3 scripts/build_manifest_subset.py \
  --manifest data/smoke-180-manifest.jsonl \
  --output data/internvl3.5-8b-instruct-preflight-3-manifest.jsonl \
  --page-id 'scihub_jee%2Ftoaa103.pdf_2.jpg' \
  --page-id 'book_en_搬书匠-3473-Reactive Programming with RxJS-2015-英文版_page_021.png' \
  --page-id 'newspaper_The Globe and Mail - 2025-1-8@magazinesclubnew_page_027.png'

python3 scripts/init_run.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-preflight-3 \
  --manifest data/internvl3.5-8b-instruct-preflight-3-manifest.jsonl \
  --baseline-config configs/internvl3.5-8b-instruct.toml
python3 scripts/record_provenance.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-preflight-3 \
  --output runtime/cloud_provenance.json
python3 scripts/run_internvl3_5_8b_instruct.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-preflight-3 \
  --image-root data/omnidocbench-v1.7/images
python3 scripts/verify_run.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-preflight-3
```

Then measure an exact, non-reporting 20-page timing sample in frozen smoke
order. Record its mean page time times 755 as the forecast; do not tune the
model or edit the configuration afterward.

```bash
python3 scripts/build_manifest_subset.py \
  --manifest data/smoke-180-manifest.jsonl \
  --output data/internvl3.5-8b-instruct-timing-preflight-20-manifest.jsonl \
  --first 20
python3 scripts/init_run.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-timing-preflight-20 \
  --manifest data/internvl3.5-8b-instruct-timing-preflight-20-manifest.jsonl \
  --baseline-config configs/internvl3.5-8b-instruct.toml
python3 scripts/record_provenance.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-timing-preflight-20 \
  --output runtime/cloud_provenance.json
python3 scripts/run_internvl3_5_8b_instruct.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-timing-preflight-20 \
  --image-root data/omnidocbench-v1.7/images
python3 scripts/verify_run.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-timing-preflight-20
```

Review `runtime/model_initialization.json`, `runtime/pages.jsonl`, and each
raw request, response, stdout, and stderr file before proceeding.

### Matched smoke and full runs

Both reportable runs use byte-identical DeepSeek reference manifests. Run the
inference command inside `tmux`; `Ctrl-b`, then `d` detaches safely.

```bash
# Pod SSH terminal: 180-page smoke inference
python3 scripts/init_run.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-smoke-180 \
  --manifest data/smoke-180-manifest.jsonl \
  --reference-manifest runs/deepseek-ocr-omnidocbench-v1.7-english-smoke-180/manifest.jsonl \
  --baseline-config configs/internvl3.5-8b-instruct.toml
python3 scripts/record_provenance.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-smoke-180 \
  --output runtime/cloud_provenance.json
tmux new -s internvl-smoke
python3 scripts/run_internvl3_5_8b_instruct.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-smoke-180 \
  --image-root data/omnidocbench-v1.7/images
```

After 180/180 verification, copy the run to the Mac and evaluate it using the
unchanged official evaluator:

```bash
# Mac terminal
bash scripts/run_official_evaluator.sh \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-smoke-180 \
  --gt-json data/smoke-180-ground-truth.json \
  --platform linux/amd64
```

Only after accepting smoke metrics and timing, use the same pattern for the
full matched manifest:

```bash
# Pod SSH terminal
python3 scripts/init_run.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-full-755 \
  --manifest data/full-english-manifest.jsonl \
  --reference-manifest runs/deepseek-ocr-omnidocbench-v1.7-english-full-755/manifest.jsonl \
  --baseline-config configs/internvl3.5-8b-instruct.toml
python3 scripts/record_provenance.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-full-755 \
  --output runtime/cloud_provenance.json
tmux new -s internvl-full
python3 scripts/run_internvl3_5_8b_instruct.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-full-755 \
  --image-root data/omnidocbench-v1.7/images
```

Use `scripts/show_progress.py` in another Pod SSH terminal. If a process is
actually interrupted, rerun the identical command with `--resume`; only
hash-verified terminal records are retained. The runner keeps failures as
zero-byte Markdown predictions plus `failures.jsonl`, and preserves partial
outputs in `raw_predictions/`. After 755/755 verification, evaluate the copied
Mac run against `data/omnidocbench-v1.7/OmniDocBench.json` using the same
official evaluator wrapper, then create
`basline-reports/internvl3.5-8b-instruct/baseline-report.md` from the saved
artifacts and official results.

### Two-GPU sharded execution

The two-GPU harness runs two independent native InternVL processes, each with
one model copy and one physical GPU. It never changes the prompt, model,
preprocessing, decoding, or evaluator. `init_sharded_run.py` derives a
deterministic, disjoint, near-equal partition by SHA-256 page-id rank and saves
the full assignment. `merge_sharded_run.py` refuses to create a parent
prediction set unless every child ledger is terminal and each output's bytes
and SHA-256 match its child record.

Because hardware and execution mode differ from the single-H100 smoke run,
perform a new reportable two-GPU smoke run before the two-GPU full run. The
existing H100 smoke remains valid evidence for that execution mode; it is not
silently replaced.

In the **two-GPU Pod SSH terminal**, after setup and environment activation:

```bash
# Build the parent plus two deterministic child runs.
python3 scripts/init_sharded_run.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-smoke-180-r2-2gpu \
  --manifest data/smoke-180-manifest.jsonl \
  --reference-manifest runs/deepseek-ocr-omnidocbench-v1.7-english-smoke-180/manifest.jsonl \
  --baseline-config configs/internvl3.5-8b-instruct.toml \
  --shards 2

python3 scripts/record_provenance.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-smoke-180-r2-2gpu \
  --output runtime/cloud_provenance.json

tmux new -s internvl-smoke-2gpu
source /workspace/venvs/internvl3.5-8b-instruct/bin/activate
cd /workspace/document-ocr-research
bash scripts/launch_internvl_shards.sh \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-smoke-180-r2-2gpu \
  --image-root data/omnidocbench-v1.7/images \
  --gpu-ids 0,1
```

Monitor the two child ledgers from another Pod SSH terminal:

```bash
python3 scripts/show_sharded_progress.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-smoke-180-r2-2gpu
```

After both shards reach their own totals, still in the Pod SSH terminal:

```bash
python3 scripts/merge_sharded_run.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-smoke-180-r2-2gpu

python3 scripts/verify_run.py \
  --run-dir runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-smoke-180-r2-2gpu
```

The assembled parent directory is the only directory copied to the Mac and
scored. It contains the combined `predictions/` and terminal runtime ledger;
its child `shards/` retain all original raw outputs and attempt logs. A real
interruption is resumed by rerunning the launcher with `--resume`, followed by
the same merge command. Never manually move predictions between shards.

Only after the two-GPU smoke has been evaluated should a two-GPU 755-page run
be initialized identically, replacing the smoke run name/manifest/reference
with `internvl3.5-8b-instruct-omnidocbench-v1.7-english-full-755-r2-2gpu`,
`data/full-english-manifest.jsonl`, and the full DeepSeek reference manifest.

## Matched HunyuanOCR-1.5 baseline

[`configs/hunyuanocr-1.5.toml`](configs/hunyuanocr-1.5.toml) pins the
`tencent/HunyuanOCR` checkpoint at `47644ecc4fc854efa4f505155158831f36773ee4`
and the official HunyuanOCR toolkit at
`f2d2ec989c3b96dcc46a1f434e1032fd12c459d8`. This baseline is **vLLM
autoregressive (AR) only**. It does not start DFlash, use speculative decoding,
or use the archived native Transformers path.

The fixed task is the upstream `doc_parse` prompt. Decoding uses the official
values: `temperature=0`, `top_p=1`, `top_k=-1`, and
`repetition_penalty=1.08`; the benchmark-specific maximum is 8,192 tokens.
The server runs one full model per GPU, so two GPUs are used for data-parallel
shards, not tensor/model parallelism.

### AR-only vLLM Pod setup

This baseline follows HunyuanOCR's official **AR-only vLLM** guide: Python
3.10, vLLM `0.18.1`, Transformers `4.57.6`, and a CUDA-12.x vLLM wheel. A
newer host driver is backward-compatible with that wheel, so a CUDA-13-capable
RTX 4090 Pod is suitable. Do not substitute the unified CUDA-13 environment:
its Transformers 5.13 native-HF route is incompatible with Hunyuan loading in
vLLM. At least 24 GB VRAM is required per GPU.

In the **Pod SSH terminal**, after the repository and `data/` have been copied
to the network volume:

```bash
cd /workspace/document-ocr-research
nvidia-smi
apt-get update && apt-get install -y tmux curl git
bash scripts/setup_hunyuanocr_1_5_environment.sh
source /root/venvs/hunyuanocr-1.5-vllm-ar/bin/activate
```

The setup downloads the pinned model once, checks out the pinned official
client source, creates a full package freeze, and hashes all checkpoint files.
It excludes the legacy `v1.0/` files and `dflash/` draft checkpoint because
this baseline is AR-only; the exclusion is recorded in the model artifact
manifest. Before the model download or server start, it validates Python 3.10,
the exact vLLM/Transformers versions, CUDA visibility, the pinned toolkit
commit, and vLLM registration of `HunYuanVLForConditionalGeneration`. The
setup intentionally has no FlashAttention source-build step. The launcher
also uses vLLM's supported `--enforce-eager` switch: Hunyuan's xDRoPE model
hits a Torch-compile profiling assertion under vLLM 0.18 on this runtime.
This changes only the execution backend; AR weights, prompt, and greedy
decoding remain pinned.

### One GPU or two GPUs

Use the same sharded procedure for both choices. A one-shard parent preserves
the exact merge and verification audit path; a two-shard parent launches one
official AR vLLM server per physical GPU. Pick one choice before starting the
reportable smoke run, and retain it unchanged for its matching full run.

```bash
# Pod SSH terminal: choose ONE of these initialization commands.
# Single GPU (GPU 0):
python3 scripts/init_sharded_run.py \
  --run-dir runs/hunyuanocr-1.5-omnidocbench-v1.7-english-smoke-180-1gpu \
  --manifest data/smoke-180-manifest.jsonl \
  --reference-manifest runs/deepseek-ocr-omnidocbench-v1.7-english-smoke-180/manifest.jsonl \
  --baseline-config configs/hunyuanocr-1.5.toml \
  --shards 1

# Two GPUs (GPUs 0 and 1): change only the run name and shard count.
python3 scripts/init_sharded_run.py \
  --run-dir runs/hunyuanocr-1.5-omnidocbench-v1.7-english-smoke-180-2gpu \
  --manifest data/smoke-180-manifest.jsonl \
  --reference-manifest runs/deepseek-ocr-omnidocbench-v1.7-english-smoke-180/manifest.jsonl \
  --baseline-config configs/hunyuanocr-1.5.toml \
  --shards 2
```

Run the selected launch command inside `tmux` (the model server and the
inference workers are shut down automatically when it completes):

```bash
# Single GPU run
tmux new -s hunyuan-smoke
source /root/venvs/hunyuanocr-1.5-vllm-ar/bin/activate
cd /workspace/document-ocr-research
bash scripts/launch_hunyuanocr_shards.sh \
  --run-dir runs/hunyuanocr-1.5-omnidocbench-v1.7-english-smoke-180-1gpu \
  --image-root data/omnidocbench-v1.7/images \
  --gpu-ids 0

# Two GPU run: use the two-GPU run directory and --gpu-ids 0,1 instead.
```

Monitor the parent from a second Pod terminal:

```bash
python3 scripts/show_sharded_progress.py \
  --run-dir runs/hunyuanocr-1.5-omnidocbench-v1.7-english-smoke-180-1gpu
```

After all child shards are terminal, merge and verify them. These commands
refuse missing, duplicate, or hash-mismatched predictions.

```bash
python3 scripts/merge_sharded_run.py \
  --run-dir runs/hunyuanocr-1.5-omnidocbench-v1.7-english-smoke-180-1gpu
python3 scripts/verify_run.py \
  --run-dir runs/hunyuanocr-1.5-omnidocbench-v1.7-english-smoke-180-1gpu
python3 scripts/summarize_hunyuan_generation.py \
  --run-dir runs/hunyuanocr-1.5-omnidocbench-v1.7-english-smoke-180-1gpu
```

`runtime/generation_termination_summary.json` reports `length` terminations
as cap hits and distinguishes them from client early-stops caused by the
official tail-repetition guard. Review that file before the 755-page run. Do
not silently change the 8,192-token cap: if `length` terminations are material,
record the outcome and create a new configuration/run decision first.

For every page, `raw_predictions/<page>/attempts/<attempt>/` contains
`server_stream_events.jsonl`, the **client-received raw completion before
cleanup**, the post-tail-cleanup text, and the final official client output.
The file scored in `predictions/` is that official processed output (including
the upstream `doc_parse` normalization). If the official streaming client
early-stops for a repeated tail, the raw file is accurately labelled as the
text received by the client, rather than being misrepresented as an unseen
full server completion.

Evaluate the complete 180-page parent run on the Mac with the unchanged
official wrapper. After accepting the smoke metrics and cap summary, use the
identical sequence for the 755-page manifest and matching DeepSeek reference,
replacing `smoke-180` with `full-755`. The full parent is the only directory
copied to the Mac and evaluated; its `shards/` directory retains all server,
raw-output, and runner evidence.

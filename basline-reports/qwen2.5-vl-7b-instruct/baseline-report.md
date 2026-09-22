# Qwen2.5-VL-7B-Instruct on OmniDocBench v1.7: English-only baseline

**Report date:** 2026-09-09  
**Evaluation status:** complete

## Result at a glance

Qwen2.5-VL-7B-Instruct completed inference for **all 755 English OmniDocBench v1.7 pages**. Every expected Markdown prediction was present and non-empty: **755 successful pages, 0 empty outputs, and 0 inference errors**.

## Overall score

**79.5640 / 100** — the official OmniDocBench evaluator's notebook-level aggregate for this 755-page run, using `end2end` evaluation and `quick_match` matching. This is the evaluator's reported aggregate; component metrics below are the detailed comparison values.

## Baseline protocol

| Item | Value |
| --- | --- |
| Dataset | OmniDocBench v1.7, English-only subset (755 pages) |
| Dataset revision | `aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec` |
| Dataset input hash | `2e775fddec46dfda75a19cac2e56ba205f0024c2ad000829095bfd6be304f85c` |
| Manifest hash | `da131d72809bb64866daa7932727f3a6e72778fc2564db754190f5cffc252592` |
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Model revision | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| Hardware | NVIDIA GeForce RTX 4090, 23,028 MiB VRAM |
| Runtime | Python 3.11; Torch `2.6.0+cu124`; Transformers `4.49.0`; `qwen-vl-utils==0.0.8`; FlashAttention `2.7.3` |
| Prompt | Task-specific Markdown instruction recorded in the run configuration |
| Vision budget | `min_pixels=256*28*28`; `max_pixels=1280*28*28` |
| Decoding | Seed `20260906`; BF16; Flash Attention 2; greedy (`do_sample=false`, `num_beams=1`); `max_new_tokens=8192` |
| Evaluator | Official OmniDocBench evaluator, `end2end` + `quick_match` |
| Evaluator repository commit | `193627ae9e97d89188468ed1ee3b7a856ff76044` |
| Evaluator image | `ghcr.io/zeng-weijun/omnidocbench-eval@sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617` |

Qwen does not publish a native OCR-to-Markdown prompt equivalent to DeepSeek-OCR's document-parsing prompt. The fixed task-specific prompt is therefore part of this baseline protocol; it is not presented as a Qwen-native setting.

## Official evaluation metrics

For edit distance, **lower is better**. For BLEU, METEOR, CDM, TEDS, and TEDS structure-only, **higher is better**. The page-average CDM/TEDS values are the values used by the evaluator's notebook aggregate.

| Content type | Metric | Result |
| --- | --- | ---: |
| Text blocks | Edit distance | 0.124420 |
| Text blocks | BLEU | 0.693234 |
| Text blocks | METEOR | 0.276953 |
| Display formulas | Edit distance | 0.226805 |
| Display formulas | CDM, page average | 0.852689 |
| Display formulas | CDM, all samples | 0.792796 |
| Tables | Edit distance | 0.539192 |
| Tables | TEDS, page average | 0.658652 |
| Tables | TEDS, all samples | 0.528318 |
| Tables | TEDS structure-only, page average | 0.735258 |
| Tables | TEDS structure-only, all samples | 0.586946 |
| Reading order | Edit distance | 0.193736 |

The evaluator compared text on 721 pages, formulas on 223 pages, tables on 147 pages, and reading order on 750 pages. Its formula stage processed 1,905 samples with no errors or timeouts.

## Inference coverage and runtime

| Measure | Result |
| --- | ---: |
| Expected prediction files | 755 |
| Actual prediction files | 755 |
| Missing predictions | 0 |
| Unexpected predictions | 0 |
| Zero-byte predictions | 0 |
| Inference failures | 0 |
| Total page-inference time | 8.099 hours |
| Mean page-inference time | 38.618 seconds |
| Median page-inference time | 29.037 seconds |
| 90th-percentile page-inference time | 66.328 seconds |
| Outputs reaching `max_new_tokens=8192` | 28 / 755 (3.71%) |

## Evaluation-stage observations

- Page matching covered all 755 pages. It recorded one `quick_match` fallback after the 300-second quick-match limit, on `newspaper_The Globe and Mail - 2025-1-8@magazinesclubnew_page_014.png`; no page-level match timeout occurred.
- Table TEDS processed 243 samples. One case reached the 120-second evaluator timeout because its prediction was 301,002 characters long; no table-stage errors were recorded. That case remains included in the evaluator's saved timeout records.
- The 28 generation-cap cases and all per-page raw outputs remain in the runtime and prediction artifacts. They were not retried with a larger token limit, truncated by the harness, or excluded from scoring.
- The official evaluator container required the recorded `huggingface_hub==0.25.2` compatibility override. The evaluator image digest remains pinned above, and the override/runtime version are preserved with the run artifacts.

## Comparison with the completed DeepSeek-OCR baseline

Both runs use the identical 755-page manifest, ground truth, evaluator image, and evaluation mode. Qwen was faster, while DeepSeek-OCR had the higher evaluator aggregate and stronger text, formula, table-structure, and reading-order scores.

| Measure | Qwen2.5-VL-7B-Instruct | DeepSeek-OCR |
| --- | ---: | ---: |
| Overall evaluator score | 79.5640 | 83.3661 |
| Text edit distance ↓ | 0.124420 | 0.058139 |
| Formula CDM, page average ↑ | 0.852689 | 0.877689 |
| Table TEDS, page average ↑ | 0.658652 | 0.681435 |
| Reading-order edit distance ↓ | 0.193736 | 0.131053 |
| Total inference time | 8.099 hr | 9.824 hr |

## Reproducibility artifacts

All paths below are relative to this repository.

- [Run configuration](../../runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-full-755/config/baseline.toml)
- [Frozen 755-page manifest](../../runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-full-755/manifest.jsonl)
- [Coverage audit](../../runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-full-755/metrics/coverage_audit.json)
- [Cloud/runtime provenance](../../runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-full-755/runtime/cloud_provenance.json)
- [Per-page runtime ledger](../../runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-full-755/runtime/pages.jsonl)
- [Raw official metric output](../../runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-full-755/metrics/official/predictions_quick_match_metric_result.json)
- [Official evaluator run summary](../../runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-full-755/metrics/official/predictions_quick_match_run_summary.json)
- [Official evaluator stage execution, including fallback and timeout records](../../runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-full-755/metrics/official/predictions_quick_match_stage_execution.json)
- [Evaluator image digest and compatibility records](../../runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-full-755/evaluator/)
- [Raw Markdown predictions](../../runs/qwen2.5-vl-7b-instruct-omnidocbench-v1.7-english-full-755/predictions/)

# PaddleOCR-VL-1.6 on OmniDocBench v1.7: English-only baseline

**Report date:** 2026-09-12  
**Evaluation status:** complete, failure-inclusive

## Result at a glance

PaddleOCR-VL-1.6 completed inference for all **755 English OmniDocBench v1.7 pages** and the official evaluator completed successfully. All 755 expected Markdown files are present. **754 are non-empty and one is empty**; the empty native Markdown output was retained, documented, and included in official evaluation rather than retried, repaired, or excluded.

## Overall score

**96.1418 / 100** — the official OmniDocBench evaluator's notebook-level aggregate for this 755-page run, using `end2end` evaluation and `quick_match` matching. Component metrics below are the evaluator's detailed values.

## Baseline protocol

| Item | Value |
| --- | --- |
| Dataset | OmniDocBench v1.7, English-only subset (755 pages) |
| Dataset revision | `aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec` |
| Dataset input hash | `2e775fddec46dfda75a19cac2e56ba205f0024c2ad000829095bfd6be304f85c` |
| Manifest hash | `da131d72809bb64866daa7932727f3a6e72778fc2564db754190f5cffc252592` |
| Model | `PaddleOCR-VL-1.6-0.9B`, via the native complete `PaddleOCRVL` pipeline |
| PaddleOCR source commit | `0006f7874c334ce9c0f497d8b6cce9cdc0b7363d` (v3.6.0 tag) |
| VLM revision | `PaddlePaddle/PaddleOCR-VL-1.6@c5630abae1d940eafe0697512a0325494b02ab42` |
| Layout model revision | `PaddlePaddle/PP-DocLayoutV3@7b48a7566925fa464281f930c58eee04fe2c862a` |
| Hardware | NVIDIA L4, 23,034 MiB VRAM; driver `580.126.09` |
| Runtime | Python 3.11.10; PaddlePaddle GPU `3.2.1` (CUDA 11.8); PaddleOCR/PaddleX `3.6.0` |
| Pipeline | Native layout detection plus native VLM recognition; `gpu:0`; no TensorRT |
| Decoding | Native local generation without sampling controls; seed `20260906`; `max_new_tokens=8192` |
| Evaluator | Official OmniDocBench evaluator, `end2end` + `quick_match` |
| Evaluator repository commit | `193627ae9e97d89188468ed1ee3b7a856ff76044` |
| Evaluator image | `ghcr.io/zeng-weijun/omnidocbench-eval@sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617` |

The complete native pipeline was used rather than directly calling the VLM component. In the pinned native engine, public sampling controls are unsupported; the recorded local-generate route is therefore the deterministic baseline protocol. The full-run configuration and manifest match the frozen baseline inputs; the actual cloud hardware and package state are preserved in provenance.

## Official evaluation metrics

For edit distance, **lower is better**. For BLEU, METEOR, CDM, TEDS, and TEDS structure-only, **higher is better**. The page-average CDM/TEDS values are used in the evaluator's notebook-level aggregate.

| Content type | Metric | Result |
| --- | --- | ---: |
| Text blocks | Edit distance, page average | 0.032906 |
| Text blocks | BLEU | 0.791119 |
| Text blocks | METEOR | 0.409753 |
| Display formulas | Edit distance, page average | 0.080411 |
| Display formulas | CDM, page average | 0.985104 |
| Display formulas | CDM, all samples | 0.980469 |
| Tables | Edit distance, page average | 0.064956 |
| Tables | TEDS, page average | 0.932056 |
| Tables | TEDS, all samples | 0.904257 |
| Tables | TEDS structure-only, page average | 0.961961 |
| Tables | TEDS structure-only, all samples | 0.939419 |
| Reading order | Edit distance, page average | 0.098225 |

The evaluator compared text on 721 pages, display formulas on 223 pages, tables on 147 pages, and reading order on 750 pages. It processed 1,905 formula samples and 243 table samples with no evaluator-stage errors or timeouts. Page matching covered all 755 pages with no quick-match or page-level timeout fallbacks.

## Inference coverage and runtime

| Measure | Result |
| --- | ---: |
| Expected prediction files | 755 |
| Actual prediction files | 755 |
| Missing predictions | 0 |
| Unexpected predictions | 0 |
| Non-empty predictions | 754 |
| Empty model outputs | 1 |
| Inference errors | 0 |
| Total page-inference time | 7.609 hours |
| Mean page-inference time | 36.280 seconds |
| Median page-inference time | 30.355 seconds |
| 90th-percentile page-inference time | 62.784 seconds |
| Maximum page-inference time | 722.901 seconds |

The empty output is `color_textbook_zhonggaokao_小学_13.人教新起点英语（4-5年级）_人教新起点五年级英语上册_课本_人教新起点英语5A电子课本_page_004.md`. It is a zero-byte file because the native pipeline wrote no Markdown for that page. Its source image hash, timestamp, native-pipeline settings, and status are retained in `failures.jsonl`; its zero-byte prediction remains in the evaluated directory. No retry or textual failure marker was used.

## Smoke-test gate

The clean L4 smoke run completed before full inference: **180/180 pages**, with **180 successful outputs**, zero empty outputs, and zero inference errors. Its official notebook aggregate was **96.8901 / 100**. The model configuration and frozen manifests were not altered after that smoke run.

## Comparison with completed baselines

All three baselines use the identical 755-page English manifest, ground truth, official evaluator image, and `end2end` + `quick_match` evaluation mode. Runtime comparisons are informative rather than hardware-normalized: PaddleOCR-VL used an L4, while DeepSeek-OCR and Qwen used RTX 4090 instances.

| Measure | PaddleOCR-VL-1.6 | DeepSeek-OCR | Qwen2.5-VL-7B-Instruct |
| --- | ---: | ---: | ---: |
| Overall evaluator score | 96.1418 | 83.3661 | 79.5640 |
| Text edit distance ↓ | 0.032906 | 0.058139 | 0.124420 |
| Formula CDM, page average ↑ | 0.985104 | 0.877689 | 0.852689 |
| Table TEDS, page average ↑ | 0.932056 | 0.681435 | 0.658652 |
| Reading-order edit distance ↓ | 0.098225 | 0.131053 | 0.193736 |
| Total inference time | 7.609 hr | 9.824 hr | 8.099 hr |

## Reproducibility artifacts

All paths below are relative to this repository.

- [Run configuration](../../runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755/config/baseline.toml)
- [Frozen 755-page manifest](../../runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755/manifest.jsonl)
- [Coverage audit](../../runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755/metrics/coverage_audit.json)
- [Cloud/runtime provenance](../../runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755/runtime/cloud_provenance.json)
- [Pipeline initialization record](../../runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755/runtime/pipeline_initialization.json)
- [Pinned model artifacts](../../runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755/runtime/pinned_model_artifacts.json)
- [Pinned environment freeze](../../runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755/runtime/pinned_environment_freeze.txt)
- [Per-page runtime ledger](../../runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755/runtime/pages.jsonl)
- [Failure-inclusive empty-output record](../../runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755/failures.jsonl)
- [Raw official metric output](../../runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755/metrics/official/predictions_quick_match_metric_result.json)
- [Official evaluator run summary](../../runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755/metrics/official/predictions_quick_match_run_summary.json)
- [Official evaluator stage execution](../../runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755/metrics/official/predictions_quick_match_stage_execution.json)
- [Evaluator image digest and compatibility records](../../runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755/evaluator/)
- [Raw Markdown predictions](../../runs/paddleocr-vl-1.6-omnidocbench-v1.7-english-full-755/predictions/)

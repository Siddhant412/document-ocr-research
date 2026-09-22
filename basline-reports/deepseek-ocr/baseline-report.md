# DeepSeek-OCR on OmniDocBench v1.7: English-only baseline

**Report date:** 2026-09-08  
**Evaluation status:** complete

## Result at a glance

DeepSeek-OCR completed inference for **all 755 English OmniDocBench v1.7 pages**. Every expected Markdown prediction file was present and non-empty: **755 successful pages, 0 empty outputs, and 0 inference errors**.

## Overall score

**83.3661 / 100** — the official OmniDocBench evaluator's notebook-level aggregate for this 755-page run, using `end2end` evaluation and `quick_match` matching. This is the evaluator's reported aggregate; the component metrics below are the primary values for detailed comparison.

## Baseline protocol

| Item | Value |
| --- | --- |
| Dataset | OmniDocBench v1.7, English-only subset (755 pages) |
| Dataset revision | `aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec` |
| Dataset input hash | `2e775fddec46dfda75a19cac2e56ba205f0024c2ad000829095bfd6be304f85c` |
| Model | `deepseek-ai/DeepSeek-OCR` |
| Model repository commit | `09eaf526153e7a01ed16c9dea8c96282aaea29c0` |
| Hugging Face model revision | `9f30c71f441d010e5429c532364a86705536c53a` |
| Hardware | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| Decoding | deterministic seed `20260906`; greedy decoding |
| Native document prompt | `<image>\n<|grounding|>Convert the document to markdown. ` |
| DeepSeek inference settings | `base_size=1024`, `image_size=640`, `crop_mode=True`, `test_compress=True` |
| Evaluator | Official OmniDocBench evaluator, `end2end` + `quick_match` |
| Evaluator repository commit | `193627ae9e97d89188468ed1ee3b7a856ff76044` |
| Evaluator image | `ghcr.io/zeng-weijun/omnidocbench-eval@sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617` |

## Official evaluation metrics

For edit distance, **lower is better**. For BLEU, METEOR, CDM, TEDS, and TEDS structure-only, **higher is better**.

| Content type | Metric | Result |
| --- | --- | ---: |
| Text blocks | Edit distance | 0.058139 |
| Text blocks | BLEU | 0.739238 |
| Text blocks | METEOR | 0.357459 |
| Display formulas | Edit distance | 0.172781 |
| Display formulas | CDM | 0.886861 |
| Tables | Edit distance | 0.246385 |
| Tables | TEDS | 0.566869 |
| Tables | TEDS structure-only | 0.645040 |
| Reading order | Edit distance | 0.131053 |

The evaluator compared text on 721 pages, formulas on 223 pages, tables on 147 pages, and reading order on 750 pages. Its formula stage processed 1,905 samples without errors or timeouts. The table TEDS stage processed 243 samples with no timeouts, but recorded **10 parsing errors** caused by malformed table HTML values. Those cases were retained in the official evaluator records; they were not removed or manually repaired.

## Inference coverage and runtime

| Measure | Result |
| --- | ---: |
| Expected prediction files | 755 |
| Actual prediction files | 755 |
| Missing predictions | 0 |
| Unexpected predictions | 0 |
| Zero-byte predictions | 0 |
| Inference failures | 0 |
| Total page-inference time | 9.824 hours |
| Mean page-inference time | 46.843 seconds |
| Median page-inference time | 38.668 seconds |
| 90th-percentile page-inference time | 79.195 seconds |

## Evaluation notes

- Page matching completed for all 755 pages with zero quick-match and page-level timeouts.
- The official evaluator container needed a recorded compatibility override, `huggingface_hub==0.25.2`, because the image's installed `evaluate` version expected the removed `HfFolder` API. The evaluator image digest remains pinned above; the override and resulting runtime version are preserved with the run artifacts.
- Raw predictions, per-page runtime records, configuration, input hashes, evaluator output, and any evaluator-stage errors remain in the run directory. No prediction outputs were skipped or altered after inference.

## Reproducibility artifacts

All paths below are relative to this repository.

- [Run configuration](../../runs/deepseek-ocr-omnidocbench-v1.7-english-full-755/config/baseline.toml)
- [Frozen 755-page manifest](../../runs/deepseek-ocr-omnidocbench-v1.7-english-full-755/manifest.jsonl)
- [Coverage audit](../../runs/deepseek-ocr-omnidocbench-v1.7-english-full-755/metrics/coverage_audit.json)
- [Per-page runtime ledger](../../runs/deepseek-ocr-omnidocbench-v1.7-english-full-755/runtime/pages.jsonl)
- [Raw official metric output](../../runs/deepseek-ocr-omnidocbench-v1.7-english-full-755/metrics/official/predictions_quick_match_metric_result.json)
- [Official evaluator run summary](../../runs/deepseek-ocr-omnidocbench-v1.7-english-full-755/metrics/official/predictions_quick_match_run_summary.json)
- [Official evaluator stage execution, including table errors](../../runs/deepseek-ocr-omnidocbench-v1.7-english-full-755/metrics/official/predictions_quick_match_stage_execution.json)
- [Evaluator image digest and compatibility records](../../runs/deepseek-ocr-omnidocbench-v1.7-english-full-755/evaluator/)
- [Raw Markdown predictions](../../runs/deepseek-ocr-omnidocbench-v1.7-english-full-755/predictions/)

## Smoke-test gate

The preceding English-only smoke test also completed: **180/180 pages**, with **0 inference errors** and **0 empty outputs**. Its evaluator notebook aggregate was **83.4126**. The full-run configuration was not changed after that smoke test.

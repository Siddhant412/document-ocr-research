# InternVL3.5-8B-Instruct on OmniDocBench v1.7: English-only baseline

**Report date:** 2026-09-14  
**Evaluation status:** complete

## Result at a glance

InternVL3.5-8B-Instruct completed inference for all **755 English OmniDocBench v1.7 pages** on two RTX 4090 GPUs. The final assembled run contains **755 successful, non-empty Markdown predictions** with **0 inference errors**.

## Overall score

**79.6306 / 100** — the official OmniDocBench evaluator's notebook-level aggregate for this 755-page run, using `end2end` evaluation and `quick_match` matching.

## Baseline protocol

| Item | Value |
| --- | --- |
| Dataset | OmniDocBench v1.7, English-only subset (755 pages) |
| Dataset revision | `aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec` |
| Dataset input hash | `2e775fddec46dfda75a19cac2e56ba205f0024c2ad000829095bfd6be304f85c` |
| Manifest hash | `da131d72809bb64866daa7932727f3a6e72778fc2564db754190f5cffc252592` |
| Model | `OpenGVLab/InternVL3_5-8B-Instruct` |
| Model revision | `61e3b61d08dac94508888b06f5e54d3b44353d28` |
| Hardware | 2 × NVIDIA GeForce RTX 4090, 24,564 MiB VRAM each |
| Runtime | Python 3.11; Torch `2.7.1+cu128`; Transformers `4.52.1`; FlashAttention `2.7.4.post1` |
| Vision preprocessing | Official InternVL dynamic tiling: `image_size=448`, 1–12 tiles, thumbnail enabled |
| Prompt | `Convert this document page to Markdown. Preserve all text, reading order, headings, lists, tables, and formulas. Represent tables as HTML and formulas as LaTeX. Return only the Markdown, with no commentary.` |
| Mode | Normal instruct mode; thinking mode disabled |
| Decoding | Seed `20260906`; BF16; Flash Attention 2; greedy (`do_sample=false`, `num_beams=1`); `max_new_tokens=8192` |
| Evaluator | Official OmniDocBench evaluator, `end2end` + `quick_match` |
| Evaluator repository commit | `193627ae9e97d89188468ed1ee3b7a856ff76044` |
| Evaluator image | `ghcr.io/zeng-weijun/omnidocbench-eval@sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617` |

The prompt is a fixed Markdown-transcription instruction for this baseline, not a claim that InternVL publishes a native OCR prompt. The runner prepends InternVL's official single-image token.

## Official evaluation metrics

For edit distance, lower is better. For BLEU, METEOR, CDM, TEDS, and TEDS structure-only, higher is better. Page-average CDM/TEDS values are the values used by the evaluator's notebook aggregate.

| Content type | Metric | Result |
| --- | --- | ---: |
| Text blocks | Edit distance | 0.118741 |
| Text blocks | BLEU | 0.557595 |
| Text blocks | METEOR | 0.255450 |
| Display formulas | Edit distance | 0.236533 |
| Display formulas | CDM, page average | 0.865289 |
| Display formulas | CDM, all samples | 0.847089 |
| Tables | Edit distance | 0.573491 |
| Tables | TEDS, page average | 0.642371 |
| Tables | TEDS, all samples | 0.522892 |
| Tables | TEDS structure-only, page average | 0.722598 |
| Tables | TEDS structure-only, all samples | 0.593036 |
| Reading order | Edit distance | 0.190285 |

The evaluator compared text on 721 pages, formulas on 223 pages, tables on 147 pages, and reading order on 750 pages. Formula CDM processed 1,905 samples; table TEDS processed 243 samples. Neither metric stage recorded an error, exception, or timeout.

## Inference coverage and runtime

| Measure | Result |
| --- | ---: |
| Expected prediction files | 755 |
| Actual prediction files | 755 |
| Missing predictions | 0 |
| Unexpected predictions | 0 |
| Zero-byte predictions | 0 |
| Inference failures | 0 |
| Total summed GPU page-inference time | 7.103 hours |
| Mean page-inference time | 33.868 seconds |
| Median page-inference time | 21.308 seconds |
| 90th-percentile page-inference time | 52.325 seconds |
| Maximum page-inference time | 196.885 seconds |
| Outputs reaching `max_new_tokens=8192` | 40 / 755 (5.30%) |

The two shards ran concurrently: shard 000 processed 378 pages and shard 001 processed 377 pages. Wall-clock inference spanned approximately 3 hours 44 minutes, excluding setup and final merge.

## Evaluation-stage observations and recovery audit

- Page matching covered all 755 pages. The evaluator recorded one `quick_match` fallback after its 300-second quick-match limit, on `newspaper_TheWashingtonPost-2025-01-08@magazinesclubnew_page_042.png`; no page-level match timeout occurred.
- The original two-GPU Pod was removed after inference because the RunPod account balance became negative. Its persistent network volume retained both complete shard ledgers and all 755 direct filesystem predictions.
- RunPod's S3-compatible API could list, but could not directly download, 52 files with special characters in their filenames. A CPU-only Pod in the same region created a checksum-validated tar archive directly from the mounted volume. The archive contained all 755 predictions; its SHA-256 was `48076dd0263125650c202c7188d1a21d1a9993dca663612c8af243f4bbb13c51`.
- After extraction, the strict merge checked each shard prediction's byte count and SHA-256 against its per-page ledger. The assembled parent run then passed coverage verification with 755 expected and 755 actual predictions. No page was rerun or omitted during recovery.

## Comparison with completed baselines

All four results use the identical 755-page manifest, English-only ground truth, evaluator image, and `end2end + quick_match` evaluation mode.

| Model | Overall evaluator score |
| --- | ---: |
| PaddleOCR-VL-1.6 | 96.1418 |
| DeepSeek-OCR | 83.3661 |
| InternVL3.5-8B-Instruct | 79.6306 |
| Qwen2.5-VL-7B-Instruct | 79.5640 |

## Reproducibility artifacts

All paths below are relative to this repository.

- [Run configuration](../../runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-full-755-r2-2gpu/config/baseline.toml)
- [Frozen 755-page manifest](../../runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-full-755-r2-2gpu/manifest.jsonl)
- [Deterministic two-shard assignment](../../runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-full-755-r2-2gpu/sharding/assignment.json)
- [Shard merge audit](../../runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-full-755-r2-2gpu/sharding/merge.json)
- [Coverage audit](../../runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-full-755-r2-2gpu/metrics/coverage_audit.json)
- [Cloud/runtime provenance](../../runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-full-755-r2-2gpu/runtime/cloud_provenance.json)
- [Combined per-page runtime ledger](../../runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-full-755-r2-2gpu/runtime/pages.jsonl)
- [Raw official metric output](../../runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-full-755-r2-2gpu/metrics/official/predictions_quick_match_metric_result.json)
- [Official evaluator summary](../../runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-full-755-r2-2gpu/metrics/official/predictions_quick_match_run_summary.json)
- [Official evaluator stage execution](../../runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-full-755-r2-2gpu/metrics/official/predictions_quick_match_stage_execution.json)
- [Final Markdown predictions](../../runs/internvl3.5-8b-instruct-omnidocbench-v1.7-english-full-755-r2-2gpu/predictions/)

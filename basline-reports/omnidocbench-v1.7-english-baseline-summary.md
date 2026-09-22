# OmniDocBench v1.7 English baseline summary

Four completed baselines were evaluated on the same frozen **755-page English-only** manifest with the official OmniDocBench evaluator (`end2end` + `quick_match`). The overall score is the evaluator's notebook aggregate out of 100. Higher is better except for edit distance.

| Model | Overall | Text edit ↓ | Formula CDM ↑ | Table TEDS ↑ | Reading-order edit ↓ | Output coverage | Inference hardware / time |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| PaddleOCR-VL-1.6 | **96.1418** | **0.032906** | **0.985104** | **0.932056** | **0.098225** | 755/755 files; 1 empty output | L4; 7.609 h |
| DeepSeek-OCR | 83.3661 | 0.058139 | 0.877689 | 0.681435 | 0.131053 | 755/755 non-empty | RTX 4090; 9.824 h |
| InternVL3.5-8B-Instruct | 79.6306 | 0.118741 | 0.865289 | 0.642371 | 0.190285 | 755/755 non-empty | 2 × RTX 4090; 3 h 44 m wall time (7.103 summed GPU h) |
| Qwen2.5-VL-7B-Instruct | 79.5640 | 0.124420 | 0.852689 | 0.658652 | 0.193736 | 755/755 non-empty | RTX 4090; 8.099 h |

## Takeaway

PaddleOCR-VL-1.6 is the strongest baseline across the reported aggregate and component metrics. DeepSeek-OCR is the next strongest overall. InternVL3.5 and Qwen2.5-VL are effectively tied overall (difference: 0.0666 points); InternVL is slightly stronger on text, formulas, and reading order, while Qwen is slightly stronger on table TEDS.

## Evaluation notes

- The evaluator compared 721 text pages, 223 formula pages, 147 table pages, and 750 reading-order pages for every run.
- PaddleOCR-VL's one empty native output was retained and evaluated; no outputs were retried, repaired, or excluded.
- DeepSeek-OCR retained 10 official table-HTML parsing errors. Qwen had one table-TEDS timeout and one quick-match fallback; InternVL had one quick-match fallback. PaddleOCR-VL had no evaluator-stage errors or timeouts.
- Runtime is informative only: GPU models, GPU counts, and cloud hardware differ.

## Detailed reports

- [DeepSeek-OCR](deepseek-ocr/baseline-report.md)
- [PaddleOCR-VL-1.6](paddleocr-vl-1.6/baseline-report.md)
- [Qwen2.5-VL-7B-Instruct](qwen2.5-vl-7b-instruct/baseline-report.md)
- [InternVL3.5-8B-Instruct](internvl3.5-8b-instruct/baseline-report.md)

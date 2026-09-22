from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_smoke_manifest.py"
SUBSET_SCRIPT = ROOT / "scripts" / "build_ground_truth_subset.py"
FULL_SCRIPT = ROOT / "scripts" / "build_full_manifest.py"
INIT_SCRIPT = ROOT / "scripts" / "init_run.py"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_run.py"
INFERENCE_SCRIPT = ROOT / "scripts" / "run_deepseek_ocr.py"
QWEN_INFERENCE_SCRIPT = ROOT / "scripts" / "run_qwen2_5_vl.py"
INTERNVL_INFERENCE_SCRIPT = ROOT / "scripts" / "run_internvl3_5_8b_instruct.py"
PROGRESS_SCRIPT = ROOT / "scripts" / "show_progress.py"
SHARDED_INIT_SCRIPT = ROOT / "scripts" / "init_sharded_run.py"
SHARDED_MERGE_SCRIPT = ROOT / "scripts" / "merge_sharded_run.py"
MATERIALIZE_HASHED_MANIFEST_SCRIPT = ROOT / "scripts" / "materialize_hashed_manifest.py"

SPEC = importlib.util.spec_from_file_location("run_deepseek_ocr", INFERENCE_SCRIPT)
assert SPEC and SPEC.loader
INFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INFERENCE)
QWEN_SPEC = importlib.util.spec_from_file_location("run_qwen2_5_vl", QWEN_INFERENCE_SCRIPT)
assert QWEN_SPEC and QWEN_SPEC.loader
QWEN_INFERENCE = importlib.util.module_from_spec(QWEN_SPEC)
QWEN_SPEC.loader.exec_module(QWEN_INFERENCE)
INTERNVL_SPEC = importlib.util.spec_from_file_location("run_internvl3_5_8b_instruct", INTERNVL_INFERENCE_SCRIPT)
assert INTERNVL_SPEC and INTERNVL_SPEC.loader
INTERNVL_INFERENCE = importlib.util.module_from_spec(INTERNVL_SPEC)
INTERNVL_SPEC.loader.exec_module(INTERNVL_INFERENCE)
HUNYUAN_INFERENCE_SCRIPT = ROOT / "scripts" / "run_hunyuanocr_1_5.py"
HUNYUAN_SPEC = importlib.util.spec_from_file_location("run_hunyuanocr_1_5", HUNYUAN_INFERENCE_SCRIPT)
assert HUNYUAN_SPEC and HUNYUAN_SPEC.loader
HUNYUAN_INFERENCE = importlib.util.module_from_spec(HUNYUAN_SPEC)
HUNYUAN_SPEC.loader.exec_module(HUNYUAN_INFERENCE)
PADDLE_INFERENCE_SCRIPT = ROOT / "scripts" / "run_paddleocr_vl_1_6.py"
PADDLE_SPEC = importlib.util.spec_from_file_location("run_paddleocr_vl_1_6", PADDLE_INFERENCE_SCRIPT)
assert PADDLE_SPEC and PADDLE_SPEC.loader
PADDLE_INFERENCE = importlib.util.module_from_spec(PADDLE_SPEC)
PADDLE_SPEC.loader.exec_module(PADDLE_INFERENCE)
PROGRESS_SPEC = importlib.util.spec_from_file_location("show_progress", PROGRESS_SCRIPT)
assert PROGRESS_SPEC and PROGRESS_SPEC.loader
PROGRESS = importlib.util.module_from_spec(PROGRESS_SPEC)
PROGRESS_SPEC.loader.exec_module(PROGRESS)
SHARDED_INIT_SPEC = importlib.util.spec_from_file_location("init_sharded_run", SHARDED_INIT_SCRIPT)
assert SHARDED_INIT_SPEC and SHARDED_INIT_SPEC.loader
SHARDED_INIT = importlib.util.module_from_spec(SHARDED_INIT_SPEC)
SHARDED_INIT_SPEC.loader.exec_module(SHARDED_INIT)


class SmokeManifestTest(unittest.TestCase):
    def test_materialize_hashed_manifest_preserves_selection_and_adds_digests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source_rows = [
                {"page_id": "one", "image_path": "one.png", "prediction_filename": "one.md"},
                {"page_id": "two", "image_path": "two.jpg", "prediction_filename": "two.md"},
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8")
            registry = root / "checksums.json"
            registry.write_text(json.dumps({
                "images_sha256_manifest": "manifest-id",
                "images": [
                    {"image_path": "one.png", "sha256": "a" * 64, "bytes": 10},
                    {"image_path": "two.jpg", "sha256": "b" * 64, "bytes": 20},
                ],
            }), encoding="utf-8")
            output = root / "hashed.jsonl"
            subprocess.run([
                sys.executable, str(MATERIALIZE_HASHED_MANIFEST_SCRIPT),
                "--manifest", str(source), "--checksum-manifest", str(registry),
                "--expected-images-sha256-manifest", "manifest-id", "--output", str(output),
            ], check=True)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["page_id"] for row in rows], ["one", "two"])
            self.assertEqual([row["input_sha256"] for row in rows], ["a" * 64, "b" * 64])
            self.assertEqual([row["input_bytes"] for row in rows], [10, 20])

    def test_hunyuan_manifest_requires_immutable_input_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.jsonl"
            base = {
                "page_id": "page-1",
                "image_path": "page-1.png",
                "prediction_filename": "page-1.md",
            }
            manifest.write_text(json.dumps(base) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "input_sha256"):
                HUNYUAN_INFERENCE.read_manifest(manifest)
            base["input_sha256"] = "a" * 64
            manifest.write_text(json.dumps(base) + "\n", encoding="utf-8")
            self.assertEqual(HUNYUAN_INFERENCE.read_manifest(manifest), [base])

    def test_resume_only_retains_hash_verified_terminal_pages(self) -> None:
        manifest = [
            {"page_id": "completed", "prediction_filename": "completed.md"},
            {"page_id": "failure", "prediction_filename": "failure.md"},
            {"page_id": "orphan", "prediction_filename": "orphan.md"},
            {"page_id": "missing", "prediction_filename": "missing.md"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions"
            predictions.mkdir()
            runtime = root / "pages.jsonl"
            (predictions / "completed.md").write_text("model text", encoding="utf-8")
            (predictions / "failure.md").touch()
            (predictions / "orphan.md").write_text("partial", encoding="utf-8")
            records = [
                {
                    "page_id": "completed", "prediction_filename": "completed.md", "status": "ok",
                    "prediction_bytes": (predictions / "completed.md").stat().st_size,
                    "prediction_sha256": INFERENCE.digest(predictions / "completed.md"),
                },
                {
                    "page_id": "failure", "prediction_filename": "failure.md", "status": "error",
                    "prediction_bytes": 0,
                    "prediction_sha256": INFERENCE.digest(predictions / "failure.md"),
                },
            ]
            runtime.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            pending, completed = INFERENCE.plan_pages(manifest, predictions, runtime, resume=True)
            self.assertEqual(set(completed), {"completed", "failure"})
            self.assertEqual([page["page_id"] for page in pending], ["orphan", "missing"])
            qwen_pending, qwen_completed = QWEN_INFERENCE.plan_pages(manifest, predictions, runtime, resume=True)
            self.assertEqual(set(qwen_completed), {"completed", "failure"})
            self.assertEqual([page["page_id"] for page in qwen_pending], ["orphan", "missing"])
            internvl_pending, internvl_completed = INTERNVL_INFERENCE.plan_pages(
                manifest, predictions, runtime, resume=True
            )
            self.assertEqual(set(internvl_completed), {"completed", "failure"})
            self.assertEqual([page["page_id"] for page in internvl_pending], ["orphan", "missing"])
            paddle_pending, paddle_completed = PADDLE_INFERENCE.plan_pages(
                manifest, predictions, runtime, resume=True
            )
            self.assertEqual(set(paddle_completed), {"completed", "failure"})
            self.assertEqual([page["page_id"] for page in paddle_pending], ["orphan", "missing"])
            with self.assertRaises(SystemExit):
                INFERENCE.plan_pages(manifest, predictions, runtime, resume=False)

    def test_paddle_runtime_artifact_copy_requires_and_preserves_pinned_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            run_dir = root / "run"
            (run_dir / "runtime").mkdir(parents=True)
            config = {"environment": {"model_cache_root": str(cache)}}
            with self.assertRaises(FileNotFoundError):
                PADDLE_INFERENCE.copy_runtime_artifacts(run_dir, config)
            cache.mkdir()
            (cache / "pinned-models.json").write_text('{"snapshots": []}\n', encoding="utf-8")
            (cache / "pinned-environment-freeze.txt").write_text("paddleocr==3.6.0\n", encoding="utf-8")
            manifest_hash = PADDLE_INFERENCE.copy_runtime_artifacts(run_dir, config)
            self.assertEqual(
                manifest_hash, PADDLE_INFERENCE.digest(cache / "pinned-models.json")
            )
            self.assertEqual(
                (run_dir / "runtime" / "pinned_model_artifacts.json").read_text(encoding="utf-8"),
                '{"snapshots": []}\n',
            )

    def test_internvl_runtime_artifact_copy_requires_and_preserves_pinned_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            run_dir = root / "run"
            (run_dir / "runtime").mkdir(parents=True)
            config = {"environment": {"model_cache_root": str(cache)}}
            with self.assertRaises(FileNotFoundError):
                INTERNVL_INFERENCE.copy_runtime_artifacts(run_dir, config)
            cache.mkdir()
            (cache / "pinned-models.json").write_text('{"snapshots": []}\n', encoding="utf-8")
            (cache / "pinned-environment-freeze.txt").write_text("transformers==4.52.1\n", encoding="utf-8")
            manifest_hash = INTERNVL_INFERENCE.copy_runtime_artifacts(run_dir, config)
            self.assertEqual(manifest_hash, INTERNVL_INFERENCE.digest(cache / "pinned-models.json"))
            self.assertEqual(
                (run_dir / "runtime" / "pinned_environment_freeze.txt").read_text(encoding="utf-8"),
                "transformers==4.52.1\n",
            )

    def test_internvl_dynamic_tiling_ratio_selection_is_deterministic(self) -> None:
        ratios = [(1, 1), (1, 2), (2, 1), (2, 2)]
        self.assertEqual(
            INTERNVL_INFERENCE.find_closest_aspect_ratio(1.98, ratios, 896, 448, 448), (2, 1)
        )
        self.assertEqual(
            INTERNVL_INFERENCE.find_closest_aspect_ratio(0.51, ratios, 448, 896, 448), (1, 2)
        )

    def test_sharded_run_is_deterministic_disjoint_and_mergeable(self) -> None:
        rows = [
            {"page_id": f"page-{index}", "prediction_filename": f"page-{index}.md", "image_path": f"x/{index}.png"}
            for index in range(7)
        ]
        self.assertEqual(SHARDED_INIT.partition_rows(rows, 2), SHARDED_INIT.partition_rows(rows, 2))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "source.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            config = root / "baseline.toml"
            config.write_text("[inference]\nseed = 1\n", encoding="utf-8")
            run_dir = root / "run"
            subprocess.run(
                [
                    sys.executable, str(SHARDED_INIT_SCRIPT), "--run-dir", str(run_dir),
                    "--manifest", str(manifest), "--reference-manifest", str(manifest),
                    "--baseline-config", str(config), "--shards", "2",
                ],
                check=True,
            )
            assignment = json.loads((run_dir / "sharding" / "assignment.json").read_text(encoding="utf-8"))
            self.assertEqual([shard["manifest_rows"] for shard in assignment["shards"]], [4, 3])
            assigned_ids = [entry["page_id"] for entry in assignment["assignments"]]
            self.assertEqual(set(assigned_ids), {row["page_id"] for row in rows})
            self.assertEqual(len(assigned_ids), len(set(assigned_ids)))
            for shard in assignment["shards"]:
                shard_dir = run_dir / "shards" / shard["name"]
                shard_rows = [
                    json.loads(line) for line in (shard_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                ledger = []
                for row in shard_rows:
                    prediction = shard_dir / "predictions" / row["prediction_filename"]
                    prediction.write_text(row["page_id"], encoding="utf-8")
                    ledger.append(
                        {
                            "page_id": row["page_id"], "prediction_filename": row["prediction_filename"],
                            "status": "ok", "prediction_bytes": prediction.stat().st_size,
                            "prediction_sha256": INFERENCE.digest(prediction),
                        }
                    )
                (shard_dir / "runtime" / "pages.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in ledger), encoding="utf-8"
                )
            subprocess.run([sys.executable, str(SHARDED_MERGE_SCRIPT), "--run-dir", str(run_dir)], check=True)
            self.assertEqual(
                {path.name for path in (run_dir / "predictions").glob("*.md")},
                {row["prediction_filename"] for row in rows},
            )
            merged = [json.loads(line) for line in (run_dir / "runtime" / "pages.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["page_id"] for record in merged], [row["page_id"] for row in rows])

    def test_progress_counts_latest_terminal_status_for_each_page(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "runtime").mkdir()
            manifest = [
                {"page_id": "one"}, {"page_id": "two"}, {"page_id": "three"}, {"page_id": "four"}
            ]
            (run_dir / "manifest.jsonl").write_text(
                "".join(json.dumps(page) + "\n" for page in manifest), encoding="utf-8"
            )
            records = [
                {"page_id": "one", "status": "ok", "started_at_utc": "2026-01-01T00:00:00+00:00"},
                {"page_id": "two", "status": "error", "started_at_utc": "2026-01-01T00:01:00+00:00"},
                {"page_id": "one", "status": "empty_output", "started_at_utc": "2026-01-01T00:02:00+00:00"},
                {"page_id": "not-in-manifest", "status": "ok"},
                "{torn-json",  # A power loss during append must not break status reporting.
            ]
            (run_dir / "runtime" / "pages.jsonl").write_text(
                "\n".join(record if isinstance(record, str) else json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            report = PROGRESS.progress_report(run_dir)
            self.assertEqual(report["processed_pages"], 2)
            self.assertEqual(report["remaining_pages"], 2)
            self.assertEqual(report["successful_pages"], 0)
            self.assertEqual(report["empty_model_outputs"], 1)
            self.assertEqual(report["inference_errors"], 1)

    def test_qwen_local_image_reference_preserves_literal_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "a space_%2F_搬书匠.png"
            image.touch()
            reference = QWEN_INFERENCE.local_image_reference(image)
            self.assertEqual(reference, str(image.resolve()))
            self.assertNotIn("file://", reference)
            self.assertIn("a space_%2F_搬书匠.png", reference)

    def test_source_quotas_and_marginals_are_reproducible(self) -> None:
        pages = []
        for source in ("a", "b"):
            for language in ("en", "zh"):
                for layout in ("single", "double"):
                    for replicate in range(3):
                        pages.append(
                            {
                                "page_info": {
                                    "image_path": f"images/{source}_{language}_{layout}_{replicate}.jpg",
                                    "page_attribute": {
                                        "data_source": source,
                                        "language": language,
                                        "layout": layout,
                                    },
                                }
                            }
                        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps(pages), encoding="utf-8")
            outputs = []
            for index in range(2):
                manifest = root / f"manifest-{index}.jsonl"
                coverage = root / f"coverage-{index}.json"
                subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--annotations",
                        str(annotations),
                        "--output",
                        str(manifest),
                        "--coverage-output",
                        str(coverage),
                        "--pages",
                        "12",
                        "--minimum-per-source",
                        "2",
                    ],
                    check=True,
                )
                outputs.append((manifest.read_text(encoding="utf-8"), json.loads(coverage.read_text())))
            self.assertEqual(outputs[0][0], outputs[1][0])
            self.assertEqual(outputs[0][1]["marginal_residuals"], {"language": {}, "layout": {}})
            self.assertEqual(outputs[0][1]["actual"]["data_source"], {"a": 6, "b": 6})
            self.assertEqual(outputs[0][1]["actual"]["language"], {"en": 6, "zh": 6})
            self.assertEqual(outputs[0][1]["actual"]["layout"], {"double": 6, "single": 6})
            english_manifest = root / "english-manifest.jsonl"
            english_coverage = root / "english-coverage.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--annotations",
                    str(annotations),
                    "--output",
                    str(english_manifest),
                    "--coverage-output",
                    str(english_coverage),
                    "--pages",
                    "12",
                    "--minimum-per-source",
                    "2",
                    "--language",
                    "en",
                ],
                check=True,
            )
            english_rows = [json.loads(line) for line in english_manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(english_rows), 12)
            self.assertEqual({row["language"] for row in english_rows}, {"en"})
            subset = root / "subset.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SUBSET_SCRIPT),
                    "--annotations",
                    str(annotations),
                    "--manifest",
                    str(root / "manifest-0.jsonl"),
                    "--output",
                    str(subset),
                ],
                check=True,
            )
            self.assertEqual(len(json.loads(subset.read_text(encoding="utf-8"))), 12)
            full_manifest = root / "full.jsonl"
            full_coverage = root / "full-coverage.json"
            subprocess.run(
                [
                    sys.executable,
                    str(FULL_SCRIPT),
                    "--annotations",
                    str(annotations),
                    "--output",
                    str(full_manifest),
                    "--coverage-output",
                    str(full_coverage),
                ],
                check=True,
            )
            self.assertEqual(len(full_manifest.read_text(encoding="utf-8").splitlines()), 24)

            baseline = root / "baseline.toml"
            baseline.write_text("[inference]\nseed = 1\n", encoding="utf-8")
            run_dir = root / "run"
            subprocess.run(
                [
                    sys.executable,
                    str(INIT_SCRIPT),
                    "--run-dir",
                    str(run_dir),
                    "--manifest",
                    str(root / "manifest-0.jsonl"),
                    "--baseline-config",
                    str(baseline),
                    "--reference-manifest",
                    str(root / "manifest-0.jsonl"),
                ],
                check=True,
            )
            run_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_metadata["manifest_sha256"], run_metadata["reference_manifest_sha256"])
            smoke_rows = [
                json.loads(line)
                for line in (root / "manifest-0.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            for row in smoke_rows:
                (run_dir / "predictions" / row["prediction_filename"]).touch()
            subprocess.run(
                [sys.executable, str(VERIFY_SCRIPT), "--run-dir", str(run_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            audit = json.loads((run_dir / "metrics" / "coverage_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(audit["complete"])
            self.assertEqual(len(audit["zero_byte_predictions"]), 12)


if __name__ == "__main__":
    unittest.main()

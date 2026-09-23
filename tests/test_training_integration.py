"""合成 WAV で CLI の学習・固定・評価を接続する。MIMII 精度の検証ではない。"""

from contextlib import ExitStack, redirect_stdout
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from scripts import evaluate, runtime, train
from src import audio, dataset, model as model_api, service


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.original_root = runtime.ROOT
        self.config = json.loads((self.original_root / "configs" / "default.json").read_text(encoding="utf-8"))
        self.config_path = self.root / "configs" / "default.json"
        self.config_path.parent.mkdir(parents=True)
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.rows = []
        self.feature_by_hash = {}
        self.final_audio = {}
        self.fit_order = []
        self.loaded_sources = []
        self.final_allowed = False
        self.training_active = False
        self.candidate_metrics = {}
        self.original_load = audio.load_audio
        self.original_extract = audio.extract_features
        self.original_fit = model_api.fit_model
        self.original_evaluate = evaluate.evaluate_rows
        self.original_selected = runtime.selected_model

        # Each synthetic source is unique. final_test WAVs deliberately do not
        # exist until a test explicitly simulates evaluation after freezing.
        definitions = [("train", 0)] * 3 + [("calibration", 0)] * 2
        definitions += [("development", 0)] * 2 + [("development", 1)] * 2
        definitions += [("final_test", 0), ("final_test", 1)]
        rng = np.random.default_rng(42)
        t = np.arange(8 * 16000) / 16000
        for index, (split, label) in enumerate(definitions):
            frequency = 440 + index if label == 0 else 1100 + index
            selected = (.1 * np.sin(2 * np.pi * frequency * t) + .005 * rng.normal(size=len(t))).astype(np.float32)
            samples = np.column_stack([selected] + [np.zeros_like(selected)] * 7)
            buffer = io.BytesIO()
            sf.write(buffer, samples, 16000, format="WAV", subtype="PCM_16")
            data = buffer.getvalue()
            clip = self.original_load(data)
            path = Path("data", "raw", "pump", "id_00", "abnormal" if label else "normal", f"{split}_{index}.wav")
            self.rows.append({
                "record_id": clip.source_sha256[:24], "archive_name": dataset.ARCHIVE_NAME,
                "machine_type": "pump", "machine_id": "id_00", "snr_db": 6,
                "relative_path": path.as_posix(), "source_sha256": clip.source_sha256,
                "selected_channel_sha256": clip.selected_channel_sha256, "label": label,
                "sample_rate": 16000, "channels": 8, "duration_seconds": 8.0, "split": split,
            })
            if split == "final_test":
                self.final_audio[path] = data
            else:
                (self.root / path).parent.mkdir(parents=True, exist_ok=True)
                (self.root / path).write_bytes(data)
                self.feature_by_hash[clip.source_sha256] = audio.extract_features(clip, self.config["feature"])
        self.manifest = self.root / "data" / "manifest.csv"
        with self.manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=dataset.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        self.manifest_hash = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.output = self.root / "outputs" / "training" / "fixture"

        for module in (runtime, train, evaluate):
            self.stack.enter_context(patch.object(module, "ROOT", self.root))
        # selected_model's root default is bound when its module is imported.
        self.stack.enter_context(patch.object(evaluate, "selected_model", side_effect=lambda: self.original_selected(self.root)))
        self.stack.enter_context(patch.object(dataset, "read_manifest", side_effect=self.read_fixture_manifest))
        # This hand-built tiny split tests orchestration, not the production
        # split proportions (covered by test_dataset.py).
        self.stack.enter_context(patch.object(dataset, "validate_manifest", side_effect=self.validate_fixture_manifest))
        self.stack.enter_context(patch.object(audio, "load_audio", side_effect=self.load_fixture_audio))
        self.stack.enter_context(patch.object(audio, "extract_features", side_effect=self.fixture_features))
        self.stack.enter_context(patch.object(model_api, "fit_model", side_effect=self.track_fit))
        self.stack.enter_context(patch.object(evaluate, "evaluate_rows", side_effect=self.track_evaluation))

    def read_fixture_manifest(self, path):
        self.assertEqual(Path(path), self.manifest)
        return copy.deepcopy(self.rows)

    def validate_fixture_manifest(self, rows):
        self.assertEqual(rows, self.rows)
        self.assertEqual(len({row["source_sha256"] for row in rows}), len(rows))
        self.assertEqual({row["split"] for row in rows}, set(dataset.SPLITS))

    def load_fixture_audio(self, source):
        path = Path(source)
        self.assertTrue(path.resolve().is_relative_to((self.root / "data" / "raw").resolve()))
        if path.name.startswith("final_test_"):
            self.assertTrue(self.final_allowed, "学習・選択中に final_test 音声を参照しました")
        self.loaded_sources.append(path)
        return self.original_load(path)

    def fixture_features(self, clip, config):
        self.assertEqual(config, self.config["feature"])
        return self.feature_by_hash[clip.source_sha256]

    def track_fit(self, train_features, calibration_features, config, manifest_hash, n_clusters=16):
        train_values, calibration_values = list(train_features), list(calibration_features)
        for values, split in ((train_values, "train"), (calibration_values, "calibration")):
            expected = [row for row in self.rows if row["split"] == split]
            self.assertEqual(len(values), len(expected))
            self.assertTrue(all(row["label"] == 0 for row in expected))
            for actual, row in zip(values, expected):
                np.testing.assert_array_equal(actual, self.feature_by_hash[row["source_sha256"]].vectors)
        self.assertEqual(manifest_hash, self.manifest_hash)
        self.fit_order.append(n_clusters)
        return self.original_fit(train_values, calibration_values, config, manifest_hash, n_clusters=n_clusters)

    def track_evaluation(self, model, rows, split, manifest_hash):
        if self.training_active:
            self.assertEqual(split, "development")
            self.assertEqual([row["record_id"] for row in rows],
                             [row["record_id"] for row in self.rows if row["split"] == "development"])
        report = self.original_evaluate(model, rows, split, manifest_hash)
        if self.training_active:
            # Control only the candidate-selection branch. WAV processing,
            # fitted artifacts and clip predictions above are still real.
            report["metrics"].update(self.candidate_metrics[model.metadata["n_clusters"]])
        return report

    def run_training(self, metrics):
        self.candidate_metrics = metrics
        self.training_active = True
        try:
            with redirect_stdout(io.StringIO()):
                train.worker(self.config_path, self.output)
        finally:
            self.training_active = False
        self.assertFalse((self.root / "outputs" / "final_test_access.jsonl").exists())
        self.assertFalse(any(path.name.startswith("final_test_") for path in self.loaded_sources))
        for path in self.final_audio:
            self.assertFalse((self.root / path).exists())
        return self.original_selected(self.root)

    def evaluate_cli(self, split):
        with patch("sys.argv", ["evaluate", "--split", split]), redirect_stdout(io.StringIO()):
            evaluate.main()
        reports = sorted((self.root / "outputs" / "evaluations").glob(f"{split}_*/report.json"))
        self.assertEqual(len(reports), 1)
        return json.loads(reports[0].read_text(encoding="utf-8"))

    def assert_all_rows_and_service_scores(self, report, model, split):
        expected = [row for row in self.rows if row["split"] == split]
        self.assertEqual(report["metrics"]["counts"]["total"], len(expected))
        self.assertEqual([p["record_id"] for p in report["predictions"]], [r["record_id"] for r in expected])
        batch = service.inspect_batch(model, [(self.root / row["relative_path"], "renamed.wav") for row in expected])
        self.assertEqual(batch["success"], len(expected))
        for prediction, result in zip(report["predictions"], batch["results"]):
            self.assertEqual(prediction["score"], result["score"])
            self.assertEqual(prediction["decision"], result["decision"])
        self.assertEqual(report["manifest_sha256"], self.manifest_hash)

    def test_first_candidate_meets_goals_freezes_without_final_access(self):
        model, pointer = self.run_training({
            1: {"roc_auc": 1.0, "recall": 1.0, "false_positive_rate": 0.0},
            16: {"roc_auc": .8, "recall": .7, "false_positive_rate": .05},
        })
        self.assertEqual(self.fit_order, [1, 16])
        self.assertEqual(model.metadata["n_clusters"], 16)
        self.assertTrue(pointer["frozen"])
        self.assertEqual(set(pointer["artifact_sha256"]), {"arrays.npz", "metadata.json"})
        selection = json.loads((self.output / "selection.json").read_text(encoding="utf-8"))
        self.assertTrue(selection["development_goals_met"])
        self.assertFalse(selection["final_test_used"])
        self.assertEqual(len(list((self.root / "models").glob("*/metadata.json"))), 2)
        report = self.evaluate_cli("development")
        self.assert_all_rows_and_service_scores(report, model, "development")
        self.assertFalse((self.root / "outputs" / "final_test_access.jsonl").exists())

    def test_limited_fallback_and_frozen_final_evaluation_block_retraining(self):
        model, pointer = self.run_training({
            1: {"roc_auc": 1.0, "recall": 1.0, "false_positive_rate": 0.0},
            16: {"roc_auc": .6, "recall": .4, "false_positive_rate": .2},
            8: {"roc_auc": .85, "recall": .75, "false_positive_rate": .05},
            32: {"roc_auc": .85, "recall": .75, "false_positive_rate": .05},
        })
        self.assertEqual(self.fit_order, [1, 16, 8, 32])
        self.assertEqual(model.metadata["n_clusters"], 8)
        self.assertEqual(len(list((self.root / "models").glob("*/metadata.json"))), 4)
        selected_before = (self.root / "models" / "selected.json").read_bytes()
        for path, raw in self.final_audio.items():
            (self.root / path).parent.mkdir(parents=True, exist_ok=True)
            (self.root / path).write_bytes(raw)
        self.final_allowed = True
        # Reuse an independent extraction call saved at setup, after final access
        # becomes permitted; the learner can never see these feature vectors.
        for row in self.rows:
            if row["split"] == "final_test":
                clip = self.original_load(self.root / row["relative_path"])
                self.feature_by_hash[clip.source_sha256] = self.original_extract(clip, self.config["feature"])
        report = self.evaluate_cli("final_test")
        self.assert_all_rows_and_service_scores(report, model, "final_test")
        self.assertEqual(report["model_id"], pointer["model_id"])
        self.assertEqual((self.root / "models" / "selected.json").read_bytes(), selected_before)
        audit = (self.root / "outputs" / "final_test_access.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(audit), 1)
        self.assertEqual(json.loads(audit[0])["model_id"], model.model_id)
        fit_count = len(self.fit_order)
        with self.assertRaisesRegex(ValueError, "final_test"):
            train.worker(self.config_path, self.root / "outputs" / "training" / "forbidden")
        self.assertEqual(len(self.fit_order), fit_count)

    def test_abnormal_calibration_is_rejected_before_fitting(self):
        next(row for row in self.rows if row["split"] == "calibration")["label"] = 1
        # Exercise the worker's own guard even if a caller provides an already
        # parsed but semantically invalid manifest.
        with patch.object(dataset, "validate_manifest", return_value=None):
            with self.assertRaisesRegex(ValueError, "正常"):
                train.worker(self.config_path, self.output)
        self.assertEqual(self.fit_order, [])
        self.assertEqual(self.loaded_sources, [])


if __name__ == "__main__":
    unittest.main()

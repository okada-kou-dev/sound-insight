"""公開素材の境界・改変検出・セッション分離。合成音のみを使用。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from streamlit.testing.v1 import AppTest

from scripts import export_demo, runtime, verify_public_demo
from src import audio, dataset, evaluation, model, service
from src.public_bundle import asset_path, verify_bundle


class PublicDemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        t = np.arange(8 * 16000) / 16000
        for label, count in (("normal", 10), ("abnormal", 4)):
            directory = self.root / "data" / "raw" / "pump" / "id_00" / label
            directory.mkdir(parents=True)
            for index in range(count):
                frequency = 220 + index * 7 + (300 if label == "abnormal" else 0)
                signal = .1 * np.sin(2 * np.pi * frequency * t)
                sf.write(directory / f"{index}.wav", np.column_stack([signal] + [signal * .3] * 7), 16000, subtype="PCM_16")
        dataset.build_manifest(self.root)
        self.rows = dataset.read_manifest(self.root / "data" / "manifest.csv")
        self.manifest_hash = runtime.digest(self.root / "data" / "manifest.csv")
        rng = np.random.default_rng(42)
        trained = model.fit_model([rng.normal(size=(20, 320))], [rng.normal(size=(10, 320))], {}, self.manifest_hash, n_clusters=1)
        directory = model.save_model(trained, self.root / "models")
        runtime.write_json(self.root / "outputs" / "selection.json", {
            "selected": {"model_id": trained.model_id}, "manifest_sha256": self.manifest_hash})
        runtime.write_json(self.root / "models" / "selected.json", {
            "model_id": trained.model_id, "manifest_sha256": self.manifest_hash, "frozen": True,
            "artifact_sha256": {name: runtime.digest(directory / name) for name in ("arrays.npz", "metadata.json")},
            "selection_report": "outputs/selection.json", "selection_sha256": runtime.digest(self.root / "outputs" / "selection.json")})
        self.model = trained
        self.reports = {}
        for split in ("development", "final_test"):
            predictions = []
            for row in self.rows:
                if row["split"] == split:
                    result = service.inspect_one(trained, self.root / row["relative_path"], "synthetic.wav")
                    self.assertEqual(result["status"], "success")
                    predictions.append({k: row[k] for k in ("record_id", "relative_path", "label")} |
                        {k: result[k] for k in ("score", "decision")})
            report = {"split": split, "model_id": trained.model_id, "manifest_sha256": self.manifest_hash,
                "threshold": trained.threshold, "predictions": predictions,
                "metrics": evaluation.compute_metrics([p["label"] for p in predictions], [p["score"] for p in predictions], trained.threshold)}
            path = self.root / "outputs" / "evaluations" / split / "report.json"
            runtime.write_json(path, report)
            self.reports[split] = path
        (self.root / "DATA_LICENSE.md").write_text("Synthetic test fixture, not MIMII.", encoding="utf-8")

    def test_export_preserves_pcm_and_fixed_model_and_excludes_other_audio(self):
        out = export_demo.export_demo(self.root)
        bundle = verify_bundle(out)
        deployed, pointer = runtime.selected_model(out)
        self.assertEqual(deployed.model_id, self.model.model_id)
        self.assertEqual(pointer["selection_report"], "outputs/selection.json")
        expected = {r["record_id"] for r in self.rows if r["split"] == "development"}
        self.assertEqual({s["record_id"] for s in bundle["samples"]}, expected)
        for sample in bundle["samples"]:
            clip = audio.load_audio(out / sample["path"])
            self.assertEqual(clip.channels, 1)
            self.assertEqual(clip.selected_channel_sha256, sample["selected_channel_sha256"])
        self.assertEqual(len(list(out.rglob("*.wav"))), len(expected))
        self.assertFalse(list(out.rglob("*.sqlite3")))
        self.assertFalse(list(out.rglob("*.zip")))

    def test_missing_or_partial_final_report_blocks_export(self):
        final = json.loads(self.reports["final_test"].read_text(encoding="utf-8"))
        self.reports["final_test"].write_text(json.dumps(dict(final, split="not_final")), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "全final_test"):
            export_demo.export_demo(self.root)
        damaged = copy.deepcopy(final)
        damaged["predictions"].pop()
        self.reports["final_test"].write_text(json.dumps(damaged), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "全件"):
            export_demo.export_demo(self.root)
        self.assertFalse((self.root / "outputs" / "public_demo").exists())

    def test_corrupt_assets_extra_files_and_path_escape_rejected(self):
        out = export_demo.export_demo(self.root)
        bundle = verify_bundle(out)
        with self.assertRaises(ValueError):
            asset_path(out, "../outside.wav")
        with self.assertRaises(ValueError):
            asset_path(out, "C:/outside.wav")
        path = out / bundle["samples"][0]["path"]
        with path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "変更"):
            verify_bundle(out)
        (out / "private.sqlite3").write_bytes(b"private")
        with self.assertRaisesRegex(ValueError, "一覧"):
            verify_bundle(out)

    def test_deployment_check_allows_rounding_but_rejects_score_drift(self):
        out = export_demo.export_demo(self.root)
        def rounded(*args):
            result = service.inspect_one(*args)
            return dict(result, score=result["score"] * (1 + 2e-7))
        with patch("scripts.verify_public_demo.inspect_one", side_effect=rounded):
            self.assertTrue(verify_public_demo.verify(out)["success"])
        def drifted(*args):
            result = service.inspect_one(*args)
            return dict(result, score=result["score"] * 1.001)
        with patch("scripts.verify_public_demo.inspect_one", side_effect=drifted):
            report = verify_public_demo.verify(out)
            self.assertFalse(report["success"])
            self.assertTrue(all(not row["score_matches"] for row in report["samples"]))

    def test_deployment_check_rejects_changed_decision_even_with_same_score(self):
        out = export_demo.export_demo(self.root)
        def flipped(*args):
            result = service.inspect_one(*args)
            decision = "within_reference" if result["decision"] == "anomaly_candidate" else "anomaly_candidate"
            return dict(result, decision=decision)
        with patch("scripts.verify_public_demo.inspect_one", side_effect=flipped):
            with self.assertRaisesRegex(ValueError, "判定"):
                verify_public_demo.verify(out)

    def test_ui_only_samples_no_db_no_rerun_and_sessions_isolated(self):
        out = export_demo.export_demo(self.root)
        source = f"from pathlib import Path\nfrom src.public_ui import main\nmain(Path({str(out)!r}))\n"
        with patch("src.service.inspect_one", wraps=service.inspect_one) as inspect, \
                patch("src.storage.save_result", side_effect=AssertionError("Public UI must not persist")):
            first = AppTest.from_string(source, default_timeout=40).run()
            self.assertEqual(len(first.exception), 0)
            self.assertEqual(len(first.get("file_uploader")), 0)
            inspect.assert_not_called()
            self.assertEqual([tab.label for tab in first.tabs], [])
            self.assertEqual(first.radio[0].value, "一括検査")
            first.radio[0].set_value("1件だけ検査").run()
            next(b for b in first.button if b.label == "単発検査を実行").click().run()
            self.assertEqual(len(first.exception), 0)
            self.assertEqual(inspect.call_count, 1)
            self.assertEqual(first.session_state["public_revealed"], [])
            self.assertFalse(any(metric.label == "ROC-AUC" for metric in first.metric))
            with patch("src.playback.show_player", side_effect=lambda tracks, **kw: {"completed": [tracks[0]["id"]]}):
                first.run()
            self.assertEqual(len(first.session_state["public_revealed"]), 1)
            self.assertFalse(any(metric.label == "ROC-AUC" for metric in first.metric))
            self.assertNotIn("選択モデルの条件", [e.label for e in first.expander])
            next(s for s in first.selectbox if s.label == "音声データを選択（正常・異常は公開元のラベル）").select_index(2).run()
            self.assertNotIn("public_single", first.session_state)
            self.assertEqual(len(first.get("bidi_component")), 0)
            self.assertFalse(any(metric.label == "ROC-AUC" for metric in first.metric))
            self.assertEqual(inspect.call_count, 1)
            first.run()
            self.assertEqual(inspect.call_count, 1)
            self.assertEqual(len(first.session_state["public_history"]), 1)
            second = AppTest.from_string(source, default_timeout=40).run()
            self.assertEqual(len(second.exception), 0)
            self.assertNotIn("public_history", second.session_state)
            self.assertFalse(any(e.label == "検査履歴・CSV" for e in second.expander))
            first.radio[0].set_value("一括検査").run()
            next(b for b in first.button if b.label == "全サンプルを一括検査").click().run()
            self.assertEqual(len(first.exception), 0)
            self.assertEqual(len(first.session_state["public_history"]), 4)
            self.assertEqual(inspect.call_count, 4)
            self.assertEqual(len(first.session_state["public_tracks"]), 3)
            self.assertFalse(any(k.startswith("_") for row in first.session_state["public_history"] for k in row))
            self.assertTrue(all("comparison" in r for r in first.session_state["public_batch"]["results"]))
            first.run()
            self.assertEqual(inspect.call_count, 4)
            # A failed visualization must not hide a valid inspection forever.
            with patch("src.playback.make_track", side_effect=ValueError("viewer unavailable")):
                next(b for b in first.button if b.label == "全サンプルを一括検査").click().run()
            self.assertEqual(len(first.exception), 0)
            self.assertEqual(len(first.session_state["public_revealed"]), 3)
            self.assertEqual(inspect.call_count, 7)
        self.assertFalse(list(out.rglob("*.sqlite3")))


if __name__ == "__main__":
    unittest.main()

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from streamlit.testing.v1 import AppTest

from src import audio, model, service, storage
from scripts import runtime


class UITests(unittest.TestCase):
    def make_app(self, root):
        source = f"from pathlib import Path\nfrom src.ui import main\nmain(Path({str(root)!r}))\n"
        return AppTest.from_string(source, default_timeout=40)

    def test_missing_model_is_helpful_and_reruns_have_no_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.make_app(Path(directory))
            with patch("src.service.inspect_one") as inspect:
                app.run()
                app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual([tab.label for tab in app.tabs], [])
            self.assertIn("このアプリについて", [item.label for item in app.expander])
            self.assertTrue(any("選択モデルがありません" in item.value for item in app.warning))
            inspect.assert_not_called()
            self.assertFalse((Path(directory) / "outputs").exists())

    def test_model_cache_reuses_verified_model_and_rejects_changed_evidence(self):
        from src.ui import cached_model, load_selected_model
        cached_model.clear()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(42)
            trained = model.fit_model([rng.normal(size=(20, 320))], [rng.normal(size=(10, 320))], {}, "synthetic_manifest", n_clusters=1)
            model_dir = model.save_model(trained, root / "models")
            evidence = root / "outputs" / "selection.json"
            runtime.write_json(evidence, {"selected": {"model_id": trained.model_id}, "manifest_sha256": "synthetic_manifest"})
            runtime.write_json(root / "models" / "selected.json", {
                "model_id": trained.model_id, "frozen": True, "manifest_sha256": "synthetic_manifest",
                "artifact_sha256": {name: runtime.digest(model_dir / name) for name in ("arrays.npz", "metadata.json")},
                "selection_report": "outputs/selection.json", "selection_sha256": runtime.digest(evidence),
            })
            with patch("scripts.runtime.selected_model", wraps=runtime.selected_model) as load:
                first, _ = load_selected_model(root)
                second, _ = load_selected_model(root)
                self.assertIs(first, second)
                self.assertEqual(load.call_count, 1)
                evidence.write_text(evidence.read_text(encoding="utf-8") + " ", encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_selected_model(root)
                self.assertEqual(load.call_count, 2)
        cached_model.clear()

    def test_explicit_inspection_then_rerun_does_not_infer_or_save_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            source = root / "data" / "example.wav"
            t = np.arange(16000 * 8) / 16000
            sf.write(source, 0.1 * np.sin(2 * np.pi * 440 * t), 16000, subtype="PCM_16")
            with (root / "data" / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["relative_path", "label", "split"])
                writer.writeheader()
                writer.writerow({"relative_path": "data/example.wav", "label": "0", "split": "development"})
            tiny = model.AnomalyModel("ui_synthetic", np.zeros(320), np.ones(320), np.zeros((1, 320)),
                                      10000.0, audio.DEFAULT_FEATURE_CONFIG.copy(), {"synthetic_test": True})
            with patch("src.ui.load_selected_model", return_value=(tiny, {"frozen": False})), \
                    patch("src.service.inspect_one", wraps=service.inspect_one) as inspect:
                app = self.make_app(root).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.get("file_uploader")), 0)
                inspect.assert_not_called()
                self.assertEqual(app.radio[0].value, "一括検査")
                app.radio[0].set_value("1件だけ検査").run()
                next(button for button in app.button if button.label == "単発検査を実行").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(inspect.call_count, 1)
                self.assertEqual(app.session_state["single_result"]["reference_label"], 0)
                self.assertEqual(app.session_state["local_revealed"], [])
                self.assertFalse(any("公開元ラベル: 正常" in item.value for item in app.success))
                with patch("src.playback.show_player", side_effect=lambda tracks, **kw: {"completed": [tracks[0]["id"]]}):
                    app.run()
                self.assertTrue(any("公開元ラベル: 正常" in item.value for item in app.success))
                app.run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(inspect.call_count, 1)
                self.assertEqual(len(storage.read_history(root / "outputs" / "history.sqlite3")), 1)
                app.radio[0].set_value("一括検査").run()
                self.assertEqual(len(app.number_input), 0)
                next(button for button in app.button if button.label == "全サンプルを一括検査").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(inspect.call_count, 2)
                app.run()
                self.assertEqual(inspect.call_count, 2)
                self.assertEqual(len(storage.read_history(root / "outputs" / "history.sqlite3")), 2)
                self.assertEqual(len(app.session_state["local_tracks"]), 1)
                self.assertFalse(any(k.startswith("_") for k in app.session_state["batch_result"]["results"][0]))

    def test_reference_comparison_covers_misses_false_alarms_and_input_errors(self):
        from src.ui import with_reference
        for label, decision, expected in ((1, "anomaly_candidate", "検出成功"),
                (1, "within_reference", "見逃し"), (0, "anomaly_candidate", "誤警報"),
                (0, "within_reference", "正常を正しく判定")):
            original = {"status": "success", "decision": decision, "score": 0.123}
            result = with_reference(original, label)
            self.assertEqual(result["comparison"], expected)
            self.assertEqual(result["score"], original["score"])
            self.assertNotIn("reference_label", original)
        result = with_reference({"status": "invalid_input", "decision": None}, 1)
        self.assertEqual(result["comparison"], "判定不能")
        with self.assertRaises(ValueError):
            with_reference({"status": "success", "decision": "within_reference"}, 2)

    def test_reveal_ignores_unknown_ids_and_previous_runs(self):
        from src.ui import revealed_results
        rows = [{"run_id": "current", "item_index": 0}, {"run_id": "current", "item_index": 1}]
        tracks = [{"id": "a", "run_id": "current", "item_index": 0},
                  {"id": "b", "run_id": "current", "item_index": 1}]
        self.assertEqual(revealed_results(rows, tracks, []), [])
        self.assertEqual(revealed_results(rows, tracks, ["b", "b", "old", {}]), [rows[1]])
        self.assertEqual(revealed_results(rows, tracks, None), [])

    def test_about_links_model_conditions_on_github(self):
        source = """from src.ui import app_footer
app_footer()
"""
        app = AppTest.from_string(source).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 0)
        self.assertNotIn("選択モデルの条件", [item.label for item in app.expander])
        self.assertTrue(any("docs/accuracy_results.md" in item.value for item in app.markdown))


if __name__ == "__main__":
    unittest.main()

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src import audio, model, service, storage


def synthetic_wav():
    t = np.arange(16000 * 8) / 16000
    output = io.BytesIO()
    sf.write(output, 0.1 * np.sin(2 * np.pi * 440 * t), 16000, format="WAV", subtype="PCM_16")
    return output.getvalue()


def tiny_model():
    return model.AnomalyModel("synthetic_only", np.zeros(320), np.ones(320),
                              np.zeros((1, 320)), 10000.0,
                              audio.DEFAULT_FEATURE_CONFIG.copy(), {"synthetic_test": True})


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wav = synthetic_wav()
        cls.model = tiny_model()

    def test_single_batch_and_renamed_audio_agree(self):
        one = service.inspect_one(self.model, self.wav, "normal.wav")
        progress = []
        batch = service.inspect_batch(self.model, [(self.wav, "abnormal.wav"), (b"bad", "invalid.wav"), (self.wav, "renamed.wav")],
                                      progress=lambda done, total, result: progress.append((done, total)))
        self.assertEqual(one["status"], "success")
        self.assertEqual(batch["results"][0]["score"], one["score"])
        self.assertEqual(batch["results"][2]["decision"], one["decision"])
        self.assertEqual(batch["success"], 2)
        self.assertEqual(batch["failed"], 1)
        self.assertEqual(batch["results"][1]["status"], "invalid_input")
        self.assertIsNone(batch["results"][1]["decision"])
        self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])
        self.assertNotIn("_clip", batch["results"][0])

    def test_database_failure_is_separate_from_model_decision(self):
        with patch("src.storage.save_result", side_effect=OSError("disk full")):
            result = service.inspect_and_save(self.model, self.wav, "example.wav", "unused.sqlite3")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["decision"], "within_reference")
        self.assertEqual(result["save_status"], "failed")
        self.assertIn("disk full", result["save_error"])

    def test_errors_never_become_equipment_decisions(self):
        with patch("src.model.score_features", side_effect=ValueError("incompatible model")):
            result = service.inspect_one(self.model, self.wav, "test.wav")
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["score"])
        self.assertIsNone(result["decision"])

    def test_batch_continues_after_input_and_database_failure(self):
        with patch("src.storage.save_result", side_effect=OSError("disk full")):
            result = service.inspect_batch(self.model, [(b"bad", "bad.wav"), (self.wav, "valid.wav")], "unused.sqlite3")
        self.assertEqual((result["total"], result["success"], result["failed"]), (2, 1, 1))
        self.assertEqual((result["saved"], result["save_failed"]), (0, 2))
        self.assertIsNone(result["results"][0]["decision"])
        self.assertEqual(result["results"][1]["decision"], "within_reference")

    def test_safe_upload_and_history_source_name(self):
        with tempfile.TemporaryDirectory() as directory:
            upload_dir = Path(directory) / "uploads"
            uploaded = service.save_upload(self.wav, upload_dir)
            self.assertEqual(uploaded.parent, upload_dir)
            self.assertEqual(uploaded.read_bytes(), self.wav)
            self.assertEqual(len(uploaded.stem), 32)
            result = service.inspect_and_save(self.model, uploaded, r"..\..\private\example.wav", Path(directory) / "history.sqlite3", run_id="same")
            self.assertEqual(result["source_name"], "example.wav")
            service.persist_result(result, Path(directory) / "history.sqlite3")
            self.assertEqual(result["save_status"], "already_saved")
            self.assertEqual(len(storage.read_history(Path(directory) / "history.sqlite3")), 1)

    def test_batch_bounds(self):
        with self.assertRaises(ValueError):
            service.inspect_batch(self.model, [])
        with self.assertRaises(ValueError):
            service.inspect_batch(self.model, [(self.wav, "test.wav")] * 101)

    def test_batch_viewer_receives_details_once_without_retaining_them_in_results(self):
        captured = []
        def capture(result):
            captured.append((result["run_id"], result["item_index"], result["status"],
                             "_clip" in result, "_features" in result))
        with patch("src.service.inspect_one", wraps=service.inspect_one) as inspect:
            batch = service.inspect_batch(self.model, [(self.wav, "a.wav"), (b"bad", "b.wav")], on_result=capture)
            self.assertEqual(inspect.call_count, 2)
        self.assertEqual(captured, [(batch["run_id"], 0, "success", True, True),
                                    (batch["run_id"], 1, "invalid_input", False, False)])
        self.assertFalse(any(k.startswith("_") for row in batch["results"] for k in row))

    def test_oversized_upload_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "uploads"
            with self.assertRaises(audio.InvalidAudio):
                service.save_upload(b"0" * (service.MAX_UPLOAD_BYTES + 1), path)
            self.assertFalse(path.exists())

    def test_viewer_failure_preserves_all_inspections_and_saved_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "history.sqlite3"
            def unavailable(result):
                raise ValueError("view unavailable")
            batch = service.inspect_batch(self.model, [(self.wav, "a.wav"), (self.wav, "b.wav")],
                                          db_path=db, on_result=unavailable)
            self.assertEqual((batch["total"], batch["success"], batch["saved"]), (2, 2, 2))
            self.assertTrue(all("view unavailable" in r["viewer_error"] for r in batch["results"]))
            self.assertTrue(all(r["decision"] == "within_reference" for r in batch["results"]))
            self.assertEqual(len(storage.read_history(db)), 2)


if __name__ == "__main__":
    unittest.main()

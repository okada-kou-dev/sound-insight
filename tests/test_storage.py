import csv
import io
from pathlib import Path
import tempfile
import unittest

from src import storage


def history_row(run_id="run1", item_index=0):
    return dict(run_id=run_id, item_index=item_index, inspected_at="2026-09-19T00:00:00+00:00",
                source_name="test.wav", source_sha256="abc", model_id="tiny",
                machine_type="pump", machine_id="id_00", sample_rate=16000,
                duration_seconds=8.0, channel_index=0, score=0.1, threshold=0.2,
                decision="within_reference", processing_ms=1.2, status="success",
                error_message=None)


class StorageTests(unittest.TestCase):
    def test_duplicate_execution_is_saved_once_but_new_run_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "nested" / "history.sqlite3"
            self.assertTrue(storage.save_result(db, history_row()))
            self.assertFalse(storage.save_result(db, history_row()))
            self.assertTrue(storage.save_result(db, history_row("run2")))
            rows = storage.read_history(db)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["run_id"], "run2")
            self.assertNotIn("label", rows[0])

    def test_database_failure_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(Exception):
                storage.save_result(Path(directory), history_row())

    def test_csv_preserves_unicode_and_escapes_untrusted_formula(self):
        row = history_row()
        row["source_name"] = "=危険.wav"
        exported = storage.history_csv([row])
        self.assertTrue(exported.startswith(b"\xef\xbb\xbf"))
        result = next(csv.DictReader(io.StringIO(exported.decode("utf-8-sig"))))
        self.assertEqual(result["source_name"], "'=危険.wav")
        self.assertEqual(result["score"], "0.1")

    def test_absent_database_is_empty_without_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "missing.sqlite3"
            self.assertEqual(storage.read_history(db), [])
            self.assertFalse(db.exists())

    def test_reference_is_explicit_csv_only_and_does_not_change_db_schema(self):
        row = dict(history_row(), reference_label=1, comparison="見逃し")
        reference = next(csv.DictReader(io.StringIO(storage.history_csv([row], include_reference=True).decode("utf-8-sig"))))
        self.assertEqual(reference["reference_label"], "1")
        self.assertEqual(reference["comparison"], "見逃し")
        default = next(csv.DictReader(io.StringIO(storage.history_csv([row]).decode("utf-8-sig"))))
        self.assertNotIn("reference_label", default)
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "history.sqlite3"
            storage.save_result(db, row)
            self.assertNotIn("reference_label", storage.read_history(db)[0])


if __name__ == "__main__":
    unittest.main()

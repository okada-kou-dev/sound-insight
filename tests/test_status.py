from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import status
from scripts.runtime import digest, write_json
from src.dataset import ARCHIVE_MD5, ARCHIVE_NAME


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = datetime(2026, 9, 19, 10, tzinfo=timezone.utc)
        self.log = self.root / "data" / "download.jsonl"
        self.log.parent.mkdir()
        self.pipeline = self.root / "outputs" / "pipeline" / "run" / "state.json"
        write_json(self.pipeline, {"state": "waiting_for_download", "download_pid": 123,
            "download_created": 50, "stages": []})
        write_json(self.pipeline.parent / "process.json", {"pid": 124, "created": 51})
        self.addCleanup(patch.stopall)
        self.alive = patch("scripts.status.process_alive", return_value=True).start()

    def event(self, name, **fields):
        with self.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": name, "at": self.now.isoformat(), **fields}) + "\n")

    def test_live_progress_and_distinct_pipeline_state(self):
        self.event("request", offset=0)
        self.event("progress", bytes=20, total_bytes=100)
        report = status.snapshot(self.root, self.now)
        self.assertEqual(report["download"]["percent"], 20)
        self.assertTrue(report["download"]["alive"])
        self.assertEqual(report["warnings"], [])
        self.assertIn("ダウンロード未完了", status.render(report))

    def test_dead_processes_do_not_look_like_ongoing_work(self):
        self.alive.return_value = False
        report = status.snapshot(self.root, self.now)
        self.assertEqual(len(report["warnings"]), 2)
        self.assertIn("後続プロセス: 終了しています", status.render(report))

    def test_stale_log_with_live_process_is_reported_as_uncertain(self):
        self.event("progress", bytes=20, total_bytes=100)
        report = status.snapshot(self.root, self.now + timedelta(minutes=11))
        self.assertTrue(report["download"]["alive"])
        self.assertFalse(report["download"]["verified"])
        self.assertIn("10分以上", report["warnings"][0])

    def test_completion_requires_verification_and_manifest_evidence(self):
        archive = self.root / "data" / ARCHIVE_NAME
        archive.write_bytes(b"synthetic status fixture")
        self.assertFalse(status.snapshot(self.root, self.now)["download"]["verified"])
        self.event("download_verified", md5=ARCHIVE_MD5, size_bytes=archive.stat().st_size)
        report = status.snapshot(self.root, self.now)
        self.assertTrue(report["download"]["verified"])
        self.assertFalse(report["download"]["prepared"])
        self.assertIn("対象抽出・manifest作成は未完了", status.render(report))
        manifest = self.root / "data" / "manifest.csv"
        manifest.write_bytes(b"synthetic manifest for status only")
        self.event("manifest_ready", manifest_sha256=digest(manifest))
        self.assertTrue(status.snapshot(self.root, self.now)["download"]["prepared"])
        manifest.write_bytes(b"changed")
        self.assertFalse(status.snapshot(self.root, self.now)["download"]["prepared"])

    def test_retry_progress_does_not_reuse_old_bytes_and_partial_line_is_ignored(self):
        self.event("progress", bytes=80, total_bytes=100)
        self.event("request", offset=0)
        with self.log.open("a", encoding="utf-8") as handle:
            handle.write('{"event": "download_verified",')
        report = status.snapshot(self.root, self.now)
        self.assertEqual(report["download"]["bytes"], 0)
        self.assertFalse(report["download"]["verified"])

    def test_failure_reason_and_exit_code_are_visible(self):
        self.pipeline.write_text(json.dumps({"state": "stopped", "error": "fixture error",
            "stages": [{"name": "train", "exit_code": 1}]}), encoding="utf-8")
        rendered = status.render(status.snapshot(self.root, self.now))
        self.assertIn("停止理由: fixture error", rendered)
        self.assertIn("train: 終了コード 1", rendered)

    def test_missing_state_after_launch_is_not_silently_treated_as_no_pipeline(self):
        self.pipeline.unlink()
        log = self.root / "outputs" / "logs" / "continue-after-download.log"
        log.parent.mkdir()
        log.write_text("outputs/pipeline/run\n", encoding="utf-8")
        report = status.snapshot(self.root, self.now)
        self.assertEqual(report["pipeline"], {})
        self.assertIn("状態ファイルがありません", report["warnings"][0])


if __name__ == "__main__":
    unittest.main()

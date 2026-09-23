import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import zipfile

import numpy as np
import soundfile as sf

from src.dataset import (
    ARCHIVE_NAME, ARCHIVE_URL, DatasetError, build_manifest,
    deduplicate_and_split, download_archive, extract_target,
    manifest_sha256, read_manifest, split_counts, validate_manifest, _HTTPSOnlyRedirect,
)


def row(index, label=0):
    source = hashlib.sha256(f"source-{index}".encode()).hexdigest()
    return {
        "record_id": source[:24], "archive_name": ARCHIVE_NAME,
        "machine_type": "pump", "machine_id": "id_00", "snr_db": 6,
        "relative_path": f"data/raw/pump/id_00/{'normal' if label == 0 else 'abnormal'}/{index:04}.wav",
        "source_sha256": source, "selected_channel_sha256": hashlib.sha256(f"pcm-{index}".encode()).hexdigest(),
        "label": label, "sample_rate": 16000, "channels": 8,
        "duration_seconds": 10.0,
    }


class SplitTests(unittest.TestCase):
    def test_rounding_and_seed_are_fixed(self):
        self.assertEqual(split_counts(11, (0.6, 0.2, 0.1, 0.1)), [7, 2, 1, 1])
        self.assertEqual(split_counts(3, (0.5, 0.5)), [2, 1])
        original = [row(i) for i in range(10)] + [row(i, 1) for i in range(10, 14)]
        result, duplicates = deduplicate_and_split(original)
        reversed_result, _ = deduplicate_and_split(list(reversed(original)))
        self.assertEqual(result, reversed_result)
        self.assertEqual(duplicates, [])
        self.assertEqual([sum(r["label"] == 0 and r["split"] == split for r in result) for split in ("train", "calibration", "development", "final_test")], [6, 2, 1, 1])
        self.assertTrue(all(r["label"] == 0 for r in result if r["split"] in ("train", "calibration")))
        validate_manifest(result)

    def test_source_or_pcm_duplicates_form_transitive_groups(self):
        first, second, third = row(1), row(2), row(3)
        second["source_sha256"] = first["source_sha256"]
        third["selected_channel_sha256"] = second["selected_channel_sha256"]
        result, excluded = deduplicate_and_split([third, second, first])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(excluded), 2)
        self.assertEqual(result[0]["relative_path"], first["relative_path"])

    def test_conflicting_duplicate_labels_stop_the_data_stage(self):
        first, second = row(1), row(2, 1)
        second["selected_channel_sha256"] = first["selected_channel_sha256"]
        with self.assertRaisesRegex(DatasetError, "ラベルが矛盾"):
            deduplicate_and_split([first, second])

    def test_manifest_rejects_leakage_and_unsafe_paths(self):
        for field, value in (("split", "train"), ("relative_path", "../../outside.wav"), ("sample_rate", 8000)):
            item = row(1, 1)
            item["split"] = "development"
            item[field] = value
            with self.subTest(field=field), self.assertRaises(DatasetError):
                validate_manifest([item])
        result, _ = deduplicate_and_split([row(1), row(2)])
        result[1]["selected_channel_sha256"] = result[0]["selected_channel_sha256"]
        with self.assertRaises(DatasetError):
            validate_manifest(result)

    def test_manifest_rejects_manual_split_reassignment(self):
        result, _ = deduplicate_and_split([row(i) for i in range(10)])
        result[0]["split"] = "final_test" if result[0]["split"] != "final_test" else "train"
        with self.assertRaisesRegex(DatasetError, "固定seed"):
            validate_manifest(result)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / ARCHIVE_NAME

    def create_zip(self, extra=None, prefix="pump/"):
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(f"{prefix}id_00/normal/000.wav", b"normal fixture")
            archive.writestr(f"{prefix}id_00/abnormal/001.wav", b"abnormal fixture")
            archive.writestr(f"{prefix}id_02/normal/000.wav", b"excluded fixture")
            archive.writestr("LICENSE.txt", b"license fixture")
            if extra is not None:
                if isinstance(extra, str) and "\\" in extra:
                    raw_name = extra
                    extra = zipfile.ZipInfo("raw-name")
                    extra.filename = raw_name
                archive.writestr(extra, b"unsafe")

    def test_extracts_only_target_and_preserves_notices(self):
        self.create_zip()
        summary = extract_target(self.archive, self.root / "raw")
        self.assertEqual(summary["selected_files"], 3)
        self.assertEqual(summary["notice_files"], ["archive_notices/LICENSE.txt"])
        self.assertFalse((self.root / "raw/pump/id_02").exists())
        target = self.root / "raw/pump/id_00/normal/000.wav"
        self.assertEqual(target.read_bytes(), b"normal fixture")
        self.assertEqual(extract_target(self.archive, self.root / "raw")["reused_files"], 3)

    def test_supports_archive_without_machine_prefix(self):
        self.create_zip(prefix="")
        self.assertEqual(extract_target(self.archive, self.root / "raw")["new_files"], 3)

    def test_rejects_path_traversal_and_symlink_before_writing(self):
        unsafe = ["../outside.wav", "/absolute.wav", "C:/absolute.wav", "pump/../outside.wav", "pump\\outside.wav"]
        symlink = zipfile.ZipInfo("link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        for extra in [*unsafe, symlink]:
            self.create_zip(extra)
            with self.subTest(path=str(extra)), self.assertRaises(DatasetError):
                extract_target(self.archive, self.root / "raw")
            self.assertFalse((self.root / "raw/pump").exists())

    def test_existing_raw_is_never_overwritten(self):
        self.create_zip()
        target = self.root / "raw/pump/id_00/normal/000.wav"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"existing user data")
        with self.assertRaisesRegex(DatasetError, "上書き"):
            extract_target(self.archive, self.root / "raw")
        self.assertEqual(target.read_bytes(), b"existing user data")

    def test_extra_existing_wav_is_rejected_without_deletion_or_extraction(self):
        self.create_zip()
        extra = self.root / "raw/pump/id_00/normal/foreign.WAV"
        extra.parent.mkdir(parents=True)
        extra.write_bytes(b"existing extra audio")
        with self.assertRaisesRegex(DatasetError, "指定ZIPにない既存WAV"):
            extract_target(self.archive, self.root / "raw")
        self.assertEqual(extra.read_bytes(), b"existing extra audio")
        self.assertFalse((extra.parent / "000.wav").exists())


class FakeResponse(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(body)), "ETag": '"v1"'}

    def geturl(self):
        return ARCHIVE_URL


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b"small mocked ZIP transfer; no network"
        self.md5 = hashlib.md5(self.payload).hexdigest()

    def test_stream_and_reuse_verified_file_without_network(self):
        with patch("src.dataset.ARCHIVE_MD5", self.md5), patch("src.dataset.MIN_INITIAL_FREE_BYTES", 0), patch("src.dataset._open_archive", return_value=FakeResponse(self.payload)) as opened:
            summary = download_archive(self.root)
            self.assertEqual(summary["sha256"], hashlib.sha256(self.payload).hexdigest())
            self.assertEqual((self.root / ARCHIVE_NAME).read_bytes(), self.payload)
            self.assertEqual(download_archive(self.root), summary)
            self.assertEqual(opened.call_count, 1)

    def test_resumes_only_matching_range_and_validator(self):
        partial = self.root / (ARCHIVE_NAME + ".part")
        partial.write_bytes(self.payload[:10])
        partial.with_suffix(".part.json").write_text(json.dumps({"url": ARCHIVE_URL, "validator": '"v1"', "total_size": len(self.payload)}), encoding="utf-8")
        response = FakeResponse(self.payload[10:], status=206, headers={"Content-Range": f"bytes 10-{len(self.payload) - 1}/{len(self.payload)}", "ETag": '"v1"'})
        with patch("src.dataset.ARCHIVE_MD5", self.md5), patch("src.dataset.MIN_INITIAL_FREE_BYTES", 0), patch("src.dataset._open_archive", return_value=response) as opened:
            download_archive(self.root)
        self.assertEqual(opened.call_args.args[0].get_header("Range"), "bytes=10-")
        self.assertEqual((self.root / ARCHIVE_NAME).read_bytes(), self.payload)

    def test_bad_checksum_keeps_partial_and_does_not_publish_archive(self):
        with patch("src.dataset.MIN_INITIAL_FREE_BYTES", 0), patch("src.dataset._open_archive", return_value=FakeResponse(self.payload)):
            with self.assertRaisesRegex(DatasetError, "MD5"):
                download_archive(self.root)
        self.assertFalse((self.root / ARCHIVE_NAME).exists())
        self.assertEqual((self.root / (ARCHIVE_NAME + ".part")).read_bytes(), self.payload)

    def test_initial_disk_requirement_precedes_network(self):
        with patch("src.dataset.MIN_INITIAL_FREE_BYTES", 10**30), patch("src.dataset._open_archive") as opened:
            with self.assertRaisesRegex(DatasetError, "20GiB"):
                download_archive(self.root)
        opened.assert_not_called()

    def test_range_rejection_keeps_old_partial_and_restarts_once(self):
        partial = self.root / (ARCHIVE_NAME + ".part")
        partial.write_bytes(self.payload[:10])
        partial.with_suffix(".part.json").write_text(json.dumps({"url": ARCHIVE_URL, "validator": '"v1"', "total_size": len(self.payload)}), encoding="utf-8")
        with patch("src.dataset.ARCHIVE_MD5", self.md5), patch("src.dataset.MIN_INITIAL_FREE_BYTES", 0), patch("src.dataset._open_archive", side_effect=[FakeResponse(self.payload), FakeResponse(self.payload)]) as opened:
            download_archive(self.root)
        self.assertEqual(opened.call_count, 2)
        preserved = list(self.root.glob("*.part.preserved-*"))
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0].read_bytes(), self.payload[:10])
        self.assertIn("Range再開を受理しなかった", (self.root / "download.jsonl").read_text(encoding="utf-8"))

    def test_changed_resume_validator_is_rejected(self):
        partial = self.root / (ARCHIVE_NAME + ".part")
        partial.write_bytes(self.payload[:10])
        partial.with_suffix(".part.json").write_text(json.dumps({"url": ARCHIVE_URL, "validator": '"v1"', "total_size": len(self.payload)}), encoding="utf-8")
        response = FakeResponse(self.payload[10:], status=206, headers={"Content-Range": f"bytes 10-{len(self.payload) - 1}/{len(self.payload)}", "ETag": '"v2"'})
        with patch("src.dataset.MIN_INITIAL_FREE_BYTES", 0), patch("src.dataset._open_archive", return_value=response):
            with self.assertRaisesRegex(DatasetError, "HTTP検証子"):
                download_archive(self.root)
        self.assertEqual(partial.read_bytes(), self.payload[:10])
        self.assertFalse((self.root / ARCHIVE_NAME).exists())

    def test_retries_are_capped_and_original_error_is_logged(self):
        with patch("src.dataset.MIN_INITIAL_FREE_BYTES", 0), patch("src.dataset._open_archive", side_effect=OSError("mock network failure")) as opened, patch("src.dataset.time.sleep"):
            with self.assertRaises(DatasetError):
                download_archive(self.root)
        self.assertEqual(opened.call_count, 3)
        self.assertIn("mock network failure", (self.root / "download.jsonl").read_text(encoding="utf-8"))

    def test_rejects_https_downgrade_before_following_redirect(self):
        with self.assertRaisesRegex(DatasetError, "HTTPS"):
            _HTTPSOnlyRedirect().redirect_request(None, None, 302, "", {}, "http://example.invalid/data")

    def test_504_uses_backoff_and_retry_after_before_success(self):
        failures = [HTTPError(ARCHIVE_URL, 504, "Gateway Timeout", {}, None),
                    HTTPError(ARCHIVE_URL, 503, "Unavailable", {"Retry-After": "60"}, None),
                    FakeResponse(self.payload)]
        with patch("src.dataset.MIN_INITIAL_FREE_BYTES", 0), patch("src.dataset.ARCHIVE_MD5", self.md5), \
                patch("src.dataset._open_archive", side_effect=failures), patch("src.dataset.time.sleep") as slept:
            download_archive(self.root)
        self.assertEqual([call.args[0] for call in slept.call_args_list], [15, 60])

    def test_auth_error_or_long_retry_after_stops_without_extra_requests(self):
        for code, headers in ((403, {}), (503, {"Retry-After": "120"})):
            error = HTTPError(ARCHIVE_URL, code, "mock failure", headers, None)
            with self.subTest(code=code), patch("src.dataset.MIN_INITIAL_FREE_BYTES", 0), \
                    patch("src.dataset._open_archive", side_effect=error) as opened, patch("src.dataset.time.sleep") as slept:
                with self.assertRaises(DatasetError):
                    download_archive(self.root)
                self.assertEqual(opened.call_count, 1)
                slept.assert_not_called()


class ManifestBuildTests(unittest.TestCase):
    def test_manifest_is_repeatable_and_refuses_to_change_frozen_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(14):
                label_dir = "normal" if index < 10 else "abnormal"
                path = root / "data/raw/pump/id_00" / label_dir / f"{index:04}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                # 非無音、元サンプリング周波数、8ch、8秒の合成検証音。
                samples = np.full((16000 * 8, 8), (index + 1) / 100, dtype=np.float32)
                sf.write(path, samples, 16000, subtype="PCM_16")
            summary = build_manifest(root)
            rows = read_manifest(root / "data/manifest.csv")
            self.assertEqual(len(rows), 14)
            self.assertEqual(summary["manifest_sha256"], manifest_sha256(root / "data/manifest.csv"))
            self.assertEqual(summary["counts"]["train"], {"normal": 6, "abnormal": 0})
            self.assertEqual(build_manifest(root), summary)
            path = root / "data/raw/pump/id_00/normal/0099.wav"
            sf.write(path, np.full((16000 * 8, 8), 0.5, dtype=np.float32), 16000, subtype="PCM_16")
            with self.assertRaisesRegex(DatasetError, "上書き"):
                build_manifest(root)
            self.assertEqual(len(read_manifest(root / "data/manifest.csv")), 14)


if __name__ == "__main__":
    unittest.main()

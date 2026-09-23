"""固定したMIMII原版の取得・安全な抽出・重複排除・分割。

音声/正解ラベルは評価用manifestだけが関連付け、推論には渡さない。
既存のZIP・原音・確定manifestは内容が一致する場合だけ再利用する。
"""

from __future__ import annotations

import csv
import hashlib
from http.client import HTTPException
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import stat
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile

ARCHIVE_NAME = "6_dB_pump.zip"
ARCHIVE_URL = "https://zenodo.org/records/3384388/files/6_dB_pump.zip?download=1"
ARCHIVE_MD5 = "a09ba6060c10fc09cd4c8770213b0b9f"
MIN_INITIAL_FREE_BYTES = 20 * 1024**3
CHUNK_BYTES = 1024 * 1024
SPLITS = ("train", "calibration", "development", "final_test")
FIELDS = (
    "record_id", "archive_name", "machine_type", "machine_id", "snr_db",
    "relative_path", "source_sha256", "selected_channel_sha256", "label",
    "sample_rate", "channels", "duration_seconds", "split",
)
SESSION_CAVEAT = (
    "確かな収録セッション情報を確認できないためファイル単位で分割。"
    "同一設備・同一条件内の評価であり、別セッション・新設備への独立性は未確認。"
)


class DatasetError(ValueError):
    """安全性・固定対象・データ整合性の条件を満たさない。"""


class _HTTPSOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise DatasetError("HTTPS以外へのリダイレクトを拒否しました。")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_archive(request: Request, *, timeout: int):
    return build_opener(_HTTPSOnlyRedirect()).open(request, timeout=timeout)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(path: str | Path) -> str:
    return _file_hash(Path(path))


def _log(path: Path, event: str, **details) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"at": _utc_now(), "event": event, **details}, ensure_ascii=False) + "\n")


def _write_json_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _preserve_partial(path: Path, log_path: Path, reason: str) -> None:
    """再開できない転送も削除しない。時刻名へ退避して理由を残す。"""
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for item in (path, path.with_suffix(path.suffix + ".json")):
        if item.exists():
            target = item.with_name(item.name + ".preserved-" + suffix)
            item.rename(target)
    _log(log_path, "partial_preserved", reason=reason)


def _retry_delay(exc: Exception, attempt: int) -> float:
    """過負荷への短周期連打を避け、長いRetry-Afterは次回実行へ委ねる。"""
    delay = float(15 * 2 ** (attempt - 1))
    if isinstance(exc, HTTPError):
        if exc.code not in (408, 429, 500, 502, 503, 504):
            raise DatasetError(f"再試行対象外のHTTP {exc.code}。取得ログを確認してください。") from exc
        value = exc.headers.get("Retry-After") if exc.headers else None
        if value:
            try:
                seconds = float(value) if value.isdigit() else (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
                delay = max(delay, seconds)
            except (TypeError, ValueError, OverflowError):
                pass
    if delay > 60:
        raise DatasetError(f"配布元から{delay:.0f}秒の待機指定があります。ログ・一時ファイルを保持して停止します。時間を空けて再開してください。") from exc
    return delay


def download_archive(data_dir: str | Path, *, attempts: int = 3) -> dict:
    """指定1ZIPのみ取得。Range/If-Rangeで同一転送と確認できる場合だけ再開。"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / ARCHIVE_NAME
    log_path = data_dir / "download.jsonl"
    if archive.exists():
        md5 = _file_hash(archive, "md5")
        if md5 != ARCHIVE_MD5:
            _log(log_path, "existing_checksum_error", md5=md5)
            raise DatasetError("既存ZIPのMD5が不一致です。上書きせず停止しました。")
        result = {"archive_name": ARCHIVE_NAME, "md5": md5, "sha256": _file_hash(archive), "size_bytes": archive.stat().st_size}
        _log(log_path, "verified_archive_reused", **result)
        return result
    if not 1 <= attempts <= 3:
        raise DatasetError("取得再試行上限は1～3回です。")
    free = shutil.disk_usage(data_dir).free
    _log(log_path, "preflight", free_bytes=free, required_free_bytes=MIN_INITIAL_FREE_BYTES, url=ARCHIVE_URL)
    if free < MIN_INITIAL_FREE_BYTES:
        raise DatasetError("取得開始時の空き容量が20GiB未満です。")
    partial = archive.with_suffix(".zip.part")
    transfer = partial.with_suffix(".part.json")
    if partial.exists() and not transfer.exists():
        _preserve_partial(partial, log_path, "再開確認用のETag/Last-Modified情報がない")
    elif transfer.exists() and not partial.exists():
        _preserve_partial(partial, log_path, "転送状態だけが存在し音声ZIPの一時ファイルがない")

    for attempt in range(1, attempts + 1):
        metadata = {}
        if transfer.exists():
            try:
                metadata = json.loads(transfer.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                _preserve_partial(partial, log_path, "再開メタデータが読めない")
        offset = partial.stat().st_size if partial.exists() else 0
        validator = metadata.get("validator")
        if offset and (metadata.get("url") != ARCHIVE_URL or not validator):
            _preserve_partial(partial, log_path, "同一URLとHTTP検証子を確認できない")
            offset, metadata, validator = 0, {}, None
        if offset and metadata.get("total_size") == offset:
            break  # 保存後に中断した場合も、取得せずハッシュを再確認する。
        headers = {"User-Agent": "SoundInsight/1.0 (MIMII fixed-archive downloader)", "Accept-Encoding": "identity"}
        if offset:
            headers.update({"Range": f"bytes={offset}-", "If-Range": validator})
        _log(log_path, "request", attempt=attempt, offset=offset)
        try:
            with _open_archive(Request(ARCHIVE_URL, headers=headers), timeout=60) as response:
                if urlparse(response.geturl()).scheme != "https":
                    raise DatasetError("HTTPS以外へのリダイレクトを拒否しました。")
                status_code = response.status
                etag = response.headers.get("ETag")
                last_modified = response.headers.get("Last-Modified")
                response_validator = etag if etag and not etag.startswith("W/") else last_modified
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise DatasetError("予期しないContent-Encodingの取得を拒否しました。")
                total_size = None
                if offset:
                    if status_code != 206:
                        _preserve_partial(partial, log_path, "サーバーがRange再開を受理しなかった")
                        if attempt == attempts:
                            raise DatasetError("Range再開不可。ログと一時ファイルを保持しました。")
                        continue
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match or int(match[1]) != offset or int(match[2]) + 1 != int(match[3]):
                        raise DatasetError("Range応答の範囲が一致しません。")
                    total_size = int(match[3])
                    if metadata.get("total_size") not in (None, total_size) or response_validator != validator:
                        raise DatasetError("再開前後のHTTP検証子または全容量が変化しました。")
                elif status_code != 200:
                    raise DatasetError(f"予期しない取得HTTPステータス: {status_code}")
                elif response.headers.get("Content-Length"):
                    total_size = int(response.headers["Content-Length"])
                if total_size is not None and total_size < offset:
                    raise DatasetError("取得容量の宣言が不正です。")
                metadata = {"url": ARCHIVE_URL, "validator": response_validator, "total_size": total_size}
                transfer.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
                count = offset
                next_progress = count + 64 * CHUNK_BYTES
                with partial.open("ab" if partial.exists() else "xb") as handle:
                    while chunk := response.read(CHUNK_BYTES):
                        handle.write(chunk)
                        count += len(chunk)
                        if total_size is not None and count > total_size:
                            raise DatasetError("宣言容量を超える取得データを拒否しました。")
                        if count >= next_progress:
                            _log(log_path, "progress", bytes=count, total_bytes=total_size)
                            print(f"download: {count / 1024**3:.2f} GiB" + (f" / {total_size / 1024**3:.2f} GiB" if total_size else ""), flush=True)
                            next_progress = count + 64 * CHUNK_BYTES
                    handle.flush()
                    os.fsync(handle.fileno())
                if total_size is not None and count != total_size:
                    raise OSError(f"転送途中終了: {count}/{total_size} bytes")
                break
        except (HTTPError, URLError, HTTPException, OSError) as exc:
            _log(log_path, "transfer_error", attempt=attempt, error=str(exc))
            if attempt == attempts:
                raise DatasetError("指定ZIPの取得に失敗しました。download.jsonlと.partを保持しています。") from exc
            delay = _retry_delay(exc, attempt)
            _log(log_path, "retry_wait", seconds=delay, next_attempt=attempt + 1)
            time.sleep(delay)
    if not partial.exists():
        raise DatasetError("取得済み一時ZIPがありません。")
    md5 = _file_hash(partial, "md5")
    sha256 = _file_hash(partial)
    result = {"archive_name": ARCHIVE_NAME, "md5": md5, "sha256": sha256, "size_bytes": partial.stat().st_size}
    _log(log_path, "checksum", **result, matches_expected=md5 == ARCHIVE_MD5)
    if md5 != ARCHIVE_MD5:
        raise DatasetError("取得ZIPのMD5が不一致です。.partを保持して停止しました。")
    if archive.exists():
        raise DatasetError("取得中に確定ZIPが作成されたため上書きを拒否しました。")
    partial.rename(archive)
    # 転送の再開根拠も取得記録として保持する。
    _log(log_path, "download_verified", **result)
    return result


def _safe_zip_parts(info: zipfile.ZipInfo) -> tuple[str, ...]:
    # ZipInfo.filename はWindowsで区切りやNULが正規化されるため元表現を調べる。
    name = info.orig_filename
    if not name or "\\" in name or "\x00" in name or name.startswith("/"):
        raise DatasetError("ZIP内の危険なパスを拒否しました。")
    parts = tuple(name.rstrip("/").split("/"))
    if any(part in ("", ".", "..") or any(char in part for char in '<>:"|?*') or part.endswith((" ", ".")) or getattr(os.path, "isreserved", lambda value: False)(part) for part in parts):
        raise DatasetError("ZIP内の絶対パス・親参照・不正パスを拒否しました。")
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise DatasetError("ZIP内のシンボリックリンク等を拒否しました。")
    if info.flag_bits & 1:
        raise DatasetError("暗号化ZIPは扱いません。")
    return parts


def _is_notice(name: str) -> bool:
    base = name.lower()
    return any(base.startswith(prefix) for prefix in ("license", "licence", "copying", "copyright", "notice", "readme", "disclaimer")) and Path(base).suffix in ("", ".txt", ".md", ".rst", ".html", ".htm", ".pdf")


def extract_target(archive_path: str | Path, raw_dir: str | Path) -> dict:
    """指定ZIPの対象音声と配布された権利文書だけを保全して抽出する。"""
    archive_path, raw_dir = Path(archive_path), Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    root = raw_dir.resolve()
    selected = []
    seen_members, seen_targets = set(), set()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            parts = _safe_zip_parts(info)
            key = "/".join(parts).casefold()
            if key in seen_members:
                raise DatasetError("ZIP内の重複パスを拒否しました。")
            seen_members.add(key)
            if info.is_dir():
                continue
            relative = None
            if len(parts) >= 3 and parts[-3] == "id_00" and parts[-2] in ("normal", "abnormal") and parts[-1].lower().endswith(".wav"):
                prefix = parts[:-3]
                if prefix in ((), ("pump",), ("6_dB_pump",), ("6_dB_pump", "pump")):
                    relative = Path("pump", "id_00", parts[-2], parts[-1])
            elif _is_notice(parts[-1]):
                relative = Path("archive_notices", *parts)
            if relative is None:
                continue
            destination = raw_dir / relative
            try:
                destination.resolve().relative_to(root)
            except ValueError as exc:
                raise DatasetError("抽出先がrawディレクトリの外部です。") from exc
            target_key = relative.as_posix().casefold()
            if target_key in seen_targets:
                raise DatasetError("異なるZIPパスが同じ保存先へ衝突します。")
            seen_targets.add(target_key)
            if info.file_size > (10 * 1024**2 if relative.parts[0] == "pump" else 32 * 1024**2):
                raise DatasetError("対象ファイルの展開容量が上限を超えます。")
            selected.append((info, destination, relative))
        labels = {relative.parts[2] for _, _, relative in selected if relative.parts[0] == "pump"}
        if labels != {"normal", "abnormal"}:
            raise DatasetError("ZIP内にpump/id_00のnormalとabnormalの両方がありません。")
        expected_audio = {relative.as_posix().casefold() for _, _, relative in selected if relative.parts[0] == "pump"}
        existing_audio = {
            path.relative_to(raw_dir).as_posix().casefold()
            for label in ("normal", "abnormal")
            for path in (raw_dir / "pump" / "id_00" / label).rglob("*")
            if path.suffix.lower() == ".wav"
        }
        extras = sorted(existing_audio - expected_audio)
        if extras:
            raise DatasetError("指定ZIPにない既存WAVを検出しました。削除・混入を避け停止します: " + ", ".join(extras))
        expanded_bytes = sum(info.file_size for info, _, _ in selected)
        needed_bytes = sum(info.file_size for info, dest, _ in selected if not dest.exists())
        if shutil.disk_usage(raw_dir).free < needed_bytes + 1024**3:
            raise DatasetError("対象の展開容量と1GiBの予備容量を確保できません。")
        extracted, reused = 0, 0
        for info, destination, relative in selected:
            if destination.exists():
                if not destination.is_file() or destination.is_symlink() or destination.stat().st_size != info.file_size:
                    raise DatasetError("既存原音がZIPと一致しません。上書きを拒否しました。")
                digest = hashlib.sha256()
                with archive.open(info) as source:
                    for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
                        digest.update(chunk)
                if digest.hexdigest() != _file_hash(destination):
                    raise DatasetError("既存原音のハッシュがZIPと一致しません。上書きを拒否しました。")
                reused += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            # 途中失敗時も部分原音を確定名にせず残す。
            temp = destination.with_name(destination.name + f".extracting-{time.time_ns()}")
            with archive.open(info) as source, temp.open("xb") as target:
                shutil.copyfileobj(source, target, length=CHUNK_BYTES)
            if temp.stat().st_size != info.file_size:
                raise DatasetError("展開したファイルの容量が一致しません。")
            if destination.exists():
                raise DatasetError("抽出中に保存先が作成されたため上書きを拒否しました。")
            temp.rename(destination)
            destination.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
            extracted += 1
        return {
            "selected_files": len(selected), "new_files": extracted, "reused_files": reused,
            "expanded_bytes": expanded_bytes, "remaining_free_bytes": shutil.disk_usage(raw_dir).free,
            "notice_files": [rel.as_posix() for _, _, rel in selected if rel.parts[0] == "archive_notices"],
            "audio_files": sorted(rel.as_posix() for _, _, rel in selected if rel.parts[0] == "pump"),
        }


def split_counts(total: int, proportions: tuple[float, ...]) -> list[int]:
    """最大剰余法。同じ剰余は指定したsplitの順序を優先する。"""
    if total < 0 or not proportions or any(p < 0 for p in proportions) or not math.isclose(sum(proportions), 1):
        raise DatasetError("分割数・比率が不正です。")
    desired = [total * p for p in proportions]
    counts = [math.floor(value) for value in desired]
    ranking = sorted(range(len(counts)), key=lambda i: (-(desired[i] - counts[i]), i))
    for index in ranking[:total - sum(counts)]:
        counts[index] += 1
    return counts


def deduplicate_and_split(rows: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    """元ファイルまたは先頭ch PCMが等しい連結グループを代表1件へまとめる。"""
    if seed != 42:
        raise DatasetError("仕様の固定seedは42です。")
    ordered = [dict(row) for row in sorted(rows, key=lambda row: row["relative_path"])]
    parent = list(range(len(ordered)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    indexes = {}
    for index, row in enumerate(ordered):
        for field in ("source_sha256", "selected_channel_sha256"):
            key = (field, row[field])
            if key in indexes:
                left, right = find(index), find(indexes[key])
                parent[max(left, right)] = min(left, right)
            else:
                indexes[key] = index
    groups = {}
    for index, row in enumerate(ordered):
        groups.setdefault(find(index), []).append(row)
    representatives, duplicates = [], []
    for group in groups.values():
        if len({int(row["label"]) for row in group}) != 1:
            names = [row["relative_path"] for row in group]
            raise DatasetError("同一原音または選択ch PCMで正解ラベルが矛盾しています: " + ", ".join(names))
        representative = group[0]
        representatives.append(representative)
        for duplicate in group[1:]:
            duplicates.append({
                "excluded_path": duplicate["relative_path"], "representative_path": representative["relative_path"],
                "source_sha256": duplicate["source_sha256"], "selected_channel_sha256": duplicate["selected_channel_sha256"],
                "label": int(duplicate["label"]), "reason": "same_source_or_selected_pcm_connected_group",
            })
    rng = random.Random(seed)
    result = []
    for label, names, proportions in ((0, SPLITS, (0.6, 0.2, 0.1, 0.1)), (1, SPLITS[2:], (0.5, 0.5))):
        group = sorted((row for row in representatives if int(row["label"]) == label), key=lambda row: row["relative_path"])
        rng.shuffle(group)
        start = 0
        for name, count in zip(names, split_counts(len(group), proportions)):
            for row in group[start:start + count]:
                row["split"] = name
                result.append(row)
            start += count
    return sorted(result, key=lambda row: row["relative_path"]), duplicates


def validate_manifest(rows: list[dict]) -> None:
    """固定対象・ラベル・split・重複・安全な相対パスの整合性を検証する。"""
    if not rows:
        raise DatasetError("manifestが空です。")
    seen = {field: set() for field in ("record_id", "relative_path", "source_sha256", "selected_channel_sha256")}
    for row in rows:
        if set(FIELDS) - set(row):
            raise DatasetError("manifestの必須列が不足しています。")
        if row["archive_name"] != ARCHIVE_NAME or row["machine_type"] != "pump" or row["machine_id"] != "id_00" or int(row["snr_db"]) != 6:
            raise DatasetError("manifestの対象が固定対象と異なります。")
        label, split = int(row["label"]), row["split"]
        if label not in (0, 1) or split not in SPLITS or (label == 1 and split in SPLITS[:2]):
            raise DatasetError("manifestのラベル・splitが不正です。")
        path = str(row["relative_path"])
        parts = PurePosixPath(path).parts
        if "\\" in path or ":" in path or path.startswith("/") or ".." in parts or len(parts) != 6 or parts[:4] != ("data", "raw", "pump", "id_00") or parts[4] != ("normal" if label == 0 else "abnormal") or not parts[5].lower().endswith(".wav"):
            raise DatasetError("manifestの相対パスとラベルが不正です。")
        if int(row["sample_rate"]) != 16000 or int(row["channels"]) != 8 or not 8 <= float(row["duration_seconds"]) <= 12:
            raise DatasetError("manifestの原版音声ヘッダが仕様外です。")
        for field, values in seen.items():
            value = str(row[field])
            if not value or value in values:
                raise DatasetError(f"manifestに重複または空の{field}があります。")
            if field.endswith("sha256") and not re.fullmatch(r"[0-9a-f]{64}", value):
                raise DatasetError("manifestのSHA-256表現が不正です。")
            values.add(value)
    expected, _ = deduplicate_and_split(rows)
    assigned = {row["relative_path"]: row["split"] for row in expected}
    if any(row["split"] != assigned[row["relative_path"]] for row in rows):
        raise DatasetError("manifestの割当が固定seed42・比率・端数処理と一致しません。")


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise DatasetError("manifestヘッダが仕様と一致しません。")
        rows = list(reader)
    try:
        for row in rows:
            for field in ("snr_db", "label", "sample_rate", "channels"):
                row[field] = int(row[field])
            row["duration_seconds"] = float(row["duration_seconds"])
        validate_manifest(rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetError(f"manifestを検証できません: {exc}") from exc
    return rows


def build_manifest(project_root: str | Path, *, archive_info: dict | None = None, extraction_info: dict | None = None) -> dict:
    from src.audio import load_audio

    project_root = Path(project_root).resolve()
    raw_dir = project_root / "data" / "raw" / "pump" / "id_00"
    expected_paths = None
    if extraction_info is not None:
        if not extraction_info.get("audio_files"):
            raise DatasetError("抽出した対象WAV一覧がありません。")
        expected_paths = {"data/raw/" + path for path in extraction_info["audio_files"]}
        actual_paths = {
            path.relative_to(project_root).as_posix()
            for label in ("normal", "abnormal")
            for path in (raw_dir / label).rglob("*") if path.suffix.lower() == ".wav"
        }
        if actual_paths != expected_paths:
            raise DatasetError("抽出したZIPのWAV一覧と原音が一致しません。manifest作成を停止します。")
    rows = []
    for label, directory in ((0, "normal"), (1, "abnormal")):
        paths = sorted((raw_dir / directory).glob("*.wav"))
        if not paths:
            raise DatasetError(f"対象の{directory} WAVがありません。")
        for index, path in enumerate(paths):
            if path.is_symlink():
                raise DatasetError("原音へのシンボリックリンクを拒否しました。")
            audio = load_audio(path)
            if audio.channels != 8:
                raise DatasetError("原版データは8チャンネルを要求します。")
            rows.append({
                "record_id": audio.source_sha256[:24], "archive_name": ARCHIVE_NAME,
                "machine_type": "pump", "machine_id": "id_00", "snr_db": 6,
                "relative_path": path.relative_to(project_root).as_posix(),
                "source_sha256": audio.source_sha256, "selected_channel_sha256": audio.selected_channel_sha256,
                "label": label, "sample_rate": audio.sample_rate, "channels": audio.channels,
                "duration_seconds": audio.duration_seconds,
            })
            if (index + 1) % 100 == 0:
                print(f"manifest: {directory} {index + 1}/{len(paths)}", flush=True)
    split_rows, duplicates = deduplicate_and_split(rows)
    validate_manifest(split_rows)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(split_rows)
    content = buffer.getvalue().encode("utf-8")
    data_dir = project_root / "data"
    manifest_path = data_dir / "manifest.csv"
    summary_path = data_dir / "manifest_summary.json"
    digest = hashlib.sha256(content).hexdigest()
    if manifest_path.exists() and manifest_path.read_bytes() != content:
        raise DatasetError("確定manifestと再生成結果が異なります。既存分割の上書きを拒否しました。")
    summary = {
        "manifest_sha256": digest, "seed": 42, "target": {"machine_type": "pump", "machine_id": "id_00", "snr_db": 6, "channel_index": 0},
        "relative_path_base": "project_root", "raw_files": len(rows), "retained_files": len(split_rows), "excluded_duplicates": len(duplicates),
        "counts": {split: {"normal": sum(row["split"] == split and row["label"] == 0 for row in split_rows), "abnormal": sum(row["split"] == split and row["label"] == 1 for row in split_rows)} for split in SPLITS},
        "duplicates": duplicates, "rounding": "largest_remainder; ties use train, calibration, development, final_test order",
        "shuffle": "Python random.Random(42); path-sorted normal followed by abnormal; representative is smallest path in connected duplicate group",
        "selected_pcm_hash": "SHA-256 of channel_index=0 samples as little-endian float32, C-order bytes; sample_rate fixed at 16000",
        "session_caveat": SESSION_CAVEAT, "archive": archive_info, "extraction": extraction_info, "created_at": _utc_now(),
    }
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("manifest_sha256") != digest:
            raise DatasetError("既存manifest_summaryのハッシュと再生成結果が異なります。")
        summary = existing
    if not manifest_path.exists():
        with manifest_path.open("xb") as handle:
            handle.write(content)
    if not summary_path.exists():
        _write_json_exclusive(summary_path, summary)
    return summary


def prepare_dataset(project_root: str | Path, *, download: bool = False) -> dict:
    project_root = Path(project_root).resolve()
    data_dir = project_root / "data"
    archive_path = data_dir / ARCHIVE_NAME
    if download:
        archive_info = download_archive(data_dir)
    else:
        if not archive_path.exists():
            raise DatasetError("指定ZIPがありません。公式配布と容量確認後に--downloadを指定してください。")
        md5 = _file_hash(archive_path, "md5")
        if md5 != ARCHIVE_MD5:
            raise DatasetError("指定ZIPのMD5が一致しません。")
        archive_info = {"archive_name": ARCHIVE_NAME, "md5": md5, "sha256": _file_hash(archive_path), "size_bytes": archive_path.stat().st_size}
    extraction_info = extract_target(archive_path, data_dir / "raw")
    summary = build_manifest(project_root, archive_info=archive_info, extraction_info=extraction_info)
    _log(data_dir / "download.jsonl", "manifest_ready", manifest_sha256=summary["manifest_sha256"], counts=summary["counts"])
    return summary

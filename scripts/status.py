"""取得・データ準備・後続処理を読み取り専用で確認する。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from scripts.runtime import ROOT, digest
from src.dataset import ARCHIVE_MD5, ARCHIVE_NAME


STAGES = {
    "waiting_for_download": "ダウンロード・データ準備の完了待ち",
    "train": "正常学習・校正・モデル選択",
    "development": "development評価",
    "final_test": "固定後の全final_test評価",
    "verify_demo": "単発・一括デモの検証",
    "export_demo": "公開素材の作成",
    "local_pipeline_complete_publication_pending": "学習・評価・公開素材作成まで完了（公開操作は未実施）",
    "stopped": "停止・要確認",
}


def process_alive(pid, created):
    """PID再利用も停止とみなす。アクセス不明はFalseにせずNone。"""
    import psutil
    if not pid or created is None:
        return None
    try:
        process = psutil.Process(int(pid))
        return abs(process.create_time() - float(created)) < .01 and process.is_running()
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, ValueError, TypeError):
        return None


def read_events(path):
    if not path.exists():
        return []
    # A line may still be being appended. Never interpret a partial line as success.
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                events.append(item)
        except ValueError:
            continue
    return events


def snapshot(root=ROOT, now=None):
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    result = {"checked_at": now.isoformat(), "download": {}, "pipeline": {}, "warnings": []}
    events = read_events(root / "data" / "download.jsonl")
    # Only display progress belonging to the latest request; retries can restart at zero.
    request_index = max((i for i, event in enumerate(events) if event.get("event") == "request"), default=0)
    progress = next((e for e in reversed(events[request_index:]) if e.get("event") == "progress"), {})
    request = events[request_index] if events and events[request_index].get("event") == "request" else {}
    verified = next((e for e in reversed(events) if e.get("event") in ("download_verified", "verified_archive_reused")), {})
    archive = root / "data" / ARCHIVE_NAME
    verified_ok = bool(verified and verified.get("md5") == ARCHIVE_MD5 and archive.exists()
                       and archive.stat().st_size == verified.get("size_bytes"))
    ready = next((e for e in reversed(events) if e.get("event") == "manifest_ready"), {})
    manifest = root / "data" / "manifest.csv"
    prepared = bool(verified_ok and ready and manifest.exists() and digest(manifest) == ready.get("manifest_sha256"))
    done_bytes = verified.get("size_bytes", 0) if verified_ok else progress.get("bytes", request.get("offset", 0))
    total = verified.get("size_bytes") if verified_ok else progress.get("total_bytes")
    result["download"] = {"verified": verified_ok, "prepared": prepared,
        "bytes": done_bytes, "total_bytes": total, "percent": 100 * done_bytes / total if total else None,
        "last_progress_at": progress.get("at"), "alive": None}
    paths = sorted((root / "outputs" / "pipeline").glob("*/state.json"))
    if not paths and (root / "outputs" / "logs" / "continue-after-download.log").exists():
        result["warnings"].append("後続処理の起動ログはありますが状態ファイルがありません。稼働と保存先を確認してください。")
    if paths:
        path = paths[-1]
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            state["path"] = path.relative_to(root).as_posix()
            state["alive"] = None
            owner = path.parent / "process.json"
            if owner.exists():
                info = json.loads(owner.read_text(encoding="utf-8"))
                state["alive"] = process_alive(info.get("pid"), info.get("created"))
            result["pipeline"] = state
            result["download"]["alive"] = process_alive(state.get("download_pid"), state.get("download_created"))
            if state["alive"] is False and state.get("state") not in ("stopped", "local_pipeline_complete_publication_pending"):
                result["warnings"].append("後続プロセスが終了しています。状態ファイルだけでは実行中と判断できません。ログを確認してください。")
        except (OSError, ValueError, TypeError):
            result["warnings"].append("後続の状態ファイルが更新中または不正です。再確認してください。")
    if not verified_ok and result["download"]["alive"] is False:
        result["warnings"].append("取得プロセスが終了し、検証済みZIPはありません。ログを確認してください。")
    if progress.get("at") and not verified_ok:
        try:
            age = (now - datetime.fromisoformat(progress["at"])).total_seconds()
            if age > 600:
                result["warnings"].append("10分以上、取得量の進捗ログが更新されていません。処理の停止・低速化を確認してください。")
        except (TypeError, ValueError):
            result["warnings"].append("取得ログの日時を解釈できません。最新時刻を確認してください。")
    if events and events[-1].get("event") in ("transfer_error", "existing_checksum_error"):
        result["warnings"].append("直近の取得ログにエラーがあります。再試行中か最終停止かを後続状態で確認してください。")
    return result


def render(report):
    download, pipeline = report["download"], report["pipeline"]
    lines = [f"確認時刻: {report['checked_at']}"]
    if download["prepared"]:
        lines.append("データ準備完了: ZIP検証・対象抽出・manifest作成の記録を確認しました。")
    elif download["verified"]:
        lines.append("ダウンロード完了: ZIPのMD5検証記録あり。対象抽出・manifest作成は未完了です。")
    else:
        amount = f"{download['bytes'] / 1e9:.2f} GB"
        if download["total_bytes"]:
            amount += f" / {download['total_bytes'] / 1e9:.2f} GB ({download['percent']:.1f}%)"
        lines.append(f"ダウンロード未完了: 最終ログで {amount}")
        activity = {True: "稼働を確認", False: "終了しています", None: "稼働状態は未確認"}[download["alive"]]
        lines.append(f"取得プロセス: {activity}")
        if download["last_progress_at"]:
            lines.append(f"最終進捗: {download['last_progress_at']}")
    if pipeline:
        lines.append(f"後続の記録: {STAGES.get(pipeline.get('state'), '不明')}")
        if pipeline.get("alive") is not None:
            lines.append(f"後続プロセス: {'稼働を確認' if pipeline['alive'] else '終了しています'}")
        for stage in pipeline.get("stages", []):
            lines.append(f"  {stage['name']}: 終了コード {stage['exit_code']}")
        if pipeline.get("error"):
            lines.append(f"停止理由: {pipeline['error']}")
        lines.append(f"状態ファイル: {pipeline['path']}")
    lines += [f"要確認: {message}" for message in report["warnings"]]
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="10秒ごとに確認し、表示が変化したときに出力（Ctrl+Cで終了）")
    parser.add_argument("--json", action="store_true", help="機械可読JSONで1回表示")
    args = parser.parse_args()
    if args.watch and args.json:
        parser.error("--watchと--jsonは同時指定できません")
    previous = None
    try:
        while True:
            report = snapshot()
            current = json.dumps({k: v for k, v in report.items() if k != "checked_at"}, ensure_ascii=False, sort_keys=True)
            if current != previous:
                print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render(report), flush=True)
                previous = current
            terminal = report["pipeline"].get("state") in ("stopped", "local_pipeline_complete_publication_pending")
            if not args.watch or terminal:
                return 0
            time.sleep(10)
    except KeyboardInterrupt:
        print("状態表示を終了しました。取得・学習プロセスは停止していません。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

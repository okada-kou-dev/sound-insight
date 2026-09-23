"""Japanese local-only Streamlit inspection interface."""
from __future__ import annotations

import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

from scripts import runtime

import numpy as np
import pandas as pd
import streamlit as st

from src import service, storage
from src.presentation import GITHUB_URL, app_header, html_table


DECISIONS = {"within_reference": "正常音", "anomaly_candidate": "異常音"}


@st.cache_resource(show_spinner="保存モデルを読み込んでいます…")
def cached_model(project_directory: str, artifact_revisions: tuple, pointer_name="selected.json") -> tuple[Any, dict[str, Any]]:
    return runtime.selected_model(Path(project_directory), pointer_name)


def load_selected_model(root: Path) -> tuple[Any | None, dict[str, Any]]:
    pointer = root / "models" / "temporal_selected.json"
    if not pointer.exists():
        pointer = root / "models" / "selected.json"
    if not pointer.exists():
        return None, {}
    selected = json.loads(pointer.read_text(encoding="utf-8"))
    model_id = selected["model_id"]
    if not isinstance(model_id, str) or Path(model_id).name != model_id or "/" in model_id or "\\" in model_id:
        raise ValueError("選択モデルIDの形式が不正です。")
    directory = root / "models" / model_id
    evidence = (root / selected["selection_report"]).resolve()
    if not evidence.is_relative_to((root / "outputs").resolve()):
        raise ValueError("モデル選択の証跡パスが不正です。")
    # Reuse the CLI's checksum/provenance verification whenever any artifact changes.
    revision_paths = [pointer, directory / "arrays.npz", directory / "metadata.json", evidence, root / "data" / "manifest.csv"]
    revisions = tuple((path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else None for path in revision_paths)
    return cached_model(str(root), revisions, pointer.name)


def development_examples(root: Path) -> list[dict[str, Any]]:
    manifest = root / "data" / "manifest.csv"
    if not manifest.exists():
        return []
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "development"]
    from src.demo_selection import common_rows, fixed_rows
    fixed = common_rows(root, rows) or fixed_rows(root, rows)
    if fixed is not None:
        return fixed
    selected = []
    rng = random.Random(42)
    for label in ("0", "1"):
        candidates = sorted((row for row in rows if row["label"] == label), key=lambda r: r["relative_path"])
        selected.extend(rng.sample(candidates, min(10, len(candidates))))
    return selected


def project_source(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("入力パスがプロジェクト外です。")
    return path


def with_reference(result: dict[str, Any], label: int | str) -> dict[str, Any]:
    """Attach publisher ground truth after inference; never pass it to the model."""
    label = int(label)
    if label not in (0, 1):
        raise ValueError("公開元ラベルは正常0または異常1です")
    comparison = "判定不能"
    if result.get("status") == "success":
        decision = result.get("decision")
        if decision not in DECISIONS:
            raise ValueError("照合対象のAI判定が不正です")
        comparison = {(0, "within_reference"): "正常を正しく判定", (1, "anomaly_candidate"): "検出成功",
                      (0, "anomaly_candidate"): "誤警報", (1, "within_reference"): "見逃し"}[label, decision]
    return dict(result, reference_label=label, comparison=comparison)


def comparison_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{"ファイル": r["source_name"], "公開元ラベル": "異常" if r.get("reference_label") == 1 else "正常" if r.get("reference_label") == 0 else "未照合",
        "AI判定": DECISIONS.get(r.get("decision"), "判定不能"), "照合結果": r.get("comparison", "未照合"),
        "スコア": r.get("score"), "しきい値": r.get("threshold"), "処理ms": r["processing_ms"]} for r in results])


def show_comparison_summary(results: list[dict[str, Any]]) -> None:
    counts = {name: sum(r.get("comparison") == name for r in results)
              for name in ("検出成功", "正常を正しく判定", "見逃し", "誤警報", "判定不能")}
    columns = st.columns(len(counts))
    for column, (name, count) in zip(columns, counts.items()):
        column.metric(name, f"{count}件")
    st.caption("公開元のラベルとAI判定を照合した結果です。")


def show_result(result: dict[str, Any]) -> None:
    """Clip-level decision only. The synchronized monitor is rendered once below."""
    st.caption(f"検査したファイル: {result['source_name']} / 処理時間: {result['processing_ms']:.1f} ms")
    if result.get("save_status") == "failed":
        st.error(f"履歴保存に失敗しました。検査結果とは別のエラーです。{result.get('save_error', '')}")
    if result["status"] != "success":
        st.error(f"検査できませんでした ({result['status']})：{result['error_message']}")
        return
    score, threshold, decision = st.columns(3)
    score.metric("音声全体のスコア", f"{result['score']:.6f}")
    threshold.metric("固定しきい値", f"{result['threshold']:.6f}")
    decision.metric("AI判定", DECISIONS[result["decision"]])
    if "reference_label" in result:
        label = "異常" if result["reference_label"] else "正常"
        message = f"公開元ラベル: {label} / 照合結果: {result['comparison']}"
        (st.warning if result["comparison"] in ("見逃し", "誤警報") else st.success)(message)


def inspection_panel(model: Any, choices: list[dict], *, namespace: str,
                     db_path: Path | None = None, remember_results=None) -> None:
    """Shared batch-first flow. Inference is explicit; playback stays in the browser."""
    from src.playback import make_track, show_player
    from src.nearest import NearestModel, band_contributions
    # Keep the component flag compatible; the adopted view is shared by both editions.
    local_insights = True
    single_key = "public_single" if namespace == "public" else "single_result"
    batch_key = "public_batch" if namespace == "public" else "batch_result"
    active_key, tracks_key = f"{namespace}_active", f"{namespace}_tracks"
    mode = st.radio("検査モード", ["一括検査", "1件だけ検査"], horizontal=True, key=f"{namespace}_mode")
    controls, details = st.columns([1, 2], vertical_alignment="bottom")
    chosen = choices[0]
    with details:
        if mode == "1件だけ検査":
            chosen = st.selectbox("音声データを選択（正常・異常は公開元のラベル）", choices,
                format_func=lambda s: f"{'異常' if int(s['label']) else '正常'} / {s['name']}", key=f"{namespace}_sample")
        else:
            normal = sum(int(s["label"]) == 0 for s in choices)
            st.markdown(f"**{len(choices)}件を連続検査・再生** · 正常 {normal} / 異常 {len(choices)-normal}")
    with controls:
        button_label = "全サンプルを一括検査" if mode == "一括検査" else "単発検査を実行"
        execute = st.button(button_label, type="primary", width="stretch")
    selection = (model.model_id, mode, str(chosen["path"]) if mode == "1件だけ検査" else tuple(str(s["path"]) for s in choices),
                 "shared_insights_v2")
    if st.session_state.get(f"{namespace}_selection") != selection:
        for key in (single_key, batch_key, active_key, tracks_key, f"{namespace}_revealed", f"{namespace}_visualization_errors"):
            st.session_state.pop(key, None)
        st.session_state[f"{namespace}_selection"] = selection
    if execute:
        st.session_state[f"{namespace}_revealed"] = []
        tracks, visualization_errors = [], []
        def capture(result):
            if result["status"] == "success":
                row = choices[result["item_index"]] if mode == "一括検査" else chosen
                try:
                    contributions = (band_contributions(model, result["_features"])
                                     if local_insights and isinstance(model, NearestModel) else None)
                    tracks.append(make_track(with_reference(result, row["label"]), contributions=contributions))
                except Exception as exc:
                    visualization_errors.append(f"{result['source_name']}: {type(exc).__name__}")
        try:
            with st.spinner("音声と解析を準備しています…", show_time=True):
                if mode == "一括検査":
                    progress = st.progress(0, text=f"再生準備 0 / {len(choices)}件")
                    def on_progress(done, total, result):
                        progress.progress(done / total, text=f"再生準備 {done} / {total}件")
                    batch = service.inspect_batch(model, [(s["path"], s["name"]) for s in choices],
                        db_path=db_path, progress=on_progress, on_result=capture)
                    progress.empty()
                    batch["results"] = [with_reference(r, choices[r["item_index"]]["label"]) for r in batch["results"]]
                    st.session_state[batch_key] = batch
                    results = batch["results"]
                    st.session_state[active_key] = "batch"
                else:
                    result = (service.inspect_and_save(model, chosen["path"], chosen["name"], db_path)
                              if db_path is not None else service.inspect_one(model, chosen["path"], chosen["name"]))
                    result = with_reference(result, chosen["label"])
                    capture(result)
                    st.session_state[single_key] = {k: v for k, v in result.items() if not k.startswith("_")}
                    results = [st.session_state[single_key]]
                    st.session_state[active_key] = "single"
                st.session_state[tracks_key] = tracks
                st.session_state[f"{namespace}_visualization_errors"] = visualization_errors
                if remember_results is not None:
                    remember_results(results)
        except Exception as exc:
            st.error(f"検査を完了できませんでした: {type(exc).__name__}: {exc}")
    active = st.session_state.get(active_key)
    if active is None:
        if mode == "1件だけ検査":
            st.info("音声を選び、単発検査を実行してください。")
        return
    result = st.session_state[batch_key if active == "batch" else single_key]
    tracks = st.session_state.get(tracks_key, [])
    rows = result["results"] if active == "batch" else [result]
    completed = []
    if tracks:
        player = show_player(tracks, key=f"{namespace}_{result['run_id']}", autoplay=True, local_insights=local_insights)
        completed = player.get("completed", [])
    # Only known track IDs may reveal result metadata. Playback never runs inference.
    visible = revealed_results(rows, tracks, completed)
    if st.session_state.get(f"{namespace}_visualization_errors"):
        st.warning("一部の音声で再生画面を作れませんでした。その音声の検査結果は下に表示します。")
        playable = {(t["run_id"], t["item_index"]) for t in tracks}
        visible += [r for r in rows if r["status"] == "success" and (r["run_id"], r["item_index"]) not in playable]
    st.session_state[f"{namespace}_revealed"] = visible
    for row in rows:
        if row["status"] != "success":
            show_result(row)
    if active == "single":
        if visible:
            show_result(visible[0])
            st.download_button("検査結果CSV", storage.history_csv(visible, include_reference=True), "inspection_result.csv", "text/csv")
        else:
            st.caption("音声の終わりまで進むと、AI判定と公開元の正解を照合します。")
    else:
        st.subheader(f"検査結果 · {len(visible)} / {len(rows)}件")
        show_comparison_summary(visible)
        if db_path is not None and result["save_failed"]:
            st.error(f"履歴の保存失敗 {result['save_failed']}件。AI判定とは別のエラーです。")
        if visible:
            html_table(comparison_frame(visible))
            st.download_button("表示済み結果CSV", storage.history_csv(visible, include_save_status=db_path is not None, include_reference=True), "batch_results.csv", "text/csv")
        else:
            st.caption("再生が終わった音声から、ここに結果が加わります。")


def revealed_results(rows: list[dict], tracks: list[dict], completed) -> list[dict]:
    if not isinstance(completed, list):
        return []
    ids = {value for value in completed if isinstance(value, str)}
    known = {(track["run_id"], track["item_index"]) for track in tracks if track["id"] in ids}
    return [row for row in rows if (row["run_id"], row["item_index"]) in known]


def inspection_tab(root: Path, model: Any | None) -> None:
    if model is None:
        st.warning("選択モデルがありません。CLIでデータ準備・学習・開発評価を完了してください。")
        return
    if not development_examples(root):
        st.info("公開元ラベル付きのdevelopment音声が未準備です。CLIで指定データを準備してください。")
        return
    rows = development_examples(root)
    choices = [dict(path=project_source(root, r["relative_path"]), name=f"サンプル {i:02d}", label=int(r["label"])) for i, r in enumerate(rows, 1)]
    inspection_panel(model, choices, namespace="local", db_path=root / "outputs" / "history.sqlite3")


def history_tab(root: Path) -> None:
    st.subheader("検査履歴")
    st.caption("処理日時はアプリで検査したUTC時刻です。実収録日時ではありません。最新1000件を表示・出力します。")
    try:
        rows = storage.read_history(root / "outputs" / "history.sqlite3")
        if rows:
            html_table(pd.DataFrame(rows))
            st.download_button("履歴CSVをダウンロード", storage.history_csv(rows), "inspection_history.csv", "text/csv")
        else:
            st.info("保存済みの検査履歴はありません。")
    except Exception as exc:
        st.error(f"履歴を読み込めませんでした: {exc}")


def credits_tab(online_demo: bool = False) -> None:
    st.write("公開ポンプ音50件（正常25・異常25）でAIの判定を試すアプリです。")
    if online_demo:
        st.caption("体験版：サーバー上で処理し、履歴はセッション内のみ。")
    st.markdown("**音源:** MIMII Dataset · public 1.0（2019）· Hitachi, Ltd.")
    st.html("<p><b>使用・加工:</b> pump / id_00 / +6 dB<br>"
            "先頭chを振幅を保って抽出し、音響特徴を可視化<br>"
            "音声・可視化の共有はCC BY-SA 4.0<br>"
            "提供元の承認・保証を意味しません</p>")
    st.caption("著作者：Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, Yohei Kawaguchi。")
    st.markdown("[MIMII論文（Purohit et al., DCASE 2019）・公式情報](https://github.com/MIMII-hitachi/mimii_baseline)")


def app_footer(*, online_demo: bool = False, attribution: Path | None = None) -> None:
    st.html('<div class="si-source">音源: MIMII Dataset · Hitachi, Ltd. · '
            '<a href="https://doi.org/10.5281/zenodo.3384388" target="_blank" rel="noopener noreferrer">public 1.0</a> · '
            '<a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noopener noreferrer">CC BY-SA 4.0</a></div>')
    with st.expander("このアプリについて"):
        st.markdown(f"[GitHub · ソースコード・評価・再現手順]({GITHUB_URL})")
        st.markdown(f"[評価方法・検出性能・失敗例]({GITHUB_URL}/blob/main/docs/accuracy_results.md)")
        credits_tab(online_demo=online_demo)
        st.caption("再生速度を変えると音程も変わりますが、AIの判定は変わりません。")
        if attribution is not None:
            st.download_button("サンプルの出典・加工・権利表示", attribution.read_bytes(), "ATTRIBUTION.md", "text/markdown")


def main(project_root: Path | None = None) -> None:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    st.set_page_config(page_title="SOUND INSIGHT", page_icon="🔊", layout="wide")
    app_header(online=False)
    try:
        model, _ = load_selected_model(root)
    except Exception as exc:
        model = None
        st.error(f"モデルを読み込めませんでした。保存モデルと選択設定を確認してください: {exc}")
    inspection_tab(root, model)
    app_footer()



if __name__ == "__main__":
    main()

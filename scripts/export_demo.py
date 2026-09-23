"""全最終評価後に、公開レビュー用の限定サンプル・固定モデルを新規出力する。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from scripts.runtime import ROOT, digest, new_id, selected_model, write_json
from scripts.verify_demo import demo_rows
from src.public_bundle import asset_path, verify_bundle


def checked_reports(root, rows, model, manifest_hash):
    from src.evaluation import compute_metrics
    reports = {}
    for path in sorted((root / "outputs" / "evaluations").glob("*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        split = report.get("split")
        if split not in ("development", "final_test") or report.get("model_id") != model.model_id or report.get("manifest_sha256") != manifest_hash:
            continue
        expected = {r["record_id"]: r for r in rows if r["split"] == split}
        predictions = report["predictions"]
        if not expected or len(predictions) != len(expected) or {p["record_id"] for p in predictions} != set(expected):
            raise ValueError("公開対象の評価レポートがsplit全件を含んでいません")
        for p in predictions:
            row = expected[p["record_id"]]
            if p["label"] != row["label"] or p["relative_path"] != row["relative_path"]:
                raise ValueError("評価レポートとmanifestが不一致です")
            decision = "anomaly_candidate" if p["score"] > model.threshold else "within_reference"
            if p["decision"] != decision:
                raise ValueError("評価判定と固定しきい値が不一致です")
        metrics = compute_metrics([p["label"] for p in predictions], [p["score"] for p in predictions], model.threshold)
        if report["threshold"] != model.threshold or report["metrics"] != metrics:
            raise ValueError("評価指標が固定モデルの全件スコアと一致しません")
        reports[split] = path
    if set(reports) != {"development", "final_test"}:
        raise ValueError("公開素材の作成にはdevelopmentと全final_testの評価が必要です")
    return reports


def export_demo(root: Path = ROOT) -> Path:
    import soundfile as sf
    from src.audio import load_audio
    from src.dataset import read_manifest, validate_manifest, manifest_sha256

    root = root.resolve()
    manifest = root / "data" / "manifest.csv"
    rows = read_manifest(manifest)
    validate_manifest(rows)
    manifest_hash = manifest_sha256(manifest)
    model, pointer = selected_model(root)
    if pointer["manifest_sha256"] != manifest_hash:
        raise ValueError("モデルとmanifestが不一致です")
    reports = checked_reports(root, rows, model, manifest_hash)
    from src.demo_selection import common_rows, fixed_rows
    samples = common_rows(root, rows) or fixed_rows(root, rows, public=True) or demo_rows(rows)
    if not samples:
        raise ValueError("公開サンプルがありません")
    bundle_id = new_id("demo")
    out = root / "outputs" / "public_demo" / bundle_id
    out.mkdir(parents=True, exist_ok=False)

    def copy(source, relative):
        destination = asset_path(out, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    for name in ("arrays.npz", "metadata.json"):
        copy(root / "models" / model.model_id / name, f"models/{model.model_id}/{name}")
    copy(root / pointer["selection_report"], "outputs/selection.json")
    pointer = dict(pointer, selection_report="outputs/selection.json")
    write_json(out / "models" / "selected.json", pointer)
    for split, path in reports.items():
        copy(path, f"outputs/evaluations/{split}/report.json")
    public_samples = []
    for index, row in enumerate(samples):
        source = asset_path(root, row["relative_path"])
        clip = load_audio(source)
        if clip.source_sha256 != row["source_sha256"] or clip.selected_channel_sha256 != row["selected_channel_sha256"]:
            raise ValueError("サンプル音声がmanifestと不一致です")
        relative = f"samples/sample_{index:02d}.wav"
        destination = out / relative
        destination.parent.mkdir(exist_ok=True)
        sf.write(destination, clip.samples, clip.sample_rate, format="WAV", subtype="PCM_16")
        if load_audio(destination).selected_channel_sha256 != clip.selected_channel_sha256:
            raise ValueError("モノラル抽出でPCMが変化しました")
        public_samples.append({"path": relative, "record_id": row["record_id"], "label": row["label"],
            "split": "development", "original_source_sha256": clip.source_sha256,
            "selected_channel_sha256": clip.selected_channel_sha256})
    notice_root = root / "data" / "raw" / "archive_notices"
    for path in sorted(notice_root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(notice_root).as_posix()
            copy(asset_path(notice_root, relative), f"notices/{relative}")
    copy(root / "DATA_LICENSE.md", "DATA_LICENSE.md")
    attribution = """# 公開サンプルの出典と加工

MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection
public 1.0 (2019), Hitachi, Ltd.
Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, Yohei Kawaguchi
出典: https://doi.org/10.5281/zenodo.3384388
音声ライセンス: CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0/

6_dB_pump.zip / pump / id_00 / +6 dB のdevelopmentから固定選択。構成・来歴はbundle.jsonとconfigs/の選定記録を参照。
先頭チャンネルを16kHz・16bit PCMのモノラルWAVとして抽出。サンプル列・振幅は保持。
再サンプリング・音量正規化・ノイズ除去は行っていません。ファイル名は公開用に変更。
検査時にログメル・窓特徴量を抽出し可視化します。
提供元による本アプリの承認・保証を意味しません。原版同梱表示がある場合はnotices/に保持。
音声の条件と独立コードのライセンスは区別します。モデルの来歴・評価は同梱JSONを参照。
"""
    (out / "ATTRIBUTION.md").write_text(attribution, encoding="utf-8")
    write_json(out / "bundle.json", {"version": 1, "bundle_id": bundle_id, "input_mode": "samples_only",
        "model_id": model.model_id, "manifest_sha256": manifest_hash, "samples": public_samples,
        "sha256": {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    verify_bundle(out)
    selected_model(out)
    return out


if __name__ == "__main__":
    print(export_demo().relative_to(ROOT).as_posix())

"""固定モデルの全クリップ評価。final_test は固定証跡を要求する。"""
from __future__ import annotations

from scripts.runtime import ROOT, Measurement, digest, environment, new_id, safe_source, selected_model, write_json
import argparse
import json
import time
from datetime import datetime, timezone


def evaluate_rows(model, rows, split, manifest_hash):
    from src.audio import load_audio, extract_features
    from src.model import score_features
    from src.evaluation import compute_metrics
    predictions = []
    feature_seconds = []
    with Measurement() as measured:
        for row in rows:
            start = time.perf_counter()
            clip = load_audio(safe_source(row))
            if clip.source_sha256 != row["source_sha256"] or clip.selected_channel_sha256 != row["selected_channel_sha256"]:
                raise ValueError("manifest作成後の音声変更を検出しました")
            feature_start = time.perf_counter()
            features = extract_features(clip, model.feature_config)
            feature_seconds.append(time.perf_counter() - feature_start)
            result = score_features(model, features)
            predictions.append({"record_id":row["record_id"],"relative_path":row["relative_path"],
                "label":int(row["label"]),"score":float(result["score"]),"decision":result["decision"],
                "processing_ms":(time.perf_counter()-start)*1000})
    metrics = compute_metrics([p["label"] for p in predictions], [p["score"] for p in predictions], model.threshold)
    return {"split":split,"model_id":model.model_id,"manifest_sha256":manifest_hash,
        "threshold":float(model.threshold),"metrics":metrics,"predictions":predictions,
        "errors":[p for p in predictions if bool(p["label"]) != (p["decision"]=="anomaly_candidate")],
        "measurement":measured.result("WAV読込・特徴量・全窓採点 (UIなし)",len(rows),"first clip cold; subsequent warm"),
        "feature_seconds":feature_seconds,"environment":environment(),
        "target":{"machine_type":"pump","machine_id":"id_00","snr_db":6,"channel_index":0},
        "limitations":"同一設備・同一条件のファイル分割。収録セッション独立性、新設備、実運用は未検証。pAUCはscikit-learnの標準化pAUC(max_fpr=0.1)。"}


def main():
    from src.dataset import read_manifest, validate_manifest, manifest_sha256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["development","final_test"], required=True)
    args = parser.parse_args()
    manifest = ROOT / "data" / "manifest.csv"
    rows = read_manifest(manifest)
    validate_manifest(rows)
    manifest_hash = manifest_sha256(manifest)
    with Measurement() as loaded:
        model, pointer = selected_model()
    if pointer["manifest_sha256"] != manifest_hash:
        raise ValueError("モデル固定時のmanifestと不一致です")
    if not pointer.get("frozen"):
        raise ValueError("モデル選択・設定固定を先に完了してください")
    evidence = ROOT / pointer["selection_report"]
    if not evidence.resolve().is_relative_to((ROOT / "outputs").resolve()) or digest(evidence) != pointer["selection_sha256"]:
        raise ValueError("モデル選択の固定証跡が一致しません")
    selected = [r for r in rows if r["split"] == args.split]
    if not selected:
        raise ValueError("評価クリップがありません")
    out = ROOT / "outputs" / "evaluations" / new_id(args.split)
    if args.split == "final_test":
        audit = ROOT / "outputs" / "final_test_access.jsonl"
        with audit.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at":datetime.now(timezone.utc).isoformat(),"model_id":model.model_id,
                "manifest_sha256":manifest_hash,"purpose":"frozen model full final evaluation"})+"\n")
    report = evaluate_rows(model, selected, args.split, manifest_hash)
    report["model_load"] = loaded.result("NPZ/metadata読込・固定hash検証",1,"cold")
    report["selection_report"] = pointer["selection_report"]
    write_json(out / "report.json", report)
    print(json.dumps({"report":str(out.relative_to(ROOT)),"count":len(selected),"metrics":report["metrics"]},ensure_ascii=False))


if __name__ == "__main__":
    main()

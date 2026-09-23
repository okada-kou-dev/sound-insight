"""正常trainのみで学習し、calibration/devで限定比較・設定を固定する。"""
from __future__ import annotations

from scripts.runtime import ROOT, Measurement, config_from, digest, environment, new_id, safe_source, write_json
import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone


def meets_goals(metrics):
    return all(metrics.get(k) is not None for k in ("roc_auc","recall","false_positive_rate")) and (
        metrics["roc_auc"] >= .75 and metrics["recall"] >= .60 and metrics["false_positive_rate"] <= .10)


def ranking(candidate):
    metrics = candidate["metrics"]
    return (metrics["roc_auc"] if metrics["roc_auc"] is not None else -1,
            -metrics["false_positive_rate"] if metrics["false_positive_rate"] is not None else -1,
            -candidate["n_clusters"])


def worker(config_path, output):
    from src.audio import load_audio, extract_features
    from src.dataset import read_manifest, validate_manifest, manifest_sha256
    from src.model import fit_model, save_model, load_model
    from scripts.evaluate import evaluate_rows
    config = config_from(config_path)
    if (ROOT / "outputs" / "final_test_access.jsonl").exists():
        raise ValueError("final_test利用履歴あり。再調整は独立な最終評価と呼べないため自動再学習を停止します")
    manifest = ROOT / "data" / "manifest.csv"
    rows = read_manifest(manifest)
    validate_manifest(rows)
    manifest_hash = manifest_sha256(manifest)
    groups = {split:[r for r in rows if r["split"]==split] for split in ("train","calibration","development")}
    if any(not value for value in groups.values()):
        raise ValueError("train/calibration/developmentが必要です")
    if any(int(r["label"])!=0 for s in ("train","calibration") for r in groups[s]):
        raise ValueError("正常のみのfit/calibration条件に違反しています")
    def features(split):
        for row in groups[split]:
            clip = load_audio(safe_source(row))
            if clip.source_sha256 != row["source_sha256"] or clip.selected_channel_sha256 != row["selected_channel_sha256"]:
                raise ValueError("原音hashとmanifestの不一致")
            yield extract_features(clip, config["feature"]).vectors
    candidates = []
    with Measurement() as total:
        def fit_candidate(k):
            with Measurement() as trained:
                model = fit_model(features("train"), features("calibration"), config, manifest_hash, n_clusters=k)
                model_dir = save_model(model, ROOT / "models")
                reloaded = load_model(model_dir)
            report = evaluate_rows(reloaded, groups["development"], "development", manifest_hash)
            report["training_measurement"] = trained.result("train特徴量抽出・学習・calibration・保存再読込",len(groups["train"]),"cold" if not candidates else "warm")
            write_json(output / f"development_k{k}.json", report)
            result = {"model_id":reloaded.model_id,"n_clusters":k,"metrics":report["metrics"],
                "report":str((output / f"development_k{k}.json").relative_to(ROOT))}
            candidates.append(result)
            print(json.dumps(result,ensure_ascii=False),flush=True)
            return result
        fit_candidate(1)
        first = fit_candidate(16)
        if not meets_goals(first["metrics"]):
            fit_candidate(8)
            fit_candidate(32)
    selected = max([c for c in candidates if c["n_clusters"]!=1],key=ranking)
    report = {"created_at":datetime.now(timezone.utc).isoformat(),"manifest_sha256":manifest_hash,
        "config":config,"candidates":candidates,"selected":selected,"development_goals_met":meets_goals(selected["metrics"]),
        "selection_rule":"development ROC-AUC降順→正常誤警報率昇順→クラスタ数昇順。baselineは参照のみ",
        "final_test_used":False,"environment":environment(),
        "measurement":total.result("baseline・仕様内候補の全学習/校正/開発評価",sum(map(len,groups.values())),"cold to warm")}
    write_json(output / "selection.json", report)
    model_dir = ROOT / "models" / selected["model_id"]
    pointer = {"model_id":selected["model_id"],"frozen":True,"manifest_sha256":manifest_hash,
        "selection_report":str((output / "selection.json").relative_to(ROOT)),
        "selection_sha256":digest(output / "selection.json"),"frozen_at":datetime.now(timezone.utc).isoformat(),
        "artifact_sha256":{p.name:digest(p) for p in model_dir.iterdir() if p.is_file()}}
    pointer_path = ROOT / "models" / "selected.json"
    if pointer_path.exists():
        write_json(output / "previous_selected.json",json.loads(pointer_path.read_text(encoding="utf-8")))
    temporary = ROOT / "models" / (new_id("pointer")+".json")
    write_json(temporary,pointer)
    temporary.replace(pointer_path)
    print(json.dumps({"selected":selected,"frozen":True},ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/default.json")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--output",help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = config_from(config_path)
    if args.worker:
        worker(config_path,Path(args.output))
        return
    output = ROOT / "outputs" / "training" / new_id("training")
    output.mkdir(parents=True,exist_ok=False)
    with (output / "run.log").open("w",encoding="utf-8") as handle:
        process = subprocess.Popen([sys.executable,"-m","scripts.train","--worker","--config",str(config_path),"--output",str(output)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=config["training"]["timeout_seconds"])
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=30)
            write_json(output / "timeout.json",{"timeout_seconds":900,"stopped_pid":process.pid,"log":"run.log"})
            raise SystemExit("学習900秒上限。自身のworkerを停止し、ログと途中モデルを保全しました")
    print((output / "run.log").read_text(encoding="utf-8"))
    if code:
        raise SystemExit(code)


from pathlib import Path
if __name__ == "__main__":
    main()

"""正常近傍モデルを学習し固定100件で開発校正。原モデル・分割を保全。"""
from scripts.runtime import ROOT, Measurement, digest, environment, new_id, safe_source, write_json
import argparse
from dataclasses import replace
import json
import os
import subprocess
import sys

import numpy as np
from src.audio import extract_features, load_audio
from src.dataset import read_manifest
from src.demo_selection import fixed_rows
from src.evaluation import compute_metrics
from src import nearest
from src.temporal import segments


def worker(args):
    rows = read_manifest(ROOT/"data/manifest.csv")
    selected = fixed_rows(ROOT, rows)
    if selected is None:
        raise ValueError("先に固定100件を選定してください")
    ids = {r["record_id"] for r in selected}
    public_ids = {r["record_id"] for r in selected[:20]}
    groups = {s: [r for r in rows if r["split"] == s] for s in ("train", "calibration", "development")}
    if any(not g for g in groups.values()) or any(r["label"] != 0 for s in ("train", "calibration") for r in groups[s]):
        raise ValueError("正常train/calibrationが必要です")
    manifest_hash = digest(ROOT/"data/manifest.csv")
    model_id = new_id("temporal_knn")
    out = ROOT/"outputs/accuracy"/model_id
    out.mkdir(parents=True, exist_ok=False)

    def features(row):
        clip = load_audio(safe_source(row))
        if clip.source_sha256 != row["source_sha256"] or clip.selected_channel_sha256 != row["selected_channel_sha256"]:
            raise ValueError("元音声とmanifestのhashが一致しません")
        return extract_features(clip)

    with Measurement() as measurement:
        model = nearest.fit((features(r).log_mel for r in groups["train"]), model_id,
            {"manifest_hash": manifest_hash, "target": {"machine_type": "pump", "machine_id": "id_00", "snr_db": 6},
             "channel_index": 0, "seed": 42, "environment": environment(), "final_test_used": False,
             "demo_selection_sha256": digest(ROOT/"configs/demo_selection.json")})
        print("fit normal604 complete", flush=True)
        cal = [nearest.score(model, features(r))["window_scores"] for r in groups["calibration"]]
        window_threshold = float(np.quantile(np.concatenate(cal), .995, method="higher"))
        scored = [(r, nearest.score(model, features(r))) for r in groups["development"]]
        development100 = [(r,s) for r,s in scored if r["record_id"] in ids]
        threshold = nearest.calibrate([s["score"] for r,s in development100], [r["label"] for r,s in development100])
        metadata = dict(model.metadata, threshold=threshold, calibration_method="fixed_demo100_min_errors_then_fp_then_margin",
            window_threshold=window_threshold, window_calibration={"split": "normal calibration", "quantile": .995, "method": "higher", "unit": "sustained windows"},
            evaluation_role="fixed100 used for threshold and method development; not independent test")
        model = replace(model, threshold=threshold, metadata=metadata)
        directory = nearest.save(model, ROOT/"models")
        restored = nearest.load(directory, json.loads((directory/"metadata.json").read_text(encoding="utf-8")))
        # An actual post-load audio inference must match the in-memory model.
        check = features(groups["development"][0])
        np.testing.assert_allclose(nearest.score(restored, check)["window_scores"], nearest.score(model, check)["window_scores"], rtol=0, atol=0)
        predictions = []
        for row, result in scored:
            positive = result["score"] > threshold
            flags = (result["window_scores"] > window_threshold) & positive
            feature = result["features"]
            predictions.append({"record_id": row["record_id"], "relative_path": row["relative_path"], "label": row["label"],
                "score": result["score"], "decision": "anomaly_candidate" if positive else "within_reference",
                "segments": segments(feature.starts, feature.ends, flags), "positive_windows": int(flags.sum()),
                "windows": len(flags)})

    def metrics(pred):
        return compute_metrics([p["label"] for p in pred], [p["score"] for p in pred], threshold)
    subsets = {"public20": [p for p in predictions if p["record_id"] in public_ids],
               "local100": [p for p in predictions if p["record_id"] in ids],
               "remaining73": [p for p in predictions if p["record_id"] not in ids]}
    report = {"model_id": model_id, "manifest_sha256": manifest_hash, "split": "development", "threshold": threshold,
        "window_threshold": window_threshold, "metrics": metrics(predictions), "predictions": predictions,
        "demo_metrics": {k: metrics(v) for k,v in subsets.items()},
        "demo_record_ids": {k: [p["record_id"] for p in v] for k,v in subsets.items()},
        "demo_selection_sha256": metadata["demo_selection_sha256"], "calibration_role": metadata["evaluation_role"],
        "remaining73_role": "not used in the current threshold objective; previously viewed development data, not a new independent final test",
        "measurement": measurement.result("normal fit, calibration windows, development, save/load", sum(map(len,groups.values())), "cold"),
        "environment": environment(), "final_test_used": False,
        "temporal_independent_evaluation": "No independent interval annotations or interval accuracy evaluation."}
    report_path = ROOT/"outputs/evaluations"/model_id/"report.json"
    write_json(report_path, report)
    evidence = out/"selection.json"
    write_json(evidence, {"selected": {"model_id": model_id}, "manifest_sha256": manifest_hash,
        "protocol": "training/README.md", "metadata": metadata, "demo_metrics": report["demo_metrics"]})
    pointer = {"model_id": model_id, "manifest_sha256": manifest_hash, "frozen": True,
        "selection_report": evidence.relative_to(ROOT).as_posix(), "selection_sha256": digest(evidence),
        "artifact_sha256": {p.name: digest(p) for p in directory.iterdir()}}
    write_json(out/"pointer.json", pointer)
    if args.activate:
        path = ROOT/"models/temporal_selected.json"
        if path.exists(): write_json(out/"previous_pointer.json", json.loads(path.read_text(encoding="utf-8")))
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    if args.deploy_demo:
        from scripts.train_temporal import deploy_demo
        deploy_demo(directory, pointer, evidence, report_path)
    print(json.dumps({"model_id": model_id, "threshold": threshold, "window_threshold": window_threshold,
                      "demo_metrics": report["demo_metrics"], "measurement": report["measurement"]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--deploy-demo", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    out = ROOT/"outputs/accuracy"/new_id("run")
    out.mkdir(parents=True)
    with (out/"run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "scripts.train_nearest", "--worker"]+sys.argv[1:], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        try:
            code = process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            process.terminate(); process.wait(timeout=30)
            write_json(out/"timeout.json", {"stopped_pid": process.pid, "timeout_seconds": 900})
            raise SystemExit("学習900秒上限。自身のworkerを停止し、中間結果を保全しました")
    write_json(out/"completion.json", {"worker_exit_code": code})
    print((out/"run.log").read_text(encoding="utf-8"))
    if code: raise SystemExit(code)


if __name__ == "__main__":
    main()

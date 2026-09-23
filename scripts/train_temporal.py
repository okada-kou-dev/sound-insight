"""区間検出モデルを正常音だけでfit/校正。既存モデル・final_testは保持。"""
from __future__ import annotations

from scripts.runtime import ROOT, Measurement, digest, environment, new_id, safe_source, write_json
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from src.audio import extract_features, load_audio
from src.dataset import read_manifest, validate_manifest
from src.evaluation import compute_metrics
from src.service import inspect_one
from src import temporal


def deploy_demo(model_dir: Path, pointer: dict, evidence: Path, report_path: Path) -> None:
    from src.public_bundle import verify_bundle
    root = ROOT / "demo_assets"
    bundle = verify_bundle(root)
    # Preserve the previous complete asset manifest before adding the new model.
    write_json(evidence.parent / "previous_demo_bundle.json", bundle)
    paths = [(model_dir / name, f"models/{model_dir.name}/{name}") for name in ("arrays.npz", "metadata.json")]
    paths += [(evidence, f"outputs/{model_dir.name}_selection.json"),
              (report_path, f"outputs/evaluations/{model_dir.name}/report.json")]
    for source, relative in paths:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError("新モデルの配備先が既に存在します")
        shutil.copyfile(source, destination)
    public_pointer = dict(pointer, selection_report=f"outputs/{model_dir.name}_selection.json")
    pointer_path = root / "models" / "temporal_selected.json"
    temporary = pointer_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(public_pointer, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(pointer_path)
    bundle["temporal_model_id"] = model_dir.name
    bundle["sha256"] = {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file() and p.name != "bundle.json"}
    temporary = root / "bundle.tmp"
    temporary.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(root / "bundle.json")
    verify_bundle(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--deploy-demo", action="store_true")
    args = parser.parse_args()
    rows = read_manifest(ROOT / "data" / "manifest.csv")
    validate_manifest(rows)
    groups = {s: [r for r in rows if r["split"] == s] for s in ("train", "calibration", "development")}
    if any(not g for g in groups.values()) or any(int(r["label"]) for s in ("train", "calibration") for r in groups[s]):
        raise ValueError("正常train/calibrationとdevelopmentが必要です")
    manifest_hash = digest(ROOT / "data" / "manifest.csv")
    model_id = new_id("temporal")
    out = ROOT / "outputs" / "temporal" / model_id
    out.mkdir(parents=True, exist_ok=False)

    def checked_clip(row):
        clip = load_audio(safe_source(row))
        if clip.source_sha256 != row["source_sha256"] or clip.selected_channel_sha256 != row["selected_channel_sha256"]:
            raise ValueError("原音とmanifestのhashが不一致です")
        return clip

    def mels(split):
        for i, row in enumerate(groups[split]):
            if i % 150 == 0:
                print(f"{split}: {i}/{len(groups[split])}", flush=True)
            yield extract_features(checked_clip(row)).log_mel

    with Measurement() as measure:
        model = temporal.fit(mels("train"), mels("calibration"), model_id,
            {"manifest_hash": manifest_hash, "target": {"machine_type": "pump", "machine_id": "id_00", "snr_db": 6},
             "channel_index": 0, "seed": 42, "created_at": datetime.now(timezone.utc).isoformat(),
             "environment": environment(), "final_test_used": False,
             "clip_score": "maximum sustained window score; same threshold as interval detector"})
        directory = temporal.save(model, ROOT / "models")
        model = temporal.load(directory, json.loads((directory / "metadata.json").read_text(encoding="utf-8")))
        predictions = []
        for row in groups["development"]:
            checked_clip(row)
            result = inspect_one(model, safe_source(row), row["record_id"])
            if result["status"] != "success":
                raise ValueError(result["error_message"])
            feature = result["_features"]
            predictions.append({"record_id": row["record_id"], "relative_path": row["relative_path"],
                "label": int(row["label"]), "score": result["score"], "decision": result["decision"],
                "segments": temporal.segments(feature.starts, feature.ends, result["_window_flags"]),
                "positive_windows": int(result["_window_flags"].sum()), "windows": len(feature.starts)})
    metrics = compute_metrics([p["label"] for p in predictions], [p["score"] for p in predictions], model.threshold)
    normal = [p for p in predictions if not p["label"]]
    bundle = json.loads((ROOT / "demo_assets" / "bundle.json").read_text(encoding="utf-8"))
    sample_map = {s["record_id"]: i for i, s in enumerate(bundle["samples"], 1)}
    samples = [dict(p, sample_number=sample_map[p["record_id"]]) for p in predictions if p["record_id"] in sample_map]
    timing = []
    for p in samples:
        if p["sample_number"] not in (11, 16):
            continue
        target_end = 10. if p["sample_number"] == 11 else 7.
        overlap = sum(max(0., min(b, target_end) - max(a, 0.)) for a, b in p["segments"])
        predicted_seconds = sum(b-a for a,b in p["segments"])
        timing.append({"sample_number": p["sample_number"], "annotation_source": "user listening, development diagnostic",
            "target": [[0., target_end]], "detected": p["segments"], "covered_seconds": overlap,
            "missed_seconds": target_end-overlap, "extra_seconds": predicted_seconds-overlap,
            "intersection_over_union": overlap/(target_end+predicted_seconds-overlap)})
    report = {"model_id": model_id, "manifest_sha256": manifest_hash, "split": "development", "threshold": model.threshold,
        "metrics": metrics, "predictions": predictions, "measurement": measure.result("正常fit/校正/保存再読込/development", sum(map(len, groups.values())), "cold"),
        "environment": environment(), "final_test_used": False,
        "normal_window_positive_fraction": sum(p["positive_windows"] for p in normal)/sum(p["windows"] for p in normal),
        "demo_samples": sorted(samples, key=lambda p: p["sample_number"]), "user_interval_diagnostics": timing,
        "temporal_independent_evaluation": "No held-out abnormal interval annotations; diagnostics are development observations"}
    report_path = ROOT / "outputs" / "evaluations" / model_id / "report.json"
    write_json(report_path, report)
    evidence = out / "selection.json"
    write_json(evidence, {"selected": {"model_id": model_id}, "manifest_sha256": manifest_hash, "final_test_used": False,
                         "protocol": "docs/TEMPORAL_PROTOCOL.md", "metadata": model.metadata, "development_metrics": metrics,
                         "user_interval_diagnostics": timing})
    pointer = {"model_id": model_id, "manifest_sha256": manifest_hash, "frozen": True,
        "selection_report": evidence.relative_to(ROOT).as_posix(), "selection_sha256": digest(evidence),
        "artifact_sha256": {p.name: digest(p) for p in directory.iterdir()}}
    write_json(out / "pointer.json", pointer)
    if args.activate:
        path = ROOT / "models" / "temporal_selected.json"
        if path.exists():
            write_json(out / "previous_pointer.json", json.loads(path.read_text(encoding="utf-8")))
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    if args.deploy_demo:
        deploy_demo(directory, pointer, evidence, report_path)
    print(json.dumps({"model_id": model_id, "threshold": model.threshold, "metrics": {k:v for k,v in metrics.items() if k != "distributions"},
        "timing": timing, "normal_window_positive_fraction": report["normal_window_positive_fraction"],
        "report": report_path.relative_to(ROOT).as_posix()}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""同梱サンプルを固定評価と照合。取得・学習・最終評価の再実行はしない。"""
from __future__ import annotations

from scripts.runtime import ROOT, environment, new_id, selected_model, write_json
import argparse
import json
import math
from pathlib import Path

from src.audio import load_audio
from src.public_bundle import asset_path, verify_bundle
from src.service import inspect_one

# Feature extraction uses float32. Cross-platform DSP/BLAS rounding is allowed
# only in this deployment check; model parameters and decisions remain exact.
SCORE_REL_TOL = 1e-6
SCORE_ABS_TOL = 1e-9


def verify(root: Path) -> dict:
    bundle = verify_bundle(root)
    model, pointer = selected_model(root)
    if model.model_id != bundle["model_id"] or pointer["manifest_sha256"] != bundle["manifest_sha256"]:
        raise ValueError("公開素材とモデルの来歴不一致")
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in (root / "outputs" / "evaluations").glob("*/report.json")]
    development = next(r for r in reports if r["split"] == "development" and r["model_id"] == model.model_id)
    predictions = {r["record_id"]: r for r in development["predictions"]}
    verified = []
    temporal_model = None
    temporal_verified = []
    if "temporal_model_id" in bundle:
        temporal_model, temporal_pointer = selected_model(root, "temporal_selected.json")
        if temporal_model.model_id != bundle["temporal_model_id"] or temporal_pointer["manifest_sha256"] != bundle["manifest_sha256"]:
            raise ValueError("区間モデルの来歴不一致")
        temporal_report = next(r for r in reports if r["model_id"] == temporal_model.model_id and r["split"] == "development")
        temporal_expected = {r["record_id"]: r for r in temporal_report["predictions"]}
    for sample in bundle["samples"]:
        expected = predictions[sample["record_id"]]
        if int(expected["label"]) != sample["label"]:
            raise ValueError("公開元ラベルとdevelopment証跡が不一致")
        source = asset_path(root, sample["path"])
        if load_audio(source).selected_channel_sha256 != sample["selected_channel_sha256"]:
            raise ValueError("公開音声PCMと原版の先頭チャンネルが不一致")
        result = inspect_one(model, source, source.name)
        if result["status"] != "success" or result["decision"] != expected["decision"]:
            raise ValueError("配備環境の判定と固定評価が不一致")
        verified.append({"sample": sample["path"], "reference_label": sample["label"],
                         "score": result["score"], "expected_score": expected["score"],
                         "absolute_difference": abs(result["score"] - expected["score"]),
                         "score_matches": math.isclose(result["score"], expected["score"], rel_tol=SCORE_REL_TOL, abs_tol=SCORE_ABS_TOL),
                         "expected_threshold_margin": abs(expected["score"] - model.threshold),
                         "decision": result["decision"]})
        if temporal_model is not None:
            from src.temporal import segments
            from src.playback import make_track
            temporal_result = inspect_one(temporal_model, source, source.name)
            expected_temporal = temporal_expected[sample["record_id"]]
            if temporal_result["status"] != "success" or temporal_result["decision"] != expected_temporal["decision"]:
                raise ValueError("区間モデルの判定と固定developmentが不一致")
            track = make_track(temporal_result)
            actual_segments = segments(track["windows"]["starts"], track["windows"]["ends"], track["windows"]["flags"])
            if actual_segments != expected_temporal["segments"] or sample["label"] != expected_temporal["label"]:
                raise ValueError("区間モデルの時間範囲または正解ラベルが不一致")
            temporal_verified.append({"sample": sample["path"], "decision": temporal_result["decision"],
                "segments": actual_segments, "score": temporal_result["score"],
                "score_matches": math.isclose(temporal_result["score"], expected_temporal["score"], rel_tol=SCORE_REL_TOL, abs_tol=SCORE_ABS_TOL)})
    return {"success": all(row["score_matches"] for row in verified + temporal_verified),
            "bundle_id": bundle["bundle_id"], "model_id": model.model_id,
            "score_tolerance": {"relative": SCORE_REL_TOL, "absolute": SCORE_ABS_TOL},
            "count": len(verified), "samples": verified, "environment": environment(),
            "temporal_model_id": temporal_model.model_id if temporal_model is not None else None,
            "temporal_samples": temporal_verified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "demo_assets")
    args = parser.parse_args()
    report = verify(args.assets.resolve())
    out = ROOT / "outputs" / "verification" / new_id("public_bundle") / "report.json"
    write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["success"]:
        raise ValueError(f"配備環境のスコアと固定評価が不一致。全件差分: {out.relative_to(ROOT).as_posix()}")
    print(f"{report['count']} public samples verified; {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

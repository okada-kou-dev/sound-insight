"""固定100件の実音声・来歴・保存モデルの判定と区間を照合。"""
import json
import math
from scripts.runtime import ROOT, new_id, selected_model, write_json, safe_source
from src.audio import load_audio
from src.dataset import read_manifest
from src.demo_selection import fixed_rows
from src.playback import make_track
from src.service import inspect_one
from src.temporal import segments


def main():
    model, _ = selected_model(ROOT, "temporal_selected.json")
    rows = fixed_rows(ROOT, read_manifest(ROOT/"data/manifest.csv"))
    report = json.loads((ROOT/"outputs/evaluations"/model.model_id/"report.json").read_text(encoding="utf-8"))
    expected = {r["record_id"]: r for r in report["predictions"]}
    verified = []
    for number, row in enumerate(rows, 1):
        path = safe_source(row)
        if load_audio(path).selected_channel_sha256 != row["selected_channel_sha256"]:
            raise ValueError("固定デモのPCM不一致")
        result = inspect_one(model, path, f"sample_{number:03d}.wav")
        original = expected[row["record_id"]]
        if result["status"] != "success" or result["decision"] != original["decision"] or not math.isclose(result["score"], original["score"], rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"固定デモ {number} の判定・スコア不一致")
        track = make_track(result)
        spans = segments(track["windows"]["starts"], track["windows"]["ends"], track["windows"]["flags"])
        if spans != original["segments"]:
            raise ValueError(f"固定デモ {number} の区間不一致")
        if number <= 20:
            renamed = inspect_one(model, path, "unlabeled.wav")
            if renamed["score"] != result["score"] or renamed["decision"] != result["decision"]:
                raise ValueError("ファイル名変更で判定が変化しました")
        verified.append({"number": number, "record_id": row["record_id"], "label": row["label"], "score": result["score"], "decision": result["decision"], "segments": spans})
    destination = ROOT/"outputs/verification"/new_id("fixed100")/"report.json"
    write_json(destination, {"success": True, "model_id": model.model_id, "count": len(verified), "samples": verified})
    print(f"{len(verified)} fixed samples verified; {destination.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

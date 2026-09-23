"""実MIMIIデモの単発/一括一致。--synthetic は別記録の合成音smoke test。"""
from __future__ import annotations

from scripts.runtime import ROOT, Measurement, config_from, environment, new_id, safe_source, selected_model, write_json
import argparse
import io
import random


def demo_rows(rows):
    """developmentから正常10/異常10をseed42で固定選択（正誤によらない）。"""
    rng = random.Random(42)
    result = []
    for label in (0, 1):
        group = sorted([r for r in rows if r["split"]=="development" and int(r["label"])==label],key=lambda r:r["relative_path"])
        result.extend(rng.sample(group,min(10,len(group))))
    return result


def verify(model, items, out, kind):
    from src.service import inspect_one, inspect_batch, persist_result
    from src.storage import read_history, history_csv
    events = []
    database = out / "history.sqlite3"
    with Measurement() as measured:
        single = [inspect_one(model,source,name) for source,name in items]
        batch = inspect_batch(model,items,database,progress=lambda done,total,result:events.append([done,total]))
    if batch["success"] != len(items) or batch["saved"] != len(items):
        raise AssertionError("E2Eの検査/保存が失敗しました")
    for one, many in zip(single,batch["results"]):
        if one["status"] != "success" or one["score"] != many["score"] or one["decision"] != many["decision"]:
            raise AssertionError("単発/一括不一致")
        if persist_result(many,database)["save_status"] != "already_saved":
            raise AssertionError("同run/itemの二重保存防止に失敗")
    history = read_history(database)
    if len(history) != len(items) or events != [[i,len(items)] for i in range(1,len(items)+1)]:
        raise AssertionError("履歴/進捗件数不一致")
    (out / "history.csv").write_bytes(history_csv(history))
    renamed = inspect_one(model,items[0][0],"different_label_and_name.wav")
    if renamed["score"] != single[0]["score"]:
        raise AssertionError("ファイル名でスコアが変化")
    report = {"kind":kind,"model_id":model.model_id,"count":len(items),"success":True,
        "checks":["single=batch","rename invariant","DB duplicate rejected","progress count","CSV"],
        "measurement":measured.result("単発全件+一括全件+SQLite保存",len(items)*2,
            "warm (synthetic feature extraction and fitting completed)" if kind.startswith("synthetic") else "first clip cold; subsequent warm"),
        "environment":environment(),"results":batch["results"],
        "not_verified":["実際の音声聴取","OSネット切断","動画撮影"]}
    write_json(out / "report.json",report)
    print(f"{kind}: {len(items)} 件 E2E OK; {out.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic",action="store_true")
    args = parser.parse_args()
    out = ROOT / "outputs" / "verification" / new_id("synthetic" if args.synthetic else "mimii")
    out.mkdir(parents=True,exist_ok=False)
    if args.synthetic:
        import numpy as np
        import soundfile as sf
        from src.audio import load_audio,extract_features
        from src.model import fit_model,save_model,load_model
        items = []
        rng = np.random.default_rng(42)
        for i in range(10):
            t = np.arange(160000,dtype=np.float64)/16000
            samples = (.1*np.sin(2*np.pi*(220+i*2)*t)+.002*rng.normal(size=len(t))).astype(np.float32)
            buffer = io.BytesIO()
            sf.write(buffer,samples,16000,format="WAV",subtype="PCM_16")
            items.append((buffer.getvalue(),f"synthetic_{i}.wav"))
        config = config_from(ROOT / "configs" / "default.json")
        features = [extract_features(load_audio(data),config["feature"]).vectors for data,_ in items]
        model = fit_model(features[:6],features[6:8],config,"synthetic-only",n_clusters=16)
        model_dir = save_model(model,out / "models")
        model = load_model(model_dir)
        verify(model,items[8:],out,"synthetic-not-MIMII")
    else:
        from src.dataset import read_manifest,validate_manifest,manifest_sha256
        model,pointer = selected_model()
        manifest = ROOT / "data" / "manifest.csv"
        rows = read_manifest(manifest)
        validate_manifest(rows)
        if manifest_sha256(manifest) != pointer["manifest_sha256"]:
            raise ValueError("manifestがモデルと不一致")
        selected = demo_rows(rows)
        if not selected:
            raise ValueError("developmentのデモ入力がありません")
        write_json(out / "demo_selection.json",selected)
        verify(model,[(safe_source(r),r["relative_path"]) for r in selected],out,"MIMII-development")


if __name__ == "__main__":
    main()

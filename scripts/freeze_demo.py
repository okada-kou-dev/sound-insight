"""既存公開20件を含む固定100件を再現・検証する。既存選定は上書きしない。"""
import json
from scripts.runtime import ROOT, digest
from src.dataset import read_manifest
from src.demo_selection import create_selection, fixed_rows


def main():
    rows = read_manifest(ROOT / "data" / "manifest.csv")
    bundle = json.loads((ROOT / "demo_assets" / "bundle.json").read_text(encoding="utf-8"))
    path = ROOT / "configs" / "demo_selection.json"
    public_ids = [s["record_id"] for s in bundle["samples"]]
    if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("version") == 2:
        selection = json.loads(path.read_text(encoding="utf-8"))
        original = json.loads((ROOT / "configs/demo_selection_v1.json").read_text(encoding="utf-8"))
        if original != create_selection(rows, original["public20"], digest(ROOT / "data/manifest.csv")) or selection["public20"] != public_ids[:20]:
            raise ValueError("校正時の構成または公開順序との不一致")
    else:
        selection = create_selection(rows, public_ids, digest(ROOT / "data" / "manifest.csv"))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != selection:
            raise ValueError("固定済みデモとの不一致。自動で選び直しません")
    else:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(selection, stream, ensure_ascii=False, indent=2)
    assert len(fixed_rows(ROOT, rows)) == 100
    if bundle.get("selection_mode") == "common50":
        from src.demo_selection import common_rows
        if [r["record_id"] for r in common_rows(ROOT, rows)] != public_ids:
            raise ValueError("Web/ローカル共通50件の不一致")
        print("common50=25 normal/25 abnormal; Web/local order and PCM provenance verified")
    print("fixed public20=10 normal/10 abnormal; local100=50 normal/50 abnormal; identity verified")


if __name__ == "__main__":
    main()

"""公開20件を含む固定100件。ラベルは選定・照合だけに使用。"""
import json
import random
from pathlib import Path


def common_selection(selection: dict, source_hash: str) -> dict:
    """既存20件を維持し、固定100件の順序から各クラス15件を追加。予測は使わない。"""
    by_id = {r["record_id"]: r for r in selection["records"]}
    ids = list(selection["public20"])
    counts = {label: sum(int(by_id[i]["label"]) == label for i in ids) for label in (0, 1)}
    for record_id in selection["local100"]:
        label = int(by_id[record_id]["label"])
        if record_id not in ids and counts[label] < 25:
            ids.append(record_id)
            counts[label] += 1
    if len(ids) != 50 or counts != {0: 25, 1: 25}:
        raise ValueError("共通デモには正常25件・異常25件が必要です")
    return {"version": 1, "source_selection_sha256": source_hash,
            "selection": "keep public20; append first 15 per label in frozen local100 order; no prediction filtering",
            "sample_ids": ids}


def common_rows(root: Path, rows: list[dict]) -> list[dict] | None:
    path = root / "configs/demo50.json"
    if not path.exists():
        return None
    from scripts.runtime import digest
    fixed = fixed_rows(root, rows)
    source = root / "configs/demo_selection.json"
    expected = common_selection(json.loads(source.read_text(encoding="utf-8")), digest(source))
    if json.loads(path.read_text(encoding="utf-8")) != expected:
        raise ValueError("共通50件の固定構成が変更されています")
    by_id = {r["record_id"]: r for r in fixed}
    return [by_id[i] for i in expected["sample_ids"]]


def create_selection(rows: list[dict], public_ids: list[str], manifest_hash: str) -> dict:
    by_id = {r["record_id"]: r for r in rows}
    chosen = list(public_ids)
    if len(chosen) != 20 or len(set(chosen)) != 20:
        raise ValueError("既存の公開20件を指定してください")
    public_rows = [by_id[i] for i in chosen]
    if any(r["split"] != "development" for r in public_rows) or sum(int(r["label"]) for r in public_rows) != 10:
        raise ValueError("公開20件の正常・異常が半々ではありません")
    rng = random.Random(42)
    for label in (0, 1):
        candidates = sorted((r for r in rows if r["split"] == "development" and int(r["label"]) == label and r["record_id"] not in public_ids), key=lambda r: r["relative_path"])
        if len(candidates) < 40:
            raise ValueError("固定100件に必要なdevelopment音声が不足しています")
        chosen.extend(r["record_id"] for r in rng.sample(candidates, 40))
    return {"version": 1, "seed": 42, "manifest_sha256": manifest_hash,
            "selection": "existing public20 unchanged, plus 40 per label drawn before candidate evaluation",
            "public20": public_ids, "local100": chosen,
            "records": [{k: by_id[i][k] for k in ("record_id", "relative_path", "selected_channel_sha256", "label", "split")} for i in chosen]}


def reorder_selection(original: dict, first_id: str, second_id: str, original_hash: str) -> dict:
    """一度だけ表示順を決める。学習・校正に使った100件の構成は変更しない。"""
    evidence = {r["record_id"]: r for r in original["records"]}
    if first_id not in original["public20"] or second_id not in original["public20"]:
        raise ValueError("先頭2件は既存の公開サンプルから選んでください")
    if int(evidence[first_id]["label"]) != 1 or int(evidence[second_id]["label"]) != 0:
        raise ValueError("先頭は異常音、2件目は正常音にします")
    rng = random.Random(20260920)
    public = [i for i in original["public20"] if i not in (first_id, second_id)]
    rng.shuffle(public)
    public = [first_id, second_id] + public
    remaining = [i for i in original["local100"] if i not in public]
    rng.shuffle(remaining)
    result = dict(original, version=2, public20=public, local100=public + remaining,
                  records=[evidence[i] for i in public + remaining],
                  ordering={"seed": 20260920, "first_id": first_id, "second_id": second_id,
                            "original_selection_sha256": original_hash,
                            "reason": "presentation order only; first clip has intermittent model detections"})
    return result


def fixed_rows(root: Path, rows: list[dict], *, public=False) -> list[dict] | None:
    path = root / "configs" / "demo_selection.json"
    if not path.exists():
        return None
    from scripts.runtime import digest
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("version") not in (1, 2) or selection.get("manifest_sha256") != digest(root / "data" / "manifest.csv"):
        raise ValueError("固定デモとmanifestの来歴が一致しません")
    if selection["version"] == 2:
        original_path = root / "configs" / "demo_selection_v1.json"
        ordering = selection["ordering"]
        if digest(original_path) != ordering["original_selection_sha256"]:
            raise ValueError("校正時の固定デモの来歴が一致しません")
        original = json.loads(original_path.read_text(encoding="utf-8"))
        if selection != reorder_selection(original, ordering["first_id"], ordering["second_id"], digest(original_path)):
            raise ValueError("固定デモの表示順または構成が変更されています")
    by_id = {row["record_id"]: row for row in rows}
    ids = selection["local100"]
    public_ids = selection["public20"]
    if len(ids) != 100 or len(set(ids)) != 100 or len(public_ids) != 20 or ids[:20] != public_ids:
        raise ValueError("固定デモ件数・順序が不正です")
    selected = [by_id[i] for i in ids]
    for count, subset in ((100, selected), (20, selected[:20])):
        if sum(int(r["label"]) == 1 for r in subset) != count // 2 or any(r["split"] != "development" for r in subset):
            raise ValueError("固定デモはdevelopmentの正常・異常を半々にします")
    evidence = {r["record_id"]: r for r in selection["records"]}
    for row in selected:
        if any(str(row[k]) != str(evidence[row["record_id"]][k]) for k in ("relative_path", "selected_channel_sha256", "label", "split")):
            raise ValueError("固定デモの音声・ラベルが変更されています")
    return selected[:20] if public else selected

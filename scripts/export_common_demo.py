"""既存の検証済み公開素材を保全し、共通50件を新規出力する。"""
import json
from pathlib import Path
import shutil

from scripts.runtime import ROOT, digest, new_id, write_json
from src.demo_selection import common_selection, common_rows
from src.public_bundle import verify_bundle


def export(root: Path = ROOT) -> Path:
    import soundfile as sf
    from src.audio import load_audio
    from src.dataset import read_manifest
    from scripts.verify_public_demo import verify

    source = root / "demo_assets"
    previous = verify_bundle(source)
    selection_path = root / "configs/demo_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected = common_selection(selection, digest(selection_path))
    config = root / "configs/demo50.json"
    if not config.exists():
        with config.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(expected, stream, ensure_ascii=False, indent=2)
    rows = common_rows(root, read_manifest(root / "data/manifest.csv"))
    if [s["record_id"] for s in previous["samples"][:20]] != expected["sample_ids"][:20]:
        raise ValueError("既存20件の構成・順序が一致しません")
    out = root / "outputs/public_demo" / new_id("common50")
    shutil.copytree(source, out)
    samples = []
    for index, row in enumerate(rows):
        clip = load_audio(root / row["relative_path"])
        if clip.source_sha256 != row["source_sha256"] or clip.selected_channel_sha256 != row["selected_channel_sha256"]:
            raise ValueError("原版PCMとmanifestが一致しません")
        relative = previous["samples"][index]["path"] if index < 20 else f"samples/sample_{index:02d}.wav"
        path = out / relative
        if not path.exists():
            sf.write(path, clip.samples, clip.sample_rate, format="WAV", subtype="PCM_16")
        if load_audio(path).selected_channel_sha256 != clip.selected_channel_sha256:
            raise ValueError("公開モノラル音声と原版PCMが一致しません")
        samples.append(dict(path=relative, record_id=row["record_id"], label=int(row["label"]),
                            split="development", original_source_sha256=clip.source_sha256,
                            selected_channel_sha256=clip.selected_channel_sha256))
    attribution = out / "ATTRIBUTION.md"
    text = attribution.read_text(encoding="utf-8")
    text += "\n共通50件版：既存20件を先頭に維持し、固定development100件の順序から正常・異常を15件ずつ追加。正常25件・異常25件。判定結果による選別なし。Web/ローカルで同じPCM・順序を使用。先頭チャンネル抽出以外の音声加工なし。\n"
    attribution.write_text(text, encoding="utf-8", newline="\n")
    bundle = dict(previous, bundle_id=out.name, selection_mode="common50", samples=samples,
                  selection_sha256=digest(config), sha256={p.relative_to(out).as_posix(): digest(p)
                  for p in sorted(out.rglob("*")) if p.is_file() and p.name != "bundle.json"})
    (out / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    report = verify(out)
    if not report["success"]:
        raise ValueError("50件の推論照合に失敗")
    write_json(root / "outputs/verification" / new_id("common50") / "report.json", report)
    return out


if __name__ == "__main__":
    print(export().relative_to(ROOT).as_posix())

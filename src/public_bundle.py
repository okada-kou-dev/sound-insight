"""公開用の限定素材を検証する。取得・学習・ユーザー音声の保存は行わない。"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from scripts.runtime import digest


def asset_path(root: Path, name: str) -> Path:
    parts = PurePosixPath(name).parts
    if not parts or "\\" in name or ":" in name or name.startswith("/") or any(p in (".", "..") for p in name.split("/")):
        raise ValueError("公開素材の相対パスが不正です")
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("公開素材のパスが配備範囲外です")
    if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent):
        raise ValueError("公開素材のリンクは使用できません")
    return path


def verify_bundle(root: Path) -> dict:
    root = Path(root).resolve()
    bundle = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if bundle.get("version") != 1 or bundle.get("input_mode") != "samples_only":
        raise ValueError("公開素材の形式・入力範囲が不一致です")
    files = bundle["sha256"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / "bundle.json"}
    if actual != set(files):
        raise ValueError("公開素材の一覧と実ファイルが不一致です")
    for name, expected in files.items():
        if digest(asset_path(root, name)) != expected:
            raise ValueError(f"公開素材の変更を検出しました: {name}")
    samples = bundle["samples"]
    if not 1 <= len(samples) <= 50 or len({s["path"] for s in samples}) != len(samples):
        raise ValueError("公開サンプル数または重複が不正です")
    if bundle.get("selection_mode") == "common50" and (len(samples) != 50 or sum(s["label"] == 1 for s in samples) != 25):
        raise ValueError("共通デモは正常25件・異常25件です")
    for sample in samples:
        path = sample["path"]
        if sample["split"] != "development" or sample["label"] not in (0, 1) or not path.startswith("samples/") or not path.endswith(".wav") or path not in files:
            raise ValueError("developmentの公開音声サンプルではありません")
    required = {"models/selected.json", "outputs/selection.json", "ATTRIBUTION.md"}
    if "temporal_model_id" in bundle:
        model_id = bundle["temporal_model_id"]
        if not isinstance(model_id, str) or not model_id.startswith("temporal_") or Path(model_id).name != model_id or "/" in model_id or "\\" in model_id:
            raise ValueError("区間モデルIDが不正です")
        required.update({"models/temporal_selected.json", f"models/{model_id}/arrays.npz",
                         f"models/{model_id}/metadata.json", f"outputs/evaluations/{model_id}/report.json"})
    if not required.issubset(files):
        raise ValueError("固定証跡または権利表示がありません")
    return bundle

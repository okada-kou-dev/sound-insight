"""サンプルのみを扱う公開体験版。履歴はStreamlitのセッション内だけ。"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.public_bundle import asset_path, verify_bundle
from src.presentation import app_header
from src.ui import (app_footer, load_selected_model,
                    inspection_panel)


def remember(results: list[dict]) -> None:
    rows = st.session_state.setdefault("public_history", [])
    known = {(r["run_id"], r["item_index"]) for r in rows}
    for result in results:
        key = (result["run_id"], result["item_index"])
        if key not in known:
            rows.append({k: v for k, v in result.items() if not k.startswith("_")})
            known.add(key)
    st.session_state["public_history"] = rows[-100:]


def main(bundle_root: Path | None = None) -> None:
    root = Path(bundle_root or os.environ.get("SOUND_INSIGHT_DEMO_DIR", os.environ.get("LOCAL_SOUND_DEMO_DIR", Path(__file__).resolve().parents[1] / "demo_assets")))
    st.set_page_config(page_title="SOUND INSIGHT", page_icon="🔊", layout="wide")
    app_header(online=True)
    try:
        bundle = verify_bundle(root)
        model, pointer = load_selected_model(root)
        if model is None or model.model_id != bundle.get("temporal_model_id", bundle["model_id"]) or pointer["manifest_sha256"] != bundle["manifest_sha256"]:
            raise ValueError("公開素材と固定モデルが一致しません")
    except Exception:
        st.error("体験版の素材が未準備、または整合性を確認できません。管理者による準備が必要です。")
        app_footer(online_demo=True)
        return
    # Avoid retaining a prior bundle's result if a deployment changes models.
    active_bundle_id = f"{bundle['bundle_id']}:{model.model_id}"
    if st.session_state.get("public_bundle_id") != active_bundle_id:
        for key in ("public_history", "public_single", "public_batch", "public_active", "public_tracks", "public_visualization_errors", "public_selection", "public_revealed"):
            st.session_state.pop(key, None)
        st.session_state["public_bundle_id"] = active_bundle_id
    samples = bundle["samples"]
    choices = [dict(path=asset_path(root, s["path"]), name=f"サンプル {i:02d}", label=s["label"]) for i, s in enumerate(samples, 1)]
    inspection_panel(model, choices, namespace="public", remember_results=remember)
    app_footer(online_demo=True, attribution=root / "ATTRIBUTION.md")

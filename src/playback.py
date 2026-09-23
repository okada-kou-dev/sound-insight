"""音声・既計算の特徴量を同期表示する、推論を行わないブラウザ部品。"""
from __future__ import annotations

import base64
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


FRONTEND = Path(__file__).with_name("frontend")
PAYLOAD_VERSION = 1
MEL_DISPLAY_MIN_DB = -100.0
MEL_DISPLAY_MAX_DB = 20.0
CONTRIBUTION_DISPLAY_MAX = 0.15  # Fixed across tracks; display saturation only.


def _encoded(values: np.ndarray) -> str:
    return base64.b64encode(values.tobytes(order="C")).decode("ascii")


def make_track(result: dict[str, Any], *, contributions: np.ndarray | None = None) -> dict[str, Any]:
    """成功した詳細結果だけを、JSONで安全に渡せる軽量な再生データにする。

    モデルや原音を変更せず、特徴量の再計算もしない。波形のmin/max集約と
    メルの色量子化は描画用。PCM16は厳密に元float32へ戻せる場合のみ使う。
    """
    if result.get("status") != "success":
        raise ValueError("検査が成功した音声だけを再生モニターに渡せます。")
    try:
        clip, features = result["_clip"], result["_features"]
        samples = np.asarray(clip.samples, dtype=np.float32)
        starts = np.asarray(features.starts, dtype=np.float64)
        ends = np.asarray(features.ends, dtype=np.float64)
        scores = np.asarray(result["_window_scores"], dtype=np.float64)
        log_mel = np.asarray(features.log_mel, dtype=np.float32)
        sample_rate = int(clip.sample_rate)
        score, threshold = float(result["score"]), float(result["threshold"])
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError("再生に必要な音声・時間窓の詳細がありません。") from exc
    if (samples.ndim != 1 or sample_rate != 16000 or not 8 * sample_rate <= len(samples) <= 12 * sample_rate
            or not np.isfinite(samples).all() or not np.any(samples)):
        raise ValueError("再生音声の形状・値・長さが不正です。")
    duration = len(samples) / sample_rate
    if (starts.ndim != 1 or not len(starts) or starts.shape != ends.shape or starts.shape != scores.shape
            or not np.isfinite(starts).all() or not np.isfinite(ends).all() or not np.isfinite(scores).all()
            or np.any(scores < 0) or np.any(starts < 0) or np.any(ends <= starts)
            or np.any(np.diff(starts) <= 0) or np.any(np.diff(ends) <= 0) or ends[-1] > duration + 1e-9
            or log_mel.ndim != 2 or log_mel.shape[0] != 64 or not log_mel.shape[1]
            or not np.isfinite(log_mel).all() or not np.isfinite([score, threshold]).all()
            or score < 0 or threshold < 0
            or result.get("decision") not in ("within_reference", "anomaly_candidate")):
        raise ValueError("時間窓または判定結果が不正です。")

    quantized = np.rint(samples.astype(np.float64) * 32768.0)
    pcm16_exact = bool(np.all((quantized >= -32768) & (quantized <= 32767))
                       and np.array_equal((quantized / 32768.0).astype(np.float32), samples))
    pcm = quantized.astype("<i2") if pcm16_exact else samples.astype("<f4")
    # Approx. 1,000 columns retain extrema, including narrow transient peaks.
    edges = np.linspace(0, len(samples), min(1000, len(samples)) + 1, dtype=np.int64)
    minima = np.minimum.reduceat(samples, edges[:-1])
    maxima = np.maximum.reduceat(samples, edges[:-1])
    mel_colors = np.rint(np.clip((log_mel - MEL_DISPLAY_MIN_DB) /
                                (MEL_DISPLAY_MAX_DB - MEL_DISPLAY_MIN_DB), 0, 1) * 255).astype(np.uint8)
    score_min, score_max = float(scores.min()), float(scores.max())
    flags = result.get("_window_flags")
    window_threshold = float(result.get("_window_threshold", threshold))
    if flags is not None:
        flags = np.asarray(flags)
        policy = result.get("_window_policy", "shared_threshold")
        if policy not in ("shared_threshold", "clip_gated_calibrated") or not np.isfinite(window_threshold) or window_threshold <= 0:
            raise ValueError("区間判定基準が不正です")
        expected = scores > window_threshold
        if policy == "clip_gated_calibrated":
            expected &= result["decision"] == "anomaly_candidate"
        if (flags.shape != scores.shape or flags.dtype != np.dtype(bool)
                or not np.array_equal(flags, expected)
                or (policy == "shared_threshold" and bool(flags.any()) != (result["decision"] == "anomaly_candidate"))):
            raise ValueError("区間判定と固定しきい値・音声全体の判定が不一致です")
    identity = hashlib.sha256()
    identity.update(pcm.tobytes())
    identity.update(scores.tobytes())
    identity.update(f"{result.get('run_id', '')}:{result.get('item_index', 0)}:{result.get('model_id', '')}".encode())
    contribution_payload = None
    if contributions is not None:
        bands = np.asarray(contributions, dtype=np.float64)
        if (bands.shape != (len(scores), 64) or not np.isfinite(bands).all() or np.any(bands < 0)
                or not np.allclose(bands.sum(axis=1), scores, rtol=1e-9, atol=1e-12)
                or not np.isclose(bands.mean(axis=0).sum(), score, rtol=1e-9, atol=1e-12)):
            raise ValueError("帯域別寄与と実際の持続スコアが一致しません。")
        # Only display bytes travel to the browser. Exact additivity is checked above.
        pixels = np.rint(np.clip(bands.T / CONTRIBUTION_DISPLAY_MAX, 0, 1) * 255).astype(np.uint8)
        contribution_payload = {"width": len(scores), "height": 64, "pixels": _encoded(pixels),
                                "maximum": CONTRIBUTION_DISPLAY_MAX, "minimum": 0.0}
        identity.update(pixels.tobytes())
    track = {
        "version": PAYLOAD_VERSION, "id": identity.hexdigest(),
        "title": str(result.get("source_name", "音声サンプル")),
        "item_index": int(result.get("item_index", 0)),
        "run_id": str(result.get("run_id", "")), "model_id": str(result.get("model_id", "")),
        "sample_rate": sample_rate, "duration": duration, "sample_count": len(samples),
        "pcm_format": "s16le" if pcm16_exact else "f32le", "pcm": _encoded(pcm),
        "waveform": {"minimum": minima.tolist(), "maximum": maxima.tolist(),
                     "times": (edges[:-1] / sample_rate).tolist(),
                     "peak": float(np.max(np.abs(samples)))},
        "mel": {"width": log_mel.shape[1], "height": log_mel.shape[0], "pixels": _encoded(mel_colors),
                "minimum_db": MEL_DISPLAY_MIN_DB, "maximum_db": MEL_DISPLAY_MAX_DB,
                "end_seconds": float(ends[-1]), "frame_seconds": 0.064,
                "hop_seconds": (float(ends[-1]) - 0.064) / max(1, log_mel.shape[1] - 1)},
        "windows": {"starts": starts.tolist(), "ends": ends.tolist(), "scores": scores.tolist(),
                    "minimum": score_min, "maximum": score_max,
                    "threshold": window_threshold if flags is not None else None,
                    "flags": flags.tolist() if flags is not None else None},
        "score": score, "threshold": threshold, "decision": str(result["decision"]),
        # These are only display metadata, never inputs to signal or window calculations.
        "reference_label": result.get("reference_label"),
        "comparison": str(result.get("comparison", "")),
    }
    if contribution_payload is not None:
        track["contribution"] = contribution_payload
    return track


@lru_cache(maxsize=1)
def _frontend_assets(revisions: tuple) -> tuple[str, str, str]:
    """Cache file contents, not a callable tied to a Streamlit Runtime registry."""
    return tuple((FRONTEND / name).read_text(encoding="utf-8")
                 for name in ("playback.html", "playback.css", "playback.js"))


def _component():
    import streamlit as st

    paths = [FRONTEND / name for name in ("playback.html", "playback.css", "playback.js")]
    revisions = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    html, css, js = _frontend_assets(revisions)
    # Registration belongs to the current Runtime. Re-registering an identical
    # definition is idempotent in Streamlit 1.64; independent AppTest runtimes
    # must never reuse only the first runtime's mounting callable.
    return st.components.v2.component(
        "sound_insight_synchronized_player",
        html=html,
        css=css,
        js=js,
        isolate_styles=True,
    )


def show_player(tracks: list[dict[str, Any]], *, key: str, autoplay: bool = True, local_insights: bool = False):
    """部品をinline表示。完了IDだけ受信し、推論や保存は繰り返さない。"""
    if not tracks or len(tracks) > 100:
        raise ValueError("再生対象は1～100件です。")
    if any(track.get("version") != PAYLOAD_VERSION for track in tracks):
        raise ValueError("再生データの形式が一致しません。")
    return _component()(key=f"sound_player_{key}", data={"tracks": tracks, "autoplay": bool(autoplay),
                                                       "instance_id": str(key), "local_insights": bool(local_insights)}, default={"completed": []},
                        on_completed_change=lambda: None, height="content", width="stretch")

"""Shared inspection orchestration used by the UI and command-line checks."""
from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from src import audio, model as model_api, storage


MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def display_name(name: str) -> str:
    return str(name).replace("\\", "/").rsplit("/", 1)[-1]


def save_upload(data: bytes, upload_dir: str | Path) -> Path:
    if len(data) > MAX_UPLOAD_BYTES:
        raise audio.InvalidAudio("WAVの上限は10MiBです。")
    directory = Path(upload_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}.wav"
    with path.open("xb") as stream:
        stream.write(data)
    return path


def inspect_one(model: Any, source: Path | bytes, source_name: str,
                run_id: str | None = None, item_index: int = 0) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "run_id": run_id or uuid.uuid4().hex, "item_index": item_index,
        "inspected_at": datetime.now(timezone.utc).isoformat(),
        "source_name": display_name(source_name), "source_sha256": None,
        "model_id": model.model_id, "machine_type": "pump", "machine_id": "id_00",
        "sample_rate": None, "duration_seconds": None, "channel_index": 0,
        "score": None, "threshold": float(model.threshold), "decision": None,
        "processing_ms": 0.0, "status": "error", "error_message": None,
    }
    if isinstance(source, bytes):
        result["source_sha256"] = hashlib.sha256(source).hexdigest()
    try:
        clip = audio.load_audio(source)
        result.update(source_sha256=clip.source_sha256, sample_rate=clip.sample_rate,
                      duration_seconds=clip.duration_seconds)
        features = audio.extract_features(clip, model.feature_config)
        scores = model_api.score_features(model, features)
        result.update(score=float(scores["score"]), threshold=float(scores["threshold"]),
                      decision=scores["decision"], status="success", _clip=clip,
                      _features=scores.get("features", features), _window_scores=scores["window_scores"])
        if "window_flags" in scores:
            result["_window_flags"] = scores["window_flags"]
        for name in ("window_threshold", "window_policy"):
            if name in scores:
                result[f"_{name}"] = scores[name]
    except audio.InvalidAudio as exc:
        result.update(status="invalid_input", error_message=str(exc))
    except Exception as exc:
        # No inferred equipment decision is supplied when software/model/input fails.
        result.update(status="error", error_message=f"{type(exc).__name__}: {exc}")
    result["processing_ms"] = (time.perf_counter() - started) * 1000
    return result


def persist_result(result: dict[str, Any], db_path: str | Path) -> dict[str, Any]:
    try:
        inserted = storage.save_result(db_path, result)
        result.update(save_status="saved" if inserted else "already_saved", save_error=None)
    except Exception as exc:
        result.update(save_status="failed", save_error=f"{type(exc).__name__}: {exc}")
    return result


def inspect_and_save(model: Any, source: Path | bytes, source_name: str,
                     db_path: str | Path, run_id: str | None = None,
                     item_index: int = 0) -> dict[str, Any]:
    result = inspect_one(model, source, source_name, run_id, item_index)
    return persist_result(result, db_path)


def inspect_batch(model: Any, items: Iterable[tuple[Path | bytes, str]],
                  db_path: str | Path | None = None,
                  progress: Callable[[int, int, dict[str, Any]], None] | None = None,
                  run_id: str | None = None,
                  on_result: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Process each item once and retain errors; progress is completed/actual total."""
    inputs = list(items)
    if not 1 <= len(inputs) <= 100:
        raise ValueError("一括検査は1～100件です。")
    batch_id = run_id or uuid.uuid4().hex
    started = time.perf_counter()
    results = []
    for index, (source, name) in enumerate(inputs):
        result = inspect_one(model, source, name, batch_id, index)
        if db_path is not None:
            persist_result(result, db_path)
        if on_result is not None:
            # The viewer may build a compact playback payload once, before the
            # full feature matrix is discarded. It never enters DB/history.
            try:
                on_result(result)
            except Exception as exc:
                # Visualization failure must not erase a saved decision or abort
                # the remaining audio inspections. Streamlit control exceptions
                # inherit BaseException and continue to propagate.
                result["viewer_error"] = f"{type(exc).__name__}: {exc}"
        # Keep batch memory bounded: waveform/features are only needed for a single view.
        result = {key: value for key, value in result.items() if not key.startswith("_")}
        results.append(result)
        if progress is not None:
            progress(index + 1, len(inputs), result)
    return {
        "run_id": batch_id, "total": len(inputs),
        "success": sum(item["status"] == "success" for item in results),
        "anomaly": sum(item["decision"] == "anomaly_candidate" for item in results),
        "failed": sum(item["status"] != "success" for item in results),
        "saved": sum(item.get("save_status") in ("saved", "already_saved") for item in results),
        "save_failed": sum(item.get("save_status") == "failed" for item in results),
        "processing_ms": (time.perf_counter() - started) * 1000,
        "results": results,
    }

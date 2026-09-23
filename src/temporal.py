"""正常音で学習・独立校正する、固定基準による持続区間検出。"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.covariance import LedoitWolf

from .audio import FeatureSet, normalize_feature_config

METHOD = "temporal_ledoitwolf_v1"
FRAMES = 15
PERSISTENCE = 5
QUANTILE = .995


@dataclass(frozen=True)
class TemporalModel:
    model_id: str
    mean: np.ndarray
    precision: np.ndarray
    threshold: float
    feature_config: dict
    metadata: dict


def vectors(log_mel: np.ndarray) -> np.ndarray:
    mel = np.asarray(log_mel, dtype=np.float64)
    if mel.ndim != 2 or mel.shape[0] != 64 or mel.shape[1] < FRAMES + PERSISTENCE - 1 or not np.isfinite(mel).all():
        raise ValueError("区間検出には有限な64帯域のログメルが必要です")
    return sliding_window_view(mel, FRAMES, axis=1).mean(axis=-1).T.copy()


def raw_distances(model: TemporalModel, values: np.ndarray) -> np.ndarray:
    delta = values - model.mean
    result = np.einsum("ij,jk,ik->i", delta, model.precision, delta, optimize=True) / 64
    if not np.isfinite(result).all() or np.any(result < -1e-9):
        raise ValueError("区間スコアが不正です")
    return np.maximum(result, 0)


def fit(train_mels, calibration_mels, model_id: str, metadata: dict) -> TemporalModel:
    # Equal per-clip sampling; neither development labels nor demo clips enter fit.
    samples = []
    for mel in train_mels:
        values = vectors(mel)
        samples.append(values[np.linspace(0, len(values) - 1, min(32, len(values)), dtype=int)])
    if not samples:
        raise ValueError("正常trainが必要です")
    covariance = LedoitWolf().fit(np.concatenate(samples))
    cfg = normalize_feature_config(None)
    provisional = TemporalModel(model_id, covariance.location_, covariance.precision_, 0., cfg, {})
    calibration = [raw_distances(provisional, vectors(mel)) for mel in calibration_mels]
    if not calibration:
        raise ValueError("独立した正常calibrationが必要です")
    threshold = float(np.quantile(np.concatenate(calibration), QUANTILE, method="higher"))
    metadata = dict(metadata, method=METHOD, model_format_version=1, model_id=model_id,
                    feature_config=cfg, frames=FRAMES, persistence=PERSISTENCE,
                    train_clips=len(samples), train_vectors_used=sum(map(len, samples)),
                    calibration_clips=len(calibration), calibration_windows=sum(map(len, calibration)),
                    calibration_method={"quantile": QUANTILE, "method": "higher", "unit": "raw normal windows"},
                    threshold=threshold, shrinkage=float(covariance.shrinkage_))
    return TemporalModel(model_id, covariance.location_, covariance.precision_, threshold, cfg, metadata)


def score(model: TemporalModel, features: FeatureSet) -> dict:
    values = vectors(features.log_mel)
    raw = raw_distances(model, values)
    # A hit requires five consecutive 512ms windows to exceed a fixed threshold.
    # The minimum is a continuous score for the same rule, not a clip-relative rank.
    sustained = sliding_window_view(raw, PERSISTENCE).min(axis=-1)
    starts = np.arange(len(sustained), dtype=np.float64) * .032
    ends = starts + ((FRAMES + PERSISTENCE - 2) * .032 + .064)
    maximum = float(sustained.max())
    return {"score": maximum, "window_scores": sustained, "threshold": model.threshold,
            "decision": "anomaly_candidate" if maximum > model.threshold else "within_reference",
            "window_flags": sustained > model.threshold,
            "features": FeatureSet(values[:len(sustained)], features.log_mel, starts, ends)}


def segments(starts, ends, flags) -> list[list[float]]:
    """Union of detected context supports; never invent intervals from file identity."""
    result = []
    for start, end, hit in zip(starts, ends, flags, strict=True):
        if not hit:
            continue
        start, end = float(start), float(end)
        if result and start <= result[-1][1] + 1e-9:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def save(model: TemporalModel, models_dir: Path) -> Path:
    directory = models_dir / model.model_id
    if Path(model.model_id).name != model.model_id or model.model_id in ("", ".", ".."):
        raise ValueError("不正な区間モデルID")
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "arrays.npz", mean=model.mean, precision=model.precision)
    (directory / "metadata.json").write_text(json.dumps(model.metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return directory


def load(directory: Path, metadata: dict) -> TemporalModel:
    if (metadata.get("method") != METHOD or metadata.get("model_format_version") != 1
            or metadata.get("model_id") != directory.name or metadata.get("frames") != FRAMES
            or metadata.get("persistence") != PERSISTENCE
            or metadata.get("calibration_method") != {"quantile": QUANTILE, "method": "higher", "unit": "raw normal windows"}
            or metadata.get("feature_config") != normalize_feature_config(None)
            or metadata.get("target") != {"machine_type": "pump", "machine_id": "id_00", "snr_db": 6}
            or metadata.get("channel_index") != 0):
        raise ValueError("区間モデルの条件が不一致です")
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"mean", "precision"}:
            raise ValueError("区間モデル配列が不正です")
        mean, precision = (np.asarray(arrays[name], dtype=np.float64) for name in ("mean", "precision"))
    threshold = float(metadata["threshold"])
    if (mean.shape != (64,) or precision.shape != (64, 64) or not np.isfinite(mean).all()
            or not np.isfinite(precision).all() or not np.allclose(precision, precision.T)
            or np.linalg.eigvalsh(precision).min() <= 0 or not np.isfinite(threshold) or threshold <= 0):
        raise ValueError("区間モデル数値が不正です")
    return TemporalModel(directory.name, mean, precision, threshold, metadata["feature_config"], metadata)

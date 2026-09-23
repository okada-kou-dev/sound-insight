"""正常の複数パターンとの近傍距離によるクリップ分類。"""
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.audio import FeatureSet, normalize_feature_config
from src.temporal import vectors

METHOD = "normal_knn_mean_v1"
NEIGHBORS = 5


@dataclass(frozen=True)
class NearestModel:
    model_id: str
    mean: np.ndarray
    scale: np.ndarray
    reference: np.ndarray
    threshold: float
    feature_config: dict
    metadata: dict
    _neighbors: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_neighbors", NearestNeighbors(n_neighbors=NEIGHBORS, algorithm="brute", n_jobs=1).fit(self.reference))


def fit(train_mels, model_id, metadata):
    sampled = []
    for mel in train_mels:
        values = vectors(mel)
        sampled.append(values[np.linspace(0, len(values)-1, min(32, len(values)), dtype=int)])
    if not sampled:
        raise ValueError("正常学習音が必要です")
    train = np.concatenate(sampled)
    scaler = StandardScaler().fit(train)
    metadata = dict(metadata, method=METHOD, model_id=model_id, model_format_version=1,
                    feature_config=normalize_feature_config(), frames=15, persistence=5,
                    neighbors=NEIGHBORS, aggregation="mean sustained nearest distance",
                    train_clips=len(sampled), train_vectors_used=len(train))
    return NearestModel(model_id, scaler.mean_, scaler.scale_, scaler.transform(train),
                        1., normalize_feature_config(), metadata)


def score(model, features):
    values = vectors(features.log_mel)
    distances = model._neighbors.kneighbors((values-model.mean)/model.scale, return_distance=True)[0]
    raw = np.mean(distances**2, axis=1)/64
    sustained = sliding_window_view(raw, 5).min(-1)
    value = float(sustained.mean())
    starts = np.arange(len(sustained), dtype=float)*.032
    result = {"score": value, "window_scores": sustained, "threshold": model.threshold,
            "decision": "anomaly_candidate" if value > model.threshold else "within_reference",
            "features": FeatureSet(values[:len(sustained)], features.log_mel, starts, starts+.640)}
    if "window_threshold" in model.metadata:
        tau = float(model.metadata["window_threshold"])
        result.update(window_threshold=tau, window_policy="clip_gated_calibrated",
                      window_flags=(sustained > tau) & (value > model.threshold))
    return result


def band_contributions(model: NearestModel, features: FeatureSet) -> np.ndarray:
    """持続スコアを64帯域へ分解する表示専用計算。判定・校正には使わない。"""
    values = (vectors(features.log_mel) - model.mean) / model.scale
    distances, indices = model._neighbors.kneighbors(values, return_distance=True)
    raw = np.mean(distances ** 2, axis=1) / 64
    groups = sliding_window_view(raw, 5)
    # The minimum chooses one entire window, not each band's independent minimum.
    selected = np.arange(len(groups)) + groups.argmin(axis=1)
    delta = values[selected, None, :] - model.reference[indices[selected]]
    return np.mean(delta ** 2, axis=1) / 64


def calibrate(scores, labels):
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    if scores.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all() or set(labels.tolist()) != {0, 1}:
        raise ValueError("開発校正には有限スコアと正常・異常の両ラベルが必要です")
    unique = np.unique(scores)
    candidates = np.r_[np.nextafter(unique[0], -np.inf), (unique[:-1]+unique[1:])/2, unique[-1]]
    ranked = []
    for threshold in candidates:
        positive = scores > threshold
        fp, fn = int(np.sum(positive & (labels == 0))), int(np.sum(~positive & (labels == 1)))
        ranked.append((fp+fn, fp, -float(np.min(abs(scores-threshold))), float(threshold)))
    return min(ranked)[-1]


def save(model, models_dir):
    if Path(model.model_id).name != model.model_id or "/" in model.model_id or "\\" in model.model_id or model.model_id in ("", ".", ".."):
        raise ValueError("不正なモデルID")
    directory = Path(models_dir)/model.model_id
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory/"arrays.npz", mean=model.mean, scale=model.scale, reference=model.reference)
    (directory/"metadata.json").write_text(json.dumps(dict(model.metadata, threshold=model.threshold), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return directory


def load(directory, metadata):
    expected = {"method": METHOD, "model_format_version": 1, "model_id": directory.name,
                "frames": 15, "persistence": 5, "neighbors": NEIGHBORS,
                "aggregation": "mean sustained nearest distance", "feature_config": normalize_feature_config(),
                "target": {"machine_type": "pump", "machine_id": "id_00", "snr_db": 6}, "channel_index": 0}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("近傍モデルの条件が不一致です")
    with np.load(directory/"arrays.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"mean", "scale", "reference"}:
            raise ValueError("近傍モデルの配列が不正です")
        mean, scale, reference = [np.asarray(arrays[k], dtype=float) for k in ("mean", "scale", "reference")]
    threshold = float(metadata["threshold"])
    if (metadata.get("calibration_method") != "fixed_demo100_min_errors_then_fp_then_margin"
            or metadata.get("window_calibration") != {"split": "normal calibration", "quantile": .995, "method": "higher", "unit": "sustained windows"}
            or not np.isfinite(metadata.get("window_threshold", float("nan"))) or metadata["window_threshold"] <= 0):
        raise ValueError("近傍モデルの校正条件が不正です")
    if (mean.shape != (64,) or scale.shape != (64,) or reference.ndim != 2 or reference.shape[1] != 64
            or not NEIGHBORS <= len(reference) <= 100000 or not np.isfinite(threshold) or threshold <= 0
            or any(not np.isfinite(x).all() for x in (mean, scale, reference)) or np.any(scale <= 0)):
        raise ValueError("近傍モデルの数値が不正です")
    return NearestModel(directory.name, mean, scale, reference, threshold, metadata["feature_config"], metadata)

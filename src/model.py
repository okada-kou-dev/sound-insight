"""正常音だけで学習・校正する、数値配列で保存可能な距離モデル。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any, Iterable
import uuid

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .audio import FEATURE_VERSION, FeatureSet, normalize_feature_config


MODEL_FORMAT_VERSION = 1
TARGET = {"machine_type": "pump", "machine_id": "id_00", "snr_db": 6}


@dataclass(frozen=True)
class AnomalyModel:
    model_id: str
    mean: np.ndarray
    scale: np.ndarray
    centers: np.ndarray
    threshold: float
    feature_config: dict[str, Any]
    metadata: dict[str, Any]


def _vectors(value: FeatureSet | np.ndarray, dimension: int = 320) -> np.ndarray:
    matrix = np.asarray(value.vectors if isinstance(value, FeatureSet) else value, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix) or matrix.shape[1] != dimension or not np.isfinite(matrix).all():
        raise ValueError(f"特徴量は有限な N×{dimension} 配列にしてください。")
    return matrix


def _training_matrix(features: Iterable[np.ndarray], per_clip: int, maximum: int, seed: int):
    rng = np.random.default_rng(seed)
    chunks = []
    reservoir = None
    selected_count = 0
    clip_count = 0
    original_count = 0
    for features_for_clip in features:
        values = _vectors(features_for_clip)
        clip_count += 1
        original_count += len(values)
        indices = np.linspace(0, len(values) - 1, min(per_clip, len(values)), dtype=int)
        sampled = values[indices]
        if reservoir is None and selected_count + len(sampled) <= maximum:
            chunks.append(sampled)
            selected_count += len(sampled)
            continue
        if reservoir is None:
            reservoir = np.empty((maximum, 320), dtype=np.float64)
            if chunks:
                reservoir[:selected_count] = np.concatenate(chunks)
            chunks.clear()
        for row in sampled:
            if selected_count < maximum:
                reservoir[selected_count] = row
            else:
                index = int(rng.integers(0, selected_count + 1))
                if index < maximum:
                    reservoir[index] = row
            selected_count += 1
    if not clip_count:
        raise ValueError("正常な train クリップがありません。")
    matrix = np.concatenate(chunks) if reservoir is None else reservoir
    return matrix, {
        "train_clips": clip_count, "train_windows_available": original_count,
        "train_windows_after_per_clip_limit": selected_count,
        "train_vectors_used": len(matrix),
        "max_vectors_per_clip": per_clip, "max_training_vectors": maximum,
    }


def _new_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]


def fit_model(
    train_features: Iterable[np.ndarray], calibration_features: Iterable[np.ndarray],
    config: dict[str, Any], manifest_hash: str, n_clusters: int = 16,
) -> AnomalyModel:
    feature_config = normalize_feature_config(config.get("feature"))
    training = config.get("training", {})
    seed = int(config.get("seed", 42))
    if seed != 42 or n_clusters not in (1, 8, 16, 32):
        raise ValueError("seed=42、クラスタ数は 1/8/16/32 に固定しています。")
    if config.get("target", TARGET) != TARGET:
        raise ValueError("対象は pump / id_00 / +6 dB に固定しています。")
    per_clip = int(training.get("max_vectors_per_clip", 64))
    maximum = int(training.get("max_training_vectors", 100000))
    if not 1 <= per_clip <= 64 or not 1 <= maximum <= 100000:
        raise ValueError("学習行列の上限設定が範囲外です。")
    matrix, counts = _training_matrix(train_features, per_clip, maximum, seed)
    if len(matrix) < n_clusters:
        raise ValueError("学習ベクトルがクラスタ数より少ないため学習できません。")
    with threadpool_limits(limits=4):
        scaler = StandardScaler().fit(matrix)
        standardized = scaler.transform(matrix)
        if n_clusters == 1:
            centers = standardized.mean(axis=0, keepdims=True)
        else:
            estimator = MiniBatchKMeans(
                n_clusters=n_clusters, batch_size=int(training.get("batch_size", 1024)),
                n_init=int(training.get("n_init", 3)), max_iter=int(training.get("max_iter", 100)),
                random_state=seed,
            ).fit(standardized)
            centers = estimator.cluster_centers_.copy()
    metadata = {
        "model_id": _new_id(), "model_format_version": MODEL_FORMAT_VERSION,
        "feature_version": FEATURE_VERSION, "feature_config": feature_config,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "normal_mean_distance" if n_clusters == 1 else "minibatch_kmeans",
        "n_clusters": n_clusters, "seed": seed,
        "target": dict(TARGET),
        "channel_index": 0, "manifest_hash": manifest_hash,
        "training": {"batch_size": int(training.get("batch_size", 1024)),
                     "n_init": int(training.get("n_init", 3)), "max_iter": int(training.get("max_iter", 100)),
                     "random_state": seed, "threads": 4},
        "calibration_method": {"quantile": 0.95, "method": "higher", "comparison": "score > threshold"},
        "score_definition": "mean_window(min_center(mean_feature((z - center)^2)))",
        "dependencies": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "librosa", "soundfile")},
        **counts,
    }
    provisional = AnomalyModel(metadata["model_id"], scaler.mean_.copy(), scaler.scale_.copy(),
                               centers, 0.0, feature_config, metadata)
    scores = [score_features(provisional, values)["score"] for values in calibration_features]
    if not scores:
        raise ValueError("正常な calibration クリップがありません。")
    threshold = float(np.quantile(scores, 0.95, method="higher"))
    metadata.update(threshold=threshold, calibration_clips=len(scores), calibration_scores=scores)
    return AnomalyModel(provisional.model_id, provisional.mean, provisional.scale, centers,
                        threshold, feature_config, metadata)


def score_features(model: AnomalyModel, feature_set_or_vectors: FeatureSet | np.ndarray) -> dict[str, Any]:
    from .nearest import NearestModel, score as nearest_score
    if isinstance(model, NearestModel):
        if not isinstance(feature_set_or_vectors, FeatureSet):
            raise ValueError("近傍モデルにはログメル特徴が必要です")
        return nearest_score(model, feature_set_or_vectors)
    from .temporal import TemporalModel, score
    if isinstance(model, TemporalModel):
        if not isinstance(feature_set_or_vectors, FeatureSet):
            raise ValueError("区間モデルにはログメル特徴が必要です")
        return score(model, feature_set_or_vectors)
    values = _vectors(feature_set_or_vectors, len(model.mean))
    if not np.isfinite(model.threshold) or model.threshold < 0:
        raise ValueError("モデルのしきい値が不正です。")
    if (model.mean.shape != model.scale.shape or np.any(model.scale <= 0)
            or model.centers.ndim != 2 or model.centers.shape[1] != len(model.mean)
            or not np.isfinite(model.mean).all() or not np.isfinite(model.scale).all()
            or not np.isfinite(model.centers).all() or not len(model.centers)):
        raise ValueError("モデルの数値配列が不整合です。再学習してください。")
    standardized = (values - model.mean) / model.scale
    distances = np.full(len(values), np.inf, dtype=np.float64)
    # Iterate over the at most 32 centers to avoid an N×K×320 allocation.
    for center in model.centers:
        distances = np.minimum(distances, np.mean((standardized - center) ** 2, axis=1))
    if not np.isfinite(distances).all():
        raise ValueError("計算結果が非有限値です。")
    score = float(distances.mean())
    return {"score": score, "window_scores": distances, "threshold": float(model.threshold),
            "decision": "anomaly_candidate" if score > model.threshold else "within_reference"}


def save_model(model: AnomalyModel, models_dir: Path | str) -> Path:
    destination = Path(models_dir) / model.model_id
    if Path(model.model_id).name != model.model_id or model.model_id in ("", ".", ".."):
        raise ValueError("モデル ID が不正です。")
    destination.mkdir(parents=True, exist_ok=False)
    metadata = dict(model.metadata, model_id=model.model_id, threshold=float(model.threshold),
                    feature_config=model.feature_config)
    np.savez_compressed(destination / "arrays.npz", mean=model.mean, scale=model.scale, centers=model.centers)
    (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return destination


def load_model(model_dir: Path | str) -> AnomalyModel:
    directory = Path(model_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if isinstance(metadata, dict) and metadata.get("method") == "normal_knn_mean_v1":
        from .nearest import load
        return load(directory, metadata)
    if isinstance(metadata, dict) and metadata.get("method") == "temporal_ledoitwolf_v1":
        from .temporal import load
        return load(directory, metadata)
    required = {"model_id", "model_format_version", "feature_version", "feature_config", "created_at",
                "method", "n_clusters", "seed", "target", "channel_index", "manifest_hash", "training",
                "calibration_method", "dependencies", "train_clips", "train_vectors_used", "threshold",
                "calibration_clips"}
    if not isinstance(metadata, dict) or required - set(metadata):
        raise ValueError("モデルの必須メタデータが不足しています。再学習してください。")
    if metadata.get("model_format_version") != MODEL_FORMAT_VERSION or metadata.get("feature_version") != FEATURE_VERSION:
        raise ValueError("モデル形式・特徴量の版が未対応です。再学習してください。")
    if (metadata["target"] != TARGET or metadata["channel_index"] != 0 or metadata["seed"] != 42
            or metadata["n_clusters"] not in (1, 8, 16, 32)):
        raise ValueError("モデルの対象・チャンネル・学習設定が仕様と一致しません。")
    if (metadata["feature_config"] is None
            or metadata["calibration_method"] != {"quantile": 0.95, "method": "higher", "comparison": "score > threshold"}):
        raise ValueError("モデルの特徴量設定・校正法に不整合があります。")
    feature_config = normalize_feature_config(metadata.get("feature_config"))
    if metadata["feature_config"] != feature_config:
        raise ValueError("保存した特徴量設定の項目が不足しています。再学習してください。")
    if metadata.get("model_id") != directory.name:
        raise ValueError("モデル ID と保存先が一致しません。")
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"mean", "scale", "centers"}:
            raise ValueError("モデル配列の項目が一致しません。")
        mean, scale, centers = (np.array(arrays[name], dtype=np.float64) for name in ("mean", "scale", "centers"))
    model = AnomalyModel(metadata["model_id"], mean, scale, centers, float(metadata["threshold"]), feature_config, metadata)
    if mean.shape != (320,) or centers.shape != (metadata["n_clusters"], 320):
        raise ValueError("モデルの特徴次元・クラスタ数が一致しません。")
    score_features(model, mean.reshape(1, -1))
    return model

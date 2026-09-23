"""クリップ単位の評価。未定義指標は理由とともに JSON null へ。"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_metrics(labels, scores, threshold: float) -> dict[str, Any]:
    y = np.asarray(labels)
    values = np.asarray(scores, dtype=np.float64)
    if (y.ndim != 1 or values.ndim != 1 or len(y) != len(values)
            or not np.isin(y, [0, 1]).all() or not np.isfinite(values).all()
            or not np.isfinite(threshold)):
        raise ValueError("ラベルは 0/1、スコアは有限値で、同じ件数の一次元配列にしてください。")
    predicted = values > threshold
    normal, abnormal = y == 0, y == 1
    tp = int(np.sum(predicted & abnormal))
    fp = int(np.sum(predicted & normal))
    tn = int(np.sum(~predicted & normal))
    fn = int(np.sum(~predicted & abnormal))
    reasons: dict[str, str] = {}

    def ratio(name: str, numerator: int, denominator: int, reason: str):
        if not denominator:
            reasons[name] = reason
            return None
        return numerator / denominator

    recall = ratio("recall", tp, tp + fn, "異常の正解クリップが 0 件です。")
    precision = ratio("precision", tp, tp + fp, "要確認の予測が 0 件です。")
    f1 = ratio("f1", 2 * tp, 2 * tp + fp + fn, "異常正解と要確認予測がともに 0 件です。")
    fpr = ratio("false_positive_rate", fp, fp + tn, "正常の正解クリップが 0 件です。")
    if np.any(normal) and np.any(abnormal):
        auc = float(roc_auc_score(y, values))
        partial_auc = float(roc_auc_score(y, values, max_fpr=0.1))
    else:
        auc = partial_auc = None
        reasons["roc_auc"] = reasons["partial_auc"] = "正常と異常の両クラスが必要です。"

    def distribution(mask):
        subset = values[mask]
        if not len(subset):
            return {"count": 0, "min": None, "max": None, "mean": None, "median": None,
                    "q05": None, "q95": None, "scores": []}
        return {"count": len(subset), "min": float(subset.min()), "max": float(subset.max()),
                "mean": float(subset.mean()), "median": float(np.median(subset)),
                "q05": float(np.quantile(subset, 0.05)), "q95": float(np.quantile(subset, 0.95)),
                "scores": subset.tolist()}

    return {"counts": {"total": len(y), "normal": int(normal.sum()), "abnormal": int(abnormal.sum())},
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "threshold": float(threshold), "roc_auc": auc, "partial_auc": partial_auc,
            "partial_auc_definition": "scikit-learn の標準化 pAUC (max_fpr=0.1)",
            "recall": recall, "precision": precision, "f1": f1, "false_positive_rate": fpr,
            "undefined_reasons": reasons,
            "distributions": {"normal": distribution(normal), "abnormal": distribution(abnormal)}}

import json
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation import compute_metrics


class EvaluationTests(unittest.TestCase):
    def test_known_confusion_and_metrics(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.6, 0.4, 0.9]
        result = compute_metrics(labels, scores, .5)
        self.assertEqual((result["tp"], result["fp"], result["tn"], result["fn"]), (1, 1, 1, 1))
        self.assertEqual(result["roc_auc"], .75)
        self.assertEqual(result["partial_auc"], roc_auc_score(labels, scores, max_fpr=.1))
        for metric in ("recall", "precision", "f1", "false_positive_rate"):
            self.assertEqual(result[metric], .5)
        self.assertEqual(result["counts"], {"total": 4, "normal": 2, "abnormal": 2})
        self.assertEqual(result["distributions"]["normal"]["count"], 2)
        json.dumps(result, allow_nan=False)

    def test_threshold_equality_is_normal(self):
        result = compute_metrics([0, 1], [.5, .5], .5)
        self.assertEqual(result["tn"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertIsNone(result["precision"])
        self.assertIn("precision", result["undefined_reasons"])
        self.assertEqual(result["f1"], 0)

    def test_one_class_and_empty_are_null_with_reasons(self):
        normal = compute_metrics([0, 0], [.1, .2], 1)
        for metric in ("roc_auc", "partial_auc", "recall", "precision", "f1"):
            self.assertIsNone(normal[metric])
            self.assertIn(metric, normal["undefined_reasons"])
        self.assertEqual(normal["false_positive_rate"], 0)
        abnormal = compute_metrics([1], [2], 1)
        self.assertIsNone(abnormal["false_positive_rate"])
        self.assertEqual(abnormal["recall"], 1)
        empty = compute_metrics([], [], 1)
        self.assertEqual(empty["counts"]["total"], 0)
        json.dumps(empty, allow_nan=False)

    def test_invalid_values_rejected(self):
        for labels, scores in (([2], [.1]), ([0, 1], [.1]), ([0], [np.nan]), ([[0]], [.1])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                compute_metrics(labels, scores, .5)


if __name__ == "__main__":
    unittest.main()

import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.model import fit_model, load_model, save_model, score_features


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.train = [rng.normal(0, 1, (40, 320)) for _ in range(4)]
        cls.calibration = [rng.normal(0, 1, (20, 320)) for _ in range(5)]
        cls.config = {"seed": 42, "feature": {}, "training": {}}
        cls.model = fit_model(cls.train, cls.calibration, cls.config, "test-manifest", n_clusters=16)

    def test_sklearn_score_and_save_reload_agree(self):
        matrix = np.concatenate(self.train)
        with threadpool_limits(limits=4):
            scaler = StandardScaler().fit(matrix)
            estimator = MiniBatchKMeans(n_clusters=16, batch_size=1024, n_init=3,
                                       max_iter=100, random_state=42).fit(scaler.transform(matrix))
        expected = np.min(estimator.transform(scaler.transform(self.calibration[0])) ** 2, axis=1) / 320
        result = score_features(self.model, self.calibration[0])
        np.testing.assert_allclose(result["window_scores"], expected, rtol=1e-12, atol=1e-12)
        with tempfile.TemporaryDirectory() as temporary:
            path = save_model(self.model, temporary)
            loaded = load_model(path)
            np.testing.assert_array_equal(score_features(loaded, self.calibration[0])["window_scores"], result["window_scores"])
            self.assertEqual(loaded.metadata["manifest_hash"], "test-manifest")
            with self.assertRaises(FileExistsError):
                save_model(self.model, temporary)

    def test_threshold_quantile_boundary_and_score_direction(self):
        scores = [score_features(self.model, item)["score"] for item in self.calibration]
        self.assertEqual(self.model.threshold, np.quantile(scores, .95, method="higher"))
        result = score_features(self.model, self.calibration[0])
        equal = dataclasses.replace(self.model, threshold=result["score"])
        self.assertEqual(score_features(equal, self.calibration[0])["decision"], "within_reference")
        very_different = self.calibration[0] + 100
        self.assertGreater(score_features(self.model, very_different)["score"], result["score"])
        self.assertEqual(score_features(self.model, very_different)["decision"], "anomaly_candidate")

    def test_baseline_uses_exact_mean_and_own_calibration(self):
        baseline = fit_model(self.train, self.calibration, self.config, "test", n_clusters=1)
        scaler = StandardScaler().fit(np.concatenate(self.train))
        expected_center = scaler.transform(np.concatenate(self.train)).mean(axis=0, keepdims=True)
        np.testing.assert_array_equal(baseline.centers, expected_center)
        expected_threshold = np.quantile([score_features(baseline, c)["score"] for c in self.calibration], .95, method="higher")
        self.assertEqual(baseline.threshold, expected_threshold)
        self.assertEqual(baseline.metadata["method"], "normal_mean_distance")

    def test_train_limits_reproducibility_and_calibration_not_fit(self):
        config = dict(self.config, training={"max_vectors_per_clip": 12, "max_training_vectors": 20})
        first = fit_model(iter(self.train), iter(self.calibration), config, "test", n_clusters=1)
        second = fit_model(iter(self.train), [x + 100 for x in self.calibration], config, "test", n_clusters=1)
        self.assertEqual(first.metadata["train_vectors_used"], 20)
        self.assertEqual(first.metadata["train_windows_after_per_clip_limit"], 48)
        np.testing.assert_array_equal(first.mean, second.mean)
        np.testing.assert_array_equal(first.centers, second.centers)
        self.assertNotEqual(first.threshold, second.threshold)
        self.assertNotEqual(first.model_id, second.model_id)

    def test_default_training_subsampling_does_not_remove_inference_windows(self):
        values = np.random.default_rng(42).normal(size=(300, 320))
        baseline = fit_model([values], [values], self.config, "test", n_clusters=1)
        self.assertEqual(baseline.metadata["train_vectors_used"], 64)
        self.assertEqual(baseline.metadata["train_windows_available"], 300)
        np.testing.assert_allclose(baseline.mean, values[np.linspace(0, 299, 64, dtype=int)].mean(axis=0))
        self.assertEqual(len(score_features(baseline, values)["window_scores"]), 300)

    def test_corrupt_model_feature_version_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = save_model(self.model, temporary)
            metadata_path = path / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["feature_version"] = -1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model(path)
        with self.assertRaises(ValueError):
            score_features(self.model, np.full((1, 320), np.nan))
        with self.assertRaises(ValueError):
            score_features(dataclasses.replace(self.model, scale=np.zeros(320)), self.train[0])
        with self.assertRaises(ValueError):
            fit_model([], self.calibration, self.config, "test")
        with self.assertRaises(ValueError):
            fit_model(self.train, [], self.config, "test")

    def test_missing_metadata_and_wrong_target_rejected(self):
        for change in ({"feature_config": None}, {"feature_config": {}}, {"target": {"machine_type": "fan"}},
                       {"calibration_method": {}}, {"seed": 1}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                path = save_model(self.model, temporary)
                metadata_path = path / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata.update(change)
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_model(path)
        with tempfile.TemporaryDirectory() as temporary:
            path = save_model(self.model, temporary)
            (path / "metadata.json").write_text('{}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model(path)


if __name__ == "__main__":
    unittest.main()

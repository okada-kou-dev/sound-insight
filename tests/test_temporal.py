"""固定基準・独立校正・時刻・持続条件・モデル再読込の検証。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src import temporal
from src.audio import FeatureSet, normalize_feature_config
from src.model import load_model, score_features
from src.playback import make_track
from tests.test_playback import detail_result


def features(mel):
    return FeatureSet(np.zeros((1, 320)), mel, np.array([0.]), np.array([.192]))


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.model = temporal.TemporalModel("synthetic", np.zeros(64), np.eye(64), 1., normalize_feature_config(None), {})

    def test_fixed_threshold_detects_constant_anomaly_and_no_normal_peaks(self):
        abnormal = score_features(self.model, features(np.full((64, 311), 2.)))
        normal = score_features(self.model, features(np.full((64, 311), .5)))
        self.assertTrue(abnormal["window_flags"].all())
        self.assertFalse(normal["window_flags"].any())
        self.assertEqual(abnormal["decision"], "anomaly_candidate")
        self.assertEqual(normal["decision"], "within_reference")
        self.assertAlmostEqual(abnormal["features"].ends[0], .640)
        self.assertAlmostEqual(abnormal["features"].ends[-1], 9.984)
        self.assertEqual(temporal.segments(abnormal["features"].starts, abnormal["features"].ends, abnormal["window_flags"]), [[0., 9.984]])

    def test_later_louder_audio_never_renormalizes_earlier_scores(self):
        mel = np.full((64, 311), 2.)
        first = temporal.score(self.model, features(mel))
        mel[:, 200:] = 30.
        changed = temporal.score(self.model, features(mel))
        np.testing.assert_array_equal(first["window_scores"][:182], changed["window_scores"][:182])
        np.testing.assert_array_equal(first["window_flags"][:182], changed["window_flags"][:182])

    def test_persistence_rejects_isolated_score_and_boundary_equality(self):
        raw = np.ones(297)
        raw[5:9] = 2.  # Four high windows are insufficient.
        raw[30:35] = 2.
        with patch("src.temporal.raw_distances", return_value=raw):
            result = temporal.score(self.model, features(np.ones((64, 311))))
        self.assertEqual(np.flatnonzero(result["window_flags"]).tolist(), [30])
        self.assertEqual(result["score"], 2.)
        with patch("src.temporal.raw_distances", return_value=np.ones(297)):
            boundary = temporal.score(self.model, features(np.ones((64, 311))))
        self.assertEqual(boundary["decision"], "within_reference")

    def test_fit_uses_independent_normal_calibration_and_roundtrips(self):
        rng = np.random.default_rng(42)
        train = [rng.normal(size=(64, 100)) for _ in range(3)]
        calibration = [rng.normal(size=(64, 100)) for _ in range(2)]
        metadata = {"target": {"machine_type": "pump", "machine_id": "id_00", "snr_db": 6}, "channel_index": 0}
        model = temporal.fit(train, calibration, "synthetic", metadata)
        more_distant = temporal.fit(train, [x+5 for x in calibration], "synthetic", metadata)
        np.testing.assert_array_equal(model.mean, more_distant.mean)
        np.testing.assert_array_equal(model.precision, more_distant.precision)
        self.assertGreater(more_distant.threshold, model.threshold)
        expected = np.quantile(np.concatenate([temporal.raw_distances(model, temporal.vectors(x)) for x in calibration]), .995, method="higher")
        self.assertEqual(model.threshold, expected)
        with tempfile.TemporaryDirectory() as temporary:
            directory = temporal.save(model, Path(temporary))
            restored = load_model(directory)
            np.testing.assert_array_equal(temporal.score(model, features(calibration[0]))["window_scores"],
                                          temporal.score(restored, features(calibration[0]))["window_scores"])
            damaged = dict(model.metadata, persistence=1)
            (directory/"metadata.json").write_text(json.dumps(damaged), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model(directory)

    def test_payload_rejects_flags_inconsistent_with_actual_scores_or_clip(self):
        result = detail_result()
        result.update(threshold=.3, score=.5, decision="anomaly_candidate", _window_flags=np.array([False, True, True]))
        self.assertEqual(make_track(result)["windows"]["flags"], [False, True, True])
        for change in ({"_window_flags": np.array([True, False, True])}, {"decision": "within_reference"}, {"_window_flags": np.array([0, 1, 1])}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                make_track(dict(result, **change))

    def test_invalid_features_and_empty_training_fail(self):
        for mel in (np.zeros((63, 100)), np.zeros((64, 5)), np.full((64, 100), np.nan)):
            with self.assertRaises(ValueError):
                temporal.vectors(mel)
        with self.assertRaises(ValueError):
            temporal.fit([], [], "empty", {})


if __name__ == "__main__":
    unittest.main()

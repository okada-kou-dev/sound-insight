from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src import nearest
from src.audio import FeatureSet
from src.model import load_model, score_features
from src.playback import make_track
from tests.test_playback import detail_result


def features(mel):
    return FeatureSet(np.zeros((1, 320)), mel, np.array([0.]), np.array([.192]))


class NearestTests(unittest.TestCase):
    def model(self):
        rng = np.random.default_rng(42)
        m = nearest.fit([rng.normal(-2, .1, (64, 311)), rng.normal(2, .1, (64, 311))], "synthetic-nearest",
                        {"target": {"machine_type": "pump", "machine_id": "id_00", "snr_db": 6}, "channel_index": 0})
        return replace(m, threshold=.5, metadata=dict(m.metadata,
            calibration_method="fixed_demo100_min_errors_then_fp_then_margin", window_threshold=.6,
            window_calibration={"split": "normal calibration", "quantile": .995, "method": "higher", "unit": "sustained windows"}))

    def test_normal_modes_and_anomaly_with_fixed_threshold_and_reload(self):
        m = self.model()
        for level in (-2, 2):
            result = score_features(m, features(np.full((64,311),level)))
            self.assertEqual(result["decision"], "within_reference")
            self.assertFalse(result["window_flags"].any())
        sample = features(np.full((64,311),7.))
        result = score_features(m, sample)
        self.assertEqual(result["decision"], "anomaly_candidate")
        self.assertTrue(result["window_flags"].all())
        with tempfile.TemporaryDirectory() as directory:
            path = nearest.save(m, Path(directory))
            loaded = load_model(path)
            np.testing.assert_array_equal(score_features(loaded, sample)["window_scores"], result["window_scores"])
            metadata = json.loads((path/"metadata.json").read_text(encoding="utf-8")); metadata["neighbors"] = 1
            (path/"metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError): load_model(path)

    def test_clip_score_is_mean_and_future_cannot_rescale_past_windows(self):
        m = self.model(); signal = np.full((64,311),2.); altered = signal.copy(); altered[:,220:] = 7.
        a, b = [score_features(m, features(s)) for s in (signal,altered)]
        np.testing.assert_array_equal(a["window_scores"][:190], b["window_scores"][:190])
        self.assertAlmostEqual(b["score"], b["window_scores"].mean())
        self.assertEqual(b["features"].ends[0], .640)

    def test_supervised_development_calibration_and_tie_prefers_fewer_false_alarms(self):
        self.assertEqual(nearest.calibrate([.1,.2,.8,.9],[0,0,1,1]),.5)
        # A score tie cannot be eliminated by moving a threshold.
        tau = nearest.calibrate([.1,.4,.4,.9],[0,0,1,1])
        self.assertGreater(tau,.4)
        with self.assertRaises(ValueError): nearest.calibrate([.1,.2],[0,0])

    def test_payload_uses_separate_window_threshold_and_clip_gate(self):
        r = detail_result(); r.update(_window_policy="clip_gated_calibrated", _window_threshold=.7,
            score=.4, threshold=.5, decision="within_reference")
        r["_window_flags"] = np.zeros_like(r["_window_scores"], dtype=bool)
        t = make_track(r)
        self.assertEqual(t["windows"]["threshold"],.7)
        self.assertFalse(any(t["windows"]["flags"]))
        r["_window_flags"][0] = True
        with self.assertRaises(ValueError): make_track(r)

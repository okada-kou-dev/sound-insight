import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio import InvalidAudio, extract_features, load_audio, playback_wav


def wav_bytes(samples, rate=16000, subtype="PCM_16", format="WAV"):
    output = io.BytesIO()
    sf.write(output, samples, rate, format=format, subtype=subtype)
    return output.getvalue()


def tone(seconds=8, rate=16000, amplitude=0.2):
    return (amplitude * np.sin(2 * np.pi * 440 * np.arange(int(seconds * rate)) / rate)).astype(np.float32)


class AudioTests(unittest.TestCase):
    def test_channel_selection_hash_and_renaming(self):
        samples = tone()
        channels = np.column_stack([samples] + [np.full(len(samples), 0.8) for _ in range(7)])
        raw = wav_bytes(channels)
        clip = load_audio(raw)
        mono = load_audio(wav_bytes(samples))
        self.assertEqual(clip.channels, 8)
        self.assertEqual(clip.samples.ndim, 1)
        self.assertEqual(clip.samples.dtype, np.float32)
        np.testing.assert_array_equal(clip.samples, mono.samples)
        self.assertEqual(clip.selected_channel_sha256, mono.selected_channel_sha256)
        self.assertNotEqual(clip.source_sha256, mono.source_sha256)
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / "normal.wav", Path(temporary) / "abnormal.wav"
            a.write_bytes(raw)
            b.write_bytes(raw)
            self.assertEqual(load_audio(a).source_sha256, load_audio(b).source_sha256)
            np.testing.assert_array_equal(extract_features(load_audio(a)).vectors,
                                          extract_features(load_audio(b)).vectors)

    def test_reject_invalid_inputs_without_resampling(self):
        invalid = [b"not a WAV", wav_bytes(tone(7)), wav_bytes(tone(13)),
                   wav_bytes(tone(rate=8000), rate=8000),
                   wav_bytes(np.column_stack([tone(), tone()])),
                   wav_bytes(np.zeros(8 * 16000)), b"x" * (10 * 1024 * 1024 + 1),
                   wav_bytes(tone(), format="FLAC")]
        for raw in invalid:
            with self.subTest(bytes=len(raw)), self.assertRaises(InvalidAudio):
                load_audio(raw)

    def test_nonfinite_and_silent_selected_channel(self):
        samples = tone()
        samples[10] = np.nan
        with self.assertRaises(InvalidAudio):
            load_audio(wav_bytes(samples, subtype="FLOAT"))
        channels = np.column_stack([np.zeros(128000)] + [tone() for _ in range(7)])
        with self.assertRaises(InvalidAudio):
            load_audio(wav_bytes(channels))

    def test_quiet_is_accepted_playback_preserves_samples(self):
        clip = load_audio(wav_bytes(tone(amplitude=1e-7), subtype="FLOAT"))
        decoded, rate = sf.read(io.BytesIO(playback_wav(clip)), dtype="float32")
        self.assertEqual(rate, clip.sample_rate)
        np.testing.assert_array_equal(decoded, clip.samples)

    def test_feature_shape_order_and_times(self):
        clip = load_audio(wav_bytes(tone()))
        features = extract_features(clip)
        frames = 1 + (len(clip.samples) - 1024) // 512
        self.assertEqual(features.log_mel.shape, (64, frames))
        self.assertEqual(features.vectors.shape, (frames - 4, 320))
        self.assertTrue(np.isfinite(features.vectors).all())
        np.testing.assert_array_equal(features.vectors[0], features.log_mel[:, :5].T.reshape(-1))
        self.assertEqual(features.starts[0], 0)
        self.assertAlmostEqual(features.starts[1], 512 / 16000)
        self.assertAlmostEqual(features.ends[0], (4 * 512 + 1024) / 16000)
        self.assertLessEqual(features.ends[-1], clip.duration_seconds)

    def test_volume_is_not_normalized_and_config_is_fixed(self):
        loud = extract_features(load_audio(wav_bytes(tone(), subtype="FLOAT")))
        quiet = extract_features(load_audio(wav_bytes(tone(amplitude=.02), subtype="FLOAT")))
        peak_band = np.argmax(loud.log_mel[:, 0])
        self.assertAlmostEqual(float(loud.log_mel[peak_band, 0] - quiet.log_mel[peak_band, 0]), 20, places=3)
        with self.assertRaises(ValueError):
            extract_features(load_audio(wav_bytes(tone())), {"center": True})


if __name__ == "__main__":
    unittest.main()

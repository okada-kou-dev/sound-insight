"""新規Pythonプロセスで接続要求を禁止する。OSネット切断の代替ではない。"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


class OfflineProcessingTests(unittest.TestCase):
    def test_fresh_process_inference_playback_and_storage_need_no_network(self):
        worker = textwrap.dedent('''
            import io
            from pathlib import Path
            import socket
            import sys
            connection_attempts = []
            def blocked(*args, **kwargs):
                connection_attempts.append(True)
                raise AssertionError("Network is unavailable in this test")
            socket.create_connection = blocked
            socket.getaddrinfo = blocked
            socket.socket.connect = blocked
            socket.socket.connect_ex = blocked
            import numpy as np
            import soundfile as sf
            from src.audio import load_audio, extract_features, playback_wav
            from src.model import fit_model, save_model, load_model
            from src.service import inspect_batch, inspect_and_save
            from src.storage import read_history, history_csv
            root = Path(sys.argv[1])
            t = np.arange(8 * 16000) / 16000
            def wav(frequency):
                buffer = io.BytesIO()
                sf.write(buffer, .1 * np.sin(2 * np.pi * frequency * t), 16000, format="WAV", subtype="PCM_16")
                return buffer.getvalue()
            train = extract_features(load_audio(wav(220))).vectors
            calibration = extract_features(load_audio(wav(222))).vectors
            trained = fit_model([train], [calibration], {}, "offline-synthetic-only", n_clusters=1)
            selected = load_model(save_model(trained, root / "models"))
            database = root / "history.sqlite3"
            result = inspect_and_save(selected, wav(224), "synthetic.wav", database)
            assert result["status"] == "success" and result["save_status"] == "saved", result
            assert playback_wav(result["_clip"])[:4] == b"RIFF"
            batch = inspect_batch(selected, [(wav(226), "batch.wav")], database)
            assert batch["success"] == batch["saved"] == 1
            assert len(read_history(database)) == 2
            assert history_csv(read_history(database)).startswith(b"\\xef\\xbb\\xbf")
            from src.temporal import fit, save
            temporal = fit([extract_features(load_audio(wav(220))).log_mel],
                           [extract_features(load_audio(wav(222))).log_mel],
                           "offline-temporal", {"target": {"machine_type": "pump", "machine_id": "id_00", "snr_db": 6}, "channel_index": 0})
            temporal = load_model(save(temporal, root / "models"))
            detected = inspect_and_save(temporal, wav(224), "temporal.wav", database)
            assert detected["status"] == "success" and detected["save_status"] == "saved", detected
            assert len(read_history(database)) == 3
            from dataclasses import replace
            from src import nearest
            new_model = nearest.fit([extract_features(load_audio(wav(220))).log_mel],
                "offline-nearest", {"target": {"machine_type": "pump", "machine_id": "id_00", "snr_db": 6}, "channel_index": 0})
            new_model = replace(new_model, metadata=dict(new_model.metadata,
                calibration_method="fixed_demo100_min_errors_then_fp_then_margin", window_threshold=.7,
                window_calibration={"split": "normal calibration", "quantile": .995, "method": "higher", "unit": "sustained windows"}))
            new_model = load_model(nearest.save(new_model, root / "models"))
            detected = inspect_and_save(new_model, wav(224), "nearest.wav", database)
            assert detected["status"] == "success" and detected["save_status"] == "saved", detected
            assert len(read_history(database)) == 4
            assert not connection_attempts, "A network request was attempted even if its error was suppressed"
            print("network-blocked-synthetic: inference/playback/SQLite/CSV OK; OS disconnect untested")
        ''')
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", worker, directory],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                encoding="utf-8", env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("network-blocked-synthetic", result.stdout)


if __name__ == "__main__":
    unittest.main()

"""CLI 共通: プロジェクト内パス、実測、再現情報。"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def new_id(prefix):
    return f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_source(row):
    path = (ROOT / row["relative_path"]).resolve()
    if not path.is_relative_to((ROOT / "data" / "raw").resolve()):
        raise ValueError("manifest の原音パスが data/raw の外です")
    return path


def config_from(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    # このMVPの変更可能範囲は候補クラスタ数のみ。CLIでは基準設定を固定する。
    expected_feature = {"sample_rate":16000,"channel_index":0,"n_fft":1024,"win_length":1024,
        "hop_length":512,"n_mels":64,"power":2.0,"center":False,"window":"hann",
        "fmin":0,"fmax":8000,"htk":False,"norm":"slaney","context_frames":5,"log_floor":1e-10}
    expected_training = {"n_clusters":16,"batch_size":1024,"n_init":3,"max_iter":100,
        "random_state":42,"max_vectors_per_clip":64,"max_training_vectors":100000,
        "threads":4,"timeout_seconds":900}
    if config.get("feature") != expected_feature or config.get("training") != expected_training:
        raise ValueError("仕様で固定した特徴量・学習設定と一致しません")
    if config.get("seed") != 42 or config.get("target") != {"machine_type":"pump","machine_id":"id_00","snr_db":6}:
        raise ValueError("対象またはseedが仕様と一致しません")
    return config


def environment():
    import importlib.metadata
    return {"python":platform.python_version(),"platform":platform.system(),
            "machine":platform.machine(),"logical_cpus":os.cpu_count(),
            "dependencies":{name:importlib.metadata.version(name) for name in
                ("numpy","scipy","librosa","soundfile","scikit-learn","streamlit","psutil")}}


class Measurement:
    """perf_counter wall time + 50ms周期で観測した当該プロセスRSS。"""
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.maximum = self.process.memory_info().rss
        self.stop = threading.Event()
        self.start = time.perf_counter()
        def sample():
            while not self.stop.wait(0.05):
                self.maximum = max(self.maximum, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.maximum = max(self.maximum, self.process.memory_info().rss)
        self.seconds = time.perf_counter() - self.start
        self.stop.set()
        self.thread.join()

    def result(self, scope, count, state="warm"):
        return {"seconds":self.seconds,"observed_max_rss_bytes":self.maximum,
            "rss_sampling_seconds":0.05,"scope":scope,"count":count,"state":state}


def selected_model(root=ROOT, pointer_name="selected.json"):
    from src.model import load_model
    root = Path(root).resolve()
    if pointer_name not in ("selected.json", "temporal_selected.json"):
        raise ValueError("未対応のモデルポインタ")
    pointer = json.loads((root / "models" / pointer_name).read_text(encoding="utf-8"))
    model_id = pointer["model_id"]
    if Path(model_id).name != model_id or "/" in model_id or "\\" in model_id:
        raise ValueError("不正なmodel_id")
    model_dir = root / "models" / model_id
    if not model_dir.resolve().is_relative_to((root / "models").resolve()):
        raise ValueError("モデルパスがプロジェクト外です")
    checksums = pointer.get("artifact_sha256", {})
    if set(checksums) != {"arrays.npz", "metadata.json"}:
        raise ValueError("固定モデルの必須ファイルhashがありません")
    for name, checksum in checksums.items():
        if Path(name).name != name or digest(model_dir / name) != checksum:
            raise ValueError("固定後のモデル変更を検出しました")
    model = load_model(model_dir)
    if model.metadata["manifest_hash"] != pointer.get("manifest_sha256"):
        raise ValueError("モデルと選択ポインタのmanifest hashが不一致です")
    if not pointer.get("frozen"):
        raise ValueError("モデル・設定が固定されていません")
    evidence = (root / pointer["selection_report"]).resolve()
    if not evidence.is_relative_to((root / "outputs").resolve()) or digest(evidence) != pointer["selection_sha256"]:
        raise ValueError("モデル選択の固定証跡が一致しません")
    selection = json.loads(evidence.read_text(encoding="utf-8"))
    if selection["selected"]["model_id"] != model_id or selection["manifest_sha256"] != pointer["manifest_sha256"]:
        raise ValueError("選択証跡とモデル・manifestの不一致")
    return model, pointer

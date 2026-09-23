"""共有する WAV 検証・先頭チャンネル選択・ログメル特徴量。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


MAX_AUDIO_BYTES = 10 * 1024 * 1024
FEATURE_VERSION = 1
DEFAULT_FEATURE_CONFIG = {
    "sample_rate": 16000,
    "channel_index": 0,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 512,
    "n_mels": 64,
    "power": 2.0,
    "center": False,
    "window": "hann",
    "fmin": 0,
    "fmax": 8000,
    "htk": False,
    "norm": "slaney",
    "context_frames": 5,
    "log_floor": 1e-10,
}


class InvalidAudio(ValueError):
    """入力音声は対象条件を満たさず、設備の判定は行えない。"""


@dataclass(frozen=True)
class AudioClip:
    samples: np.ndarray
    sample_rate: int
    channels: int
    duration_seconds: float
    source_sha256: str
    selected_channel_sha256: str


@dataclass(frozen=True)
class FeatureSet:
    vectors: np.ndarray
    log_mel: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


def normalize_feature_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """省略値を補い、本形式で未対応の設定を明確に拒否する。"""
    result = dict(DEFAULT_FEATURE_CONFIG)
    if config:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f"未対応の特徴量設定です: {sorted(unknown)}")
        result.update(config)
    if result != DEFAULT_FEATURE_CONFIG:
        raise ValueError("特徴量設定が仕様版と一致しません。設定を確認し、再学習してください。")
    return result


def load_audio(source: Path | str | bytes) -> AudioClip:
    """最大 10 MiB の WAV を検証し、原振幅の先頭 ch を返す。"""
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.stat().st_size > MAX_AUDIO_BYTES:
                raise InvalidAudio("WAV の上限は 10 MiB です。")
            raw = path.read_bytes()
        elif isinstance(source, bytes):
            raw = source
        else:
            raise InvalidAudio("WAV のパスまたはバイト列を指定してください。")
        if not raw or len(raw) > MAX_AUDIO_BYTES:
            raise InvalidAudio("WAV が空、または 10 MiB を超えています。")
        with sf.SoundFile(io.BytesIO(raw)) as stream:
            if stream.format not in {"WAV", "WAVEX"}:
                raise InvalidAudio("WAV 形式だけに対応しています。")
            sample_rate, channels, frames = stream.samplerate, stream.channels, stream.frames
            if sample_rate != 16000:
                raise InvalidAudio("16 kHz だけに対応しています。自動変換は行いません。")
            if channels not in (1, 8):
                raise InvalidAudio("1 または 8 チャンネルだけに対応しています。")
            if not 8 * sample_rate <= frames <= 12 * sample_rate:
                raise InvalidAudio("音声の長さは 8～12 秒にしてください。")
            all_channels = stream.read(dtype="float32", always_2d=True)
        if all_channels.shape != (frames, channels) or not np.isfinite(all_channels).all():
            raise InvalidAudio("音声に非有限値または不整合があります。")
        samples = np.ascontiguousarray(all_channels[:, 0], dtype="<f4")
        if not np.any(samples):
            raise InvalidAudio("選択チャンネルが完全無音のため判定できません。")
        return AudioClip(
            samples=samples,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=frames / sample_rate,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            selected_channel_sha256=hashlib.sha256(samples.tobytes(order="C")).hexdigest(),
        )
    except InvalidAudio:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise InvalidAudio(f"WAV を読み込めません: {exc}") from exc


def extract_features(clip: AudioClip, feature_config: dict[str, Any] | None = None) -> FeatureSet:
    """学習・推論共通の 320 次元特徴量と、実音声内の文脈窓時刻。"""
    import librosa

    cfg = normalize_feature_config(feature_config)
    if clip.sample_rate != cfg["sample_rate"]:
        raise InvalidAudio("モデルと音声のサンプルレートが一致しません。")
    y = np.asarray(clip.samples)
    if y.ndim != 1 or not np.isfinite(y).all() or not np.any(y):
        raise InvalidAudio("選択音声の形状・値・無音を確認してください。")
    mel_power = librosa.feature.melspectrogram(
        y=y, sr=clip.sample_rate, n_fft=cfg["n_fft"],
        win_length=cfg["win_length"], hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"], power=cfg["power"], center=cfg["center"],
        window=cfg["window"], fmin=cfg["fmin"], fmax=cfg["fmax"],
        htk=cfg["htk"], norm=cfg["norm"],
    )
    log_mel = (10.0 * np.log10(np.maximum(mel_power, cfg["log_floor"]))).astype(np.float32)
    count = log_mel.shape[1] - cfg["context_frames"] + 1
    if count < 1:
        raise InvalidAudio("文脈窓を作るための音声が不足しています。")
    vectors = np.concatenate(
        [log_mel[:, offset:offset + count].T for offset in range(cfg["context_frames"])],
        axis=1,
    )
    starts = np.arange(count, dtype=np.float64) * cfg["hop_length"] / clip.sample_rate
    ends = starts + ((cfg["context_frames"] - 1) * cfg["hop_length"] + cfg["n_fft"]) / clip.sample_rate
    if not np.isfinite(vectors).all() or ends[-1] > len(y) / clip.sample_rate + 1e-12:
        raise InvalidAudio("特徴量または文脈窓の時刻に不整合があります。")
    return FeatureSet(np.ascontiguousarray(vectors), log_mel, starts, ends)


def playback_wav(clip: AudioClip) -> bytes:
    """選択した音だけを元の振幅で再生する PCM WAV。"""
    output = io.BytesIO()
    # FLOAT avoids clipping or quantization of valid floating-point WAV inputs.
    sf.write(output, clip.samples, clip.sample_rate, format="WAV", subtype="FLOAT")
    return output.getvalue()

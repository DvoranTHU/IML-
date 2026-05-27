import librosa
import numpy as np

from src.config import load_config

# 最大值归一化到0.99
def normalize_peak(y, peak=0.99):
    max_val = float(np.max(np.abs(y)))
    if max_val == 0.0:
        return y
    return y * (peak / max_val)

# 均方根归一化到0.1
def normalize_rms(y, target_rms=0.1):
    rms = float(np.sqrt(np.mean(np.square(y))))
    if rms == 0.0:
        return y
    return y * (target_rms / rms)

# 加载并预处理音频，返回波形和采样率
def load_audio(path, cfg=None):
    if cfg is None:
        cfg = load_config()
    audio_cfg = cfg["audio"]
    sr = int(audio_cfg["sample_rate"])
    mono = bool(audio_cfg.get("mono", True))

    y, _ = librosa.load(path, sr=sr, mono=mono)

    norm = str(audio_cfg.get("normalize", "peak")).lower()
    if norm == "peak":
        y = normalize_peak(y, float(audio_cfg.get("peak", 0.99)))
    elif norm == "rms":
        y = normalize_rms(y, float(audio_cfg.get("target_rms", 0.1)))
    else:
        raise ValueError()

    return y.astype(np.float32), sr

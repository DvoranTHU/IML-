import numpy as np

from src.audio.load import load_audio
from src.config import load_config
from src.features.aggregate import N_FUNCTIONALS, aggregate_frames
from src.features.frame import extract_frame_features, n_frame_features

# 提取单条语音的特征向量
def extract_audio(path, cfg=None):
    if cfg is None:
        cfg = load_config()
    y, sr = load_audio(path, cfg=cfg)
    frames, _ = extract_frame_features(y, sr, cfg=cfg)
    vec = aggregate_frames(frames)
    washed_vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return washed_vec

# 返回语音特征向量的维度
def audio_feature_dim(cfg=None):
    if cfg is None:
        cfg = load_config()
    return n_frame_features(cfg) * N_FUNCTIONALS # 472

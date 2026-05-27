import numpy as np
from scipy import stats

FUNCTIONAL_NAMES = ("mean", "std", "min", "max", "skew", "kurtosis", "p25", "p75")
N_FUNCTIONALS = len(FUNCTIONAL_NAMES)

def safe_skew(x):
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    return float(stats.skew(x, bias=False, nan_policy="omit"))

def safe_kurtosis(x):
    x = x[np.isfinite(x)]
    if x.size < 4:
        return 0.0
    return float(stats.kurtosis(x, bias=False, nan_policy="omit"))

# 输入某个特征维度上，按帧切分得到的特征序列 [n_frames]
# 返回该特征维度的统计量 [8]
def functional_vector(frame_series):
    x = np.asarray(frame_series, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(N_FUNCTIONALS, dtype=np.float32)

    return np.array(
        [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.min(x)),
            float(np.max(x)),
            safe_skew(x),
            safe_kurtosis(x),
            float(np.percentile(x, 25)),
            float(np.percentile(x, 75)),
        ],
        dtype=np.float32,
    )

# 输入单个语音按帧切分得到的特征矩阵 [n_frames, feature_dim]
# 返回单个语音特征向量 [feature_dim * 8]
def aggregate_frames(frame_matrix):
    n_feats = frame_matrix.shape[1] # feature_dim
    out = np.zeros(n_feats * N_FUNCTIONALS, dtype=np.float32) # [feature_dim * 8]
    for j in range(n_feats):
        start = j * N_FUNCTIONALS
        out[start : start + N_FUNCTIONALS] = functional_vector(frame_matrix[:, j]) 
    return out # [feature_dim * 8]
 
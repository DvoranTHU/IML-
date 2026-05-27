import librosa
import numpy as np

from src.config import load_config

MFCC_NAMES = [f"mfcc_{i}" for i in range(13)]
MFCC_DELTA_NAMES = [f"mfcc_delta_{i}" for i in range(13)]
MFCC_DELTA2_NAMES = [f"mfcc_delta2_{i}" for i in range(13)]

PROSODY_NAMES = [
    "f0",
    "voiced_prob",
    "energy",
    "log_energy",
    "zcr",
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_bandwidth",
    "spectral_flatness",
    "hnr",
    "spectral_flux",
    "jitter_local",
    "shimmer_local",
    "harmonic_energy",
    "noise_energy",
    "spectral_entropy",
    "f0_delta",
    "energy_delta",
    "zcr_delta",
    "flux_delta",
]

N_PROSODY = len(PROSODY_NAMES)

# 获取采样率、fft窗口长度、hop长度
def frame_params(cfg):
    sr = int(cfg["audio"]["sample_rate"])
    feat = cfg["features"]
    n_fft = int(sr * feat["frame_length_ms"] / 1000)
    hop_length = int(sr * feat["hop_length_ms"] / 1000)
    return sr, n_fft, hop_length

# 20个prosody特征序列，按长度对齐，不足的补nan，超过截断
def align_length(arrays, length):
    out = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if arr.size < length:
            pad = np.full(length - arr.size, np.nan, dtype=np.float64)
            arr = np.concatenate([arr, pad])
        else:
            arr = arr[:length]
        out.append(arr)
    return out

# 计算频率抖动和幅度抖动
def local_jitter_shimmer(f0, amp):
    jitter = np.abs(np.diff(f0, prepend=np.nan))
    shimmer = np.abs(np.diff(amp, prepend=np.nan))
    return jitter, shimmer

# 计算谱熵
def spectral_entropy_per_frame(mag, eps=1e-10):
    p = mag / (np.sum(mag, axis=0, keepdims=True) + eps)
    ent = -np.sum(p * np.log(p + eps), axis=0)
    return ent

# 计算谐波失真比
def hnr_from_hpss(harm, perc, eps=1e-10):
    h = harm ** 2
    n = perc ** 2
    return 10.0 * np.log10((h + eps) / (n + eps))

# 获取单个语音按帧切分后得到的特征矩阵 [n_frames, feature_dim]
# 获取特征名列表 [feature_dim]
def extract_frame_features(y, sr, cfg=None):
    if cfg is None:
        cfg = load_config()
    _, n_fft, hop_length = frame_params(cfg)
    feat_cfg = cfg["features"]
    n_mfcc = int(feat_cfg.get("n_mfcc", 13))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length) # [n_mfcc, n_frames]
    if feat_cfg.get("include_delta", True):
        mfcc_delta = librosa.feature.delta(mfcc) # [n_mfcc, n_frames]
    else:
        mfcc_delta = np.zeros_like(mfcc)
    if feat_cfg.get("include_delta_delta", True):
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2) # [n_mfcc, n_frames]
    else:
        mfcc_delta2 = np.zeros_like(mfcc)

    n_frames = mfcc.shape[1]

    fmin = float(feat_cfg.get("f0_fmin_hz", 80))
    fmax = float(feat_cfg.get("f0_fmax_hz", 400))
    f0 = librosa.yin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=n_fft,
        hop_length=hop_length,
    )
    voiced_prob = (np.isfinite(f0) & (f0 > 0)).astype(np.float64)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=n_fft, hop_length=hop_length
    )[0]
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=n_fft, hop_length=hop_length
    )[0]

    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(S, axis=1, prepend=S[:, :1]) ** 2, axis=0))

    y_harm, y_perc = librosa.effects.hpss(y)
    harm_rms = librosa.feature.rms(
        y=y_harm, frame_length=n_fft, hop_length=hop_length
    )[0]
    noise_rms = librosa.feature.rms(
        y=y_perc, frame_length=n_fft, hop_length=hop_length
    )[0]
    hnr = hnr_from_hpss(harm_rms, noise_rms)
    spec_ent = spectral_entropy_per_frame(S)

    jitter, shimmer = local_jitter_shimmer(f0, rms)
    f0_delta = np.abs(np.diff(f0, prepend=np.nan))
    energy_delta = np.abs(np.diff(rms, prepend=np.nan))
    zcr_delta = np.abs(np.diff(zcr, prepend=np.nan))
    flux_delta = np.abs(np.diff(flux, prepend=np.nan))

    log_energy = np.log10(rms + 1e-10)

    prosody = align_length(
        [
            f0,
            voiced_prob,
            rms,
            log_energy,
            zcr,
            centroid,
            rolloff,
            bandwidth,
            flatness,
            hnr,
            flux,
            jitter,
            shimmer,
            harm_rms,
            noise_rms,
            spec_ent,
            f0_delta,
            energy_delta,
            zcr_delta,
            flux_delta,
        ],
        n_frames,
    )

    blocks = [mfcc.T, mfcc_delta.T, mfcc_delta2.T, np.column_stack(prosody)] # [n_frames, feature_dim]
    names = (
        MFCC_NAMES
        + MFCC_DELTA_NAMES
        + MFCC_DELTA2_NAMES
        + PROSODY_NAMES
    )
    frame_matrix = np.column_stack(blocks).astype(np.float32)
    return frame_matrix, names # [n_frames, feature_dim], [feature_dim]

# 返回每帧特征维度，3*13+20=59
def n_frame_features(cfg=None):
    if cfg is None:
        cfg = load_config()
    n_mfcc = int(cfg["features"].get("n_mfcc", 13))
    n = n_mfcc
    if cfg["features"].get("include_delta", True):
        n += n_mfcc
    if cfg["features"].get("include_delta_delta", True):
        n += n_mfcc
    return n + N_PROSODY

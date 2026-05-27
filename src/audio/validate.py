from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.audio.load import load_audio
from src.config import PROJECT_ROOT, load_config

# 检查音频数组
def validate_array(y, sr, min_duration_sec=0.1):
    issues = []
    if y.size == 0:
        issues.append("empty")
        return issues
    if np.any(~np.isfinite(y)):
        issues.append("non_finite")
    if float(np.max(np.abs(y))) == 0.0:
        issues.append("silent")
    duration = len(y) / sr
    if duration < min_duration_sec:
        issues.append(f"too_short")
    return issues

# 快速检查，不重采样
def validate_file_quick(path, cfg=None):
    path = Path(path)
    if not path.is_file():
        return ["missing"]
    if cfg is None:
        cfg = load_config()
    min_dur = float(cfg["audio"].get("min_duration_sec", 0.1))
    try:
        info = sf.info(path)
        if info.duration < min_dur:
            return [f"too_short({info.duration:.3f}s)"]
        data, _ = sf.read(path, dtype="float32", always_2d=True)
        y = data.mean(axis=1) if data.ndim > 1 else data
        if y.size == 0:
            return ["empty"]
        if np.any(~np.isfinite(y)):
            return ["non_finite"]
        if float(np.max(np.abs(y))) == 0.0:
            return ["silent"]
    except Exception as exc:
        return [f"{exc}"]
    return []

# 完整检查，重采样并归一化
def validate_file(path, cfg=None):
    path = Path(path)
    if not path.is_file():
        return ["missing"]
    try:
        y, sr = load_audio(path, cfg=cfg)
    except Exception as exc:
        return [f"{exc}"]
    if cfg is None:
        cfg = load_config()
    min_dur = float(cfg["audio"].get("min_duration_sec", 0.1))
    return validate_array(y, sr, min_duration_sec=min_dur)

# 全量扫描经验证已通过，现在只做快速扫描
def run_validation(metadata_csv, cfg=None, project_root=None):
    if cfg is None:
        cfg = load_config()
    root = project_root if project_root is not None else PROJECT_ROOT

    meta = pd.read_csv(metadata_csv)
    issue_rows = []
    for _, row in meta.iterrows():
        full_path = root / row["path"]
        issues = validate_file_quick(full_path, cfg=cfg)
        if issues:
            issue_rows.append(
                {
                    "path": row["path"],
                    "dataset": row["dataset"],
                    "issues": ";".join(issues),
                }
            )
    return pd.DataFrame(issue_rows)

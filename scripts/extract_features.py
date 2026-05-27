import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT as ROOT, load_config, resolve_path
from src.features import extract_audio, audio_feature_dim

# 加载检验失败音频的路径
def exclude_path(issues_csv):
    if not issues_csv.is_file():
        return set()
    return set(pd.read_csv(issues_csv)["path"].astype(str))

# 所有数据集，所有合法语音，提取特征并保存
def extract_dataset(meta, dataset, cfg, exclude):
    sub = meta[meta["dataset"] == dataset].copy()
    sub = sub[~sub["path"].isin(exclude)]
    dim = audio_feature_dim(cfg)
    n = len(sub)
    
    vectors = [] # 特征向量
    paths = [] # 文件相对路径
    emotions = [] # 情感标签
    actor_ids = [] # 说话人ID
    failed = [] # 提取失败的语音

    for row in tqdm(sub.itertuples(), total=n, desc=dataset):
        full = ROOT / row.path
        try:
            cur_feature = extract_audio(full, cfg=cfg)
            vectors.append(cur_feature)
            paths.append(row.path)
            emotions.append(row.emotion)
            actor_ids.append(int(row.actor_id))    
        except Exception as exc:
            failed.append(f"{exc}")

    X = np.stack(vectors, axis=0).astype(np.float32) # [n, 472]
    
    if failed:
        fail_path = resolve_path(cfg["outputs"]["features"]) / f"{dataset}_failed.txt"
        fail_path.write_text("\n".join(failed) + "\n", encoding="utf-8")

    return {
        "features": X,
        "paths": np.array(paths, dtype=object),
        "emotion": np.array(emotions, dtype=object),
        "actor_id": np.array(actor_ids, dtype=np.int32),
        "feature_dim": dim,
        "dataset": dataset,
    }


def main():
    cfg = load_config()
    meta_path = resolve_path(cfg["outputs"]["metadata"]) / "dataset.csv"
    issues_path = resolve_path(cfg["outputs"]["metadata"]) / "audio_issues.csv"
    out_dir = resolve_path(cfg["outputs"]["features"])

    meta = pd.read_csv(meta_path)
    exclude = exclude_path(issues_path)

    for dataset in ("ravdess", "crema_d", "emodb"):
        bundle = extract_dataset(meta, dataset, cfg, exclude)
        out_path = out_dir / f"{dataset}.npz"
        np.savez_compressed(out_path, **bundle)

if __name__ == "__main__":
    main()

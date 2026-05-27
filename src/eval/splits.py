from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import load_config, resolve_path

# EMODB原始标签map到跨语料统一标签
EMODB_CROSS_EMOTION_MAP = {
    "anger": "angry",
    "fear": "fearful",
    "happiness": "happy",
    "sadness": "sad",
    "neutral": "neutral",
}

# 划分数据集
@dataclass
class Split:
    name: str
    protocol: str
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

# 加载元数据
def load_metadata(cfg=None):
    if cfg is None:
        cfg = load_config()
    path = resolve_path(cfg["outputs"]["metadata"]) / "dataset.csv"
    return pd.read_csv(path)

# 获取评估配置
def eval_cfg(cfg):
    return cfg.get("eval", {})

# 获取种子
def seed(cfg):
    return int(cfg.get("seed", 42))

# 获取跨语料分类
def cross_corpus_classes(cfg=None):
    if cfg is None:
        cfg = load_config()
    classes = eval_cfg(cfg).get("cross_corpus_classes")
    if classes is None:
        return ["angry", "happy", "sad", "neutral", "fearful"]
    return list(classes)

# 在跨语料实验里，把不同数据集的情感标签改成同一套名字
def normalize_emotion_for_cross_corpus(emotions, dataset):
    out = []
    for emo in np.asarray(emotions):
        label = str(emo)
        if dataset == "emodb" and label in EMODB_CROSS_EMOTION_MAP:
            label = EMODB_CROSS_EMOTION_MAP[label]
        out.append(label)
    return np.array(out, dtype=object)

# 根据情感标签过滤数据
def filter_by_emotions(df, classes, dataset=None, emotion_col="emotion"):
    sub = df.copy()
    ds = dataset if dataset is not None else sub["dataset"].iloc[0]
    sub["_emotion_eval"] = normalize_emotion_for_cross_corpus(sub[emotion_col], ds)
    mask = sub["_emotion_eval"].isin(classes)
    return sub.loc[mask].reset_index(drop=True)

# 返回RAVDESS的5折Split划分
def ravdess_speaker_folds(meta, cfg=None):
    if cfg is None:
        cfg = load_config()
    ev = eval_cfg(cfg)
    n_folds = int(ev.get("ravdess_n_folds", 5))

    sub = meta[meta["dataset"] == "ravdess"].reset_index(drop=True)
    groups = sub["actor_id"].to_numpy()
    indices = np.arange(len(sub))

    # 按演员整组划分，确保每个演员在同一折中
    gkf = GroupKFold(n_splits=n_folds)
    splits: list[Split] = []
    for fold, (train_pos, test_pos) in enumerate(gkf.split(indices, groups=groups)):
        train_actors = sorted(set(groups[train_pos]))
        test_actors = sorted(set(groups[test_pos]))
        splits.append(
            Split(
                name=f"ravdess_fold_{fold}",
                protocol="ravdess_loso",
                fold=fold,
                train_idx=indices[train_pos],
                test_idx=indices[test_pos],
                meta={
                    "n_train": int(len(train_pos)),
                    "n_test": int(len(test_pos)),
                    "train_actors": train_actors,
                    "test_actors": test_actors,
                },
            )
        )
    return splits

# 跨语料实验的Split划分
@dataclass
class CrossCorpusSplit:
    classes: list[str]
    train: Split
    test_crema: Split
    test_emodb: Split

# 仅保留已在特征缓存中的样本
def restrict_to_feature_cache(df, dataset, cfg):
    feat_path = resolve_path(cfg["outputs"]["features"]) / f"{dataset}.npz"
    cached = np.load(feat_path, allow_pickle=True)["paths"]
    valid = set(cached.tolist())
    return df[df["path"].isin(valid)].reset_index(drop=True)

# 跨语料实验的Split划分
def cross_corpus_split(meta, cfg=None):
    if cfg is None:
        cfg = load_config()
    classes = cross_corpus_classes(cfg)

    rav = meta[meta["dataset"] == "ravdess"].reset_index(drop=True)
    rav_f = restrict_to_feature_cache(filter_by_emotions(rav, classes, dataset="ravdess"), "ravdess", cfg)
    cre = meta[meta["dataset"] == "crema_d"].reset_index(drop=True)
    cre_f = restrict_to_feature_cache(filter_by_emotions(cre, classes, dataset="crema_d"), "crema_d", cfg)
    emo = meta[meta["dataset"] == "emodb"].reset_index(drop=True)
    emo_f = restrict_to_feature_cache(filter_by_emotions(emo, classes, dataset="emodb"), "emodb", cfg)

    def meta(df):
        return {
            "n_samples": len(df),
            "classes": classes,
            "emotions": sorted(df["_emotion_eval"].unique().tolist()),
            "source_df": df,
        }

    train_ravdess = Split(
        name="cross_corpus_ravdess_train",
        protocol="cross_corpus_train",
        fold=0,
        train_idx=np.arange(len(rav_f)),
        test_idx=np.array([], dtype=int),
        meta=meta(rav_f),
    )
    test_crema = Split(
        name="cross_corpus_crema_d_test",
        protocol="cross_corpus_test",
        fold=0,
        train_idx=np.array([], dtype=int),
        test_idx=np.arange(len(cre_f)),
        meta=meta(cre_f),
    )
    test_emodb = Split(
        name="cross_corpus_emodb_test",
        protocol="cross_corpus_test",
        fold=0,
        train_idx=np.array([], dtype=int),
        test_idx=np.arange(len(emo_f)),
        meta=meta(emo_f),
    )

    return CrossCorpusSplit(classes=classes, train=train_ravdess, test_crema_d=test_crema, test_emodb=test_emodb)

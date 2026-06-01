import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.config import load_config, resolve_path
from src.eval.metrics import compute_metrics, timing_summary
from src.eval.utils import bootstrap_ci, fold_mean_std
from src.eval.splits import (
    CrossCorpusSplit,
    Split,
    cross_corpus_split,
    load_metadata,
    ravdess_speaker_folds,
)


# 预测模型类型
# 参数：训练集特征、训练集标签、测试集特征、测试集标签、标签列表
# 返回：预测标签、训练时间、预测时间
FitPredictFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]],
    tuple[np.ndarray, float, float],
]

# 一折的评估结果
@dataclass
class FoldEvalResult:
    split_name: str
    protocol: str
    fold: int
    metrics: dict[str, Any]
    timing: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)

# 评估器
class EvalRunner:
    def __init__(self, cfg=None):
        self.cfg = load_config() if cfg is None else cfg
        ev = self.cfg.get("eval", {})
        self.bootstrap_n = int(ev.get("bootstrap_n", 1000))
        self.ci_level = float(ev.get("ci_level", 0.95))
        self.seed = int(self.cfg.get("seed", 42))

    # 评估一折
    def evaluate_split(self, X, y, split, fit_predict: FitPredictFn, labels=None):
        train_idx = split.train_idx
        test_idx = split.test_idx
        if labels is None:
            labels = sorted(set(y[train_idx]) | set(y[test_idx]), key=str)

        # 训练并预测
        y_pred, train_sec, pred_sec = fit_predict(
            X[train_idx],
            y[train_idx],
            X[test_idx],
            y[test_idx],
            labels,
        )
        # 计算指标
        metrics = compute_metrics(y[test_idx], y_pred, labels=labels)
        return FoldEvalResult(
            split_name=split.name,
            protocol=split.protocol,
            fold=split.fold,
            metrics=metrics,
            timing=timing_summary(train_sec, pred_sec),
            meta=dict(split.meta),
        )

    # 评估多个折
    def run_splits(self, X, y, splits, fit_predict: FitPredictFn, labels=None):
        return [self.evaluate_split(X, y, sp, fit_predict, labels=labels) for sp in splits]

    # 汇总评估结果，主指标为uar
    def summarize(self, fold_results, metric_key="uar"):
        scores = [float(r.metrics[metric_key]) for r in fold_results]
        summary = fold_mean_std(scores)
        summary["metric"] = metric_key
        summary["ci"] = bootstrap_ci(
            scores,
            n_bootstrap=self.bootstrap_n,
            ci_level=self.ci_level,
            seed=self.seed,
        )
        summary["folds"] = [
            {
                "split_name": r.split_name,
                "fold": r.fold,
                metric_key: r.metrics[metric_key],
                "accuracy": r.metrics["accuracy"],
                "macro_f1": r.metrics["macro_f1"],
            }
            for r in fold_results
        ]
        return summary

    # 保存汇总结果
    def save_summary(self, summary, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

# 读取特征npz
def load_feature_bundle(dataset, cfg=None):
    if cfg is None:
        cfg = load_config()
    path = resolve_path(cfg["outputs"]["features"]) / f"{dataset}.npz"
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}

# 按npz中paths顺序对齐metadata子集
def align_meta_to_features(meta, paths, dataset):
    sub = meta[meta["dataset"] == dataset].set_index("path")
    ordered = sub.loc[list(paths)].reset_index()
    return ordered

# 加载RAVDESS训练集和测试集特征
# 返回特征矩阵、标签向量、5个折的Split
def ravdess_cv_arrays(cfg=None):
    if cfg is None:
        cfg = load_config()
    meta = load_metadata(cfg)
    bundle = load_feature_bundle("ravdess", cfg)
    aligned = align_meta_to_features(meta, bundle["paths"], "ravdess")
    X = bundle["features"]
    y = aligned["emotion"].to_numpy()
    
    # 构建5个折的Split
    splits = ravdess_speaker_folds(meta[meta["dataset"] == "ravdess"].reset_index(drop=True), cfg)
    path_to_row = {p: i for i, p in enumerate(bundle["paths"])}
    meta_rav = meta[meta["dataset"] == "ravdess"].reset_index(drop=True)
    remapped: list[Split] = []
    for sp in splits:
        train_paths = meta_rav.iloc[sp.train_idx]["path"].tolist()
        test_paths = meta_rav.iloc[sp.test_idx]["path"].tolist()
        remapped.append(
            Split(
                name=sp.name,
                protocol=sp.protocol,
                fold=sp.fold,
                train_idx=np.array([path_to_row[p] for p in train_paths], dtype=int),
                test_idx=np.array([path_to_row[p] for p in test_paths], dtype=int),
                meta=sp.meta,
            )
        )
    return X, y, remapped


# 加载RAVDESS训练集和CREMA-D、EMODB测试集特征
# 返回特征矩阵、标签向量、5个折的CrossCorpusSplit
def cross_corpus_arrays(cfg=None):
    if cfg is None:
        cfg = load_config()
    meta = load_metadata(cfg)
    cc = cross_corpus_split(meta, cfg)

    def _pack(dataset, df):
        bundle = load_feature_bundle(dataset, cfg)
        path_to_row = {p: i for i, p in enumerate(bundle["paths"])}
        rows = [path_to_row[p] for p in df["path"]]
        X = bundle["features"][rows]
        y = df["_emotion_eval"].to_numpy()
        return X, y

    rav_df = cc.train.meta["source_df"]
    cre_df = cc.test_crema_d.meta["source_df"]
    emo_df = cc.test_emodb.meta["source_df"]
    X_train, y_train = _pack("ravdess", rav_df)
    X_cre, y_cre = _pack("crema_d", cre_df)
    X_emo, y_emo = _pack("emodb", emo_df)
    return X_train, y_train, X_cre, y_cre, X_emo, y_emo, cc

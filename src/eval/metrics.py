import time
from contextlib import contextmanager

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

# 计算指标
def compute_metrics(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred), key=str)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "uar": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "labels": list(labels),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }

# 记录代码块耗时
@contextmanager
def timed_block():
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    timed_block.last_seconds = elapsed

# 返回训练时间、预测时间、总时间
def timing_summary(train_seconds, predict_seconds):
    return {
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
        "total_seconds": float(train_seconds + predict_seconds),
    }

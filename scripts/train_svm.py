import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.eval import EvalRunner, ravdess_cv_arrays
from src.models import KernelSVMClassifier, svm_cfg


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj

# 核SVM分类器5折交叉验证，跑单个核函数
def run_kernel_cv(kernel, X, y, splits, cfg):
    runner = EvalRunner(cfg)
    fold_details = []

    # 训练和预测函数，返回预测结果，训练时间，预测时间
    def fit_predict(X_tr, y_tr, X_te, y_te, labels):
        t0 = time.perf_counter()
        clf = KernelSVMClassifier(
            kernel=kernel, cfg=cfg, random_state=int(cfg.get("seed", 42))
        )
        clf.fit(X_tr, y_tr)
        train_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = clf.predict(X_te)
        pred_sec = time.perf_counter() - t1

        fit_predict.best_params = dict(clf.best_params_)
        return y_pred, train_sec, pred_sec

    results = []
    for sp in splits:
        res = runner.evaluate_split(X, y, sp, fit_predict)
        results.append(res)
        fold_details.append(
            {
                "split_name": res.split_name,
                "fold": res.fold,
                "best_params": fit_predict.best_params,
                "metrics": res.metrics,
                "timing": res.timing,
                "meta": res.meta,
            }
        )

    summary = runner.summarize(results, metric_key="uar")
    return {"kernel": kernel, "summary": summary, "folds": fold_details}


def main():
    cfg = load_config()
    X, y, splits = ravdess_cv_arrays(cfg)
    kernels = list(svm_cfg(cfg).get("kernels", ["linear", "rbf", "poly"]))

    by_kernel = {}
    for kernel in kernels:
        by_kernel[kernel] = run_kernel_cv(kernel, X, y, splits, cfg)

    best_kernel = max(kernels, key=lambda k: by_kernel[k]["summary"]["mean"])
    out = {
        "model": "kernel_svm",
        "dataset": "ravdess",
        "protocol": "ravdess_5fold",
        "kernels": by_kernel,
        "best_kernel": best_kernel,
        "best_summary": by_kernel[best_kernel]["summary"],
    }

    out_path = resolve_path(cfg["outputs"]["metrics"]) / "svm_ravdess.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(out), f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

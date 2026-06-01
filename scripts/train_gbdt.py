import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.eval import EvalRunner, ravdess_cv_arrays
from src.models import GBDTClassifier, gbdt_cfg
from src.models.gbdt import params_from_cfg

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

# 返回重要性最高的n个特征
def top_importance(importance, n_top=30):
    idx = np.argsort(importance)[::-1][:n_top]
    return [
        {"rank": r + 1, "feature_index": int(i), "importance": float(importance[i])}
        for r, i in enumerate(idx)
    ]

def mean_fold_importance(fold_importances):
    stacked = np.stack(fold_importances, axis=0)
    return stacked.mean(axis=0)

# 绘制重要性图
def maybe_plot_importance(importance, out_path, n_top=30):
    idx = np.argsort(importance)[::-1][:n_top]
    values = importance[idx]
    labels = [f"f{i}" for i in idx]

    fig, ax = plt.subplots(figsize=(8, max(4, n_top * 0.2)))
    ax.barh(range(n_top), values[::-1], color="steelblue")
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("gain importance (mean over folds)")
    ax.set_title("LightGBM top feature importances")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True

# 加载折的UAR
def load_fold_uar(metrics_path):
    if not metrics_path.exists():
        return None
    with metrics_path.open(encoding="utf-8") as f:
        data = json.load(f)
    folds = data.get("summary", data.get("best_summary", {})).get("folds")
    if folds is None and "kernels" in data:
        best = data.get("best_kernel")
        folds = data["kernels"][best]["summary"]["folds"]
    if folds is None:
        folds = data.get("folds", [])
    return [float(x["uar"]) for x in folds]

def main():
    cfg = load_config()
    cfg_gbdt = gbdt_cfg(cfg)
    run_params = params_from_cfg(cfg_gbdt)

    X, y, splits = ravdess_cv_arrays(cfg)
    runner = EvalRunner(cfg)
    fold_details = []
    fold_importances = []

    # 训练和预测函数，返回预测结果，训练时间，预测时间
    def fit_predict(X_tr, y_tr, X_te, y_te, labels):
        t0 = time.perf_counter()
        clf = GBDTClassifier(cfg=cfg, random_state=int(cfg.get("seed", 42)))
        clf.fit(X_tr, y_tr)
        train_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = clf.predict(X_te)
        pred_sec = time.perf_counter() - t1

        fit_predict.params = dict(clf.params_)
        fit_predict.importance = clf.feature_importances_
        return y_pred, train_sec, pred_sec

    results = []
    bar = tqdm(splits, desc="GBDT 5-fold", unit="fold")
    for sp in bar:
        res = runner.evaluate_split(X, y, sp, fit_predict)
        results.append(res)
        fold_importances.append(fit_predict.importance.copy())
        fold_details.append(
            {
                "split_name": res.split_name,
                "fold": res.fold,
                "params": fit_predict.params,
                "top_importance": top_importance(fit_predict.importance),
                "metrics": res.metrics,
                "timing": res.timing,
                "meta": res.meta,
            }
        )
        bar.set_postfix(fold=sp.fold, uar=f"{res.metrics['uar']:.3f}")

    summary = runner.summarize(results, metric_key="uar")
    mean_imp = mean_fold_importance(fold_importances)

    metrics_dir = resolve_path(cfg["outputs"]["metrics"])
    figures_dir = resolve_path(cfg["outputs"].get("figures", "outputs/figures"))
    fig_path = figures_dir / "gbdt_importance_ravdess.png"
    plotted = maybe_plot_importance(mean_imp, fig_path)

    out = {
        "model": "lightgbm_gbdt",
        "dataset": "ravdess",
        "protocol": "ravdess_loso_5fold",
        "params": run_params,
        "feature_dim": int(X.shape[1]),
        "summary": summary,
        "mean_feature_importance": {
            "top": top_importance(mean_imp),
            "all": mean_imp.tolist(),
        },
        "importance_figure": str(fig_path) if plotted else None,
        "folds": fold_details,
    }

    out_path = metrics_dir / "gbdt_ravdess.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(out), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

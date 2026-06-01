import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.eval import EvalRunner, load_feature_bundle, ravdess_cv_arrays
from src.models import (
    GBDTClassifier,
    GMMMAPClassifier,
    KernelSVMClassifier,
    gmm_cfg,
    stacking_cfg,
)
from src.models.stacking import StackingClassifier, align_proba, stack_meta_features

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

# 在训练折内按演员留验证集，选UAR最高的K，与训练gmm逻辑相同
def select_n_components(X, y, actors, k_list, params, seed, val_actor_ratio=0.2):
    y = np.asarray(y)
    actors = np.asarray(actors)
    uniq = np.unique(actors)
    rng = np.random.RandomState(int(seed))
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * float(val_actor_ratio)))
    val_set = set(uniq[:n_val])
    val_mask = np.array([a in val_set for a in actors], dtype=bool)
    tr_mask = ~val_mask
    labels = sorted(set(y[tr_mask]) | set(y[val_mask]), key=str)
    best_k, best_uar = k_list[0], -1.0
    for k in k_list:
        try:
            clf = GMMMAPClassifier(n_components=k, **params)
            clf.fit(X[tr_mask], y[tr_mask])
            pred = clf.predict(X[val_mask])
            uar = recall_score(
                y[val_mask], pred, average="macro", labels=labels, zero_division=0
            )
        except ValueError:
            continue
        if uar > best_uar:
            best_uar, best_k = float(uar), k
    return best_k

# 选最优k，训练gmm基模型，返回预测类别，预测概率，最优k
def gmm_predict_proba(X_tr, y_tr, X_pred, actors_tr, cfg, seed):
    gc = gmm_cfg(cfg)
    k_list = [int(k) for k in gc.get("n_components", [8, 16, 32, 64])]
    params = {
        "covariance_type": gc.get("covariance_type", "diag"),
        "max_iter": int(gc.get("max_iter", 200)),
        "tol": float(gc.get("tol", 1e-4)),
        "reg_covar": float(gc.get("reg_covar", 1e-6)),
        "random_state": int(cfg.get("seed", 42)),
    }
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_pred_s = scaler.transform(X_pred)
    best_k = select_n_components(
        X_tr_s,
        y_tr,
        actors_tr,
        k_list,
        params,
        seed,
        float(gc.get("val_actor_ratio", 0.2)),
    )
    clf = GMMMAPClassifier(n_components=best_k, **params)
    clf.fit(X_tr_s, y_tr)
    classes, proba = clf.predict_proba(X_pred_s)
    return classes, proba, best_k

# 训练svm基模型，返回预测类别，预测概率，最优参数
def svm_predict_proba(X_tr, y_tr, X_pred, cfg, kernel):
    clf = KernelSVMClassifier(
        kernel=kernel, cfg=cfg, random_state=int(cfg.get("seed", 42))
    )
    clf.fit(X_tr, y_tr)
    classes, proba = clf.predict_proba(X_pred)
    return classes, proba, dict(clf.best_params_)

# 训练gbdt基模型，返回预测类别，预测概率，最优参数
def gbdt_predict_proba(X_tr, y_tr, X_pred, cfg):
    clf = GBDTClassifier(cfg=cfg, random_state=int(cfg.get("seed", 42)))
    clf.fit(X_tr, y_tr)
    classes, proba = clf.predict_proba(X_pred)
    return classes, proba, dict(clf.params_)


def load_baseline_uar(metrics_dir):
    baselines = {}
    for name, path in [
        ("gmm_map", "gmm_ravdess.json"),
        ("kernel_svm", "svm_ravdess.json"),
        ("gbdt", "gbdt_ravdess.json"),
    ]:
        fp = metrics_dir / path
        if not fp.is_file():
            continue
        with fp.open(encoding="utf-8") as f:
            data = json.load(f)
        if name == "kernel_svm":
            summary = data.get("best_summary", data.get("summary", {}))
        else:
            summary = data.get("summary", {})
        baselines[name] = float(summary.get("mean", float("nan")))
    return baselines


def main():
    cfg = load_config()
    sc = stacking_cfg(cfg)
    svm_kernel = str(sc.get("svm_kernel", "rbf"))

    X_feat, y, splits = ravdess_cv_arrays(cfg)
    bundle = load_feature_bundle("ravdess", cfg)
    actors = bundle["actor_id"]
    class_order = sorted(set(y), key=str)
    n = len(y)
    n_classes = len(class_order)

    gmm = np.zeros((n, n_classes), dtype=np.float64)
    svm = np.zeros((n, n_classes), dtype=np.float64)
    gbdt = np.zeros((n, n_classes), dtype=np.float64)
    fold_meta = []

    for sp in tqdm(splits, desc="base models", unit="fold"):
        tr, te = sp.train_idx, sp.test_idx
        seed = int(cfg.get("seed", 42)) + int(sp.fold)

        gmm_cls, gmm_p, gmm_k = gmm_predict_proba(
            X_feat[tr], y[tr], X_feat[te], actors[tr], cfg, seed
        )
        gmm[te] = align_proba(gmm_cls, gmm_p, class_order)

        svm_cls, svm_p, svm_bp = svm_predict_proba(
            X_feat[tr], y[tr], X_feat[te], cfg, svm_kernel
        )
        svm[te] = align_proba(svm_cls, svm_p, class_order)

        gbdt_cls, gbdt_p, gbdt_pr = gbdt_predict_proba(X_feat[tr], y[tr], X_feat[te], cfg)
        gbdt[te] = align_proba(gbdt_cls, gbdt_p, class_order)

        fold_meta.append(
            {
                "fold": sp.fold,
                "split_name": sp.name,
                "gmm_n_components": gmm_k,
                "svm_best_params": svm_bp,
                "gbdt_params": gbdt_pr,
            }
        )

    # 拼接概率
    X_meta = stack_meta_features(
        [
            (class_order, gmm),
            (class_order, svm),
            (class_order, gbdt),
        ],
        class_order,
    )

    # 训练元学习器
    meta = StackingClassifier(cfg=cfg, random_state=int(cfg.get("seed", 42)))
    meta.fit(X_meta, y)

    runner = EvalRunner(cfg)
    fold_details = []

    # 训练和预测函数，返回预测类别，训练时间，预测时间
    def fit_predict(X_tr, y_tr, X_te, y_te, labels):
        fold = fit_predict._fold
        seed = int(cfg.get("seed", 42)) + fold
        actors_tr = fit_predict._actors_tr

        t0 = time.perf_counter()
        gmm_cls, gmm_p, _ = gmm_predict_proba(
            X_tr, y_tr, X_te, actors_tr, cfg, seed
        )
        svm_cls, svm_p, _ = svm_predict_proba(X_tr, y_tr, X_te, cfg, svm_kernel)
        gbdt_cls, gbdt_p, _ = gbdt_predict_proba(X_tr, y_tr, X_te, cfg)
        train_sec = time.perf_counter() - t0

        X_te_meta = stack_meta_features(
            [
                (class_order, align_proba(gmm_cls, gmm_p, class_order)),
                (class_order, align_proba(svm_cls, svm_p, class_order)),
                (class_order, align_proba(gbdt_cls, gbdt_p, class_order)),
            ],
            class_order,
        )

        t1 = time.perf_counter()
        y_pred = meta.predict(X_te_meta)
        pred_sec = time.perf_counter() - t1
        return y_pred, train_sec, pred_sec

    results = []
    bar = tqdm(splits, desc="Stacking eval", unit="fold")
    for sp in bar:
        fit_predict._fold = sp.fold
        fit_predict._actors_tr = actors[sp.train_idx]
        res = runner.evaluate_split(X_feat, y, sp, fit_predict)
        results.append(res)
        fold_details.append(
            {
                "split_name": res.split_name,
                "fold": res.fold,
                "metrics": res.metrics,
                "timing": res.timing,
            }
        )
        bar.set_postfix(fold=sp.fold, uar=f"{res.metrics['uar']:.3f}")

    summary = runner.summarize(results, metric_key="uar")
    metrics_dir = resolve_path(cfg["outputs"]["metrics"])
    baselines = load_baseline_uar(metrics_dir)

    out = {
        "model": "stacking_logistic",
        "dataset": "ravdess",
        "protocol": "ravdess_5fold",
        "base_models": ["gmm_map", f"kernel_svm_{svm_kernel}", "gbdt"],
        "meta_feature_dim": int(X_meta.shape[1]),
        "class_order": class_order,
        "summary": summary,
        "baseline_uar": baselines,
        "fold_meta": fold_meta,
        "folds": fold_details,
    }

    out_path = metrics_dir / "stacking_ravdess.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(out), f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

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
from src.eval import cross_corpus_arrays, load_feature_bundle
from src.eval.metrics import compute_metrics, timing_summary
from src.eval.splits import cross_corpus_classes
from src.models import (
    GBDTClassifier,
    GMMMAPClassifier,
    KernelSVMClassifier,
    gmm_cfg,
    stacking_cfg,
)

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

# 在训练折内按演员留验证集，选UAR最高的K
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

# 根据路径获取演员ID
def actors_for_paths(paths, bundle):
    path_to_actor = {p: int(a) for p, a in zip(bundle["paths"], bundle["actor_id"])}
    return np.array([path_to_actor[p] for p in paths], dtype=int)

# 加载基线模型UAR
def load_in_domain_uar(metrics_dir):
    out = {}

    gmm_path = metrics_dir / "gmm_ravdess.json"
    if gmm_path.is_file():
        with gmm_path.open(encoding="utf-8") as f:
            out["gmm_map"] = float(json.load(f)["summary"]["mean"])

    svm_path = metrics_dir / "svm_ravdess.json"
    if svm_path.is_file():
        with svm_path.open(encoding="utf-8") as f:
            data = json.load(f)
        best = data.get("best_kernel", "rbf")
        out["kernel_svm_rbf"] = float(data["kernels"][best]["summary"]["mean"])

    gbdt_path = metrics_dir / "gbdt_ravdess.json"
    if gbdt_path.is_file():
        with gbdt_path.open(encoding="utf-8") as f:
            out["lightgbm_gbdt"] = float(json.load(f)["summary"]["mean"])

    stack_path = metrics_dir / "stacking_ravdess.json"
    if stack_path.is_file():
        with stack_path.open(encoding="utf-8") as f:
            out["stacking_logistic"] = float(json.load(f)["summary"]["mean"])

    return out

# GMM模型训练和预测
def eval_gmm(X_tr, y_tr, X_te, y_te, labels, actors_tr, cfg):
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
    X_te_s = scaler.transform(X_te)

    t0 = time.perf_counter()
    best_k = select_n_components(
        X_tr_s,
        y_tr,
        actors_tr,
        k_list,
        params,
        int(cfg.get("seed", 42)),
        float(gc.get("val_actor_ratio", 0.2)),
    )
    clf = GMMMAPClassifier(n_components=best_k, **params)
    clf.fit(X_tr_s, y_tr)
    train_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = clf.predict(X_te_s)
    pred_sec = time.perf_counter() - t1

    return y_pred, train_sec, pred_sec, {"n_components": int(best_k)}

# SVM模型训练和预测
def eval_svm(X_tr, y_tr, X_te, y_te, labels, cfg, kernel):
    t0 = time.perf_counter()
    clf = KernelSVMClassifier(
        kernel=kernel, cfg=cfg, random_state=int(cfg.get("seed", 42))
    )
    clf.fit(X_tr, y_tr)
    train_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = clf.predict(X_te)
    pred_sec = time.perf_counter() - t1

    return y_pred, train_sec, pred_sec, {"best_params": dict(clf.best_params_)}

# GBDT模型训练和预测
def eval_gbdt(X_tr, y_tr, X_te, y_te, labels, cfg):
    t0 = time.perf_counter()
    clf = GBDTClassifier(cfg=cfg, random_state=int(cfg.get("seed", 42)))
    clf.fit(X_tr, y_tr)
    train_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = clf.predict(X_te)
    pred_sec = time.perf_counter() - t1

    return y_pred, train_sec, pred_sec, {"params": dict(clf.params_)}

# 跨语料实验
def run_test_corpus(test_dataset, X_tr, y_tr, X_te, y_te, labels, actors_tr, cfg, svm_kernel, in_domain, cc_meta):
    model_fns = [
        ("gmm_map", lambda: eval_gmm(X_tr, y_tr, X_te, y_te, labels, actors_tr, cfg)),
        (
            f"kernel_svm_{svm_kernel}",
            lambda: eval_svm(X_tr, y_tr, X_te, y_te, labels, cfg, svm_kernel),
        ),
        ("lightgbm_gbdt", lambda: eval_gbdt(X_tr, y_tr, X_te, y_te, labels, cfg)),
    ]

    models_out = {}
    domain_shift = {}

    for name, fn in tqdm(model_fns, desc=test_dataset, unit="model"):
        y_pred, train_sec, pred_sec, meta = fn()
        metrics = compute_metrics(y_te, y_pred, labels=labels)
        models_out[name] = {
            "metrics": metrics,
            "timing": timing_summary(train_sec, pred_sec),
            "meta": meta,
        }
        if name in in_domain:
            drop = float(in_domain[name]) - float(metrics["uar"])
            domain_shift[name] = {
                "in_domain_uar": float(in_domain[name]),
                "cross_corpus_uar": float(metrics["uar"]),
                "uar_drop": float(drop),
            }

    return {
        "model": "cross_corpus_eval",
        "protocol": "ravdess_train_cross_test",
        "train_dataset": "ravdess",
        "test_dataset": test_dataset,
        "classes": list(labels),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "train_class_counts": {
            str(c): int((y_tr == c).sum()) for c in labels
        },
        "test_class_counts": {
            str(c): int((y_te == c).sum()) for c in labels
        },
        "split_meta": {
            "n_samples": cc_meta.get("n_samples"),
            "emotions": cc_meta.get("emotions"),
        },
        "in_domain_baseline_uar": in_domain, # 基线模型UAR
        "domain_shift": domain_shift, # 跨语料UAR下降
        "models": models_out, # 跨语料模型UAR
    }

def main():
    cfg = load_config()
    svm_kernel = str(stacking_cfg(cfg).get("svm_kernel", "rbf"))
    labels = cross_corpus_classes(cfg) # 5种
    
    # 获取训练特征、训练标签，两个预料的测试特征、测试标签，交叉语料Split
    X_tr, y_tr, X_cre, y_cre, X_emo, y_emo, cc = cross_corpus_arrays(cfg)
    rav_paths = cc.train.meta["source_df"]["path"].tolist()
    bundle = load_feature_bundle("ravdess", cfg)
    actors_tr = actors_for_paths(rav_paths, bundle)

    metrics_dir = resolve_path(cfg["outputs"]["metrics"])
    in_domain = load_in_domain_uar(metrics_dir)

    outputs = [
        ("crema_d", X_cre, y_cre, cc.test_crema_d.meta),
        ("emodb", X_emo, y_emo, cc.test_emodb.meta),
    ]

    for test_dataset, X_te, y_te, split_meta in outputs:
        report = run_test_corpus(
            test_dataset,
            X_tr,
            y_tr,
            X_te,
            y_te,
            labels,
            actors_tr,
            cfg,
            svm_kernel,
            in_domain,
            split_meta,
        )
        out_path = metrics_dir / f"cross_corpus_{test_dataset}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(report), f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

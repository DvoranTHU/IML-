import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.eval import EvalRunner, load_feature_bundle, ravdess_cv_arrays
from src.models import GMMMAPClassifier, gmm_cfg


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
    k_list = list(k_list)
    params = dict(params)
    seed = int(seed)
    val_actor_ratio = float(val_actor_ratio)
    actors = np.asarray(actors)
    uniq = np.unique(actors)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_actor_ratio))
    val_set = set(uniq[:n_val])
    val_mask = np.array([a in val_set for a in actors], dtype=bool)
    tr_mask = ~val_mask

    labels = sorted(set(y[tr_mask]) | set(y[val_mask]), key=str)
    best_k = k_list[0]
    best_uar = -1.0

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
            best_uar = float(uar)
            best_k = k

    return best_k


def main() -> None:
    cfg = load_config()
    X, y, splits = ravdess_cv_arrays(cfg)
    bundle = load_feature_bundle("ravdess", cfg)
    all_actors = bundle["actor_id"]

    k_list = [int(k) for k in gmm_cfg(cfg).get("n_components", [8, 16, 32, 64])]
    base_params = {
        "covariance_type": gmm_cfg(cfg).get("covariance_type", "diag"),
        "max_iter": int(gmm_cfg(cfg).get("max_iter", 200)),
        "tol": float(gmm_cfg(cfg).get("tol", 1e-4)),
        "reg_covar": float(gmm_cfg(cfg).get("reg_covar", 1e-6)),
        "random_state": int(cfg.get("seed", 42)),
    }
    val_ratio = float(gmm_cfg(cfg).get("val_actor_ratio", 0.2))

    fold_details: list[dict] = []

    # 训练和预测函数，返回预测结果，训练时间，预测时间
    def fit_predict(X_tr, y_tr, X_te, y_te, labels):
        tr_idx = fit_predict._train_idx
        actors_tr = all_actors[tr_idx]
        fold = fit_predict._fold
        seed = int(cfg.get("seed", 42)) + fold

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        t0 = time.perf_counter()
        best_k = select_n_components(
            X_tr_s, y_tr, actors_tr, k_list, base_params, seed, val_ratio
        )
        clf = GMMMAPClassifier(n_components=best_k, **base_params)
        clf.fit(X_tr_s, y_tr)
        train_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = clf.predict(X_te_s)
        pred_sec = time.perf_counter() - t1

        fit_predict._last_k = best_k
        return y_pred, train_sec, pred_sec

    runner = EvalRunner(cfg)
    results = []
    for sp in splits:
        fit_predict._train_idx = sp.train_id
        fit_predict._fold = sp.fold
        res = runner.evaluate_split(X, y, sp, fit_predict)
        results.append(res)
        fold_details.append(
            {
                "split_name": res.split_name,
                "fold": res.fold,
                "n_components": fit_predict._last_k,
                "metrics": res.metrics,
                "timing": res.timing,
                "meta": res.meta,
            }
        )

    summary = runner.summarize(results, metric_key="uar")
    out = {
        "model": "gmm_map",
        "dataset": "ravdess",
        "protocol": "ravdess_loso_5fold",
        "n_components_candidates": k_list,
        "summary": summary,
        "folds": fold_details,
    }

    metrics_dir = resolve_path(cfg["outputs"]["metrics"])
    out_path = metrics_dir / "gmm_ravdess.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(out), f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

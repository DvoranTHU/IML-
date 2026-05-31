import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.config import load_config


def svm_cfg(cfg):
    return cfg.get("models", {}).get("svm", {})


def param_grid_for_kernel(kernel, cfg_svm):
    c_list = cfg_svm.get("C", [0.1, 1, 10, 100])
    grid = {"svc__C": list(c_list)}

    if kernel in ("rbf", "poly"):
        gamma_list = cfg_svm.get("gamma", ["scale", "auto", 0.01, 0.1])
        grid["svc__gamma"] = list(gamma_list)
    if kernel == "poly":
        grid["svc__degree"] = list(cfg_svm.get("degree", [2, 3]))
    return grid


def build_pipeline(kernel, kpca_components, random_state):
    steps = [("scaler", StandardScaler())]
    if kpca_components is not None and kpca_components > 0:
        steps.append(
            (
                "kpca",
                KernelPCA(
                    n_components=kpca_components,
                    kernel="rbf",
                    fit_inverse_transform=False,
                    random_state=random_state,
                ),
            )
        )
    steps.append(
        (
            "svc",
            SVC(
                kernel=kernel,
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            ),
        )
    )
    return Pipeline(steps)

# 核SVM分类器
class KernelSVMClassifier:
    def __init__(self, kernel="rbf", cfg=None, random_state=42):
        if cfg is None:
            cfg = load_config()
        self.kernel = kernel
        self.cfg = cfg
        self.cfg_svm = svm_cfg(cfg)
        self.random_state = random_state
        self.best_estimator_ = None
        self.best_params_ = {}
        self.classes_ = []

    # 训练
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = sorted(set(y), key=str)

        kpca_dim = self.cfg_svm.get("kpca_n_components")
        use_kpca = bool(self.cfg_svm.get("use_kpca", False))
        kpca_components = int(kpca_dim) if use_kpca and kpca_dim else None

        pipe = build_pipeline(self.kernel, kpca_components, self.random_state)
        param_grid = param_grid_for_kernel(self.kernel, self.cfg_svm)
        inner_cv = int(self.cfg_svm.get("grid_cv_folds", 3))

        gs = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=StratifiedKFold(
                n_splits=inner_cv, shuffle=True, random_state=self.random_state
            ),
            scoring="recall_macro",
            n_jobs=int(self.cfg_svm.get("n_jobs", -1)),
            refit=True,
            error_score=0.0,
        )
        gs.fit(X, y)
        self.best_estimator_ = gs.best_estimator_
        self.best_params_ = dict(gs.best_params_)
        return self

    # 预测，返回每个样本概率最大的类标签
    def predict(self, X):
        if self.best_estimator_ is None:
            raise RuntimeError()
        best_classes = self.best_estimator_.predict(X)
        return best_classes

    # 预测概率，返回类标签，每个样本每个类的概率
    def predict_proba(self, X):
        if self.best_estimator_ is None:
            raise RuntimeError()
        proba = self.best_estimator_.predict_proba(X)
        classes = self.best_estimator_.named_steps["svc"].classes_
        return classes, proba

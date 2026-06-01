import numpy as np
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import load_config

_SCALAR_KEYS = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "num_leaves",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
)

def gbdt_cfg(cfg):
    return cfg.get("models", {}).get("gbdt", {})

def params_from_cfg(cfg_gbdt):
    params = {
        "n_estimators": int(cfg_gbdt.get("n_estimators", 300)),
        "max_depth": int(cfg_gbdt.get("max_depth", 6)),
        "learning_rate": float(cfg_gbdt.get("learning_rate", 0.05)),
    }
    for key in _SCALAR_KEYS[3:]:
        if key in cfg_gbdt:
            params[key] = cfg_gbdt[key]
    return params

# GBDT分类器
class GBDTClassifier:
    def __init__(self, cfg=None, random_state=42):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.cfg_gbdt = gbdt_cfg(cfg)
        self.random_state = int(random_state)
        self.estimator_ = None
        self.params_ = {}
        self.classes_ = []
        self.label_encoder_ = None

    # 训练
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = sorted(set(y), key=str)
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(self.classes_)
        y_enc = self.label_encoder_.transform(y)

        self.params_ = params_from_cfg(self.cfg_gbdt)
        n_jobs = int(self.cfg_gbdt.get("n_jobs", -1))
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            (
                "lgbm",
                LGBMClassifier(
                    **self.params_,
                    objective="multiclass",
                    class_weight="balanced",
                    verbosity=-1,
                    random_state=self.random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ])
        pipe.fit(X, y_enc)
        self.estimator_ = pipe
        return self

    # 预测，返回每个样本预测类
    def predict(self, X):
        if self.estimator_ is None:
            raise RuntimeError()
        y_enc = self.estimator_.predict(X)
        return self.label_encoder_.inverse_transform(y_enc.astype(int))

    # 预测概率，返回每个样本每个类的概率
    def predict_proba(self, X):
        if self.estimator_ is None:
            raise RuntimeError()
        proba = self.estimator_.predict_proba(X)
        classes = self.label_encoder_.inverse_transform(
            self.estimator_.named_steps["lgbm"].classes_
        )
        return classes, proba

    @property
    def feature_importances_(self):
        if self.estimator_ is None:
            raise RuntimeError()
        return self.estimator_.named_steps["lgbm"].feature_importances_


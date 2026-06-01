import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.config import load_config

def stacking_cfg(cfg):
    return cfg.get("models", {}).get("stacking", {})

# 对齐到class_order列顺序
def align_proba(classes, proba, class_order):
    idx = {c: i for i, c in enumerate(classes)}
    out = np.zeros((proba.shape[0], len(class_order)), dtype=np.float64)
    for j, label in enumerate(class_order):
        if label in idx:
            out[:, j] = proba[:, idx[label]]
    return out

# 元学习器输入，三个基模型概率拼接
def stack_meta_features(proba_blocks, class_order):
    parts = [align_proba(cls, p, class_order) for cls, p in proba_blocks]
    stack = np.hstack(parts)
    return stack

# Stacking分类器，线性分类元学习器
class StackingClassifier:
    def __init__(self, cfg=None, random_state=42):
        if cfg is None:
            cfg = load_config()
        sc = stacking_cfg(cfg)
        self.cfg = cfg
        self.random_state = int(random_state)
        self.class_order_ = []
        self.label_encoder_ = None
        self.meta_ = LogisticRegression(
            max_iter=int(sc.get("max_iter", 1000)),
            C=float(sc.get("C", 1.0)),
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=int(sc.get("n_jobs", -1)),
        )
        self.n_base_models_ = int(3)

    # 训练元学习器
    def fit(self, X_meta, y):
        X_meta = np.asarray(X_meta, dtype=np.float64)
        y = np.asarray(y)
        self.class_order_ = sorted(set(y), key=str)
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(self.class_order_)
        y_enc = self.label_encoder_.transform(y)
        self.meta_.fit(X_meta, y_enc)
        return self

    # 预测，返回预测类别
    def predict(self, X_meta):
        if self.label_encoder_ is None:
            raise RuntimeError()
        y_enc = self.meta_.predict(X_meta)
        emo_pred = self.label_encoder_.inverse_transform(y_enc.astype(int))
        return emo_pred

    # 预测概率，返回预测类别列表和每个类别的概率
    def predict_proba(self, X_meta):
        if self.label_encoder_ is None:
            raise RuntimeError()
        proba = self.meta_.predict_proba(X_meta)
        classes = self.label_encoder_.inverse_transform(self.meta_.classes_)
        return classes, proba

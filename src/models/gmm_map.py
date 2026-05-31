import numpy as np
from sklearn.mixture import GaussianMixture

# GMM MAP分类器
class GMMMAPClassifier:

    def __init__(self, n_components=16, covariance_type="diag", max_iter=200, tol=1e-4, reg_covar=1e-6, random_state=42):
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.random_state = int(random_state)
        self.classes_ = []
        self.models_ = {}
        self.log_priors_ = {}

    # 训练，每类情感一个GMM
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = sorted(set(y), key=str)
        n = len(y)
        self.models_.clear()
        self.log_priors_.clear()

        for label in self.classes_:
            mask = y == label
            X_c = X[mask]

            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                max_iter=self.max_iter,
                tol=self.tol,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
            )
            gmm.fit(X_c)
            self.models_[label] = gmm
            self.log_priors_[label] = float(np.log(mask.sum() / n))

        return self

    # 计算对数后验，返回类标签，每个样本每个类的对数后验
    def log_posteriors(self, X):
        X = np.asarray(X, dtype=np.float64)
        cols = []
        for label in self.classes_:
            log_like = self.models_[label].score_samples(X)
            cols.append(log_like + self.log_priors_[label])
        return self.classes_, np.column_stack(cols)

    # 预测，返回概率最大的类标签
    def predict(self, X):
        classes, log_post = self.log_posteriors(X)
        best = np.argmax(log_post, axis=1)
        selected = [classes[i] for i in best]
        return np.array(selected, dtype=object)

    # 预测概率，返回类标签，每个样本每个类的概率
    def predict_proba(self, X):
        classes, log_post = self.log_posteriors(X)
        log_post = log_post - np.max(log_post, axis=1, keepdims=True)
        prob = np.exp(log_post)
        prob /= prob.sum(axis=1, keepdims=True) + 1e-12
        return classes, prob

def gmm_cfg(cfg):
    return cfg.get("models", {}).get("gmm", {})

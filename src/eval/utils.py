import numpy as np
from scipy import stats

# 计算折间均值和标准差
def fold_mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }

# 计算折间均值的bootstrap置信区间
def bootstrap_ci(values, n_bootstrap=1000, ci_level=0.95, seed=42):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "ci_level": ci_level}

    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    n = arr.size
    for i in range(n_bootstrap):
        sample = arr[rng.integers(0, n, size=n)]
        means[i] = np.mean(sample)

    alpha = 1.0 - ci_level
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_level": float(ci_level),
    }

# 配对t检验
# 返回statistic t值、pvalue p值、n_pairs配对数即折数
def paired_ttest(scores_a, scores_b):
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    stat, pvalue = stats.ttest_rel(a, b, nan_policy="omit")
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "n_pairs": int(a.size),
    }

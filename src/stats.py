"""Small statistics helpers (patient-level bootstrap, paired tests)."""
import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(values, n_boot=5000, seed=0, stat=np.mean):
    """95% CI of `stat` by resampling PATIENTS with replacement. NaNs are dropped first."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = stat(v[idx], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def paired_wilcoxon(a, b):
    """Wilcoxon signed-rank test on paired per-patient scores. Returns (statistic, p)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    if len(d) == 0 or np.all(d == 0):
        return float("nan"), float("nan")
    r = wilcoxon(a[ok], b[ok])
    return float(r.statistic), float(r.pvalue)

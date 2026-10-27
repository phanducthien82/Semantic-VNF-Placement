"""Statistics used in Section 4 (reply to R2 comment 11).

* 30 independent seeds; seed s fixes the substrate attributes, the arrival trace and the
  failure pattern, and all algorithms are run on the same seed -> paired observations.
* Mean and 95% confidence interval: Student-t interval over the 30 per-seed values.
* Paired comparison of the proposed method with each baseline: two-sided Wilcoxon
  signed-rank test (no normality assumption) with Holm-Bonferroni correction over the
  baselines of a table; effect size = matched-pairs rank-biserial correlation r_rb.
  The paired t-test and Cohen's d_z are reported in the supplementary CSV together with
  the Shapiro-Wilk p-value of the paired differences.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def mean_ci(x, conf: float = 0.95):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    m = float(x.mean())
    if n < 2:
        return m, 0.0
    h = float(stats.t.ppf(0.5 + conf / 2, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m, h


def rank_biserial(d):
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    wp, wn = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wn) / (wp + wn))


def paired(a, b) -> dict:
    """Paired comparison of samples a (proposed) and b (baseline) over the same seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    out = dict(n=len(d), mean_diff=float(d.mean()) if len(d) else np.nan)
    if len(d) < 3 or np.allclose(d, 0):
        out.update(p_wilcoxon=1.0, r_rb=0.0, p_t=1.0, d_z=0.0, p_shapiro=np.nan)
        return out
    try:
        out["p_wilcoxon"] = float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        out["p_wilcoxon"] = 1.0
    out["r_rb"] = rank_biserial(d)
    out["p_t"] = float(stats.ttest_rel(a, b).pvalue)
    sd = d.std(ddof=1)
    out["d_z"] = float(d.mean() / sd) if sd > 0 else 0.0
    out["p_shapiro"] = float(stats.shapiro(d).pvalue) if len(d) >= 3 else np.nan
    return out


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a family {name: p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for k, (name, p) in enumerate(items):
        val = min(1.0, (m - k) * p)
        running = max(running, val)
        adj[name] = running
    return adj

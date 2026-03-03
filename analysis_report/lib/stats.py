"""Statistical helpers for report generation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def _finite(x: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def group_box_stats(rows: list[dict[str, Any]], group_key: str, value_key: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = {}
    for r in rows:
        g = str(r.get(group_key, "NA"))
        buckets.setdefault(g, []).append(float(r.get(value_key, np.nan)))
    out: list[dict[str, Any]] = []
    for g in sorted(buckets.keys()):
        x = _finite(buckets[g])
        if len(x) == 0:
            out.append({"group": g, "n": 0})
            continue
        out.append(
            {
                "group": g,
                "n": int(len(x)),
                "median": float(np.median(x)),
                "iqr": float(np.percentile(x, 75.0) - np.percentile(x, 25.0)),
                "p5": float(np.percentile(x, 5.0)),
                "p95": float(np.percentile(x, 95.0)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            }
        )
    return out


def ks_wasserstein(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> dict[str, float]:
    xa = _finite(a)
    xb = _finite(b)
    if len(xa) == 0 or len(xb) == 0:
        return {"ks_stat": float("nan"), "ks_p": float("nan"), "wasserstein": float("nan")}
    ks = stats.ks_2samp(xa, xb)
    wd = stats.wasserstein_distance(xa, xb)
    return {"ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue), "wasserstein": float(wd)}


def spearman_with_bootstrap(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if len(xx) < 3:
        return {"rho": float("nan"), "p": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": int(len(xx))}
    rho, p = stats.spearmanr(xx, yy)
    rng = np.random.default_rng(int(seed))
    boots = []
    idx = np.arange(len(xx), dtype=int)
    n_boot = max(50, int(n))
    for _ in range(n_boot):
        ii = rng.choice(idx, size=len(idx), replace=True)
        rr, _ = stats.spearmanr(xx[ii], yy[ii])
        if np.isfinite(rr):
            boots.append(float(rr))
    if boots:
        lo = float(np.percentile(boots, 100.0 * (alpha / 2.0)))
        hi = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    else:
        lo = float("nan")
        hi = float("nan")
    return {"rho": float(rho), "p": float(p), "ci_lo": lo, "ci_hi": hi, "n": int(len(xx))}


def simple_anova_or_variance_decomp(rows: list[dict[str, Any]], value_key: str = "XPD_early_db") -> dict[str, Any]:
    vals = np.asarray([float(r.get(value_key, np.nan)) for r in rows], dtype=float)
    d = np.asarray([float(r.get("d_m", np.nan)) for r in rows], dtype=float)
    yaw = np.asarray([float(r.get("yaw_deg", np.nan)) for r in rows], dtype=float)
    mask = np.isfinite(vals)
    vals = vals[mask]
    d = d[mask]
    yaw = yaw[mask]
    if len(vals) < 5:
        return {
            "n": int(len(vals)),
            "distance_group_var": float("nan"),
            "yaw_group_var": float("nan"),
            "dominant": "INCONCLUSIVE",
        }

    def _group_var(g: np.ndarray) -> float:
        m = np.isfinite(g)
        if np.sum(m) < 4:
            return float("nan")
        gv = g[m]
        vv = vals[m]
        uniq = np.unique(gv)
        if len(uniq) < 2:
            return 0.0
        means = []
        for u in uniq:
            x = vv[gv == u]
            if len(x):
                means.append(float(np.mean(x)))
        return float(np.var(np.asarray(means, dtype=float), ddof=1)) if len(means) > 1 else 0.0

    d_bin = np.round(d, 2)
    yaw_bin = np.round(yaw, 1)
    vd = _group_var(d_bin)
    vy = _group_var(yaw_bin)
    if not np.isfinite(vd) and not np.isfinite(vy):
        dom = "INCONCLUSIVE"
    elif (not np.isfinite(vd)) or (np.isfinite(vy) and vy > vd):
        dom = "yaw"
    elif (not np.isfinite(vy)) or (np.isfinite(vd) and vd > vy):
        dom = "distance"
    else:
        dom = "tie"
    return {
        "n": int(len(vals)),
        "distance_group_var": float(vd),
        "yaw_group_var": float(vy),
        "dominant": dom,
    }


def linear_trend_test(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> dict[str, float]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if len(xx) < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "r": float("nan"), "p": float("nan"), "n": int(len(xx))}
    lr = stats.linregress(xx, yy)
    return {
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
        "r": float(lr.rvalue),
        "p": float(lr.pvalue),
        "n": int(len(xx)),
    }


def cliffs_delta(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    xa = _finite(a)
    xb = _finite(b)
    if len(xa) == 0 or len(xb) == 0:
        return float("nan")
    gt = 0
    lt = 0
    for va in xa:
        gt += int(np.sum(va > xb))
        lt += int(np.sum(va < xb))
    return float((gt - lt) / (len(xa) * len(xb)))

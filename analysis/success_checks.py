"""Automated success-criteria checks on standardized link metrics."""

from __future__ import annotations

import csv
from typing import Any

import numpy as np
from scipy import stats


def _maybe_num(v: str) -> Any:
    s = str(v).strip()
    if s == "":
        return ""
    try:
        x = float(s)
        return int(x) if x.is_integer() else x
    except Exception:
        return s


def load_link_metrics_csv(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            out.append({k: _maybe_num(v) for k, v in r.items()})
    return out


def _vals(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    x = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)
    return x[np.isfinite(x)]


def _pick_metric_key(rows: list[dict[str, Any]], preferred: str, fallback: str) -> str:
    xv = _vals(rows, preferred)
    if len(xv) > 0:
        return preferred
    return fallback


def _paired_vals(rows: list[dict[str, Any]], key_x: str, key_y: str) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for r in rows:
        x = float(r.get(key_x, np.nan))
        y = float(r.get(key_y, np.nan))
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return float("nan")
    if float(np.nanstd(x)) == 0.0 or float(np.nanstd(y)) == 0.0:
        return float("nan")
    c = float(stats.spearmanr(x, y).correlation)
    return c if np.isfinite(c) else float("nan")


def _bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, B: int = 300, seed: int = 0) -> tuple[float, float]:
    if len(x) < 4 or len(y) < 4 or len(x) != len(y):
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    vals = []
    n = len(x)
    for _ in range(int(B)):
        idx = rng.integers(0, n, size=n)
        c = _safe_spearman(x[idx], y[idx])
        if np.isfinite(c):
            vals.append(c)
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def check_C0_floor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c0 = [r for r in rows if str(r.get("scenario_id", "")).upper() == "C0"]
    x_key = "XPD_early_db" if len(_vals(c0, "XPD_early_db")) > 0 else _pick_metric_key(c0, "XPD_early_excess_db", "XPD_early_db")
    x = _vals(c0, x_key)
    if len(x) == 0:
        return {"n": 0, "status": "NO_DATA"}
    p5, p95 = np.percentile(x, [5, 95])
    x_d, d = _paired_vals(c0, x_key, "d_m")
    x_y, yaw = _paired_vals(c0, x_key, "yaw_deg")
    corr_d = _safe_spearman(d, x_d)
    corr_yaw = _safe_spearman(yaw, x_y)
    return {
        "n": int(len(x)),
        "metric_key": str(x_key),
        "xpd_floor_mean_db": float(np.mean(x)),
        "xpd_floor_std_db": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "xpd_floor_p5_db": float(p5),
        "xpd_floor_p95_db": float(p95),
        "delta_floor_db": float(p95 - p5),
        "distance_rank_corr": corr_d,
        "yaw_rank_corr": corr_yaw,
        "dominant_factor": "yaw_or_pitch" if abs(corr_yaw) > abs(corr_d) else "distance_or_drift",
        "status": "OK",
    }


def check_A2_A3_parity_sign(rows: list[dict[str, Any]]) -> dict[str, Any]:
    a2_rows = [r for r in rows if str(r.get("scenario_id", "")).upper() == "A2"]
    a3_rows = [r for r in rows if str(r.get("scenario_id", "")).upper() == "A3"]
    key = _pick_metric_key(a2_rows + a3_rows, "XPD_early_excess_db", "XPD_early_db")
    a2 = _vals(a2_rows, key)
    a3 = _vals(a3_rows, key)
    med_a2 = float(np.median(a2)) if len(a2) else float("nan")
    med_a3 = float(np.median(a3)) if len(a3) else float("nan")
    ks_p = float("nan")
    wdist = float("nan")
    if len(a2) > 0 and len(a3) > 0:
        ks_p = float(stats.ks_2samp(a2, a3).pvalue)
        wdist = float(stats.wasserstein_distance(a2, a3))
    out_a2 = float(np.mean(a2 <= np.percentile(a2, 10))) if len(a2) else float("nan")
    out_a3 = float(np.mean(a3 <= np.percentile(a3, 10))) if len(a3) else float("nan")
    return {
        "metric_key": key,
        "n_A2": int(len(a2)),
        "n_A3": int(len(a3)),
        "median_A2_xpd_early_db": med_a2,
        "median_A3_xpd_early_db": med_a3,
        "delta_median_A3_minus_A2_db": float(med_a3 - med_a2) if np.isfinite(med_a2) and np.isfinite(med_a3) else float("nan"),
        "ks_p_A2_vs_A3": ks_p,
        "wasserstein_A2_vs_A3": wdist,
        "outlier_rate_A2_p10": out_a2,
        "outlier_rate_A3_p10": out_a3,
        "pass_A2_negative": bool(np.isfinite(med_a2) and med_a2 < 0.0),
        "pass_A3_positive": bool(np.isfinite(med_a3) and med_a3 > 0.0),
    }


def check_A4_A5_breaking(rows: list[dict[str, Any]]) -> dict[str, Any]:
    key = _pick_metric_key(rows, "XPD_early_excess_db", "XPD_early_db")
    key_late = _pick_metric_key(rows, "XPD_late_excess_db", "XPD_late_db")
    a5 = [r for r in rows if str(r.get("scenario_id", "")).upper() == "A5"]
    base = [r for r in a5 if int(r.get("roughness_flag", 0)) == 0 and int(r.get("human_flag", 0)) == 0]
    if not base:
        base = [r for r in rows if str(r.get("scenario_id", "")).upper() == "A4"]
    stress = [r for r in a5 if int(r.get("roughness_flag", 0)) == 1 or int(r.get("human_flag", 0)) == 1]

    xb = _vals(base, key)
    xs = _vals(stress, key)
    lb = _vals(base, key_late)
    ls = _vals(stress, key_late)
    m_base = float(np.median(xb)) if len(xb) else float("nan")
    m_stress = float(np.median(xs)) if len(xs) else float("nan")
    l_base = float(np.median(lb)) if len(lb) else float("nan")
    l_stress = float(np.median(ls)) if len(ls) else float("nan")
    var_b = float(np.var(xb, ddof=1)) if len(xb) > 1 else float("nan")
    var_s = float(np.var(xs, ddof=1)) if len(xs) > 1 else float("nan")
    p10_b = float(np.percentile(xb, 10)) if len(xb) else float("nan")
    p10_s = float(np.percentile(xs, 10)) if len(xs) else float("nan")
    pass_break = bool(
        np.isfinite(m_base)
        and np.isfinite(m_stress)
        and (abs(m_stress) <= abs(m_base) or (np.isfinite(l_base) and np.isfinite(l_stress) and l_stress > l_base))
    )
    return {
        "metric_key_early": key,
        "metric_key_late": key_late,
        "n_A5_base": int(len(xb)),
        "n_A5_stress": int(len(xs)),
        "median_abs_xpd_early_base": float(abs(m_base)) if np.isfinite(m_base) else float("nan"),
        "median_abs_xpd_early_stress": float(abs(m_stress)) if np.isfinite(m_stress) else float("nan"),
        "median_xpd_late_base_db": l_base,
        "median_xpd_late_stress_db": l_stress,
        "var_xpd_early_base": var_b,
        "var_xpd_early_stress": var_s,
        "var_ratio_stress_over_base": float(var_s / var_b) if np.isfinite(var_s) and np.isfinite(var_b) and var_b > 0 else float("nan"),
        "p10_xpd_early_base": p10_b,
        "p10_xpd_early_stress": p10_s,
        "pass_breaking_trend": pass_break,
    }


def check_B_space_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    b = [r for r in rows if str(r.get("scenario_id", "")).upper().startswith("B")]
    key = _pick_metric_key(b, "XPD_early_excess_db", "XPD_early_db")
    x, el = _paired_vals(b, key, "EL_proxy_db")
    corr = float("nan")
    ci_lo = float("nan")
    ci_hi = float("nan")
    if len(x) >= 2:
        corr = _safe_spearman(x, -el)
        ci_lo, ci_hi = _bootstrap_spearman_ci(x, -el, B=300, seed=0)
    los = _vals([r for r in b if int(r.get("LOSflag", 0)) == 1], key)
    nlos = _vals([r for r in b if int(r.get("LOSflag", 0)) == 0], key)
    ks_p = float("nan")
    wdist = float("nan")
    if len(los) > 0 and len(nlos) > 0:
        ks_p = float(stats.ks_2samp(los, nlos).pvalue)
        wdist = float(stats.wasserstein_distance(los, nlos))
    return {
        "metric_key": key,
        "n_B": int(len(b)),
        "spearman_xpd_early_vs_minus_el_proxy": corr,
        "spearman_ci95_lo": ci_lo,
        "spearman_ci95_hi": ci_hi,
        "n_LOS": int(len(los)),
        "n_NLOS": int(len(nlos)),
        "ks_p_los_vs_nlos_xpd_early": ks_p,
        "wasserstein_los_vs_nlos_xpd_early": wdist,
    }


def evaluate_success_criteria(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "C0_floor": check_C0_floor(rows),
        "A2_A3_parity_sign": check_A2_A3_parity_sign(rows),
        "A4_A5_breaking": check_A4_A5_breaking(rows),
        "B_space": check_B_space_consistency(rows),
    }

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


def check_C0_floor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c0 = [r for r in rows if str(r.get("scenario_id", "")).upper() == "C0"]
    x = _vals(c0, "XPD_early_db")
    if len(x) == 0:
        return {"n": 0, "status": "NO_DATA"}
    p5, p95 = np.percentile(x, [5, 95])
    d = _vals(c0, "d_m")
    corr = float(stats.spearmanr(d, x).correlation) if len(d) == len(x) and len(x) >= 2 else float("nan")
    return {
        "n": int(len(x)),
        "xpd_floor_mean_db": float(np.mean(x)),
        "xpd_floor_std_db": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "xpd_floor_p5_db": float(p5),
        "xpd_floor_p95_db": float(p95),
        "delta_floor_db": float(p95 - p5),
        "distance_rank_corr": corr,
        "status": "OK",
    }


def check_A2_A3_parity_sign(rows: list[dict[str, Any]]) -> dict[str, Any]:
    a2 = _vals([r for r in rows if str(r.get("scenario_id", "")).upper() == "A2"], "XPD_early_db")
    a3 = _vals([r for r in rows if str(r.get("scenario_id", "")).upper() == "A3"], "XPD_early_db")
    med_a2 = float(np.median(a2)) if len(a2) else float("nan")
    med_a3 = float(np.median(a3)) if len(a3) else float("nan")
    return {
        "n_A2": int(len(a2)),
        "n_A3": int(len(a3)),
        "median_A2_xpd_early_db": med_a2,
        "median_A3_xpd_early_db": med_a3,
        "pass_A2_negative": bool(np.isfinite(med_a2) and med_a2 < 0.0),
        "pass_A3_positive": bool(np.isfinite(med_a3) and med_a3 > 0.0),
    }


def check_A4_A5_breaking(rows: list[dict[str, Any]]) -> dict[str, Any]:
    a5 = [r for r in rows if str(r.get("scenario_id", "")).upper() == "A5"]
    base = [r for r in a5 if int(r.get("roughness_flag", 0)) == 0 and int(r.get("human_flag", 0)) == 0]
    if not base:
        # fallback baseline from A4 when A5 baseline run is not included
        base = [r for r in rows if str(r.get("scenario_id", "")).upper() == "A4"]
    stress = [r for r in a5 if int(r.get("roughness_flag", 0)) == 1 or int(r.get("human_flag", 0)) == 1]
    xb = _vals(base, "XPD_early_db")
    xs = _vals(stress, "XPD_early_db")
    lb = _vals(base, "XPD_late_db")
    ls = _vals(stress, "XPD_late_db")
    m_base = float(np.median(xb)) if len(xb) else float("nan")
    m_stress = float(np.median(xs)) if len(xs) else float("nan")
    l_base = float(np.median(lb)) if len(lb) else float("nan")
    l_stress = float(np.median(ls)) if len(ls) else float("nan")
    pass_break = bool(
        np.isfinite(m_base)
        and np.isfinite(m_stress)
        and (abs(m_stress) <= abs(m_base) or (np.isfinite(l_base) and np.isfinite(l_stress) and l_stress > l_base))
    )
    return {
        "n_A5_base": int(len(xb)),
        "n_A5_stress": int(len(xs)),
        "median_abs_xpd_early_base": float(abs(m_base)) if np.isfinite(m_base) else float("nan"),
        "median_abs_xpd_early_stress": float(abs(m_stress)) if np.isfinite(m_stress) else float("nan"),
        "median_xpd_late_base_db": l_base,
        "median_xpd_late_stress_db": l_stress,
        "pass_breaking_trend": pass_break,
    }


def check_B_space_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    b = [r for r in rows if str(r.get("scenario_id", "")).upper().startswith("B")]
    x = _vals(b, "XPD_early_db")
    el = _vals(b, "EL_proxy_db")
    corr = float("nan")
    if len(x) == len(el) and len(x) >= 2:
        corr = float(stats.spearmanr(x, -el).correlation)
    los = _vals([r for r in b if int(r.get("LOSflag", 0)) == 1], "XPD_early_db")
    nlos = _vals([r for r in b if int(r.get("LOSflag", 0)) == 0], "XPD_early_db")
    ks_p = float("nan")
    if len(los) > 1 and len(nlos) > 1:
        ks_p = float(stats.ks_2samp(los, nlos).pvalue)
    return {
        "n_B": int(len(b)),
        "spearman_xpd_early_vs_minus_el_proxy": corr,
        "n_LOS": int(len(los)),
        "n_NLOS": int(len(nlos)),
        "ks_p_los_vs_nlos_xpd_early": ks_p,
    }


def evaluate_success_criteria(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "C0_floor": check_C0_floor(rows),
        "A2_A3_parity_sign": check_A2_A3_parity_sign(rows),
        "A4_A5_breaking": check_A4_A5_breaking(rows),
        "B_space": check_B_space_consistency(rows),
    }

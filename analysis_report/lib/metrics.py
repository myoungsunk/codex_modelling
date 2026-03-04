"""Derived metric helpers for reporting."""

from __future__ import annotations

from typing import Any

import numpy as np


def _finite(vals: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def _pctl(x: list[float] | np.ndarray, q: float) -> float:
    v = _finite(x)
    if len(v) == 0:
        return float("nan")
    return float(np.percentile(v, float(q)))


def estimate_floor_from_c0(link_rows: list[dict[str, Any]], delta_method: str = "p5_p95") -> dict[str, Any]:
    xs = [float(r.get("XPD_early_db", np.nan)) for r in link_rows if str(r.get("scenario_id", "")).upper() == "C0"]
    v = _finite(xs)
    if len(v) == 0:
        return {
            "xpd_floor_db": float("nan"),
            "delta_floor_db": float("nan"),
            "p5_db": float("nan"),
            "p95_db": float("nan"),
            "count": 0,
            "method": str(delta_method),
        }
    p5 = _pctl(v, 5.0)
    p95 = _pctl(v, 95.0)
    iqr = _pctl(v, 75.0) - _pctl(v, 25.0)
    std = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    method = str(delta_method).lower().strip()
    if method == "iqr":
        delta = 0.5 * float(iqr)
    elif method == "std":
        delta = float(std)
    else:
        delta = 0.5 * float(p95 - p5)
    return {
        "xpd_floor_db": float(np.median(v)),
        "delta_floor_db": float(delta),
        "p5_db": float(p5),
        "p95_db": float(p95),
        "count": int(len(v)),
        "method": method,
    }


def estimate_floor_from_calibration_json(calib: dict[str, Any], delta_method: str = "p5_p95") -> dict[str, Any]:
    floor = np.asarray(calib.get("xpd_floor_db", []), dtype=float)
    unc = np.asarray(calib.get("xpd_floor_uncert_db", []), dtype=float)
    if floor.ndim == 0:
        floor = np.asarray([float(floor)], dtype=float)
    floor = floor[np.isfinite(floor)]
    if len(floor) == 0:
        return {
            "xpd_floor_db": float("nan"),
            "delta_floor_db": float("nan"),
            "p5_db": float("nan"),
            "p95_db": float("nan"),
            "count": 0,
            "method": str(delta_method),
        }
    method = str(delta_method).lower().strip()
    p5 = _pctl(floor, 5.0)
    p95 = _pctl(floor, 95.0)
    if len(unc) > 0:
        unc = unc[np.isfinite(unc)]
        delta = float(np.median(unc)) if len(unc) else 0.5 * (p95 - p5)
    elif method == "std":
        delta = float(np.std(floor, ddof=1)) if len(floor) > 1 else 0.0
    elif method == "iqr":
        delta = 0.5 * (_pctl(floor, 75.0) - _pctl(floor, 25.0))
    else:
        delta = 0.5 * (p95 - p5)
    return {
        "xpd_floor_db": float(np.median(floor)),
        "delta_floor_db": float(delta),
        "p5_db": float(p5),
        "p95_db": float(p95),
        "count": int(len(floor)),
        "method": method,
    }


def apply_floor_excess(link_rows: list[dict[str, Any]], floor_db: float, delta_db: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in link_rows:
        rr = dict(r)
        x_e = float(rr.get("XPD_early_db", np.nan))
        x_l = float(rr.get("XPD_late_db", np.nan))
        # Always recompute excess against the report-selected floor reference.
        # Do not keep pre-existing per-run excess values, which may have been
        # computed with a different floor policy.
        ex_e = float(x_e - floor_db) if np.isfinite(x_e) and np.isfinite(floor_db) else np.nan
        ex_l = float(x_l - floor_db) if np.isfinite(x_l) and np.isfinite(floor_db) else np.nan
        rr["XPD_early_excess_db"] = ex_e
        rr["XPD_late_excess_db"] = ex_l
        rr["floor_db"] = float(floor_db)
        rr["delta_floor_db"] = float(delta_db)
        rr["outlier_excess_neg"] = bool(np.isfinite(ex_e) and np.isfinite(delta_db) and ex_e < -abs(delta_db))
        rr["outlier_excess_pos"] = bool(np.isfinite(ex_e) and np.isfinite(delta_db) and ex_e > abs(delta_db))
        # claim_caution: excess is within ±delta_ref floor uncertainty band.
        # Points within this band cannot support calibration-aware excess claims.
        delta_abs = abs(float(delta_db)) if np.isfinite(delta_db) else float("nan")
        rr["claim_caution_early"] = bool(
            np.isfinite(ex_e) and np.isfinite(delta_abs) and abs(ex_e) < delta_abs
        )
        rr["claim_caution_late"] = bool(
            np.isfinite(ex_l) and np.isfinite(delta_abs) and abs(ex_l) < delta_abs
        )
        out.append(rr)
    return out


def split_by_scenario(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        s = str(r.get("scenario_id", "NA"))
        out.setdefault(s, []).append(r)
    return out


def delta_median(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], key: str) -> float:
    a = _finite([float(r.get(key, np.nan)) for r in rows_a])
    b = _finite([float(r.get(key, np.nan)) for r in rows_b])
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(np.median(a) - np.median(b))


def tail_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    x = _finite([float(r.get(key, np.nan)) for r in rows])
    if len(x) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p5": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "p5": _pctl(x, 5.0),
        "p10": _pctl(x, 10.0),
        "p90": _pctl(x, 90.0),
        "p95": _pctl(x, 95.0),
    }

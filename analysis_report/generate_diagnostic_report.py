"""Generate diagnostic report (A-E checks) from standard outputs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis_report.lib import indexer
from analysis_report.lib import io as io_lib
from analysis_report.lib import metrics as metrics_lib
from analysis_report.lib import plots as plot_lib
from analysis_report.lib import report_md
from analysis_report.lib import scene as scene_lib
from analysis_report.lib import stats as stats_lib
from analysis import link_metrics as link_metrics_lib
from analysis import windowing as windowing_lib


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            out = {}
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, (dict, list, tuple)):
                    out[k] = json.dumps(v)
                else:
                    out[k] = v
            w.writerow(out)


def _write_rows_csv_with_columns(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            out = {}
            for k in columns:
                v = r.get(k, "")
                if isinstance(v, (dict, list, tuple)):
                    out[k] = json.dumps(v)
                else:
                    out[k] = v
            w.writerow(out)


def _status(pass_cond: bool, warn_cond: bool = False) -> str:
    if pass_cond:
        return "PASS"
    if warn_cond:
        return "WARN"
    return "FAIL"


def _num(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _median(rows: list[dict[str, Any]], key: str) -> float:
    x = np.asarray([_num(r.get(key, np.nan)) for r in rows], dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.median(x))


def _median_vals(vals: list[float] | np.ndarray) -> float:
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.median(x))


def _safe_span(vals: list[float] | np.ndarray) -> float:
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.max(x) - np.min(x))


def _judge_vs_delta(effect_db: float, delta_db: float) -> tuple[str, float]:
    if not (np.isfinite(effect_db) and np.isfinite(delta_db) and abs(delta_db) > 0.0):
        return "INCONCLUSIVE", float("nan")
    ratio = abs(float(effect_db)) / abs(float(delta_db))
    if ratio >= 2.0:
        return "PASS", float(ratio)
    if ratio >= 1.0:
        return "WARN", float(ratio)
    return "FAIL", float(ratio)


def _max_finite(values: list[float], default: float = float("nan")) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float(default)
    return float(np.max(arr))


def _provenance(row: dict[str, Any]) -> dict[str, Any]:
    p = row.get("provenance_json", {})
    if isinstance(p, dict):
        return p
    if isinstance(p, str):
        s = p.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _link_params(row: dict[str, Any]) -> dict[str, Any]:
    prov = _provenance(row)
    lp = prov.get("link_params", {})
    return lp if isinstance(lp, dict) else {}


def _a4_branch(row: dict[str, Any]) -> str:
    layout = str(row.get("a4_layout_mode", "")).strip().lower()
    if not layout:
        layout = str(_link_params(row).get("a4_layout_mode", "")).strip().lower()
    inc = _num(row.get("include_late_panel", np.nan))
    if not np.isfinite(inc):
        inc = _num(_link_params(row).get("include_late_panel", np.nan))
    if np.isfinite(inc):
        return "bridge" if int(round(inc)) == 1 else "iso"
    if layout == "bridge":
        return "bridge"
    if layout == "iso":
        return "iso"
    return "other"


def _a4_dispersion_on(row: dict[str, Any]) -> bool:
    mode = str(row.get("a4_dispersion_mode", "")).strip().lower()
    if not mode:
        mode = str(row.get("material_dispersion", "")).strip().lower()
    if not mode:
        mode = str(_link_params(row).get("a4_dispersion_mode", _link_params(row).get("material_dispersion", ""))).strip().lower()
    return mode in {"on", "debye"}


def _c0_repeat_delta(link_rows: list[dict[str, Any]]) -> float:
    by_key: dict[tuple[float, float, float], list[float]] = {}
    for r in link_rows:
        if str(r.get("scenario_id", "")).upper() != "C0":
            continue
        lp = _link_params(r)
        d = _num(lp.get("distance_m", r.get("d_m", np.nan)))
        yaw = _num(lp.get("yaw_deg", r.get("yaw_deg", np.nan)))
        pitch = _num(lp.get("pitch_deg", r.get("pitch_deg", np.nan)))
        x = _num(r.get("XPD_early_db", np.nan))
        if not (np.isfinite(d) and np.isfinite(yaw) and np.isfinite(pitch) and np.isfinite(x)):
            continue
        k = (round(float(d), 6), round(float(yaw), 6), round(float(pitch), 6))
        by_key.setdefault(k, []).append(float(x))
    stds: list[float] = []
    for vv in by_key.values():
        arr = np.asarray(vv, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) >= 2:
            stds.append(float(np.std(arr, ddof=1)))
    return _median_vals(stds)


def _a5_pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    lp = _link_params(row)
    def _norm(v: Any, nd: int = 6) -> Any:
        x = _num(v)
        if not np.isfinite(x):
            return None
        return round(float(x), int(nd))
    return (
        _norm(lp.get("rho", np.nan), nd=4),
        int(round(_num(lp.get("rep_id", -1)))) if np.isfinite(_num(lp.get("rep_id", np.nan))) else None,
        int(round(_num(lp.get("rep_outer", -1)))) if np.isfinite(_num(lp.get("rep_outer", np.nan))) else None,
        _norm(lp.get("offset", np.nan), nd=4),
        _norm(lp.get("rx_x", np.nan), nd=4),
        _norm(lp.get("rx_y", np.nan), nd=4),
    )


def _a5_row_semantics(row: dict[str, Any]) -> str:
    s = str(row.get("stress_semantics", "")).strip().lower()
    if s in {"off", "response", "polarization_only"}:
        return s
    sp = _num(row.get("stress_path_structure_active", np.nan))
    sm = _num(row.get("stress_polarization_mixer_active", np.nan))
    if np.isfinite(sp) or np.isfinite(sm):
        if int(round(sp)) == 1:
            return "response"
        if int(round(sm)) == 1:
            return "polarization_only"
        return "off"
    mode = str(row.get("stress_mode", "")).strip().lower()
    stress_on = int(_num(row.get("roughness_flag", 0))) == 1 or int(_num(row.get("human_flag", 0))) == 1
    if not stress_on:
        return "off"
    if mode in {"geometry", "hybrid"}:
        return "response"
    if mode == "synthetic":
        return "polarization_only"
    if mode == "none":
        return "off"
    return "unknown"


def _a5_semantics_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    a5_rows = [r for r in rows if str(r.get("scenario_id", "")).upper() == "A5"]
    stress_rows = [r for r in a5_rows if int(_num(r.get("roughness_flag", 0))) == 1 or int(_num(r.get("human_flag", 0))) == 1]
    cnt = {"response": 0, "polarization_only": 0, "off": 0, "unknown": 0}
    for r in stress_rows:
        sem = _a5_row_semantics(r)
        cnt[sem] = int(cnt.get(sem, 0) + 1)
    total = int(len(stress_rows))
    dominant = "none"
    if total > 0:
        dominant = max(["response", "polarization_only", "off", "unknown"], key=lambda k: int(cnt.get(k, 0)))
    if total == 0:
        status = "INCONCLUSIVE"
        note = "A5 stress rows not found."
    elif int(cnt.get("response", 0)) > 0 and int(cnt.get("polarization_only", 0)) == 0:
        status = "PASS"
        note = "Geometric path-structure stress is active; delay/path contamination-response interpretation is valid."
    elif int(cnt.get("response", 0)) > 0 and int(cnt.get("polarization_only", 0)) > 0:
        status = "WARN"
        note = "Mixed A5 semantics detected (response + polarization_only); interpret delay-structure claims only on response subset."
    elif int(cnt.get("response", 0)) == 0 and int(cnt.get("polarization_only", 0)) > 0:
        status = "WARN"
        note = "Synthetic depol only: polarization-axis stress is active, but delay/path-structure stress is not active."
    else:
        status = "WARN"
        note = "A5 stress semantics could not be resolved reliably."
    return {
        "status": str(status),
        "n_a5_rows": int(len(a5_rows)),
        "n_stress_rows": int(total),
        "n_response": int(cnt.get("response", 0)),
        "n_polarization_only": int(cnt.get("polarization_only", 0)),
        "n_off": int(cnt.get("off", 0)),
        "n_unknown": int(cnt.get("unknown", 0)),
        "dominant_semantics": str(dominant),
        "contamination_response_ready": bool(int(cnt.get("response", 0)) > 0),
        "note": str(note),
    }


def _inc_bin(v: float) -> str:
    if not np.isfinite(v):
        return "NA"
    if v < 30.0:
        return "low"
    if v < 60.0:
        return "mid"
    return "high"


def _dominant_incidence_by_link(ray_rows: list[dict[str, Any]]) -> dict[str, float]:
    best: dict[str, tuple[float, float]] = {}
    for r in ray_rows:
        lid = str(r.get("link_id", ""))
        if not lid:
            continue
        p = _num(r.get("P_lin", np.nan))
        inc = _num(r.get("incidence_deg", np.nan))
        if not (np.isfinite(p) and np.isfinite(inc)):
            continue
        cur = best.get(lid)
        if cur is None or p > cur[0]:
            best[lid] = (float(p), float(inc))
    return {k: float(v[1]) for k, v in best.items()}


def _build_design_matrix(rows: list[dict[str, Any]], inc_by_link: dict[str, float]) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    num_keys = ["d_m", "EL_proxy_db"]
    cat_keys = ["LOSflag", "material_class", "roughness_flag", "human_flag", "obstacle_flag", "dominant_parity_early", "incidence_bin"]

    num_cols: list[np.ndarray] = []
    names: list[str] = []
    n = len(rows)

    for k in num_keys:
        col = np.asarray([_num(r.get(k, np.nan)) for r in rows], dtype=float)
        med = float(np.nanmedian(col)) if np.any(np.isfinite(col)) else 0.0
        col = np.where(np.isfinite(col), col, med)
        num_cols.append(col)
        names.append(k)

    for k in cat_keys:
        vals: list[str] = []
        for r in rows:
            if k == "incidence_bin":
                inc = _num(inc_by_link.get(str(r.get("link_id", "")), np.nan))
                vals.append(_inc_bin(inc))
            else:
                vals.append(str(r.get(k, "NA")))
        lv = sorted(set(vals))
        if len(lv) <= 1:
            continue
        for c in lv[1:]:
            col = np.asarray([1.0 if v == c else 0.0 for v in vals], dtype=float)
            num_cols.append(col)
            names.append(f"{k}={c}")

    if num_cols:
        x = np.column_stack([np.ones(n, dtype=float)] + num_cols)
        names = ["const"] + names
    else:
        x = np.ones((n, 1), dtype=float)
        names = ["const"]
    num_data = {k: np.asarray([_num(r.get(k, np.nan)) for r in rows], dtype=float) for k in num_keys}
    return x, names, num_data


def _vif_numeric(num_data: dict[str, np.ndarray], x_full: np.ndarray, names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, y in num_data.items():
        m = np.isfinite(y)
        if int(np.sum(m)) < 4:
            out[k] = float("nan")
            continue
        try:
            idx = names.index(k)
        except ValueError:
            out[k] = float("nan")
            continue
        xm = x_full[m, :]
        ym = y[m]
        xo = np.delete(xm, idx, axis=1)
        beta, *_ = np.linalg.lstsq(xo, ym, rcond=None)
        yh = xo @ beta
        ssr = float(np.sum((ym - yh) ** 2))
        sst = float(np.sum((ym - float(np.mean(ym))) ** 2))
        if sst <= 0:
            out[k] = float("nan")
            continue
        r2 = max(0.0, min(1.0, 1.0 - ssr / sst))
        out[k] = float(1.0 / max(1e-6, 1.0 - r2))
    return out


def _build_design_matrix_custom(
    rows: list[dict[str, Any]],
    *,
    num_keys: list[str],
    cat_keys: list[str],
    inc_by_link: dict[str, float],
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    cols: list[np.ndarray] = []
    names: list[str] = []
    n = len(rows)

    for k in num_keys:
        col = np.asarray([_num(r.get(k, np.nan)) for r in rows], dtype=float)
        med = float(np.nanmedian(col)) if np.any(np.isfinite(col)) else 0.0
        col = np.where(np.isfinite(col), col, med)
        cols.append(col)
        names.append(k)

    for k in cat_keys:
        vals: list[str] = []
        for r in rows:
            if k == "incidence_bin":
                inc = _num(inc_by_link.get(str(r.get("link_id", "")), np.nan))
                vals.append(_inc_bin(inc))
            elif k == "stress_flag":
                s = int(_num(r.get("roughness_flag", 0))) == 1 or int(_num(r.get("human_flag", 0))) == 1
                vals.append("1" if s else "0")
            elif k == "odd_flag":
                p = str(r.get("dominant_parity_early", "")).lower()
                vals.append("1" if p == "odd" else "0")
            else:
                vals.append(str(r.get(k, "NA")))
        lv = sorted(set(vals))
        if len(lv) <= 1:
            continue
        for c in lv[1:]:
            col = np.asarray([1.0 if v == c else 0.0 for v in vals], dtype=float)
            cols.append(col)
            names.append(f"{k}={c}")

    if cols:
        x = np.column_stack([np.ones(n, dtype=float)] + cols)
        names = ["const"] + names
    else:
        x = np.ones((n, 1), dtype=float)
        names = ["const"]
    num_data = {k: np.asarray([_num(r.get(k, np.nan)) for r in rows], dtype=float) for k in num_keys}
    return x, names, num_data


def _design_diag(
    rows: list[dict[str, Any]],
    *,
    num_keys: list[str],
    cat_keys: list[str],
    inc_by_link: dict[str, float],
    vif_threshold: float,
) -> dict[str, Any]:
    if len(rows) < 4:
        return {
            "status": "INCONCLUSIVE",
            "n_rows": int(len(rows)),
            "design_rank": 0,
            "design_cols": 0,
            "condition_number": float("nan"),
            "vif_threshold": float(vif_threshold),
            "vif": {},
            "vif_warnings": {},
        }
    x, names, num_data = _build_design_matrix_custom(
        rows,
        num_keys=list(num_keys),
        cat_keys=list(cat_keys),
        inc_by_link=inc_by_link,
    )
    rank = int(np.linalg.matrix_rank(x))
    cols = int(x.shape[1])
    cond = float(np.linalg.cond(x)) if x.size else float("nan")
    vif = _vif_numeric(num_data, x, names)
    vif_warn = {k: float(v) for k, v in vif.items() if np.isfinite(v) and v > float(vif_threshold)}
    if rank == cols and np.isfinite(cond) and cond < 1.0e8 and len(vif_warn) == 0:
        st = "PASS"
    elif rank >= max(1, cols - 1) and (not np.isfinite(cond) or cond < 1.0e12):
        st = "WARN"
    else:
        st = "FAIL"
    return {
        "status": st,
        "n_rows": int(len(rows)),
        "design_rank": rank,
        "design_cols": cols,
        "condition_number": cond,
        "vif_threshold": float(vif_threshold),
        "vif": vif,
        "vif_warnings": vif_warn,
    }


def _scenario_case_rows(link_rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in link_rows:
        k = (str(r.get("scenario_id", "NA")), str(r.get("case_id", "")))
        out.setdefault(k, []).append(r)
    return out


def _window_dict(row: dict[str, Any]) -> dict[str, Any]:
    w = row.get("window", {})
    if isinstance(w, dict):
        return w
    if isinstance(w, str):
        s = w.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            try:
                obj = ast.literal_eval(s)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
    return {}


def _db_ratio(num: float, den: float, eps: float = 1e-30) -> float:
    n = max(float(num), float(eps))
    d = max(float(den), float(eps))
    return float(10.0 * np.log10(n / d))


def _as_float_list(v: Any, default: list[float]) -> list[float]:
    if isinstance(v, list):
        out: list[float] = []
        for x in v:
            vv = _num(x)
            if np.isfinite(vv):
                out.append(float(vv))
        if out:
            return out
    return [float(x) for x in default]


def _estimate_freq_grid(link_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[float, float, str]:
    # Prefer frequency axis embedded in link rows (actual run output),
    # with C0 rows prioritized for calibration consistency.
    cand: list[tuple[float, float, str]] = []
    for r in link_rows:
        arr = r.get("xpd_floor_freq_hz")
        if not isinstance(arr, list):
            continue
        f = np.asarray(arr, dtype=float)
        f = f[np.isfinite(f)]
        if len(f) < 2:
            continue
        f = np.sort(np.unique(f))
        df = float(np.median(np.diff(f)))
        bw = float(f[-1] - f[0])
        if not (bw > 0 and df > 0):
            continue
        sid = str(r.get("scenario_id", "NA"))
        cand.append((bw, df, f"link_rows.xpd_floor_freq_hz[{sid}]"))
    if cand:
        c0 = [x for x in cand if x[2].endswith("[C0]")]
        pick_pool = c0 if c0 else cand
        # choose smallest df (=widest unambiguous delay range)
        pick = sorted(pick_pool, key=lambda x: x[1])[0]
        return float(pick[0]), float(pick[1]), str(pick[2])
    ms = dict(cfg.get("measurement_sweep", {}))
    bw = float(ms.get("BW_Hz", 1.0e9))
    df = float(ms.get("df_Hz", 1.0e6))
    return bw, df, "config.measurement_sweep"


def _group_rays_by_case(ray_rows: list[dict[str, Any]], scenario_id: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in ray_rows:
        if str(r.get("scenario_id", "")) != str(scenario_id):
            continue
        out.setdefault(str(r.get("case_id", "")), []).append(r)
    return out


def _floor_window_contamination(
    ray_rows: list[dict[str, Any]],
    *,
    w_floor_s: float,
) -> dict[str, Any]:
    by_case = _group_rays_by_case(ray_rows, "C0")
    c_vals_db: list[float] = []
    for _, cr in by_case.items():
        los = [x for x in cr if int(round(_num(x.get("n_bounce", np.nan)))) == 0]
        if not los:
            continue
        m_los = sorted(los, key=lambda x: _num(x.get("P_lin", np.nan)), reverse=True)[0]
        tau_los = _num(m_los.get("tau_s", np.nan))
        p_los = _num(m_los.get("P_lin", np.nan))
        if not np.isfinite(tau_los) or not np.isfinite(p_los) or p_los <= 0:
            continue
        t0 = float(tau_los - 0.5 * float(w_floor_s))
        t1 = float(tau_los + 0.5 * float(w_floor_s))
        p_non = 0.0
        for r in cr:
            if r is m_los:
                continue
            tau = _num(r.get("tau_s", np.nan))
            p = _num(r.get("P_lin", np.nan))
            if not np.isfinite(tau) or not np.isfinite(p) or p <= 0:
                continue
            if int(round(_num(r.get("n_bounce", np.nan)))) == 0:
                continue
            if t0 <= tau <= t1:
                p_non += float(p)
        c_vals_db.append(_db_ratio(p_non, p_los))
    arr = np.asarray(c_vals_db, dtype=float)
    med = float(np.nanmedian(arr)) if len(arr) else float("nan")
    p95 = float(np.nanpercentile(arr, 95.0)) if len(arr) else float("nan")
    status = "PASS" if (np.isfinite(med) and med < -15.0) else ("WARN" if (np.isfinite(med) and med < -10.0) else "FAIL")
    return {
        "n_cases": int(len(arr)),
        "W_floor_s": float(w_floor_s),
        "C_floor_median_db": med,
        "C_floor_p95_db": p95,
        "rate_below_m10_db": float(np.mean(arr < -10.0)) if len(arr) else float("nan"),
        "rate_below_m15_db": float(np.mean(arr < -15.0)) if len(arr) else float("nan"),
        "status": status,
    }


def _target_window_stats(
    ray_rows: list[dict[str, Any]],
    *,
    scenario_id: str,
    target_n: int,
    w_target_s: float,
    te_s: float,
    dt_res_s: float,
) -> dict[str, Any]:
    by_case = _group_rays_by_case(ray_rows, scenario_id)
    c_vals_db: list[float] = []
    early_ok: list[bool] = []
    first_ok: list[bool] = []
    gap_vals: list[float] = []
    exists = 0
    total = 0
    tol = 0.25 * float(dt_res_s) if np.isfinite(dt_res_s) else 0.0
    for _, cr in by_case.items():
        total += 1
        cand = [x for x in cr if int(round(_num(x.get("n_bounce", np.nan)))) == int(target_n)]
        if not cand:
            continue
        exists += 1
        m_tar = sorted(cand, key=lambda x: _num(x.get("P_lin", np.nan)), reverse=True)[0]
        tau_tar = _num(m_tar.get("tau_s", np.nan))
        p_tar = _num(m_tar.get("P_lin", np.nan))
        if not np.isfinite(tau_tar) or not np.isfinite(p_tar) or p_tar <= 0:
            continue
        all_tau = np.asarray([_num(x.get("tau_s", np.nan)) for x in cr], dtype=float)
        all_tau = all_tau[np.isfinite(all_tau)]
        if len(all_tau) == 0:
            continue
        tau0 = float(np.min(all_tau))
        early_ok.append(bool((tau_tar - tau0) <= float(te_s)))
        first_ok.append(bool(abs(tau_tar - tau0) <= float(tol)))
        t0 = float(tau_tar - 0.5 * float(w_target_s))
        t1 = float(tau_tar + 0.5 * float(w_target_s))
        p_other = 0.0
        gaps: list[float] = []
        for r in cr:
            if r is m_tar:
                continue
            tau = _num(r.get("tau_s", np.nan))
            p = _num(r.get("P_lin", np.nan))
            if not np.isfinite(tau) or not np.isfinite(p) or p <= 0:
                continue
            gaps.append(abs(float(tau) - float(tau_tar)))
            if t0 <= tau <= t1:
                p_other += float(p)
        c_vals_db.append(_db_ratio(p_other, p_tar))
        if gaps:
            gap_vals.append(float(np.min(gaps)))
    arr_c = np.asarray(c_vals_db, dtype=float)
    arr_gap = np.asarray(gap_vals, dtype=float)
    med_c = float(np.nanmedian(arr_c)) if len(arr_c) else float("nan")
    med_gap = float(np.nanmedian(arr_gap)) if len(arr_gap) else float("nan")
    target_exists_rate = float(exists / total) if total > 0 else float("nan")
    target_early_rate = float(np.mean(early_ok)) if early_ok else float("nan")
    target_first_rate = float(np.mean(first_ok)) if first_ok else float("nan")
    # Ideal: contamination < -10~-15 dB + target inside early.
    status = "PASS" if (
        np.isfinite(med_c)
        and med_c < -10.0
        and np.isfinite(target_early_rate)
        and target_early_rate >= 0.8
        and np.isfinite(target_exists_rate)
        and target_exists_rate >= 0.9
    ) else ("WARN" if (np.isfinite(med_c) and med_c < -6.0 and np.isfinite(target_exists_rate) and target_exists_rate >= 0.7) else "FAIL")
    return {
        "scenario": str(scenario_id),
        "target_n": int(target_n),
        "W_target_s": float(w_target_s),
        "target_exists_rate": target_exists_rate,
        "target_in_Wearly_rate": target_early_rate,
        "target_is_first_rate": target_first_rate,
        "C_target_median_db": med_c,
        "C_target_p95_db": float(np.nanpercentile(arr_c, 95.0)) if len(arr_c) else float("nan"),
        "target_gap_median_s": med_gap,
        "status": status,
    }


def _estimate_target_gap_median(
    ray_rows: list[dict[str, Any]],
    *,
    scenario_id: str,
    target_n: int,
) -> float:
    by_case = _group_rays_by_case(ray_rows, scenario_id)
    gap_vals: list[float] = []
    for _, cr in by_case.items():
        cand = [x for x in cr if int(round(_num(x.get("n_bounce", np.nan)))) == int(target_n)]
        if not cand:
            continue
        m_tar = sorted(cand, key=lambda x: _num(x.get("P_lin", np.nan)), reverse=True)[0]
        tau_tar = _num(m_tar.get("tau_s", np.nan))
        if not np.isfinite(tau_tar):
            continue
        local_gaps: list[float] = []
        for r in cr:
            if r is m_tar:
                continue
            tau = _num(r.get("tau_s", np.nan))
            if np.isfinite(tau):
                local_gaps.append(abs(float(tau) - float(tau_tar)))
        if local_gaps:
            gap_vals.append(float(np.min(np.asarray(local_gaps, dtype=float))))
    return float(np.nanmedian(np.asarray(gap_vals, dtype=float))) if gap_vals else float("nan")


def _resolve_target_window_by_scenario(
    *,
    ws: dict[str, Any],
    ray_rows: list[dict[str, Any]],
    target_map: dict[str, int],
    default_w_target_s: float,
    dt_res_s: float,
) -> tuple[dict[str, float], dict[str, str]]:
    out_w = {sid: float(default_w_target_s) for sid in target_map.keys()}
    out_mode = {sid: "default" for sid in target_map.keys()}

    ns_override = ws.get("target_window_ns_by_scenario", {})
    if isinstance(ns_override, dict):
        for k, v in ns_override.items():
            sid = str(k).upper().strip()
            vv = _num(v)
            if sid in out_w and np.isfinite(vv) and vv > 0:
                out_w[sid] = float(vv) * 1e-9
                out_mode[sid] = "fixed"

    a3_legacy = _num(ws.get("A3_target_window_ns", np.nan))
    if np.isfinite(a3_legacy) and a3_legacy > 0:
        out_w["A3"] = float(a3_legacy) * 1e-9
        out_mode["A3"] = "fixed"

    mode_cfg = ws.get("target_window_mode_by_scenario", {})
    mode_map: dict[str, str] = {}
    if isinstance(mode_cfg, dict):
        for k, v in mode_cfg.items():
            mode_map[str(k).upper().strip()] = str(v).lower().strip()

    for sid, tn in target_map.items():
        if mode_map.get(sid, "") != "adaptive":
            continue
        gap_med = _estimate_target_gap_median(ray_rows, scenario_id=sid, target_n=int(tn))
        w = float(out_w[sid])
        if np.isfinite(gap_med) and gap_med > 0:
            # Keep target window below nearest-path median gap to reduce contamination.
            w = min(w, 0.8 * float(gap_med))
        if np.isfinite(dt_res_s) and dt_res_s > 0:
            w = max(w, 2.0 * float(dt_res_s))
        w = max(w, 0.5e-9)
        out_w[sid] = float(w)
        out_mode[sid] = "adaptive"
    return out_w, out_mode


def _dominant_target_tau_by_link(
    ray_rows: list[dict[str, Any]],
    *,
    scenario_id: str,
    target_n: int,
) -> dict[str, float]:
    out: dict[str, tuple[float, float]] = {}
    for r in ray_rows:
        if str(r.get("scenario_id", "")) != str(scenario_id):
            continue
        nb = int(round(_num(r.get("n_bounce", np.nan)))) if np.isfinite(_num(r.get("n_bounce", np.nan))) else -1
        if nb != int(target_n):
            continue
        lid = str(r.get("link_id", ""))
        if not lid:
            continue
        p = _num(r.get("P_lin", np.nan))
        tau = _num(r.get("tau_s", np.nan))
        if not (np.isfinite(p) and np.isfinite(tau) and p > 0):
            continue
        cur = out.get(lid)
        if cur is None or p > cur[0]:
            out[lid] = (float(p), float(tau))
    return {k: float(v[1]) for k, v in out.items()}


def _target_window_sign_metric(
    *,
    link_rows: list[dict[str, Any]],
    ray_rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    scenario_id: str,
    target_n: int,
    w_target_s: float,
    floor_db: float,
    expected_sign: int,
    sign_metric: str = "excess",
    row_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run = None
    for rr in runs:
        if str(rr.get("scenario_id", "")) == str(scenario_id):
            run = rr
            break
    rows = [r for r in link_rows if str(r.get("scenario_id", "")) == str(scenario_id)]
    ff = dict(row_filter or {})
    if ff:
        out_rows: list[dict[str, Any]] = []
        for rr in rows:
            ok = True
            for k, v in ff.items():
                if str(rr.get(k, "")).strip().lower() != str(v).strip().lower():
                    ok = False
                    break
            if ok:
                out_rows.append(rr)
        rows = out_rows
    if run is None or len(rows) == 0:
        return {"scenario": str(scenario_id), "status": "WARN", "reason": "missing run or rows"}

    tau_by_link = _dominant_target_tau_by_link(ray_rows, scenario_id=scenario_id, target_n=target_n)
    vals_ex: list[float] = []
    vals_raw: list[float] = []
    hits_ex: list[bool] = []
    hits_raw: list[bool] = []
    for r in rows:
        lid = str(r.get("link_id", ""))
        tau_tar = _num(tau_by_link.get(lid, np.nan))
        if not np.isfinite(tau_tar):
            continue
        pdp = io_lib.load_pdp_npz(run, lid)
        if pdp is None:
            continue
        tau = np.asarray(pdp.get("delay_tau_s", []), dtype=float)
        pco = np.asarray(pdp.get("P_co", []), dtype=float)
        pcr = np.asarray(pdp.get("P_cross", []), dtype=float)
        if len(tau) == 0 or len(pco) != len(tau) or len(pcr) != len(tau):
            continue
        t0 = float(tau_tar - 0.5 * float(w_target_s))
        t1 = float(tau_tar + 0.5 * float(w_target_s))
        m = (tau >= t0) & (tau <= t1)
        sco = float(np.sum(pco[m]))
        scr = float(np.sum(pcr[m]))
        xpd_target = _db_ratio(sco, scr)
        xpd_target_ex = float(xpd_target - floor_db) if np.isfinite(floor_db) else float("nan")
        if not np.isfinite(xpd_target):
            continue
        vals_raw.append(float(xpd_target))
        vals_ex.append(float(xpd_target_ex) if np.isfinite(xpd_target_ex) else float("nan"))
        hits_raw.append(bool(xpd_target < 0.0) if int(expected_sign) < 0 else bool(xpd_target > 0.0))
        if np.isfinite(xpd_target_ex):
            hits_ex.append(bool(xpd_target_ex < 0.0) if int(expected_sign) < 0 else bool(xpd_target_ex > 0.0))

    arr_ex = np.asarray(vals_ex, dtype=float)
    arr_raw = np.asarray(vals_raw, dtype=float)
    med_ex = float(np.nanmedian(arr_ex)) if len(arr_ex) else float("nan")
    p10_ex = float(np.nanpercentile(arr_ex, 10.0)) if len(arr_ex) else float("nan")
    p90_ex = float(np.nanpercentile(arr_ex, 90.0)) if len(arr_ex) else float("nan")
    med_raw = float(np.nanmedian(arr_raw)) if len(arr_raw) else float("nan")
    p10_raw = float(np.nanpercentile(arr_raw, 10.0)) if len(arr_raw) else float("nan")
    p90_raw = float(np.nanpercentile(arr_raw, 90.0)) if len(arr_raw) else float("nan")
    hit_ex = float(np.mean(hits_ex)) if hits_ex else float("nan")
    hit_raw = float(np.mean(hits_raw)) if hits_raw else float("nan")

    mode = str(sign_metric).strip().lower()
    if mode not in {"excess", "raw"}:
        mode = "excess"
    hit = hit_raw if mode == "raw" else hit_ex
    if np.isfinite(hit) and hit >= 0.8:
        st = "PASS"
    elif np.isfinite(hit) and hit >= 0.6:
        st = "WARN"
    elif np.isfinite(hit):
        st = "FAIL"
    else:
        st = "INCONCLUSIVE"
    return {
        "scenario": str(scenario_id),
        "target_n": int(target_n),
        "W_target_s": float(w_target_s),
        "expected_sign": "negative" if int(expected_sign) < 0 else "positive",
        "n_links": int(len(rows)),
        "n_eval": int(len(arr_raw)),
        "median_xpd_target_ex_db": med_ex,
        "p10_xpd_target_ex_db": p10_ex,
        "p90_xpd_target_ex_db": p90_ex,
        "median_xpd_target_raw_db": med_raw,
        "p10_xpd_target_raw_db": p10_raw,
        "p90_xpd_target_raw_db": p90_raw,
        "expected_sign_hit_rate_ex": hit_ex,
        "expected_sign_hit_rate_raw": hit_raw,
        "sign_metric_for_status": mode,
        "expected_sign_hit_rate": hit,
        "status": st,
    }


def _a6_parity_benchmark(
    *,
    link_rows: list[dict[str, Any]],
    ray_rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    w_target_s: float,
    floor_db: float,
    sign_metric: str = "raw",
) -> dict[str, Any]:
    """Near-normal parity benchmark for A6 (odd/even separated by target bounce)."""
    odd = _target_window_sign_metric(
        link_rows=link_rows,
        ray_rows=ray_rows,
        runs=runs,
        scenario_id="A6",
        target_n=1,
        w_target_s=float(w_target_s),
        floor_db=float(floor_db),
        expected_sign=-1,
        sign_metric=str(sign_metric),
    )
    even = _target_window_sign_metric(
        link_rows=link_rows,
        ray_rows=ray_rows,
        runs=runs,
        scenario_id="A6",
        target_n=2,
        w_target_s=float(w_target_s),
        floor_db=float(floor_db),
        expected_sign=+1,
        sign_metric=str(sign_metric),
    )
    n_odd = int(_num(odd.get("n_eval", np.nan))) if np.isfinite(_num(odd.get("n_eval", np.nan))) else 0
    n_even = int(_num(even.get("n_eval", np.nan))) if np.isfinite(_num(even.get("n_eval", np.nan))) else 0
    available = bool((n_odd + n_even) > 0)
    hit_odd = _num(odd.get("expected_sign_hit_rate", np.nan))
    hit_even = _num(even.get("expected_sign_hit_rate", np.nan))
    hit_min = float(min(hit_odd, hit_even)) if np.isfinite(hit_odd) and np.isfinite(hit_even) else float("nan")
    if not available:
        st = "INCONCLUSIVE"
    elif np.isfinite(hit_min) and hit_min >= 0.8:
        st = "PASS"
    elif np.isfinite(hit_min) and hit_min >= 0.6:
        st = "WARN"
    elif np.isfinite(hit_min):
        st = "FAIL"
    else:
        st = "INCONCLUSIVE"
    a6_rows = [r for r in link_rows if str(r.get("scenario_id", "")) == "A6"]
    set_names = sorted(
        {
            str(r.get("a6_case_set", "full")).strip().lower()
            for r in a6_rows
            if str(r.get("a6_case_set", "full")).strip()
        }
    )
    if not set_names and available:
        set_names = ["full"]
    per_set: dict[str, Any] = {}
    for ss in set_names:
        odd_s = _target_window_sign_metric(
            link_rows=link_rows,
            ray_rows=ray_rows,
            runs=runs,
            scenario_id="A6",
            target_n=1,
            w_target_s=float(w_target_s),
            floor_db=float(floor_db),
            expected_sign=-1,
            sign_metric=str(sign_metric),
            row_filter={"a6_case_set": str(ss)},
        )
        even_s = _target_window_sign_metric(
            link_rows=link_rows,
            ray_rows=ray_rows,
            runs=runs,
            scenario_id="A6",
            target_n=2,
            w_target_s=float(w_target_s),
            floor_db=float(floor_db),
            expected_sign=+1,
            sign_metric=str(sign_metric),
            row_filter={"a6_case_set": str(ss)},
        )
        h_odd = _num(odd_s.get("expected_sign_hit_rate", np.nan))
        h_even = _num(even_s.get("expected_sign_hit_rate", np.nan))
        h_min = float(min(h_odd, h_even)) if np.isfinite(h_odd) and np.isfinite(h_even) else float("nan")
        if np.isfinite(h_min) and h_min >= 0.8:
            st_set = "PASS"
        elif np.isfinite(h_min) and h_min >= 0.6:
            st_set = "WARN"
        elif np.isfinite(h_min):
            st_set = "FAIL"
        else:
            st_set = "INCONCLUSIVE"
        per_set[str(ss)] = {
            "status": str(st_set),
            "hit_rate_min": float(h_min) if np.isfinite(h_min) else float("nan"),
            "odd": odd_s,
            "even": even_s,
        }

    full_vs_minimal: dict[str, Any] = {}
    if "full" in per_set and "minimal" in per_set:
        for mode in ["odd", "even"]:
            f0 = dict(per_set.get("full", {})).get(mode, {})
            m0 = dict(per_set.get("minimal", {})).get(mode, {})
            full_vs_minimal[mode] = {
                "n_eval_full": int(_num(dict(f0).get("n_eval", np.nan))) if np.isfinite(_num(dict(f0).get("n_eval", np.nan))) else 0,
                "n_eval_minimal": int(_num(dict(m0).get("n_eval", np.nan))) if np.isfinite(_num(dict(m0).get("n_eval", np.nan))) else 0,
                "median_raw_full_db": _num(dict(f0).get("median_xpd_target_raw_db", np.nan)),
                "median_raw_minimal_db": _num(dict(m0).get("median_xpd_target_raw_db", np.nan)),
                "delta_raw_minimal_minus_full_db": _num(dict(m0).get("median_xpd_target_raw_db", np.nan))
                - _num(dict(f0).get("median_xpd_target_raw_db", np.nan)),
                "median_ex_full_db": _num(dict(f0).get("median_xpd_target_ex_db", np.nan)),
                "median_ex_minimal_db": _num(dict(m0).get("median_xpd_target_ex_db", np.nan)),
                "delta_ex_minimal_minus_full_db": _num(dict(m0).get("median_xpd_target_ex_db", np.nan))
                - _num(dict(f0).get("median_xpd_target_ex_db", np.nan)),
                "hit_rate_full": _num(dict(f0).get("expected_sign_hit_rate", np.nan)),
                "hit_rate_minimal": _num(dict(m0).get("expected_sign_hit_rate", np.nan)),
                "delta_hit_rate_minimal_minus_full": _num(dict(m0).get("expected_sign_hit_rate", np.nan))
                - _num(dict(f0).get("expected_sign_hit_rate", np.nan)),
            }

    return {
        "available": bool(available),
        "status": str(st),
        "sign_metric_for_status": str(sign_metric),
        "n_eval_total": int(n_odd + n_even),
        "n_eval_odd": int(n_odd),
        "n_eval_even": int(n_even),
        "hit_rate_odd": float(hit_odd) if np.isfinite(hit_odd) else float("nan"),
        "hit_rate_even": float(hit_even) if np.isfinite(hit_even) else float("nan"),
        "hit_rate_min": float(hit_min) if np.isfinite(hit_min) else float("nan"),
        "odd": odd,
        "even": even,
        "case_set_compare": {
            "available_sets": list(set_names),
            "per_set": per_set,
            "full_vs_minimal": full_vs_minimal,
        },
        "note": "A6 is near-normal parity benchmark; use as primary G2 evidence when available.",
    }


def _stress_flag_row(r: dict[str, Any]) -> int:
    mode = str(r.get("stress_mode", "")).strip().lower()
    rf = int(round(_num(r.get("roughness_flag", 0)))) if np.isfinite(_num(r.get("roughness_flag", np.nan))) else 0
    hf = int(round(_num(r.get("human_flag", 0)))) if np.isfinite(_num(r.get("human_flag", np.nan))) else 0
    if rf == 1 or hf == 1:
        return 1
    if mode in {"geometry", "hybrid", "synthetic", "stress", "on"}:
        return 1
    return 0


def _build_case_level_rows(link_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in link_rows:
        rows.append(
            {
                "scenario_id": str(r.get("scenario_id", "")),
                "case_id": str(r.get("case_id", "")),
                "link_id": str(r.get("link_id", "")),
                "xpd_early_ex_db": _num(r.get("XPD_early_excess_db", np.nan)),
                "xpd_late_ex_db": _num(r.get("XPD_late_excess_db", np.nan)),
                "l_pol_db": _num(r.get("L_pol_db", np.nan)),
                "rho_early_linear": _num(r.get("rho_early_lin", np.nan)),
                "rho_early_db": _num(r.get("rho_early_db", np.nan)),
                "ds_ns": _num(r.get("delay_spread_rms_s", np.nan)) * 1e9,
                "early_energy_fraction": _num(r.get("early_energy_fraction", np.nan)),
                "EL_proxy_db": _num(r.get("EL_proxy_db", np.nan)),
                "LOSflag": int(round(_num(r.get("LOSflag", np.nan)))) if np.isfinite(_num(r.get("LOSflag", np.nan))) else "",
                "material": str(r.get("material_class", "")),
                "stress_flag": int(_stress_flag_row(r)),
                "claim_caution_early": int(round(_num(r.get("claim_caution_early", 0)))) if np.isfinite(_num(r.get("claim_caution_early", np.nan))) else 0,
                "claim_caution_late": int(round(_num(r.get("claim_caution_late", 0)))) if np.isfinite(_num(r.get("claim_caution_late", np.nan))) else 0,
            }
        )
    return rows


def _build_target_level_rows(
    *,
    link_rows: list[dict[str, Any]],
    ray_rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    floor_db: float,
    te_default_s: float,
    dt_res_s: float,
    w_target_s_by_sid: dict[str, float],
    target_map: dict[str, int],
) -> list[dict[str, Any]]:
    run_by_sid = {str(r.get("scenario_id", "")): r for r in runs}
    rays_by_link: dict[str, list[dict[str, Any]]] = {}
    for rr in ray_rows:
        lid = str(rr.get("link_id", ""))
        if lid:
            rays_by_link.setdefault(lid, []).append(rr)

    tol = 0.25 * float(dt_res_s) if np.isfinite(dt_res_s) and dt_res_s > 0 else 0.0
    out: list[dict[str, Any]] = []
    for r in link_rows:
        sid = str(r.get("scenario_id", ""))
        if sid not in target_map:
            continue
        lid = str(r.get("link_id", ""))
        run = run_by_sid.get(sid)
        rr = rays_by_link.get(lid, [])
        target_n = int(target_map[sid])
        cand = [x for x in rr if int(round(_num(x.get("n_bounce", np.nan)))) == target_n and _num(x.get("P_lin", np.nan)) > 0]
        if not cand:
            continue
        m_tar = sorted(cand, key=lambda x: _num(x.get("P_lin", np.nan)), reverse=True)[0]
        tau_tar = _num(m_tar.get("tau_s", np.nan))
        p_tar = _num(m_tar.get("P_lin", np.nan))
        inc_deg = _num(m_tar.get("incidence_deg", np.nan))
        nb = int(round(_num(m_tar.get("n_bounce", np.nan)))) if np.isfinite(_num(m_tar.get("n_bounce", np.nan))) else target_n
        parity = "even" if (int(nb) % 2 == 0) else "odd"

        all_tau = np.asarray([_num(x.get("tau_s", np.nan)) for x in rr], dtype=float)
        all_tau = all_tau[np.isfinite(all_tau)]
        tau_first = float(np.min(all_tau)) if len(all_tau) else float("nan")
        target_rank = float("nan")
        if np.isfinite(tau_tar) and len(all_tau):
            taus = np.sort(np.unique(all_tau))
            target_rank = float(int(np.argmin(np.abs(taus - float(tau_tar)))) + 1)

        w = _window_dict(r)
        tau0 = _num(w.get("tau0_s", np.nan))
        te_s = _num(w.get("Te_s", np.nan))
        if not np.isfinite(te_s):
            te_s = float(te_default_s)
        if not np.isfinite(tau0):
            tau0 = tau_first
        in_early = int(np.isfinite(tau_tar) and np.isfinite(tau0) and (float(tau_tar) - float(tau0) <= float(te_s)))
        is_first = int(np.isfinite(tau_tar) and np.isfinite(tau_first) and abs(float(tau_tar) - float(tau_first)) <= float(tol))

        w_target_s = float(w_target_s_by_sid.get(sid, max(2e-9, 0.8 * float(te_default_s))))
        pco_t = float("nan")
        pcx_t = float("nan")
        xpd_raw = float("nan")
        xpd_ex = float("nan")
        if run is not None and np.isfinite(tau_tar):
            pdp = io_lib.load_pdp_npz(run, lid)
            if pdp is not None:
                tau = np.asarray(pdp.get("delay_tau_s", []), dtype=float)
                pco = np.asarray(pdp.get("P_co", []), dtype=float)
                pcx = np.asarray(pdp.get("P_cross", []), dtype=float)
                if len(tau) and len(pco) == len(tau) and len(pcx) == len(tau):
                    t0 = float(tau_tar - 0.5 * w_target_s)
                    t1 = float(tau_tar + 0.5 * w_target_s)
                    m = (tau >= t0) & (tau <= t1)
                    pco_t = float(np.sum(pco[m]))
                    pcx_t = float(np.sum(pcx[m]))
                    xpd_raw = _db_ratio(pco_t, pcx_t)
                    xpd_ex = float(xpd_raw - floor_db) if np.isfinite(floor_db) else float("nan")

        c_target = float("nan")
        if np.isfinite(tau_tar) and np.isfinite(p_tar) and p_tar > 0:
            t0 = float(tau_tar - 0.5 * w_target_s)
            t1 = float(tau_tar + 0.5 * w_target_s)
            p_other = 0.0
            for x in rr:
                if x is m_tar:
                    continue
                tau_x = _num(x.get("tau_s", np.nan))
                p_x = _num(x.get("P_lin", np.nan))
                if np.isfinite(tau_x) and np.isfinite(p_x) and p_x > 0 and (t0 <= tau_x <= t1):
                    p_other += float(p_x)
            c_target = _db_ratio(p_other, p_tar)

        out.append(
            {
                "scenario_id": sid,
                "case_id": str(r.get("case_id", "")),
                "link_id": lid,
                "target_tau_ns": float(tau_tar) * 1e9 if np.isfinite(tau_tar) else float("nan"),
                "target_rank": target_rank,
                "target_in_Wearly": int(in_early),
                "target_is_first": int(is_first),
                "bounce_count": int(nb),
                "parity": parity,
                "incidence_angle_deg": float(inc_deg) if np.isfinite(inc_deg) else float("nan"),
                "Pco_target": pco_t,
                "Pcross_target": pcx_t,
                "xpd_target_raw_db": xpd_raw,
                "xpd_target_ex_db": xpd_ex,
                "C_target_db": c_target,
            }
        )
    return out


def _build_sensitivity_level_rows(
    *,
    link_rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    floor_db: float,
    te_ns_list: list[float],
    noise_tail_ns_list: list[float],
    threshold_db_list: list[float],
    tmax_s: float,
) -> list[dict[str, Any]]:
    run_by_sid = {str(r.get("scenario_id", "")): r for r in runs}
    out: list[dict[str, Any]] = []
    for r in link_rows:
        sid = str(r.get("scenario_id", ""))
        run = run_by_sid.get(sid)
        if run is None:
            continue
        lid = str(r.get("link_id", ""))
        pdp = io_lib.load_pdp_npz(run, lid)
        if pdp is None:
            continue
        tau = np.asarray(pdp.get("delay_tau_s", []), dtype=float)
        pco = np.asarray(pdp.get("P_co", []), dtype=float)
        pcx = np.asarray(pdp.get("P_cross", []), dtype=float)
        if len(tau) == 0 or len(pco) != len(tau) or len(pcx) != len(tau):
            continue

        base_xpd_e_ex = _num(r.get("XPD_early_excess_db", np.nan))
        base_xpd_l_ex = _num(r.get("XPD_late_excess_db", np.nan))
        base_lpol = _num(r.get("L_pol_db", np.nan))
        base_rho_db = _num(r.get("rho_early_db", np.nan))
        base_ds_ns = _num(r.get("delay_spread_rms_s", np.nan)) * 1e9
        base_ef = _num(r.get("early_energy_fraction", np.nan))

        p_total = pco + pcx
        for te_ns in te_ns_list:
            for noise_ns in noise_tail_ns_list:
                for thr_db in threshold_db_list:
                    det = windowing_lib.estimate_tau0(
                        tau,
                        p_total,
                        method="threshold",
                        noise_tail_s=float(noise_ns) * 1e-9,
                        margin_db=float(thr_db),
                    )
                    tau0 = float(det.get("tau0_s", 0.0))
                    early, late = windowing_lib.make_early_late_masks(
                        tau,
                        tau0_s=tau0,
                        Te_s=float(te_ns) * 1e-9,
                        Tmax_s=float(tmax_s),
                    )
                    met = link_metrics_lib.compute_link_metrics(
                        {"P_co": pco, "P_cross": pcx},
                        tau,
                        (early, late),
                        ds_reference="total",
                        window_params={
                            "tau0_s": tau0,
                            "Te_s": float(te_ns) * 1e-9,
                            "Tmax_s": float(tmax_s),
                            "noise_tail_ns": float(noise_ns),
                            "threshold_db": float(thr_db),
                            "method": "threshold",
                        },
                    )
                    xpd_e_ex = float(met.XPD_early_db - floor_db) if np.isfinite(floor_db) else float("nan")
                    xpd_l_ex = float(met.XPD_late_db - floor_db) if np.isfinite(floor_db) else float("nan")
                    ds_ns = float(met.delay_spread_rms_s) * 1e9 if np.isfinite(_num(met.delay_spread_rms_s)) else float("nan")
                    out.append(
                        {
                            "scenario_id": sid,
                            "case_id": str(r.get("case_id", "")),
                            "link_id": lid,
                            "Te_ns": float(te_ns),
                            "noise_tail_ns": float(noise_ns),
                            "threshold_db": float(thr_db),
                            "xpd_early_ex_db": xpd_e_ex,
                            "xpd_late_ex_db": xpd_l_ex,
                            "l_pol_db": float(met.L_pol_db),
                            "rho_early_linear": float(met.rho_early_lin),
                            "rho_early_db": float(met.rho_early_db),
                            "ds_ns": ds_ns,
                            "early_energy_fraction": float(met.early_energy_fraction),
                            "delta_xpd_early_ex_db": float(xpd_e_ex - base_xpd_e_ex) if np.isfinite(xpd_e_ex) and np.isfinite(base_xpd_e_ex) else float("nan"),
                            "delta_xpd_late_ex_db": float(xpd_l_ex - base_xpd_l_ex) if np.isfinite(xpd_l_ex) and np.isfinite(base_xpd_l_ex) else float("nan"),
                            "delta_l_pol_db": float(met.L_pol_db - base_lpol) if np.isfinite(_num(met.L_pol_db)) and np.isfinite(base_lpol) else float("nan"),
                            "delta_rho_early_db": float(met.rho_early_db - base_rho_db) if np.isfinite(_num(met.rho_early_db)) and np.isfinite(base_rho_db) else float("nan"),
                            "delta_ds_ns": float(ds_ns - base_ds_ns) if np.isfinite(ds_ns) and np.isfinite(base_ds_ns) else float("nan"),
                            "delta_early_energy_fraction": float(met.early_energy_fraction - base_ef) if np.isfinite(_num(met.early_energy_fraction)) and np.isfinite(base_ef) else float("nan"),
                        }
                    )
    return out


def _impute_missing_el_proxy(
    link_rows: list[dict[str, Any]],
    ray_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_link_best: dict[str, tuple[float, float]] = {}
    for r in ray_rows:
        lid = str(r.get("link_id", ""))
        if not lid:
            continue
        p = _num(r.get("P_lin", np.nan))
        el = _num(r.get("EL_db", np.nan))
        if not (np.isfinite(p) and np.isfinite(el) and p > 0):
            continue
        cur = by_link_best.get(lid)
        if cur is None or p > cur[0]:
            by_link_best[lid] = (float(p), float(el))

    # scenario-level fallback median for rows with no usable rays
    scenario_el_vals: dict[str, list[float]] = {}
    for r in link_rows:
        sid = str(r.get("scenario_id", ""))
        el = _num(r.get("EL_proxy_db", np.nan))
        if np.isfinite(el):
            scenario_el_vals.setdefault(sid, []).append(float(el))
    scenario_el_median: dict[str, float] = {}
    for sid, vals in scenario_el_vals.items():
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            scenario_el_median[sid] = float(np.median(v))

    global_med = float(
        np.median(
            np.asarray(
                [x for x in scenario_el_median.values() if np.isfinite(_num(x))],
                dtype=float,
            )
        )
    ) if scenario_el_median else float("nan")

    out: list[dict[str, Any]] = []
    imputed = 0
    rows_changed: list[dict[str, Any]] = []
    unresolved = 0
    for r in link_rows:
        rr = dict(r)
        el = _num(rr.get("EL_proxy_db", np.nan))
        if not np.isfinite(el):
            lid = str(rr.get("link_id", ""))
            sid = str(rr.get("scenario_id", ""))
            if lid in by_link_best:
                rr["EL_proxy_db"] = float(by_link_best[lid][1])
                rr["EL_proxy_imputed"] = 1
                imputed += 1
                rows_changed.append(
                    {
                        "scenario_id": str(rr.get("scenario_id", "")),
                        "case_id": str(rr.get("case_id", "")),
                        "link_id": lid,
                        "EL_proxy_db": float(rr["EL_proxy_db"]),
                        "method": "dominant_ray_EL_db",
                    }
                )
            elif sid in scenario_el_median:
                rr["EL_proxy_db"] = float(scenario_el_median[sid])
                rr["EL_proxy_imputed"] = 1
                imputed += 1
                rows_changed.append(
                    {
                        "scenario_id": str(rr.get("scenario_id", "")),
                        "case_id": str(rr.get("case_id", "")),
                        "link_id": lid,
                        "EL_proxy_db": float(rr["EL_proxy_db"]),
                        "method": "scenario_median_EL_proxy_db",
                    }
                )
            elif np.isfinite(global_med):
                rr["EL_proxy_db"] = float(global_med)
                rr["EL_proxy_imputed"] = 1
                imputed += 1
                rows_changed.append(
                    {
                        "scenario_id": str(rr.get("scenario_id", "")),
                        "case_id": str(rr.get("case_id", "")),
                        "link_id": lid,
                        "EL_proxy_db": float(rr["EL_proxy_db"]),
                        "method": "global_median_EL_proxy_db",
                    }
                )
            else:
                unresolved += 1
        out.append(rr)
    info = {
        "imputed_count": int(imputed),
        "unresolved_count": int(unresolved),
        "rows": rows_changed,
    }
    return out, info


def _sep_score(a: np.ndarray, b: np.ndarray) -> float:
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    xa = xa[np.isfinite(xa)]
    xb = xb[np.isfinite(xb)]
    if len(xa) < 2 or len(xb) < 2:
        return float("nan")
    mu = abs(float(np.mean(xa) - np.mean(xb)))
    var = float(np.var(xa, ddof=1) + np.var(xb, ddof=1))
    if not np.isfinite(mu) or not np.isfinite(var):
        return float("nan")
    return float(mu / np.sqrt(max(var, 1e-18)))


def _wearly_te_sweep(
    *,
    link_rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    te_ns_list: list[float],
    tmax_s: float,
) -> dict[str, Any]:
    run_by_sid = {str(r.get("scenario_id", "")): r for r in runs}
    b_rows = [r for r in link_rows if str(r.get("scenario_id", "")) in {"B1", "B2", "B3"}]
    if not b_rows:
        return {"status": "WARN", "reason": "No B-scenario links"}
    vals: dict[float, dict[str, dict[int, list[float]]]] = {}
    for te_ns in te_ns_list:
        vals[float(te_ns)] = {
            "xpd_early": {0: [], 1: []},
            "rho_early_db": {0: [], 1: []},
            "l_pol": {0: [], 1: []},
        }
    used_links = 0
    for r in b_rows:
        sid = str(r.get("scenario_id", ""))
        run = run_by_sid.get(sid)
        if run is None:
            continue
        pdp = io_lib.load_pdp_npz(run, str(r.get("link_id", "")))
        if pdp is None:
            continue
        tau = np.asarray(pdp.get("delay_tau_s", []), dtype=float)
        pco = np.asarray(pdp.get("P_co", []), dtype=float)
        pcr = np.asarray(pdp.get("P_cross", []), dtype=float)
        if len(tau) == 0 or len(pco) != len(tau) or len(pcr) != len(tau):
            continue
        lf = int(round(_num(r.get("LOSflag", np.nan)))) if np.isfinite(_num(r.get("LOSflag", np.nan))) else -1
        if lf not in {0, 1}:
            continue
        w = _window_dict(r)
        tau0 = _num(w.get("tau0_s", np.nan))
        if not np.isfinite(tau0):
            p_total = pco + pcr
            imax = int(np.argmax(p_total)) if len(p_total) else -1
            if imax < 0:
                continue
            tau0 = float(tau[imax])
        used_links += 1
        for te_ns in te_ns_list:
            te_s = float(te_ns) * 1e-9
            early = (tau >= tau0) & (tau <= tau0 + te_s)
            late = (tau > tau0 + te_s) & (tau <= tau0 + float(tmax_s))
            sco = float(np.sum(pco[early]))
            scr = float(np.sum(pcr[early]))
            lco = float(np.sum(pco[late]))
            lcr = float(np.sum(pcr[late]))
            xpd_e = _db_ratio(sco, scr)
            rho_e_db = _db_ratio(scr, sco)
            xpd_l = _db_ratio(lco, lcr)
            l_pol = float(xpd_e - xpd_l)
            vals[float(te_ns)]["xpd_early"][lf].append(float(xpd_e))
            vals[float(te_ns)]["rho_early_db"][lf].append(float(rho_e_db))
            vals[float(te_ns)]["l_pol"][lf].append(float(l_pol))
    if used_links == 0:
        return {"status": "WARN", "reason": "No PDP links loaded for B scenarios"}

    score_rows: list[dict[str, Any]] = []
    for te_ns in te_ns_list:
        d = vals[float(te_ns)]
        sx = _sep_score(np.asarray(d["xpd_early"][1], dtype=float), np.asarray(d["xpd_early"][0], dtype=float))
        sr = _sep_score(np.asarray(d["rho_early_db"][1], dtype=float), np.asarray(d["rho_early_db"][0], dtype=float))
        sl = _sep_score(np.asarray(d["l_pol"][1], dtype=float), np.asarray(d["l_pol"][0], dtype=float))
        score_rows.append(
            {
                "Te_ns": float(te_ns),
                "S_xpd_early": sx,
                "S_rho_early_db": sr,
                "S_l_pol": sl,
            }
        )
    best = max(score_rows, key=lambda x: _num(x.get("S_xpd_early", np.nan)))
    best_score = _num(best.get("S_xpd_early", np.nan))
    status = "PASS" if (np.isfinite(best_score) and best_score >= 1.0) else ("WARN" if (np.isfinite(best_score) and best_score >= 0.5) else "FAIL")
    return {
        "status": status,
        "n_links_used": int(used_links),
        "scores": score_rows,
        "best_te_ns": float(best.get("Te_ns", np.nan)),
        "best_S_xpd_early": float(best_score),
        "best_S_rho_early_db": float(_num(best.get("S_rho_early_db", np.nan))),
        "best_S_l_pol": float(_num(best.get("S_l_pol", np.nan))),
    }


def _strata_base_bins() -> list[str]:
    return [f"LOS{lf}_q{q}" for lf in [0, 1] for q in [1, 2, 3]]


def _new_strata_counter() -> dict[str, int]:
    out = {k: 0 for k in _strata_base_bins()}
    out["LOS0_qNA"] = 0
    out["LOS1_qNA"] = 0
    return out


def _el_q1_q2(rows: list[dict[str, Any]]) -> tuple[float, float]:
    b_el = np.asarray([_num(r.get("EL_proxy_db", np.nan)) for r in rows], dtype=float)
    b_el = b_el[np.isfinite(b_el)]
    if len(b_el) < 3:
        return float("nan"), float("nan")
    q1, q2 = np.percentile(b_el, [33.3, 66.7])
    return float(q1), float(q2)


def _strata_counts(rows: list[dict[str, Any]], *, q1: float, q2: float) -> dict[str, int]:
    out = _new_strata_counter()
    for r in rows:
        lf = int(round(_num(r.get("LOSflag", np.nan)))) if np.isfinite(_num(r.get("LOSflag", np.nan))) else -1
        if lf not in {0, 1}:
            continue
        el = _num(r.get("EL_proxy_db", np.nan))
        if not np.isfinite(el) or not np.isfinite(q1) or not np.isfinite(q2):
            key = f"LOS{lf}_qNA"
        elif el <= q1:
            key = f"LOS{lf}_q1"
        elif el <= q2:
            key = f"LOS{lf}_q2"
        else:
            key = f"LOS{lf}_q3"
        out[key] = int(out.get(key, 0) + 1)
    return out


def _d3_hole_analysis(
    strata_pool: dict[str, int],
    strata_selected: dict[str, int] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k in _strata_base_bins():
        pool_n = int(strata_pool.get(k, 0))
        sel_n = int(strata_selected.get(k, 0)) if isinstance(strata_selected, dict) else -1
        if pool_n == 0:
            hole = "structural_hole"
            st = "FAIL"
        elif isinstance(strata_selected, dict) and sel_n == 0:
            hole = "sampling_hole"
            st = "WARN"
        else:
            hole = "none"
            st = "PASS"
        out.append(
            {
                "strata": k,
                "pool_n": int(pool_n),
                "selected_n": int(sel_n) if sel_n >= 0 else "",
                "hole_type": hole,
                "status": st,
            }
        )
    return out


def _load_selected_subset(
    cfg: dict[str, Any],
    link_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sampling_cfg = dict(cfg.get("sampling", {})) if isinstance(cfg.get("sampling", {}), dict) else {}
    raw_path = str(
        sampling_cfg.get(
            "selected_points_csv",
            cfg.get("selected_points_csv", ""),
        )
    ).strip()
    if not raw_path:
        return [], {"status": "NA", "reason": "selected_points_csv not provided"}
    p = Path(raw_path)
    if not p.exists():
        return [], {"status": "WARN", "reason": f"selected_points_csv not found: {p}"}

    with p.open("r", encoding="utf-8", newline="") as f:
        sel_rows = [dict(r) for r in csv.DictReader(f)]
    if not sel_rows:
        return [], {"status": "WARN", "reason": f"selected_points_csv empty: {p}"}

    by_link: dict[str, dict[str, Any]] = {}
    by_case: dict[tuple[str, str], dict[str, Any]] = {}
    for r in link_rows:
        lid = str(r.get("link_id", ""))
        sid = str(r.get("scenario_id", ""))
        cid = str(r.get("case_id", ""))
        if lid:
            by_link[lid] = r
        by_case[(sid, cid)] = r

    matched: list[dict[str, Any]] = []
    unmatched = 0
    for s in sel_rows:
        lid = str(s.get("link_id", "")).strip()
        sid = str(s.get("scenario_id", "")).strip()
        cid = str(s.get("case_id", "")).strip()
        if lid and lid in by_link:
            matched.append(dict(by_link[lid]))
            continue
        if sid and cid and (sid, cid) in by_case:
            matched.append(dict(by_case[(sid, cid)]))
            continue
        unmatched += 1

    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in matched:
        k = (
            str(r.get("scenario_id", "")),
            str(r.get("case_id", "")),
            str(r.get("link_id", "")),
        )
        dedup[k] = r
    return list(dedup.values()), {
        "status": "PASS" if dedup else "WARN",
        "path": str(p),
        "selected_raw_n": int(len(sel_rows)),
        "matched_n": int(len(dedup)),
        "unmatched_n": int(unmatched),
    }


def _target_sign_stability_te_sweep(
    *,
    link_rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    te_ns_list: list[float],
    floor_db: float,
) -> dict[str, Any]:
    """Check A2/A3 odd/even sign stability over Te sweep.

    A2 expected: XPD_early_ex < 0
    A3 expected: XPD_early_ex > 0
    """

    run_by_sid = {str(r.get("scenario_id", "")): r for r in runs}
    out: dict[str, Any] = {}
    for sid, expected_sign in [("A2", -1), ("A3", +1)]:
        rows = [r for r in link_rows if str(r.get("scenario_id", "")) == sid]
        run = run_by_sid.get(sid)
        if run is None or len(rows) == 0:
            out[sid] = {"status": "WARN", "reason": "missing run or rows", "per_te": []}
            continue
        per_te: list[dict[str, Any]] = []
        for te_ns in te_ns_list:
            vals: list[float] = []
            hit: list[bool] = []
            for r in rows:
                pdp = io_lib.load_pdp_npz(run, str(r.get("link_id", "")))
                if pdp is None:
                    continue
                tau = np.asarray(pdp.get("delay_tau_s", []), dtype=float)
                pco = np.asarray(pdp.get("P_co", []), dtype=float)
                pcr = np.asarray(pdp.get("P_cross", []), dtype=float)
                if len(tau) == 0 or len(pco) != len(tau) or len(pcr) != len(tau):
                    continue
                w = _window_dict(r)
                tau0 = _num(w.get("tau0_s", np.nan))
                if not np.isfinite(tau0):
                    p_total = pco + pcr
                    i0 = int(np.argmax(p_total)) if len(p_total) else -1
                    if i0 < 0:
                        continue
                    tau0 = float(tau[i0])
                te_s = float(te_ns) * 1e-9
                early = (tau >= tau0) & (tau <= tau0 + te_s)
                sco = float(np.sum(pco[early]))
                scr = float(np.sum(pcr[early]))
                xpd_e = _db_ratio(sco, scr)
                xpd_ex = float(xpd_e - floor_db) if np.isfinite(floor_db) else float("nan")
                if not np.isfinite(xpd_ex):
                    continue
                vals.append(float(xpd_ex))
                hit.append(bool(xpd_ex < 0.0) if expected_sign < 0 else bool(xpd_ex > 0.0))
            rate = float(np.mean(hit)) if hit else float("nan")
            per_te.append(
                {
                    "Te_ns": float(te_ns),
                    "n": int(len(vals)),
                    "median_xpd_early_ex_db": float(np.median(np.asarray(vals, dtype=float))) if vals else float("nan"),
                    "expected_sign_hit_rate": rate,
                }
            )
        rates = np.asarray([_num(x.get("expected_sign_hit_rate", np.nan)) for x in per_te], dtype=float)
        rates = rates[np.isfinite(rates)]
        min_rate = float(np.min(rates)) if len(rates) else float("nan")
        med_rate = float(np.median(rates)) if len(rates) else float("nan")
        status = "PASS" if (np.isfinite(min_rate) and min_rate >= 0.8) else ("WARN" if (np.isfinite(med_rate) and med_rate >= 0.6) else "FAIL")
        out[sid] = {
            "expected_sign": "negative" if expected_sign < 0 else "positive",
            "min_hit_rate": min_rate,
            "median_hit_rate": med_rate,
            "per_te": per_te,
            "status": status,
        }
    a2s = str(out.get("A2", {}).get("status", "WARN"))
    a3s = str(out.get("A3", {}).get("status", "WARN"))
    if a2s == "PASS" and a3s == "PASS":
        overall = "PASS"
    elif a2s == "FAIL" or a3s == "FAIL":
        overall = "FAIL"
    else:
        overall = "WARN"
    out["overall_status"] = overall
    return out


def _make_scene_plots(
    config: dict[str, Any],
    out_fig_dir: Path,
    link_rows: list[dict[str, Any]],
    scene_map: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], list[str], list[dict[str, Any]]]:
    scene_cfg = dict(config.get("scene_plots", {}))
    enabled = bool(scene_cfg.get("enabled", True))
    max_cases = int(scene_cfg.get("max_cases_per_scenario", 9999))
    figure_size = tuple(float(x) for x in scene_cfg.get("figure_size", [10, 6]))

    index_rows: list[dict[str, Any]] = []
    first_scene_by_scenario: dict[str, str] = {}
    warns: list[str] = []
    warn_cases: list[dict[str, Any]] = []

    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for r in link_rows:
        by_scenario.setdefault(str(r.get("scenario_id", "NA")), []).append(r)

    for scenario_id in sorted(by_scenario.keys()):
        case_seen: set[str] = set()
        scenario_rows = by_scenario[scenario_id]
        case_rows = sorted(scenario_rows, key=lambda r: str(r.get("case_id", "")))
        case_count = 0
        for r in case_rows:
            case_id = str(r.get("case_id", ""))
            if case_id in case_seen:
                continue
            case_seen.add(case_id)
            if case_count >= max_cases:
                continue
            case_count += 1
            case_label = str(r.get("case_label", r.get("link_id", case_id)))
            key = (scenario_id, case_id)
            scene_obj = scene_map.get(key)
            scene_png = ""
            scene_json = ""
            plot_list: list[str] = []
            warn_reason = ""
            if enabled and scene_obj is not None:
                ok, problems = scene_lib.validate_scene_debug(scene_obj)
                if not ok:
                    warn_reason = f"scene_debug invalid: {';'.join(problems)}"
                    warns.append(f"{scenario_id}/{case_id}: {warn_reason}")
                scenario_tag = f"{scenario_id}__{case_id}"
                out_png = out_fig_dir / f"{scenario_tag}__scene.png"
                scene_png = scene_lib.plot_scene(scene_obj, out_png=out_png, figure_size=figure_size)
                scene_json = str(scene_obj.get("_path", ""))
                plot_list.append(scene_png)
                if scenario_id not in first_scene_by_scenario:
                    first_scene_by_scenario[scenario_id] = scene_png
            else:
                scenario_tag = f"{scenario_id}__{case_id}"
                out_png = out_fig_dir / f"{scenario_tag}__scene.png"
                fb_scene, fb_warns = scene_lib.build_fallback_scene_from_link_row(r)
                scene_png = scene_lib.plot_scene(fb_scene, out_png=out_png, figure_size=figure_size)
                plot_list.append(scene_png)
                warn_reason = "scene_debug missing; fallback layout used (ray polylines unavailable)"
                warns.append(f"{scenario_id}/{case_id}: {warn_reason}")
                for w in fb_warns:
                    warns.append(f"{scenario_id}/{case_id}: {w}")
                if scenario_id not in first_scene_by_scenario:
                    first_scene_by_scenario[scenario_id] = scene_png

            index_rows.append(
                {
                    "scenario_id": scenario_id,
                    "case_id": case_id,
                    "case_label": case_label,
                    "scene_debug_json": scene_json,
                    "scene_png_path": scene_png,
                    "key_plots": plot_list,
                }
            )
            if warn_reason:
                warn_cases.append(
                    {
                        "scenario_id": scenario_id,
                        "case_id": case_id,
                        "case_label": case_label,
                        "warning": warn_reason,
                        "scene_png_path": scene_png,
                        "link_id": str(r.get("link_id", "")),
                        "XPD_early_excess_db": _num(r.get("XPD_early_excess_db", np.nan)),
                        "XPD_late_excess_db": _num(r.get("XPD_late_excess_db", np.nan)),
                        "L_pol_db": _num(r.get("L_pol_db", np.nan)),
                        "EL_proxy_db": _num(r.get("EL_proxy_db", np.nan)),
                        "LOSflag": _num(r.get("LOSflag", np.nan)),
                    }
                )

        # room scenario global layout
        if enabled and scenario_id in {"B1", "B2", "B3"}:
            candidates = [x for x in case_rows if (scenario_id, str(x.get("case_id", ""))) in scene_map]
            if candidates:
                c0 = candidates[0]
                sc = scene_map.get((scenario_id, str(c0.get("case_id", ""))))
            else:
                sc = None
            if sc is None and case_rows:
                sc, _ = scene_lib.build_fallback_scene_from_link_row(case_rows[0])
            if sc is not None:
                rx_points = []
                for rr in case_rows:
                    rx_points.append((_num(rr.get("rx_x", np.nan)), _num(rr.get("rx_y", np.nan))))
                out_png = out_fig_dir / f"{scenario_id}__GLOBAL__scene.png"
                scene_lib.plot_scene_global(sc, rx_points=rx_points, out_png=out_png, figure_size=figure_size)
                first_scene_by_scenario[scenario_id] = str(out_png)

    return index_rows, first_scene_by_scenario, warns, warn_cases


def _build_a3_geometry_review_rows(
    idx_rows: list[dict[str, Any]],
    scene_map: dict[tuple[str, str], dict[str, Any]],
    ray_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ray_by_case: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rr in ray_rows:
        sid = str(rr.get("scenario_id", ""))
        cid = str(rr.get("case_id", ""))
        ray_by_case.setdefault((sid, cid), []).append(rr)

    out: list[dict[str, Any]] = []
    for r in idx_rows:
        sid = str(r.get("scenario_id", ""))
        if sid != "A3":
            continue
        cid = str(r.get("case_id", ""))
        sc = scene_map.get((sid, cid))
        scene_ok = False
        scene_issues = ""
        rays_topk_n = 0
        if isinstance(sc, dict):
            ok, probs = scene_lib.validate_scene_debug(sc)
            scene_ok = bool(ok)
            scene_issues = ";".join(probs) if probs else ""
            rays_topk_n = int(len(sc.get("rays_topk", []))) if isinstance(sc.get("rays_topk", []), list) else 0
        case_rays = ray_by_case.get((sid, cid), [])
        los_rays = int(sum(int(round(_num(x.get("los_flag_ray", np.nan)))) == 1 for x in case_rays if np.isfinite(_num(x.get("los_flag_ray", np.nan)))))
        has_n2 = int(
            any(
                int(round(_num(x.get("n_bounce", np.nan)))) == 2
                for x in case_rays
                if np.isfinite(_num(x.get("n_bounce", np.nan)))
            )
        )
        review = "PASS" if (scene_ok and los_rays == 0 and has_n2 == 1) else ("WARN" if has_n2 == 1 else "FAIL")
        out.append(
            {
                "scenario_id": sid,
                "case_id": cid,
                "case_label": str(r.get("case_label", "")),
                "scene_png_path": str(r.get("scene_png_path", "")),
                "scene_debug_json": str(r.get("scene_debug_json", "")),
                "scene_debug_valid": int(scene_ok),
                "scene_debug_issues": scene_issues,
                "rays_topk_n": int(rays_topk_n),
                "los_rays": int(los_rays),
                "has_target_bounce_n2": int(has_n2),
                "review_status": review,
            }
        )
    return sorted(out, key=lambda x: (str(x.get("case_id", "")), str(x.get("case_label", ""))))


def _collect_unique_values(rows: list[dict[str, Any]], key: str) -> list[str]:
    vals = []
    for r in rows:
        v = r.get(key, "")
        if isinstance(v, (dict, list, tuple)):
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        vals.append(s)
    return sorted(set(vals))


def _build_figure_metadata_rows(
    link_rows: list[dict[str, Any]],
    idx_rows: list[dict[str, Any]],
    runs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    run_by_sid: dict[str, dict[str, Any]] = {}
    for rr in list(runs or []):
        sid = str(rr.get("scenario_id", ""))
        if sid and sid not in run_by_sid:
            run_by_sid[sid] = dict(rr)

    by_case: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in link_rows:
        by_case.setdefault((str(r.get("scenario_id", "")), str(r.get("case_id", ""))), []).append(r)

    out: list[dict[str, Any]] = []
    for i in idx_rows:
        sid = str(i.get("scenario_id", ""))
        cid = str(i.get("case_id", ""))
        rows = by_case.get((sid, cid), [])
        r0 = rows[0] if rows else {}
        sm = dict(run_by_sid.get(sid, {}).get("summary", {})) if sid in run_by_sid else {}
        ac = dict(sm.get("antenna_config", {})) if isinstance(sm.get("antenna_config", {}), dict) else {}
        fcp = _num(r0.get("force_cp_swap_on_odd_reflection", np.nan))
        if not np.isfinite(fcp):
            fcp = _num(sm.get("force_cp_swap_on_odd_reflection", np.nan))
        out.append(
            {
                "scenario_id": sid,
                "case_id": cid,
                "case_label": str(i.get("case_label", "")),
                "link_id": str(r0.get("link_id", i.get("link_id", ""))),
                "scene_png_path": str(i.get("scene_png_path", "")),
                "basis": str(r0.get("basis", sm.get("basis", ""))),
                "convention": str(r0.get("convention", sm.get("convention", ""))),
                "matrix_source": str(r0.get("matrix_source", sm.get("matrix_source", ""))),
                "force_cp_swap_on_odd_reflection": fcp,
                "a4_layout_mode": str(r0.get("a4_layout_mode", sm.get("a4_layout_mode", ""))),
                "a4_dispersion_mode": str(r0.get("a4_dispersion_mode", sm.get("a4_dispersion_mode", ""))),
                "material_dispersion": str(r0.get("material_dispersion", sm.get("material_dispersion", ""))),
                "include_late_panel": _num(r0.get("include_late_panel", sm.get("include_late_panel", np.nan))),
                "stress_mode": str(r0.get("stress_mode", sm.get("stress_mode", ""))),
                "scatterer_count": _num(r0.get("scatterer_count", sm.get("scatterer_count", np.nan))),
                "diffuse_enabled": _num(r0.get("diffuse_enabled", sm.get("diffuse_enabled", np.nan))),
                "diffuse_factor": _num(r0.get("diffuse_factor", sm.get("diffuse_factor", np.nan))),
                "diffuse_lobe_alpha": _num(r0.get("diffuse_lobe_alpha", sm.get("diffuse_lobe_alpha", np.nan))),
                "diffuse_rays_per_hit": _num(r0.get("diffuse_rays_per_hit", sm.get("diffuse_rays_per_hit", np.nan))),
                "tx_cross_pol_leakage_db": _num(r0.get("tx_cross_pol_leakage_db", ac.get("tx_cross_pol_leakage_db", np.nan))),
                "rx_cross_pol_leakage_db": _num(r0.get("rx_cross_pol_leakage_db", ac.get("rx_cross_pol_leakage_db", np.nan))),
                "tx_cross_pol_leakage_db_slope_per_ghz": _num(r0.get("tx_cross_pol_leakage_db_slope_per_ghz", ac.get("tx_cross_pol_leakage_db_slope_per_ghz", np.nan))),
                "rx_cross_pol_leakage_db_slope_per_ghz": _num(r0.get("rx_cross_pol_leakage_db_slope_per_ghz", ac.get("rx_cross_pol_leakage_db_slope_per_ghz", np.nan))),
            }
        )
    return out


def _diagnostic_checks(
    link_rows: list[dict[str, Any]],
    ray_rows: list[dict[str, Any]],
    floor_ref: dict[str, Any],
    cfg: dict[str, Any],
    runs: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    scenario_rows = metrics_lib.split_by_scenario(link_rows)
    scenario_roles = dict(cfg.get("scenario_roles", {}))
    a3_role = str(scenario_roles.get("A3", "mechanism")).strip().lower()
    a5_semantics = _a5_semantics_summary(link_rows)
    a5_role_cfg = scenario_roles.get("A5", None)
    if a5_role_cfg is None:
        dom = str(a5_semantics.get("dominant_semantics", "none")).strip().lower()
        if dom == "polarization_only":
            a5_role = "stress_polarization_only"
        elif dom == "response":
            a5_role = "stress_response"
        else:
            a5_role = "stress_response"
    else:
        a5_role = str(a5_role_cfg).strip().lower()

    # A1 LOS blocked
    a1 = []
    blocked = ["A2", "A3", "A4", "A5"]
    for s in blocked:
        rr = [r for r in ray_rows if str(r.get("scenario_id", "")) == s]
        if not rr:
            a1.append({"scenario": s, "status": "WARN", "reason": "rays.csv missing"})
            continue
        los = int(sum(int(_num(r.get("los_flag_ray", 0))) == 1 for r in rr))
        a1.append({"scenario": s, "los_rays": los, "status": "PASS" if los == 0 else "FAIL"})
    checks["A1_los_blocked"] = a1

    # A2 target bounce existence
    def _target_bounce_status(sid: str, target_n: int) -> dict[str, Any]:
        rr = [r for r in ray_rows if str(r.get("scenario_id", "")) == sid]
        if not rr:
            return {"scenario": sid, "target_n": target_n, "status": "WARN", "rate": float("nan")}
        by_case: dict[str, list[dict[str, Any]]] = {}
        for r in rr:
            by_case.setdefault(str(r.get("case_id", "")), []).append(r)
        hit = 0
        total = 0
        for _, cr in by_case.items():
            total += 1
            top = sorted(cr, key=lambda x: _num(x.get("P_lin", np.nan)), reverse=True)[:3]
            ok = any(int(round(_num(t.get("n_bounce", np.nan)))) == int(target_n) for t in top)
            hit += int(ok)
        rate = float(hit / total) if total else float("nan")
        status = "PASS" if (total > 0 and rate >= 0.5) else ("WARN" if total > 0 else "FAIL")
        return {"scenario": sid, "target_n": target_n, "hit": hit, "total": total, "rate": rate, "status": status}

    checks["A2_target_bounce"] = [_target_bounce_status("A2", 1), _target_bounce_status("A3", 2)]

    # A3 geometry sanity: data-limited
    checks["A3_coord_sanity"] = {
        "status": "WARN",
        "note": "Coordinate penetration sanity needs scenario geometry file review; not inferable from standard outputs only.",
    }
    checks["A5_stress_semantics"] = dict(a5_semantics)

    # B: time-resolution checks with purpose-specific windows
    ws = dict(cfg.get("windows", {}))
    te_s = float(ws.get("Te_ns", 10.0)) * 1e-9
    tmax_s = float(ws.get("Tmax_ns", 200.0)) * 1e-9
    w_floor_s = float(ws.get("floor_window_ns", max(1.0, 0.5 * float(ws.get("Te_ns", 10.0))))) * 1e-9
    w_target_default_s = float(ws.get("target_window_ns", max(2.0, 0.8 * float(ws.get("Te_ns", 10.0))))) * 1e-9
    bw, df, freq_src = _estimate_freq_grid(link_rows, cfg)
    dt_res = 1.0 / bw if bw > 0 else float("nan")
    tau_max = 1.0 / df if df > 0 else float("nan")

    floor_stats = _floor_window_contamination(ray_rows, w_floor_s=w_floor_s)
    target_map = {"A2": 1, "A3": 2, "A4": 1, "A5": 2}
    w_target_s_by_sid, w_target_mode_by_sid = _resolve_target_window_by_scenario(
        ws=ws,
        ray_rows=ray_rows,
        target_map=target_map,
        default_w_target_s=float(w_target_default_s),
        dt_res_s=float(dt_res),
    )
    target_stats = {
        sid: _target_window_stats(
            ray_rows,
            scenario_id=sid,
            target_n=tn,
            w_target_s=float(w_target_s_by_sid.get(sid, w_target_default_s)),
            te_s=te_s,
            dt_res_s=dt_res,
        )
        for sid, tn in target_map.items()
    }

    # B2: minimal delay spacing across rays (deduplicated tau per case)
    mins = []
    for sid in ["A2", "A3", "A4", "A5", "B1", "B2", "B3"]:
        rr = [r for r in ray_rows if str(r.get("scenario_id", "")) == sid]
        by_case: dict[str, list[float]] = {}
        for r in rr:
            by_case.setdefault(str(r.get("case_id", "")), []).append(_num(r.get("tau_s", np.nan)))
        for _, tv in by_case.items():
            x = np.asarray(tv, dtype=float)
            x = np.sort(np.unique(x[np.isfinite(x)]))
            if len(x) < 2:
                continue
            mins.append(float(np.min(np.diff(x))))
    mins = np.asarray(mins, dtype=float)
    b2_ok = bool(np.isfinite(dt_res) and len(mins) > 0 and np.nanmedian(mins) > dt_res)
    b3_ok = bool(np.isfinite(tau_max) and (tmax_s < tau_max))

    te_ns_list_cfg = ws.get("te_sweep_ns", [2.0, 3.0, 5.0])
    te_ns_list: list[float] = []
    if isinstance(te_ns_list_cfg, list):
        for x in te_ns_list_cfg:
            v = _num(x)
            if np.isfinite(v) and v > 0:
                te_ns_list.append(float(v))
    if not te_ns_list:
        te_ns_list = [2.0, 3.0, 5.0]
    wearly_stats = _wearly_te_sweep(
        link_rows=link_rows,
        runs=runs,
        te_ns_list=te_ns_list,
        tmax_s=tmax_s,
    )
    sign_stability = _target_sign_stability_te_sweep(
        link_rows=link_rows,
        runs=runs,
        te_ns_list=te_ns_list,
        floor_db=float(floor_ref.get("xpd_floor_db", np.nan)),
    )
    floor_db_used = float(floor_ref.get("xpd_floor_db", np.nan))
    sign_metric_cfg = ws.get("target_sign_metric_by_scenario", {})
    if not isinstance(sign_metric_cfg, dict):
        sign_metric_cfg = {}
    sign_metric_a2 = str(sign_metric_cfg.get("A2", "excess"))
    sign_metric_a3 = str(sign_metric_cfg.get("A3", "raw"))
    sign_metric_a6 = str(sign_metric_cfg.get("A6", "raw"))
    a2_target_sign = _target_window_sign_metric(
        link_rows=link_rows,
        ray_rows=ray_rows,
        runs=runs,
        scenario_id="A2",
        target_n=1,
        w_target_s=float(w_target_s_by_sid.get("A2", w_target_default_s)),
        floor_db=floor_db_used,
        expected_sign=-1,
        sign_metric=sign_metric_a2,
    )
    a3_target_sign = _target_window_sign_metric(
        link_rows=link_rows,
        ray_rows=ray_rows,
        runs=runs,
        scenario_id="A3",
        target_n=2,
        w_target_s=float(w_target_s_by_sid.get("A3", w_target_default_s)),
        floor_db=floor_db_used,
        expected_sign=+1,
        sign_metric=sign_metric_a3,
    )
    tw_cfg = ws.get("target_window_ns_by_scenario", {})
    a6_w_target_s = float(w_target_s_by_sid.get("A3", w_target_default_s))
    if isinstance(tw_cfg, dict):
        a6_w_ns = _num(tw_cfg.get("A6", np.nan))
        if np.isfinite(a6_w_ns) and a6_w_ns > 0:
            a6_w_target_s = float(a6_w_ns) * 1e-9
    a6_parity = _a6_parity_benchmark(
        link_rows=link_rows,
        ray_rows=ray_rows,
        runs=runs,
        w_target_s=float(a6_w_target_s),
        floor_db=float(floor_db_used),
        sign_metric=sign_metric_a6,
    )

    a3_t = target_stats.get("A3", {})
    a3_exists = _num(a3_t.get("target_exists_rate", np.nan))
    a3_ct = _num(a3_t.get("C_target_median_db", np.nan))
    a3_wearly = _num(a3_t.get("target_in_Wearly_rate", np.nan))
    if np.isfinite(a3_exists) and a3_exists >= 0.9 and np.isfinite(a3_ct) and a3_ct < -10.0:
        a3_mech_status = "PASS"
    elif np.isfinite(a3_exists) and a3_exists >= 0.7 and np.isfinite(a3_ct) and a3_ct < -6.0:
        a3_mech_status = "WARN"
    else:
        a3_mech_status = "FAIL"
    if np.isfinite(a3_wearly) and a3_wearly >= 0.8:
        a3_early_status = "PASS"
    elif np.isfinite(a3_wearly) and a3_wearly >= 0.5:
        a3_early_status = "WARN"
    else:
        a3_early_status = "FAIL"
    a3_target_status_reporting = str(a3_target_sign.get("status", "INCONCLUSIVE"))
    if a3_role in {"mechanism", "mechanism_only", "supplementary", "candidate"} and a3_target_status_reporting == "FAIL":
        a3_target_status_reporting = "WARN"

    a5_t = target_stats.get("A5", {})
    a5_ct = _num(a5_t.get("C_target_median_db", np.nan))
    if np.isfinite(a5_ct):
        if a5_ct < -10.0:
            a5_target_mode = "isolation"
        elif a5_ct < -3.0:
            a5_target_mode = "mixed"
        else:
            a5_target_mode = "contamination_response"
    else:
        a5_target_mode = "unknown"

    if bool(a6_parity.get("available", False)):
        g2_primary_source = "A6_near_normal_benchmark"
        g2_primary_status = str(a6_parity.get("status", "INCONCLUSIVE"))
        a3_evidence_tier = "supplementary"
    else:
        g2_primary_source = "A3_target_window_supplementary_only"
        if str(a3_target_status_reporting) in {"PASS", "WARN"}:
            g2_primary_status = "WARN"
        else:
            g2_primary_status = str(a3_target_status_reporting)
        a3_evidence_tier = "primary_if_no_A6"

    # Role-aware reporting for A2/A3 early-window sign stability:
    # when A3 is mechanism/supplementary, early-window sign FAIL is expected
    # under 1-bounce/2-bounce mixing and should not block reporting.
    sstab_raw_overall = str(sign_stability.get("overall_status", "WARN")) if isinstance(sign_stability, dict) else "WARN"
    sstab_reporting = sstab_raw_overall
    sstab_note = ""
    sstab_a2 = dict(sign_stability.get("A2", {})) if isinstance(sign_stability, dict) else {}
    sstab_a3 = dict(sign_stability.get("A3", {})) if isinstance(sign_stability, dict) else {}
    sstab_a2_status = str(sstab_a2.get("status", "WARN"))
    sstab_a3_status = str(sstab_a3.get("status", "WARN"))
    a3_mech_only = a3_role in {"mechanism", "mechanism_only", "supplementary", "candidate"}
    a3_expected_early_fail = bool(a3_mech_only and sstab_a3_status == "FAIL")
    if a3_expected_early_fail:
        sstab_reporting = "PASS" if sstab_a2_status == "PASS" else ("WARN" if sstab_a2_status in {"WARN", "INCONCLUSIVE"} else "FAIL")
        sstab_note = (
            "A3 early-window sign FAIL treated as expected under mechanism-supplementary role "
            "(1-bounce(-) + 2-bounce(+) mixing near ~0 dB); target-window sign remains primary for A3."
        )
    else:
        sstab_note = "A2/A3 early-window sign stability interpreted directly from raw status."

    checks["B_time_resolution"] = {
        "freq_source": str(freq_src),
        "BW_Hz": float(bw),
        "df_Hz": float(df),
        "dt_res_s": float(dt_res),
        "tau_max_s": float(tau_max),
        "Te_s": float(te_s),
        "Tmax_s": float(tmax_s),
        "W_floor_s": float(w_floor_s),
        "W_target_s_default": float(w_target_default_s),
        "W_target_s": float(w_target_default_s),
        "W_target_s_by_scenario": {k: float(v) for k, v in w_target_s_by_sid.items()},
        "W_target_mode_by_scenario": {k: str(v) for k, v in w_target_mode_by_sid.items()},
        "W_floor_status": str(floor_stats.get("status", "WARN")),
        "W_floor_C_median_db": float(floor_stats.get("C_floor_median_db", np.nan)),
        "A2_target_in_Wearly_rate": float(target_stats["A2"].get("target_in_Wearly_rate", np.nan)),
        "A3_target_in_Wearly_rate": float(target_stats["A3"].get("target_in_Wearly_rate", np.nan)),
        "A2_C_target_median_db": float(target_stats["A2"].get("C_target_median_db", np.nan)),
        "A3_C_target_median_db": float(target_stats["A3"].get("C_target_median_db", np.nan)),
        "B2_min_delay_gap_median_s": float(np.nanmedian(mins)) if len(mins) else float("nan"),
        "B2_status": "PASS" if b2_ok else "WARN",
        "B3_status": "PASS" if b3_ok else "FAIL",
        "W3_status": str(wearly_stats.get("status", "WARN")),
        "W3_best_te_ns": float(_num(wearly_stats.get("best_te_ns", np.nan))),
        "W3_best_S_xpd_early": float(_num(wearly_stats.get("best_S_xpd_early", np.nan))),
        "W3_best_S_rho_early_db": float(_num(wearly_stats.get("best_S_rho_early_db", np.nan))),
        "W3_best_S_l_pol": float(_num(wearly_stats.get("best_S_l_pol", np.nan))),
        "W_floor_detail": floor_stats,
        "W_target_detail": target_stats,
        "W3_te_sweep": wearly_stats,
        "A2A3_sign_stability": sign_stability,
        "A2A3_sign_stability_raw_status": str(sstab_raw_overall),
        "A2A3_sign_stability_reporting_status": str(sstab_reporting),
        "A2A3_sign_stability_reporting_note": str(sstab_note),
        "A3_expected_early_sign_fail_under_mechanism_role": bool(a3_expected_early_fail),
        "A2_target_window_sign": a2_target_sign,
        "A3_target_window_sign": a3_target_sign,
        "A3_target_window_sign_reporting_status": str(a3_target_status_reporting),
        "A6_parity_benchmark": a6_parity,
        "A3_role": a3_role,
        "A3_evidence_tier": str(a3_evidence_tier),
        "G2_primary_evidence_source": str(g2_primary_source),
        "G2_primary_evidence_status": str(g2_primary_status),
        "A3_mechanism_status": a3_mech_status,
        "A3_system_early_status": a3_early_status,
        "A3_reporting_rule": "A3 is mechanism-only/supplementary: use target-window metrics for mechanism context; fixed system early-window dominance is not primary evidence. If A6 is present, use A6 as primary G2 sign evidence.",
        "A5_role": a5_role,
        "A5_target_mode": a5_target_mode,
        "A5_stress_semantics": str(a5_semantics.get("dominant_semantics", "none")),
        "A5_contamination_response_ready": bool(a5_semantics.get("contamination_response_ready", False)),
        "A5_semantics_note": str(a5_semantics.get("note", "")),
    }

    # C effect vs floor (C2-M / C2-S endpoints)
    floor_delta = float(floor_ref.get("delta_floor_db", np.nan))
    repeat_delta = float(_c0_repeat_delta(link_rows))
    delta_ref = float(max(abs(floor_delta), abs(repeat_delta))) if np.isfinite(floor_delta) and np.isfinite(repeat_delta) else (
        abs(float(floor_delta)) if np.isfinite(floor_delta) else (abs(float(repeat_delta)) if np.isfinite(repeat_delta) else float("nan"))
    )
    a2 = scenario_rows.get("A2", [])
    a3 = scenario_rows.get("A3", [])
    dmed = metrics_lib.delta_median(a3, a2, "XPD_early_excess_db")
    c1_status, ratio = _judge_vs_delta(float(dmed), float(delta_ref))

    a4_all = scenario_rows.get("A4", [])
    a4_iso = [r for r in a4_all if _a4_branch(r) == "iso"]
    a4_bridge = [r for r in a4_all if _a4_branch(r) == "bridge"]
    a4_bridge_disp_on_n = int(sum(1 for r in a4_bridge if _a4_dispersion_on(r)))

    def _medians_by_material(rows_in: list[dict[str, Any]], key: str) -> dict[str, float]:
        by_mat_local: dict[str, list[dict[str, Any]]] = {}
        for rr in rows_in:
            by_mat_local.setdefault(str(rr.get("material_class", "NA")), []).append(rr)
        return {k: _median(v, key) for k, v in sorted(by_mat_local.items())}

    # Primary material effect uses A4_iso only (late-panel off).
    a4_iso_early_meds = _medians_by_material(a4_iso, "XPD_early_excess_db")
    a4_iso_late_meds = _medians_by_material(a4_iso, "XPD_late_excess_db")
    a4_iso_lpol_meds = _medians_by_material(a4_iso, "L_pol_db")
    a4_iso_primary_span = _safe_span(list(a4_iso_early_meds.values()))
    a4_iso_late_span = _safe_span(list(a4_iso_late_meds.values()))
    a4_iso_lpol_span = _safe_span(list(a4_iso_lpol_meds.values()))

    # Secondary bridge effect uses A4_bridge (late-panel on).
    a4_bridge_early_meds = _medians_by_material(a4_bridge, "XPD_early_excess_db")
    a4_bridge_late_meds = _medians_by_material(a4_bridge, "XPD_late_excess_db")
    a4_bridge_lpol_meds = _medians_by_material(a4_bridge, "L_pol_db")
    a4_bridge_primary_span = _safe_span(list(a4_bridge_early_meds.values()))
    a4_bridge_late_span = _safe_span(list(a4_bridge_late_meds.values()))
    a4_bridge_lpol_span = _safe_span(list(a4_bridge_lpol_meds.values()))

    c2m_primary_status, c2m_primary_ratio = _judge_vs_delta(float(a4_iso_primary_span), float(delta_ref))
    c2m_late_status, c2m_late_ratio = _judge_vs_delta(float(a4_iso_late_span), float(delta_ref))
    c2m_lpol_status, c2m_lpol_ratio = _judge_vs_delta(float(a4_iso_lpol_span), float(delta_ref))
    c2m_bridge_status, c2m_bridge_ratio = _judge_vs_delta(float(a4_bridge_primary_span), float(delta_ref))
    c2m_bridge_late_status, c2m_bridge_late_ratio = _judge_vs_delta(float(a4_bridge_late_span), float(delta_ref))
    c2m_bridge_lpol_status, c2m_bridge_lpol_ratio = _judge_vs_delta(float(a4_bridge_lpol_span), float(delta_ref))
    a4_target = target_stats.get("A4", {})
    a4_target_c = float(_num(a4_target.get("C_target_median_db", np.nan)))
    a4_target_exists = float(_num(a4_target.get("target_exists_rate", np.nan)))
    a4_target_ok = bool(np.isfinite(a4_target_c) and a4_target_c < -10.0 and np.isfinite(a4_target_exists) and a4_target_exists >= 0.9)
    # C2-M final status is driven by effect size; target-gate is reported as
    # supplementary diagnostic only (to avoid false WARN when layout/window
    # choices intentionally increase in-window contamination).
    if c2m_primary_status == "PASS":
        c2m_status = "PASS"
    elif c2m_primary_status in {"WARN", "INCONCLUSIVE"}:
        c2m_status = "WARN"
    else:
        c2m_status = "FAIL"

    a5 = scenario_rows.get("A5", [])
    base = [r for r in a5 if int(_num(r.get("roughness_flag", 0))) == 0 and int(_num(r.get("human_flag", 0))) == 0]
    stress = [r for r in a5 if int(_num(r.get("roughness_flag", 0))) == 1 or int(_num(r.get("human_flag", 0))) == 1]
    d_stress = metrics_lib.delta_median(stress, base, "XPD_late_excess_db")
    d_lpol = metrics_lib.delta_median(stress, base, "L_pol_db")
    d_rho = metrics_lib.delta_median(stress, base, "rho_early_lin")
    d_ds = metrics_lib.delta_median(stress, base, "delay_spread_rms_s")
    v_base = metrics_lib.tail_stats(base, "XPD_late_excess_db").get("std", np.nan)
    v_stress = metrics_lib.tail_stats(stress, "XPD_late_excess_db").get("std", np.nan)
    var_ratio = float(v_stress / v_base) if np.isfinite(v_stress) and np.isfinite(v_base) and v_base > 0 else float("nan")

    # Gate: target-path total power drop (A5 stress vs base). If too large, treat as blockage.
    ray_by_link: dict[str, list[dict[str, Any]]] = {}
    for rr in ray_rows:
        if str(rr.get("scenario_id", "")).upper() != "A5":
            continue
        ray_by_link.setdefault(str(rr.get("link_id", "")), []).append(rr)

    def _dom_target_power(link_row: dict[str, Any], target_n: int = 2) -> float:
        lid = str(link_row.get("link_id", ""))
        rays = ray_by_link.get(lid, [])
        cand = []
        for rr in rays:
            nb = int(round(_num(rr.get("n_bounce", np.nan)))) if np.isfinite(_num(rr.get("n_bounce", np.nan))) else -1
            if nb != int(target_n):
                continue
            p = _num(rr.get("P_lin", np.nan))
            if np.isfinite(p) and p > 0:
                cand.append(float(p))
        if not cand:
            return float("nan")
        return float(np.max(np.asarray(cand, dtype=float)))

    base_pow: dict[tuple[Any, ...], list[float]] = {}
    stress_pow: dict[tuple[Any, ...], list[float]] = {}
    for r in base:
        p = _dom_target_power(r, target_n=2)
        if np.isfinite(p):
            base_pow.setdefault(_a5_pair_key(r), []).append(float(p))
    for r in stress:
        p = _dom_target_power(r, target_n=2)
        if np.isfinite(p):
            stress_pow.setdefault(_a5_pair_key(r), []).append(float(p))
    dp_target_vals: list[float] = []
    for k, vb in base_pow.items():
        if k not in stress_pow:
            continue
        pb = _median_vals(vb)
        ps = _median_vals(stress_pow[k])
        if np.isfinite(pb) and np.isfinite(ps) and pb > 0 and ps > 0:
            dp_target_vals.append(_db_ratio(ps, pb))
    # Fallback: if key-level pairing is unavailable, pair by rho-level medians.
    if len(dp_target_vals) == 0:
        base_by_rho: dict[float, list[float]] = {}
        stress_by_rho: dict[float, list[float]] = {}
        for r in base:
            rho = _num(_link_params(r).get("rho", np.nan))
            p = _dom_target_power(r, target_n=2)
            if np.isfinite(rho) and np.isfinite(p) and p > 0:
                base_by_rho.setdefault(round(float(rho), 4), []).append(float(p))
        for r in stress:
            rho = _num(_link_params(r).get("rho", np.nan))
            p = _dom_target_power(r, target_n=2)
            if np.isfinite(rho) and np.isfinite(p) and p > 0:
                stress_by_rho.setdefault(round(float(rho), 4), []).append(float(p))
        for rho, vb in base_by_rho.items():
            if rho not in stress_by_rho:
                continue
            pb = _median_vals(vb)
            ps = _median_vals(stress_by_rho[rho])
            if np.isfinite(pb) and np.isfinite(ps) and pb > 0 and ps > 0:
                dp_target_vals.append(_db_ratio(ps, pb))
    gate_dp_target_db = _median_vals(dp_target_vals)
    gate_status = "PASS" if (np.isfinite(gate_dp_target_db) and gate_dp_target_db > -6.0) else ("WARN" if not np.isfinite(gate_dp_target_db) else "BLOCKAGE")

    # C2-S primary: ΔL_pol < 0 and larger than uncertainty reference.
    c2s_primary_status = "INCONCLUSIVE"
    if np.isfinite(d_lpol):
        if d_lpol < 0.0:
            c2s_primary_status, _ = _judge_vs_delta(float(d_lpol), float(delta_ref))
        else:
            c2s_primary_status = "FAIL"
    if gate_status == "BLOCKAGE":
        c2s_status = "WARN"
    elif c2s_primary_status == "PASS":
        c2s_status = "PASS"
    elif c2s_primary_status in {"WARN", "INCONCLUSIVE"}:
        c2s_status = "WARN"
    else:
        c2s_status = "FAIL"

    c2s_semantics_gate = "PASS"
    if bool(a5_semantics.get("n_stress_rows", 0)) and not bool(a5_semantics.get("contamination_response_ready", False)):
        # Synthetic-only stress cannot support delay/path-structure contamination-response claims.
        c2s_semantics_gate = "WARN"
        if c2s_status == "PASS":
            c2s_status = "WARN"

    if c2m_status == "PASS" and c2s_status == "PASS":
        c2_status = "PASS"
    elif c2m_status == "FAIL" and c2s_status == "FAIL":
        c2_status = "FAIL"
    else:
        c2_status = "WARN"

    checks["C_effect_vs_floor"] = {
        "floor_delta_db": floor_delta,
        "repeat_delta_db": repeat_delta,
        "delta_ref_db": float(delta_ref),
        "A3_minus_A2_delta_median_db": dmed,
        "ratio_to_floor": ratio,
        "C1_status": c1_status,
        # Backward-compatible legacy keys
        "A4_material_shift_late_excess_db": float(a4_iso_late_span),
        "A5_stress_delta_late_excess_db": d_stress,
        "A5_stress_var_ratio": var_ratio,
        "C2_status": c2_status,
        # C2-M detailed
        "C2M_primary_subset": "A4_iso(include_late_panel=0)",
        "C2M_secondary_subset": "A4_bridge(include_late_panel=1)",
        "C2M_primary_metric": "XPD_early_excess_db",
        "C2M_primary_span_db": float(a4_iso_primary_span),
        "C2M_primary_ratio_to_delta": float(c2m_primary_ratio),
        "C2M_primary_status": str(c2m_primary_status),
        "C2M_secondary_late_span_db": float(a4_iso_late_span),
        "C2M_secondary_late_ratio_to_delta": float(c2m_late_ratio),
        "C2M_secondary_late_status": str(c2m_late_status),
        "C2M_secondary_lpol_span_db": float(a4_iso_lpol_span),
        "C2M_secondary_lpol_ratio_to_delta": float(c2m_lpol_ratio),
        "C2M_secondary_lpol_status": str(c2m_lpol_status),
        "C2M_bridge_primary_span_db": float(a4_bridge_primary_span),
        "C2M_bridge_primary_ratio_to_delta": float(c2m_bridge_ratio),
        "C2M_bridge_primary_status": str(c2m_bridge_status),
        "C2M_bridge_late_span_db": float(a4_bridge_late_span),
        "C2M_bridge_late_ratio_to_delta": float(c2m_bridge_late_ratio),
        "C2M_bridge_late_status": str(c2m_bridge_late_status),
        "C2M_bridge_lpol_span_db": float(a4_bridge_lpol_span),
        "C2M_bridge_lpol_ratio_to_delta": float(c2m_bridge_lpol_ratio),
        "C2M_bridge_lpol_status": str(c2m_bridge_lpol_status),
        "A4_iso_n": int(len(a4_iso)),
        "A4_bridge_n": int(len(a4_bridge)),
        "A4_bridge_dispersion_on_n": int(a4_bridge_disp_on_n),
        "A4_dispersion_claim_ready": bool(a4_bridge_disp_on_n > 0),
        "C2M_status": str(c2m_status),
        "A4_target_C_median_db": a4_target_c,
        "A4_target_exists_rate": a4_target_exists,
        "A4_target_gate_status": "PASS" if a4_target_ok else "WARN",
        "A4_iso_material_medians_early_ex": a4_iso_early_meds,
        "A4_iso_material_medians_late_ex": a4_iso_late_meds,
        "A4_iso_material_medians_lpol": a4_iso_lpol_meds,
        "A4_bridge_material_medians_early_ex": a4_bridge_early_meds,
        "A4_bridge_material_medians_late_ex": a4_bridge_late_meds,
        "A4_bridge_material_medians_lpol": a4_bridge_lpol_meds,
        # Backward-compatibility aliases (kept for existing downstream readers).
        "A4_material_medians_early_ex": a4_iso_early_meds,
        "A4_material_medians_late_ex": a4_iso_late_meds,
        "A4_material_medians_lpol": a4_iso_lpol_meds,
        # C2-S detailed
        "C2S_primary_metric": "L_pol_db",
        "C2S_delta_lpol_db": float(d_lpol),
        "C2S_primary_status": str(c2s_primary_status),
        "C2S_status": str(c2s_status),
        "C2S_delta_rho_early_lin": float(d_rho),
        "C2S_delta_ds_s": float(d_ds),
        "C2S_delta_late_ex_db": float(d_stress),
        "C2S_late_ex_var_ratio": float(var_ratio),
        "C2S_gate_delta_p_target_db": float(gate_dp_target_db),
        "C2S_gate_status": str(gate_status),
        "C2S_semantics_gate_status": str(c2s_semantics_gate),
        "C2S_semantics": str(a5_semantics.get("dominant_semantics", "none")),
        "C2S_semantics_note": str(a5_semantics.get("note", "")),
        "C2S_gate_pair_count": int(len(dp_target_vals)),
    }

    # D identifiability (D1 global/isolation + D2 multi-factor + D3 strata coverage)
    inc_by_link = _dominant_incidence_by_link(ray_rows)

    # D1-a: EL-identifying coverage (A3/A4/B*)
    el_ident_sids = {"A3", "A4", "B1", "B2", "B3"}
    rows_el_global = [r for r in link_rows if str(r.get("scenario_id", "")) in el_ident_sids]
    x_el_global = np.asarray([_num(r.get("EL_proxy_db", np.nan)) for r in rows_el_global], dtype=float)
    x_el_global = x_el_global[np.isfinite(x_el_global)]
    el_iqr = float(np.percentile(x_el_global, 75.0) - np.percentile(x_el_global, 25.0)) if len(x_el_global) else float("nan")
    el_bins_global = {"low": 0, "mid": 0, "high": 0}
    if len(x_el_global) >= 3:
        q1g, q2g = np.percentile(x_el_global, [33.3, 66.7])
        for v in x_el_global:
            if v <= q1g:
                el_bins_global["low"] = int(el_bins_global["low"] + 1)
            elif v <= q2g:
                el_bins_global["mid"] = int(el_bins_global["mid"] + 1)
            else:
                el_bins_global["high"] = int(el_bins_global["high"] + 1)
        min_el_bin = int(min(el_bins_global.values()))
    else:
        q1g, q2g = (float("nan"), float("nan"))
        min_el_bin = 0
    if np.isfinite(el_iqr) and el_iqr >= 3.0 and min_el_bin >= 5:
        d1_global_status = "PASS"
    elif np.isfinite(el_iqr) and el_iqr >= 2.0 and min_el_bin >= 2:
        d1_global_status = "WARN"
    elif len(x_el_global) == 0:
        d1_global_status = "INCONCLUSIVE"
    else:
        d1_global_status = "FAIL"

    # D1-b: local isolation checks (A2 parity-isolation, A5 stress-isolation)
    a2_rows = scenario_rows.get("A2", [])
    a2_el = np.asarray([_num(r.get("EL_proxy_db", np.nan)) for r in a2_rows], dtype=float)
    a2_el = a2_el[np.isfinite(a2_el)]
    a2_el_std = float(np.std(a2_el, ddof=1)) if len(a2_el) >= 2 else float("nan")
    if len(a2_el) >= 2 and np.isfinite(a2_el_std):
        if a2_el_std <= 0.25:
            a2_iso_status = "PASS"
        elif a2_el_std <= 1.0:
            a2_iso_status = "WARN"
        else:
            a2_iso_status = "FAIL"
    else:
        a2_iso_status = "INCONCLUSIVE"

    a5_rows_all = scenario_rows.get("A5", [])
    a5_base = [r for r in a5_rows_all if int(_num(r.get("roughness_flag", 0))) == 0 and int(_num(r.get("human_flag", 0))) == 0]
    a5_stress = [r for r in a5_rows_all if int(_num(r.get("roughness_flag", 0))) == 1 or int(_num(r.get("human_flag", 0))) == 1]
    a5_base_el = np.asarray([_num(r.get("EL_proxy_db", np.nan)) for r in a5_base], dtype=float)
    a5_stress_el = np.asarray([_num(r.get("EL_proxy_db", np.nan)) for r in a5_stress], dtype=float)
    a5_base_el = a5_base_el[np.isfinite(a5_base_el)]
    a5_stress_el = a5_stress_el[np.isfinite(a5_stress_el)]
    d_el_a5 = float(np.median(a5_stress_el) - np.median(a5_base_el)) if (len(a5_base_el) and len(a5_stress_el)) else float("nan")
    if a5_role in {"stress_isolation", "isolation"}:
        if np.isfinite(d_el_a5):
            if abs(d_el_a5) <= 1.0:
                a5_role_status = "PASS"
            elif abs(d_el_a5) <= 2.0:
                a5_role_status = "WARN"
            else:
                a5_role_status = "FAIL"
        else:
            a5_role_status = "INCONCLUSIVE"
        a5_role_target = "small EL change is desired for stress isolation"
    else:
        # stress_response mode: large EL invariance is not mandatory.
        if np.isfinite(d_lpol):
            if d_lpol < 0.0:
                a5_role_status, _ = _judge_vs_delta(float(d_lpol), float(delta_ref))
            else:
                a5_role_status = "FAIL"
        else:
            a5_role_status = "INCONCLUSIVE"
        if str(gate_status) == "BLOCKAGE" and a5_role_status == "PASS":
            a5_role_status = "WARN"
        a5_role_target = "response mode: L_pol decrease is primary, EL shift can be non-zero"

    if d1_global_status == "FAIL" or a2_iso_status == "FAIL":
        d1_status = "FAIL"
    elif d1_global_status == "PASS" and a2_iso_status == "PASS":
        if a5_role in {"stress_isolation", "isolation"}:
            d1_status = "PASS" if a5_role_status == "PASS" else ("WARN" if a5_role_status in {"WARN", "INCONCLUSIVE"} else "FAIL")
        else:
            # response mode mismatch should not collapse global identifiability.
            d1_status = "PASS" if a5_role_status in {"PASS", "WARN", "INCONCLUSIVE"} else "WARN"
    else:
        d1_status = "WARN"

    # D2: multi-factor separability (material×angle, stress×angle, collinearity)
    def _new_bin_counter() -> dict[str, int]:
        return {"low": 0, "mid": 0, "high": 0, "NA": 0}

    def _coverage_status(cov: dict[str, dict[str, int]]) -> tuple[str, int, int]:
        groups = sorted([k for k in cov.keys() if k and k != "NA"])
        if not groups:
            return "INCONCLUSIVE", 0, 0
        good = 0
        for g in groups:
            d = cov.get(g, {})
            n_bins = sum(int(d.get(b, 0) >= 2) for b in ["low", "mid", "high"])
            if n_bins >= 2:
                good += 1
        if len(groups) >= 2 and good == len(groups):
            return "PASS", good, len(groups)
        if good >= 1:
            return "WARN", good, len(groups)
        return "FAIL", good, len(groups)

    a4_cov: dict[str, dict[str, int]] = {}
    for r in scenario_rows.get("A4", []):
        mat = str(r.get("material_class", "NA"))
        inc = _num(inc_by_link.get(str(r.get("link_id", "")), np.nan))
        b = _inc_bin(inc)
        if mat not in a4_cov:
            a4_cov[mat] = _new_bin_counter()
        a4_cov[mat][b] = int(a4_cov[mat].get(b, 0) + 1)
    d2_mat_status, d2_mat_good, d2_mat_total = _coverage_status(a4_cov)

    a5_cov: dict[str, dict[str, int]] = {"base": _new_bin_counter(), "stress": _new_bin_counter()}
    for r in a5_base:
        inc = _num(inc_by_link.get(str(r.get("link_id", "")), np.nan))
        a5_cov["base"][_inc_bin(inc)] = int(a5_cov["base"][_inc_bin(inc)] + 1)
    for r in a5_stress:
        inc = _num(inc_by_link.get(str(r.get("link_id", "")), np.nan))
        a5_cov["stress"][_inc_bin(inc)] = int(a5_cov["stress"][_inc_bin(inc)] + 1)
    d2_stress_status, d2_stress_good, d2_stress_total = _coverage_status(a5_cov)

    vif_thr = float(dict(cfg.get("stats", {})).get("vif_threshold", 5.0))
    d2_stage1_rows = [r for r in link_rows if str(r.get("scenario_id", "")) in {"A3", "A4", "B1", "B2", "B3"}]
    d2_stage2_rows = [r for r in link_rows if str(r.get("scenario_id", "")) in {"A2", "A3", "A4", "A5", "B1", "B2", "B3"}]
    d2_stage1 = _design_diag(
        d2_stage1_rows,
        num_keys=["EL_proxy_db", "d_m"],
        cat_keys=["LOSflag", "obstacle_flag", "material_class"],
        inc_by_link=inc_by_link,
        vif_threshold=vif_thr,
    )
    d2_stage2 = _design_diag(
        d2_stage2_rows,
        num_keys=[],
        cat_keys=["odd_flag", "stress_flag", "LOSflag", "obstacle_flag"],
        inc_by_link=inc_by_link,
        vif_threshold=vif_thr,
    )
    d2_design_status = "PASS" if (d2_stage1.get("status") == "PASS" and d2_stage2.get("status") == "PASS") else (
        "FAIL" if ("FAIL" in {d2_stage1.get("status"), d2_stage2.get("status")}) else "WARN"
    )

    if "FAIL" in {d2_mat_status, d2_stress_status, d2_design_status}:
        d2_status = "FAIL"
    elif d2_mat_status == "PASS" and d2_stress_status == "PASS" and d2_design_status == "PASS":
        d2_status = "PASS"
    else:
        d2_status = "WARN"

    # D3: room/grid sampling strata coverage (LOS/NLOS × EL tertiles)
    b_rows = [r for r in link_rows if str(r.get("scenario_id", "")) in {"B1", "B2", "B3"}]
    q1, q2 = _el_q1_q2(b_rows)
    strata_counts = _strata_counts(b_rows, q1=float(q1), q2=float(q2))
    selected_b_rows = [
        r for r in (selected_rows or []) if str(r.get("scenario_id", "")) in {"B1", "B2", "B3"}
    ]
    strata_counts_selected = _strata_counts(selected_b_rows, q1=float(q1), q2=float(q2)) if selected_b_rows else None
    base_6 = _strata_base_bins()
    viable_bins = [k for k in base_6 if int(strata_counts.get(k, 0)) > 0]
    min_strata_all = int(min(int(strata_counts.get(k, 0)) for k in base_6)) if base_6 else 0
    min_strata_viable = int(min(int(strata_counts.get(k, 0)) for k in viable_bins)) if viable_bins else 0
    qna_total = int(strata_counts.get("LOS0_qNA", 0) + strata_counts.get("LOS1_qNA", 0))
    if len(viable_bins) == 0:
        d3_status = "FAIL"
    elif min_strata_viable >= 3 and qna_total == 0:
        d3_status = "PASS"
    elif min_strata_viable >= 2:
        d3_status = "WARN"
    else:
        d3_status = "FAIL"

    hole_analysis = _d3_hole_analysis(strata_counts, strata_counts_selected)

    # Priority-3: per-scenario room summary export
    b_per_scenario: list[dict[str, Any]] = []
    for sid in ["B1", "B2", "B3"]:
        s_rows = [r for r in b_rows if str(r.get("scenario_id", "")) == sid]
        sq1, sq2 = _el_q1_q2(s_rows)
        scounts = _strata_counts(s_rows, q1=float(sq1), q2=float(sq2))
        sviable = [k for k in _strata_base_bins() if int(scounts.get(k, 0)) > 0]
        smin_all = int(min(int(scounts.get(k, 0)) for k in _strata_base_bins())) if _strata_base_bins() else 0
        smin_viable = int(min(int(scounts.get(k, 0)) for k in sviable)) if sviable else 0
        sqna = int(scounts.get("LOS0_qNA", 0) + scounts.get("LOS1_qNA", 0))
        if len(sviable) == 0:
            s_status = "FAIL"
        elif smin_viable >= 3 and sqna == 0:
            s_status = "PASS"
        elif smin_viable >= 2:
            s_status = "WARN"
        else:
            s_status = "FAIL"
        b_per_scenario.append(
            {
                "scenario_id": sid,
                "status": s_status,
                "n_rows": int(len(s_rows)),
                "q1_db": float(sq1) if np.isfinite(sq1) else float("nan"),
                "q2_db": float(sq2) if np.isfinite(sq2) else float("nan"),
                "min_strata_all_n": int(smin_all),
                "min_strata_viable_n": int(smin_viable),
                "qna_total": int(sqna),
                "strata_counts": scounts,
            }
        )

    # Keep legacy correlation field for backward compatibility
    los = np.asarray([_num(r.get("LOSflag", np.nan)) for r in rows_el_global], dtype=float)
    d_m = np.asarray([_num(r.get("d_m", np.nan)) for r in rows_el_global], dtype=float)
    m = np.isfinite(los) & np.isfinite(d_m)
    corr_d_los = float(np.corrcoef(los[m], d_m[m])[0, 1]) if int(np.sum(m)) > 3 else float("nan")

    if "FAIL" in {d1_status, d2_status, d3_status}:
        d_status = "FAIL"
    elif d1_status == "PASS" and d2_status == "PASS" and d3_status == "PASS":
        d_status = "PASS"
    else:
        d_status = "WARN"

    checks["D_identifiability"] = {
        # legacy keys
        "EL_iqr_db": float(el_iqr),
        "corr_d_vs_LOS": float(corr_d_los),
        "strata_counts": strata_counts,
        "min_strata_n": int(min_strata_viable),
        "status": d_status,
        # D1 detail
        "D1": {
            "status": d1_status,
            "global": {
                "status": d1_global_status,
                "subset": sorted(el_ident_sids),
                "EL_iqr_db": float(el_iqr),
                "el_bins_low_mid_high": el_bins_global,
                "min_bin_n": int(min_el_bin),
                "n_rows": int(len(rows_el_global)),
            },
            "A2_isolation": {
                "status": a2_iso_status,
                "n_rows": int(len(a2_el)),
                "EL_std_db": float(a2_el_std),
                "target": "small EL variation is desired for parity isolation",
            },
            "A5_isolation": {
                "status": a5_role_status,
                "role": a5_role,
                "n_base": int(len(a5_base_el)),
                "n_stress": int(len(a5_stress_el)),
                "delta_median_EL_stress_minus_base_db": float(d_el_a5),
                "target": a5_role_target,
            },
        },
        # D2 detail
        "D2": {
            "status": d2_status,
            "material_x_angle_status": d2_mat_status,
            "material_x_angle_groups_good": int(d2_mat_good),
            "material_x_angle_groups_total": int(d2_mat_total),
            "material_x_angle_coverage": a4_cov,
            "stress_x_angle_status": d2_stress_status,
            "stress_x_angle_groups_good": int(d2_stress_good),
            "stress_x_angle_groups_total": int(d2_stress_total),
            "stress_x_angle_coverage": a5_cov,
            "design_status": d2_design_status,
            "stage1": d2_stage1,
            "stage2": d2_stage2,
            "design_rank": int(round(_max_finite([_num(d2_stage1.get("design_rank", np.nan)), _num(d2_stage2.get("design_rank", np.nan))], default=0.0))),
            "design_cols": int(round(_max_finite([_num(d2_stage1.get("design_cols", np.nan)), _num(d2_stage2.get("design_cols", np.nan))], default=0.0))),
            "condition_number": float(_max_finite([_num(d2_stage1.get("condition_number", np.nan)), _num(d2_stage2.get("condition_number", np.nan))], default=float("nan"))),
            "vif_threshold": float(vif_thr),
            "vif": {
                "stage1": d2_stage1.get("vif", {}),
                "stage2": d2_stage2.get("vif", {}),
            },
            "vif_warnings": {
                "stage1": d2_stage1.get("vif_warnings", {}),
                "stage2": d2_stage2.get("vif_warnings", {}),
            },
        },
        # D3 detail
        "D3": {
            "status": d3_status,
            "n_rows": int(len(b_rows)),
            "q1_db": float(q1) if np.isfinite(q1) else float("nan"),
            "q2_db": float(q2) if np.isfinite(q2) else float("nan"),
            "all_6_bins": base_6,
            "viable_bins": viable_bins,
            "min_strata_all_n": int(min_strata_all),
            "min_strata_viable_n": int(min_strata_viable),
            "qna_total": int(qna_total),
            "strata_counts": strata_counts,
            "selected_rows_n": int(len(selected_b_rows)),
            "strata_counts_selected": strata_counts_selected if isinstance(strata_counts_selected, dict) else {},
            "hole_analysis": hole_analysis,
            "per_scenario_summary": b_per_scenario,
        },
    }

    # E power-based only + strict meta-contract checks.
    basis_vals = _collect_unique_values(link_rows, "basis")
    conv_vals = _collect_unique_values(link_rows, "convention")
    msrc_vals = [v.upper() for v in _collect_unique_values(link_rows, "matrix_source")]
    if not basis_vals:
        basis_vals = sorted(
            set(
                str((dict(r.get("summary", {})).get("basis", "") if isinstance(r, dict) else "")).strip()
                for r in runs
                if str((dict(r.get("summary", {})).get("basis", "") if isinstance(r, dict) else "")).strip()
            )
        )
    if not conv_vals:
        conv_vals = sorted(
            set(
                str((dict(r.get("summary", {})).get("convention", "") if isinstance(r, dict) else "")).strip()
                for r in runs
                if str((dict(r.get("summary", {})).get("convention", "") if isinstance(r, dict) else "")).strip()
            )
        )
    if not msrc_vals:
        msrc_vals = sorted(
            set(
                str((dict(r.get("summary", {})).get("matrix_source", "") if isinstance(r, dict) else "")).upper().strip()
                for r in runs
                if str((dict(r.get("summary", {})).get("matrix_source", "") if isinstance(r, dict) else "")).strip()
            )
        )
    fcp = np.asarray([_num(r.get("force_cp_swap_on_odd_reflection", np.nan)) for r in link_rows], dtype=float)
    force_cp_any_true = bool(np.any(np.isfinite(fcp) & (np.round(fcp) == 1)))
    if not np.any(np.isfinite(fcp)):
        force_cp_any_true = any(
            bool(dict(r.get("summary", {})).get("force_cp_swap_on_odd_reflection", False))
            for r in runs
            if isinstance(r, dict)
        )
    meta_contract_ok = (
        len(basis_vals) == 1
        and len(conv_vals) == 1
        and len(msrc_vals) == 1
        and msrc_vals[0] in {"A", "J"}
    )
    if meta_contract_ok and (not force_cp_any_true):
        e_status = "PASS"
    elif meta_contract_ok:
        e_status = "WARN"
    else:
        e_status = "FAIL"

    c0_rows = scenario_rows.get("C0", [])
    tx_leak = np.asarray([_num(r.get("tx_cross_pol_leakage_db", np.nan)) for r in c0_rows], dtype=float)
    rx_leak = np.asarray([_num(r.get("rx_cross_pol_leakage_db", np.nan)) for r in c0_rows], dtype=float)
    c0_floor = np.asarray([_num(r.get("XPD_early_db", np.nan)) for r in c0_rows], dtype=float)
    tx_med = float(np.nanmedian(tx_leak)) if np.any(np.isfinite(tx_leak)) else float("nan")
    rx_med = float(np.nanmedian(rx_leak)) if np.any(np.isfinite(rx_leak)) else float("nan")
    if not np.isfinite(tx_med) or not np.isfinite(rx_med):
        tx_runs = []
        rx_runs = []
        for rr in runs:
            sm = dict(rr.get("summary", {})) if isinstance(rr, dict) else {}
            ac = dict(sm.get("antenna_config", {})) if isinstance(sm.get("antenna_config", {}), dict) else {}
            txv = _num(ac.get("tx_cross_pol_leakage_db", np.nan))
            rxv = _num(ac.get("rx_cross_pol_leakage_db", np.nan))
            if np.isfinite(txv):
                tx_runs.append(float(txv))
            if np.isfinite(rxv):
                rx_runs.append(float(rxv))
        if (not np.isfinite(tx_med)) and tx_runs:
            tx_med = float(np.median(np.asarray(tx_runs, dtype=float)))
        if (not np.isfinite(rx_med)) and rx_runs:
            rx_med = float(np.median(np.asarray(rx_runs, dtype=float)))
    c0_floor_med = float(np.nanmedian(c0_floor)) if np.any(np.isfinite(c0_floor)) else float("nan")
    # Heuristic sanity target for coupling-limited floor.
    coupling_floor_nominal_db = float(min(tx_med, rx_med)) if np.isfinite(tx_med) and np.isfinite(rx_med) else float("nan")
    coupling_floor_delta_db = float(c0_floor_med - coupling_floor_nominal_db) if np.isfinite(c0_floor_med) and np.isfinite(coupling_floor_nominal_db) else float("nan")
    if np.isfinite(coupling_floor_delta_db) and abs(coupling_floor_delta_db) <= 10.0:
        coupling_status = "PASS"
    elif np.isfinite(coupling_floor_delta_db):
        coupling_status = "WARN"
    else:
        coupling_status = "INCONCLUSIVE"

    checks["E_power_based"] = {
        "status": str(e_status),
        "used_metrics": [
            "XPD_early_db",
            "XPD_late_db",
            "rho_early_lin",
            "L_pol_db",
            "delay_spread_rms_s",
            "early_energy_fraction",
            "EL_proxy_db",
            "LOSflag",
        ],
        "basis_values": basis_vals,
        "convention_values": conv_vals,
        "matrix_source_values": msrc_vals,
        "force_cp_swap_any_true": bool(force_cp_any_true),
        "matrix_source_rule": "Use circular basis for CP interpretation and keep matrix_source fixed (A_f for antenna-embedded, J_f for de-embedded).",
        "main_result_rule": "force_cp_swap_on_odd_reflection must be false for main-result evidence.",
        "coupling_floor_check": {
            "status": str(coupling_status),
            "c0_xpd_floor_median_db": float(c0_floor_med),
            "tx_cross_pol_leakage_median_db": float(tx_med),
            "rx_cross_pol_leakage_median_db": float(rx_med),
            "coupling_floor_nominal_db": float(coupling_floor_nominal_db),
            "delta_c0_minus_nominal_db": float(coupling_floor_delta_db),
            "note": "Heuristic sanity check only; measured C0 floor should be reviewed against configured antenna coupling/leakage model.",
        },
        "note": "No complex-phase fields are used by this report pipeline.",
        "a5_stress_semantics": str(a5_semantics.get("dominant_semantics", "none")),
        "a5_contamination_response_ready": bool(a5_semantics.get("contamination_response_ready", False)),
        "a5_semantics_note": str(a5_semantics.get("note", "")),
    }
    return checks


def _make_diagnostic_plots(
    out_fig_dir: Path,
    link_rows: list[dict[str, Any]],
    ray_rows: list[dict[str, Any]],
) -> dict[str, str]:
    plots: dict[str, str] = {}
    by_s = metrics_lib.split_by_scenario(link_rows)

    c0 = by_s.get("C0", [])
    if c0:
        plots["C0_floor_cdf"] = plot_lib.plot_cdf(
            [_num(r.get("XPD_early_db", np.nan)) for r in c0],
            out_fig_dir / "C0__ALL__xpd_floor_cdf.png",
            title="C0 XPD floor distribution",
            xlabel="XPD_early (dB)",
        )
        plots["C0_floor_vs_yaw"] = plot_lib.plot_scatter(
            x=[_num(r.get("yaw_deg", np.nan)) for r in c0],
            y=[_num(r.get("XPD_early_db", np.nan)) for r in c0],
            c=[_num(r.get("d_m", np.nan)) for r in c0],
            out_png=out_fig_dir / "C0__ALL__xpd_floor_vs_yaw.png",
            title="C0 XPD floor vs yaw",
            xlabel="yaw (deg)",
            ylabel="XPD_early (dB)",
            add_fit=False,
        )
        plots["C0_floor_vs_distance"] = plot_lib.plot_scatter(
            x=[_num(r.get("d_m", np.nan)) for r in c0],
            y=[_num(r.get("XPD_early_db", np.nan)) for r in c0],
            out_png=out_fig_dir / "C0__ALL__xpd_floor_vs_distance.png",
            title="C0 XPD floor vs distance",
            xlabel="distance (m)",
            ylabel="XPD_early (dB)",
            add_fit=True,
        )

    a2 = by_s.get("A2", [])
    a3 = by_s.get("A3", [])
    if a2 or a3:
        plots["A2_A3_xpd_ex_box"] = plot_lib.plot_box_by_group(
            rows=[*a2, *a3],
            group_key="scenario_id",
            value_key="XPD_early_excess_db",
            out_png=out_fig_dir / "A2A3__ALL__xpd_early_ex_box.png",
            title="A2/A3 XPD_early_excess",
            ylabel="dB",
        )
        plots["A2_A3_lpol_box"] = plot_lib.plot_box_by_group(
            rows=[*a2, *a3],
            group_key="scenario_id",
            value_key="L_pol_db",
            out_png=out_fig_dir / "A2A3__ALL__lpol_box.png",
            title="A2/A3 L_pol",
            ylabel="dB",
        )
        plots["A2_A3_xpd_ex_cdf"] = plot_lib.plot_multi_cdf(
            {
                "A2": [_num(r.get("XPD_early_excess_db", np.nan)) for r in a2],
                "A3": [_num(r.get("XPD_early_excess_db", np.nan)) for r in a3],
            },
            out_png=out_fig_dir / "A2A3__ALL__xpd_early_ex_cdf.png",
            title="A2 vs A3 XPD_early_excess CDF",
            xlabel="XPD_early_excess (dB)",
        )

    all_rows = [r for s in ["A2", "A3", "A4", "A5", "B1", "B2", "B3"] for r in by_s.get(s, [])]
    if all_rows:
        plots["EL_vs_XPD_ex"] = plot_lib.plot_scatter(
            x=[_num(r.get("EL_proxy_db", np.nan)) for r in all_rows],
            y=[_num(r.get("XPD_early_excess_db", np.nan)) for r in all_rows],
            out_png=out_fig_dir / "ALL__xpd_early_ex_vs_el_proxy.png",
            title="XPD_early_excess vs EL_proxy",
            xlabel="EL_proxy (dB)",
            ylabel="XPD_early_excess (dB)",
            add_fit=True,
        )

    # rays diagnostics
    if ray_rows:
        bounce = [int(round(_num(r.get("n_bounce", np.nan)))) for r in ray_rows if np.isfinite(_num(r.get("n_bounce", np.nan)))]
        if bounce:
            vals = np.asarray(bounce, dtype=int)
            bins = np.arange(int(np.min(vals)), int(np.max(vals)) + 2)
            fig_out = out_fig_dir / "ALL__rays_bounce_hist.png"
            fig_out.parent.mkdir(parents=True, exist_ok=True)
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            ax.hist(vals, bins=bins - 0.5, rwidth=0.8)
            ax.set_xlabel("n_bounce")
            ax.set_ylabel("count")
            ax.set_title("Ray bounce histogram")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(fig_out, dpi=140)
            plt.close(fig)
            plots["rays_bounce_hist"] = str(fig_out)

    return plots


def _build_markdown(
    out_root: Path,
    run_group: str,
    link_rows: list[dict[str, Any]],
    checks: dict[str, Any],
    floor_ref: dict[str, Any],
    scenario_scene: dict[str, str],
    global_plots: dict[str, str],
    warns: list[str],
    warn_cases: list[dict[str, Any]],
    a3_review_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(f"# Diagnostic Report ({run_group})")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append("")
    scen_counts: dict[str, int] = {}
    for r in link_rows:
        scen_counts[str(r.get("scenario_id", "NA"))] = int(scen_counts.get(str(r.get("scenario_id", "NA")), 0) + 1)
    rows = [{"scenario": s, "n_links": n} for s, n in sorted(scen_counts.items())]
    lines.append(report_md.md_table(rows, ["scenario", "n_links"]))
    lines.append("")
    lines.append("- Figure metadata: [figure_metadata.csv](tables/figure_metadata.csv)")
    lines.append("")
    lines.append("## Final Scenario Structure (Agreed)")
    lines.append("")
    lines.append(
        report_md.md_table(
            report_md.final_structure_rows(),
            ["unit", "role", "notes"],
        )
    )
    lines.append("")
    lines.append("## Floor Reference")
    lines.append("")
    lines.append(
        report_md.md_table(
            [
                {
                    "xpd_floor_db": floor_ref.get("xpd_floor_db", np.nan),
                    "delta_floor_db": floor_ref.get("delta_floor_db", np.nan),
                    "p5_db": floor_ref.get("p5_db", np.nan),
                    "p95_db": floor_ref.get("p95_db", np.nan),
                    "count": floor_ref.get("count", 0),
                    "method": floor_ref.get("method", ""),
                }
            ],
            ["xpd_floor_db", "delta_floor_db", "p5_db", "p95_db", "count", "method"],
        )
    )
    lines.append("")

    lines.append("## Diagnostics A-E")
    lines.append("")

    # A
    lines.append("### A) Geometry / Path Validity")
    lines.append("")
    lines.append(report_md.md_table(checks.get("A1_los_blocked", []), ["scenario", "los_rays", "status", "reason"]))
    lines.append("")
    lines.append(report_md.md_table(checks.get("A2_target_bounce", []), ["scenario", "target_n", "hit", "total", "rate", "status"]))
    lines.append("")
    a3c = checks.get("A3_coord_sanity", {})
    lines.append(f"- A3 coordinate sanity: **{a3c.get('status', 'WARN')}** ({a3c.get('note', '')})")
    lines.append("")
    a5s = checks.get("A5_stress_semantics", {})
    if isinstance(a5s, dict):
        lines.append("- A5 stress semantics (path-structure vs polarization-only)")
        lines.append("")
        lines.append(
            report_md.md_table(
                [
                    {
                        "status": a5s.get("status", ""),
                        "dominant_semantics": a5s.get("dominant_semantics", ""),
                        "n_stress_rows": a5s.get("n_stress_rows", 0),
                        "n_response": a5s.get("n_response", 0),
                        "n_polarization_only": a5s.get("n_polarization_only", 0),
                        "contamination_response_ready": a5s.get("contamination_response_ready", False),
                        "note": a5s.get("note", ""),
                    }
                ],
                [
                    "status",
                    "dominant_semantics",
                    "n_stress_rows",
                    "n_response",
                    "n_polarization_only",
                    "contamination_response_ready",
                    "note",
                ],
            )
        )
        lines.append("")
    if a3_review_rows:
        lines.append("- A3 geometry manual review (ray-path visualization required before experiment)")
        lines.append("")
        lines.append(
            report_md.md_table(
                a3_review_rows,
                [
                    "scenario_id",
                    "case_id",
                    "review_status",
                    "scene_debug_valid",
                    "los_rays",
                    "has_target_bounce_n2",
                    "rays_topk_n",
                    "scene_debug_issues",
                ],
            )
        )
        lines.append("")
        for rr in a3_review_rows:
            sp = str(rr.get("scene_png_path", ""))
            if not sp:
                continue
            lines.append(f"#### A3 case {rr.get('case_id', '')}")
            lines.append("")
            lines.append(f"- review_status: **{rr.get('review_status', '')}**")
            lines.append(f"![A3-case-{rr.get('case_id','')}]({report_md.relpath(sp, out_root)})")
            lines.append("")

    # B
    b = checks.get("B_time_resolution", {})
    lines.append("### B) Time Resolution / Delay Separability")
    lines.append("")
    lines.append("- `W_floor`(C0): `C_floor = Sum(P_nonLOS in W_floor) / P_LOS`")
    lines.append("- `W_target`(A2-A5): `C_target = Sum(P_non-target in W_target) / P_target`")
    lines.append("- `W_early`(B1-B3): `S(Te)=|mu_LOS-mu_NLOS|/sqrt(sig_LOS^2+sig_NLOS^2)`")
    lines.append("")
    lines.append(
        report_md.md_table(
            [
                {
                    "freq_source": b.get("freq_source", ""),
                    "dt_res_s": b.get("dt_res_s", np.nan),
                    "tau_max_s": b.get("tau_max_s", np.nan),
                    "Te_s": b.get("Te_s", np.nan),
                    "Tmax_s": b.get("Tmax_s", np.nan),
                    "W_floor_C_median_db": b.get("W_floor_C_median_db", np.nan),
                    "W_floor_status": b.get("W_floor_status", ""),
                    "A2_target_in_Wearly_rate": b.get("A2_target_in_Wearly_rate", np.nan),
                    "A3_target_in_Wearly_rate": b.get("A3_target_in_Wearly_rate", np.nan),
                    "A2_C_target_median_db": b.get("A2_C_target_median_db", np.nan),
                    "A3_C_target_median_db": b.get("A3_C_target_median_db", np.nan),
                    "A2_target_sign_hit_rate": _num(dict(b.get("A2_target_window_sign", {})).get("expected_sign_hit_rate", np.nan)),
                    "A2_target_sign_status": dict(b.get("A2_target_window_sign", {})).get("status", ""),
                    "A3_target_sign_hit_rate": _num(dict(b.get("A3_target_window_sign", {})).get("expected_sign_hit_rate", np.nan)),
                    "A3_target_sign_status": dict(b.get("A3_target_window_sign", {})).get("status", ""),
                    "A3_target_sign_status_reporting": b.get("A3_target_window_sign_reporting_status", ""),
                    "A2A3_sign_stability_raw": b.get("A2A3_sign_stability_raw_status", ""),
                    "A2A3_sign_stability_reporting": b.get("A2A3_sign_stability_reporting_status", ""),
                    "A6_parity_status": dict(b.get("A6_parity_benchmark", {})).get("status", ""),
                    "A6_hit_rate_odd": _num(dict(b.get("A6_parity_benchmark", {})).get("hit_rate_odd", np.nan)),
                    "A6_hit_rate_even": _num(dict(b.get("A6_parity_benchmark", {})).get("hit_rate_even", np.nan)),
                    "G2_primary_evidence_source": b.get("G2_primary_evidence_source", ""),
                    "G2_primary_evidence_status": b.get("G2_primary_evidence_status", ""),
                    "A3_mechanism_status": b.get("A3_mechanism_status", ""),
                    "A3_system_early_status": b.get("A3_system_early_status", ""),
                    "A5_target_mode": b.get("A5_target_mode", ""),
                    "A5_stress_semantics": b.get("A5_stress_semantics", ""),
                    "A5_contamination_response_ready": b.get("A5_contamination_response_ready", False),
                    "min_delay_gap_median_s": b.get("B2_min_delay_gap_median_s", np.nan),
                    "B2_status": b.get("B2_status", ""),
                    "B3_status": b.get("B3_status", ""),
                    "W3_best_te_ns": b.get("W3_best_te_ns", np.nan),
                    "W3_best_S_xpd_early": b.get("W3_best_S_xpd_early", np.nan),
                    "W3_status": b.get("W3_status", ""),
                }
            ],
            [
                "freq_source",
                "dt_res_s",
                "tau_max_s",
                "Te_s",
                "Tmax_s",
                "W_floor_C_median_db",
                "W_floor_status",
                "A2_target_in_Wearly_rate",
                "A3_target_in_Wearly_rate",
                "A2_C_target_median_db",
                "A3_C_target_median_db",
                "A2_target_sign_hit_rate",
                "A2_target_sign_status",
                "A3_target_sign_hit_rate",
                "A3_target_sign_status",
                "A3_target_sign_status_reporting",
                "A2A3_sign_stability_raw",
                "A2A3_sign_stability_reporting",
                "A6_parity_status",
                "A6_hit_rate_odd",
                "A6_hit_rate_even",
                "G2_primary_evidence_source",
                "G2_primary_evidence_status",
                "A3_mechanism_status",
                "A3_system_early_status",
                "A5_target_mode",
                "A5_stress_semantics",
                "A5_contamination_response_ready",
                "min_delay_gap_median_s",
                "B2_status",
                "B3_status",
                "W3_best_te_ns",
                "W3_best_S_xpd_early",
                "W3_status",
            ],
        )
    )
    lines.append("")
    if str(b.get("A3_reporting_rule", "")).strip():
        lines.append(f"- A3 reporting rule: {b.get('A3_reporting_rule', '')}")
        lines.append("")
    if str(b.get("G2_primary_evidence_source", "")).strip():
        lines.append(
            f"- G2 primary evidence: `{b.get('G2_primary_evidence_source', '')}` "
            f"(status={b.get('G2_primary_evidence_status', 'INCONCLUSIVE')})"
        )
        lines.append("")
    if str(b.get("A5_semantics_note", "")).strip():
        lines.append(f"- A5 semantics note: {b.get('A5_semantics_note', '')}")
        lines.append("")
    wfloor = b.get("W_floor_detail", {})
    if isinstance(wfloor, dict):
        lines.append("- W_floor(C0) contamination summary")
        lines.append("")
        lines.append(
            report_md.md_table(
                [
                    {
                        "W_floor_s": wfloor.get("W_floor_s", np.nan),
                        "C_floor_median_db": wfloor.get("C_floor_median_db", np.nan),
                        "C_floor_p95_db": wfloor.get("C_floor_p95_db", np.nan),
                        "rate_below_m10_db": wfloor.get("rate_below_m10_db", np.nan),
                        "rate_below_m15_db": wfloor.get("rate_below_m15_db", np.nan),
                        "status": wfloor.get("status", ""),
                        "n_cases": wfloor.get("n_cases", 0),
                    }
                ],
                ["W_floor_s", "C_floor_median_db", "C_floor_p95_db", "rate_below_m10_db", "rate_below_m15_db", "status", "n_cases"],
            )
        )
        lines.append("")
    wtarget = b.get("W_target_detail", {})
    if isinstance(wtarget, dict):
        wmap = b.get("W_target_s_by_scenario", {})
        mmap = b.get("W_target_mode_by_scenario", {})
        if isinstance(wmap, dict) and wmap:
            wr = []
            for sid in ["A2", "A3", "A4", "A5"]:
                if sid not in wmap:
                    continue
                wr.append(
                    {
                        "scenario": sid,
                        "W_target_s": _num(wmap.get(sid, np.nan)),
                        "W_target_ns": _num(wmap.get(sid, np.nan)) * 1e9 if np.isfinite(_num(wmap.get(sid, np.nan))) else np.nan,
                        "mode": str(mmap.get(sid, "default")) if isinstance(mmap, dict) else "default",
                    }
                )
            if wr:
                lines.append("- W_target per-scenario configuration")
                lines.append("")
                lines.append(report_md.md_table(wr, ["scenario", "W_target_s", "W_target_ns", "mode"]))
                lines.append("")
        trows = []
        for sid in ["A2", "A3", "A4", "A5"]:
            wsid = wtarget.get(sid, {})
            if not isinstance(wsid, dict):
                continue
            trows.append(
                {
                    "scenario": sid,
                    "target_n": wsid.get("target_n", np.nan),
                    "target_exists_rate": wsid.get("target_exists_rate", np.nan),
                    "target_is_first_rate": wsid.get("target_is_first_rate", np.nan),
                    "target_in_Wearly_rate": wsid.get("target_in_Wearly_rate", np.nan),
                    "C_target_median_db": wsid.get("C_target_median_db", np.nan),
                    "target_gap_median_s": wsid.get("target_gap_median_s", np.nan),
                    "status": wsid.get("status", ""),
                }
            )
        if trows:
            lines.append("- W_target(controlled scenarios) summary")
            lines.append("")
            lines.append(
                report_md.md_table(
                    trows,
                    [
                        "scenario",
                        "target_n",
                        "target_exists_rate",
                        "target_is_first_rate",
                        "target_in_Wearly_rate",
                        "C_target_median_db",
                        "target_gap_median_s",
                        "status",
                    ],
                )
            )
            lines.append("")
    w3 = b.get("W3_te_sweep", {})
    if isinstance(w3, dict):
        srows = w3.get("scores", [])
        if isinstance(srows, list) and srows:
            lines.append("- W_early(room/grid) Te sweep separation")
            lines.append("")
            lines.append(report_md.md_table(srows, ["Te_ns", "S_xpd_early", "S_rho_early_db", "S_l_pol"]))
            lines.append("")
    sstab = b.get("A2A3_sign_stability", {})
    if isinstance(sstab, dict):
        overall_raw = str(sstab.get("overall_status", "WARN"))
        overall_rep = str(b.get("A2A3_sign_stability_reporting_status", overall_raw))
        lines.append(f"- A2/A3 odd-even sign stability over Te sweep: **{overall_rep}** (raw={overall_raw})")
        lines.append("")
        rows_sign: list[dict[str, Any]] = []
        for sid in ["A2", "A3"]:
            d = sstab.get(sid, {})
            if not isinstance(d, dict):
                continue
            rows_sign.append(
                {
                    "scenario": sid,
                    "expected_sign": d.get("expected_sign", ""),
                    "min_hit_rate": d.get("min_hit_rate", np.nan),
                    "median_hit_rate": d.get("median_hit_rate", np.nan),
                    "status": d.get("status", ""),
                }
            )
        if rows_sign:
            lines.append(report_md.md_table(rows_sign, ["scenario", "expected_sign", "min_hit_rate", "median_hit_rate", "status"]))
            lines.append("")
        if str(b.get("A2A3_sign_stability_reporting_note", "")).strip():
            lines.append(f"- sign-stability reporting note: {b.get('A2A3_sign_stability_reporting_note', '')}")
            lines.append("")
    tw_rows: list[dict[str, Any]] = []
    for k in ["A2_target_window_sign", "A3_target_window_sign"]:
        d = b.get(k, {})
        if not isinstance(d, dict):
            continue
        tw_rows.append(
            {
                "scenario": d.get("scenario", ""),
                "target_n": d.get("target_n", np.nan),
                "W_target_s": d.get("W_target_s", np.nan),
                "expected_sign": d.get("expected_sign", ""),
                "n_eval": d.get("n_eval", 0),
                "sign_metric_for_status": d.get("sign_metric_for_status", ""),
                "expected_sign_hit_rate": d.get("expected_sign_hit_rate", np.nan),
                "expected_sign_hit_rate_raw": d.get("expected_sign_hit_rate_raw", np.nan),
                "expected_sign_hit_rate_ex": d.get("expected_sign_hit_rate_ex", np.nan),
                "median_xpd_target_raw_db": d.get("median_xpd_target_raw_db", np.nan),
                "median_xpd_target_ex_db": d.get("median_xpd_target_ex_db", np.nan),
                "status": d.get("status", ""),
            }
        )
    a6b = b.get("A6_parity_benchmark", {})
    if isinstance(a6b, dict):
        for vv in ["odd", "even"]:
            d = a6b.get(vv, {})
            if not isinstance(d, dict) or not d:
                continue
            n_eval = _num(d.get("n_eval", np.nan))
            if (not np.isfinite(n_eval) or n_eval <= 0) and not np.isfinite(_num(d.get("expected_sign_hit_rate", np.nan))):
                continue
            tw_rows.append(
                {
                    "scenario": f"A6_{vv}",
                    "target_n": d.get("target_n", np.nan),
                    "W_target_s": d.get("W_target_s", np.nan),
                    "expected_sign": d.get("expected_sign", ""),
                    "n_eval": d.get("n_eval", 0),
                    "sign_metric_for_status": d.get("sign_metric_for_status", ""),
                    "expected_sign_hit_rate": d.get("expected_sign_hit_rate", np.nan),
                    "expected_sign_hit_rate_raw": d.get("expected_sign_hit_rate_raw", np.nan),
                    "expected_sign_hit_rate_ex": d.get("expected_sign_hit_rate_ex", np.nan),
                    "median_xpd_target_raw_db": d.get("median_xpd_target_raw_db", np.nan),
                    "median_xpd_target_ex_db": d.get("median_xpd_target_ex_db", np.nan),
                    "status": d.get("status", ""),
                }
            )
    if tw_rows:
        lines.append("- Target-window sign metric (A2/A3 and A6 parity benchmark when available)")
        lines.append("")
        lines.append(
            report_md.md_table(
                tw_rows,
                [
                    "scenario",
                    "target_n",
                    "W_target_s",
                    "expected_sign",
                    "n_eval",
                    "sign_metric_for_status",
                    "expected_sign_hit_rate",
                    "expected_sign_hit_rate_raw",
                    "expected_sign_hit_rate_ex",
                    "median_xpd_target_raw_db",
                    "median_xpd_target_ex_db",
                    "status",
                ],
            )
        )
        lines.append("")
    a6_cmp = dict(dict(b.get("A6_parity_benchmark", {})).get("case_set_compare", {}))
    if a6_cmp:
        per_set = dict(a6_cmp.get("per_set", {}))
        rows_cmp: list[dict[str, Any]] = []
        for ss in sorted(per_set.keys()):
            ss_d = dict(per_set.get(ss, {}))
            for mode in ["odd", "even"]:
                mm = dict(ss_d.get(mode, {}))
                if not mm:
                    continue
                rows_cmp.append(
                    {
                        "case_set": str(ss),
                        "mode": str(mode),
                        "n_eval": mm.get("n_eval", 0),
                        "median_xpd_target_raw_db": mm.get("median_xpd_target_raw_db", np.nan),
                        "median_xpd_target_ex_db": mm.get("median_xpd_target_ex_db", np.nan),
                        "expected_sign_hit_rate": mm.get("expected_sign_hit_rate", np.nan),
                        "status": ss_d.get("status", ""),
                    }
                )
        if rows_cmp:
            lines.append("- A6 case-set comparison (full vs minimal, odd/even)")
            lines.append("")
            lines.append(
                report_md.md_table(
                    rows_cmp,
                    [
                        "case_set",
                        "mode",
                        "n_eval",
                        "median_xpd_target_raw_db",
                        "median_xpd_target_ex_db",
                        "expected_sign_hit_rate",
                        "status",
                    ],
                )
            )
            lines.append("")
        fvm = dict(a6_cmp.get("full_vs_minimal", {}))
        rows_delta: list[dict[str, Any]] = []
        for mode in ["odd", "even"]:
            dd = dict(fvm.get(mode, {}))
            if not dd:
                continue
            rows_delta.append(
                {
                    "mode": str(mode),
                    "delta_raw_minimal_minus_full_db": dd.get("delta_raw_minimal_minus_full_db", np.nan),
                    "delta_ex_minimal_minus_full_db": dd.get("delta_ex_minimal_minus_full_db", np.nan),
                    "delta_hit_rate_minimal_minus_full": dd.get("delta_hit_rate_minimal_minus_full", np.nan),
                    "n_eval_full": dd.get("n_eval_full", 0),
                    "n_eval_minimal": dd.get("n_eval_minimal", 0),
                }
            )
        if rows_delta:
            lines.append("- A6 full vs minimal delta summary (`minimal - full`)")
            lines.append("")
            lines.append(
                report_md.md_table(
                    rows_delta,
                    [
                        "mode",
                        "delta_raw_minimal_minus_full_db",
                        "delta_ex_minimal_minus_full_db",
                        "delta_hit_rate_minimal_minus_full",
                        "n_eval_full",
                        "n_eval_minimal",
                    ],
                )
            )
            lines.append("")

    # C
    c = checks.get("C_effect_vs_floor", {})
    lines.append("### C) Effect Size vs Floor Uncertainty")
    lines.append("")
    lines.append("- C2-M primary: `XPD_early_excess`; secondary: `XPD_late_excess`, `L_pol`")
    lines.append("- C2-S primary: `L_pol`; secondary: `rho_early`, `DS`, `XPD_late_excess`; gate: `ΔP_target,total > -6 dB`")
    lines.append("")
    lines.append(
        report_md.md_table(
            [
                {
                    "floor_delta_db": c.get("floor_delta_db", np.nan),
                    "repeat_delta_db": c.get("repeat_delta_db", np.nan),
                    "delta_ref_db": c.get("delta_ref_db", np.nan),
                    "A3_minus_A2_delta_median_db": c.get("A3_minus_A2_delta_median_db", np.nan),
                    "ratio_to_floor": c.get("ratio_to_floor", np.nan),
                    "C1_status": c.get("C1_status", ""),
                    "C2M_primary_span_db": c.get("C2M_primary_span_db", np.nan),
                    "C2M_primary_status": c.get("C2M_primary_status", ""),
                    "C2M_secondary_late_span_db": c.get("C2M_secondary_late_span_db", np.nan),
                    "C2M_secondary_late_status": c.get("C2M_secondary_late_status", ""),
                    "C2M_status": c.get("C2M_status", ""),
                    "C2S_delta_lpol_db": c.get("C2S_delta_lpol_db", np.nan),
                    "C2S_primary_status": c.get("C2S_primary_status", ""),
                    "C2S_gate_delta_p_target_db": c.get("C2S_gate_delta_p_target_db", np.nan),
                    "C2S_gate_status": c.get("C2S_gate_status", ""),
                    "C2S_semantics_gate_status": c.get("C2S_semantics_gate_status", ""),
                    "C2S_semantics": c.get("C2S_semantics", ""),
                    "C2S_status": c.get("C2S_status", ""),
                    "C2_status": c.get("C2_status", ""),
                }
            ],
            [
                "floor_delta_db",
                "repeat_delta_db",
                "delta_ref_db",
                "A3_minus_A2_delta_median_db",
                "ratio_to_floor",
                "C1_status",
                "C2M_primary_span_db",
                "C2M_primary_status",
                "C2M_secondary_late_span_db",
                "C2M_secondary_late_status",
                "C2M_status",
                "C2S_delta_lpol_db",
                "C2S_primary_status",
                "C2S_gate_delta_p_target_db",
                "C2S_gate_status",
                "C2S_semantics_gate_status",
                "C2S_semantics",
                "C2S_status",
                "C2_status",
            ],
        )
    )
    lines.append("")
    if str(c.get("C2S_semantics_note", "")).strip():
        lines.append(f"- C2-S semantics note: {c.get('C2S_semantics_note', '')}")
        lines.append("")

    # D
    d = checks.get("D_identifiability", {})
    lines.append("### D) Identifiability")
    lines.append("")
    lines.append(
        report_md.md_table(
            [
                {
                    "status": d.get("status", ""),
                    "EL_iqr_db": d.get("EL_iqr_db", np.nan),
                    "corr_d_vs_LOS": d.get("corr_d_vs_LOS", np.nan),
                    "min_strata_n": d.get("min_strata_n", 0),
                }
            ],
            ["status", "EL_iqr_db", "corr_d_vs_LOS", "min_strata_n"],
        )
    )
    lines.append("")

    d1 = d.get("D1", {})
    if isinstance(d1, dict):
        g = d1.get("global", {})
        a2i = d1.get("A2_isolation", {})
        a5i = d1.get("A5_isolation", {})
        lines.append("- D1 split: EL-identifying coverage(global) + parity/stress isolation(local)")
        lines.append("")
        lines.append(
            report_md.md_table(
                [
                    {
                        "component": "D1_global",
                        "status": g.get("status", ""),
                        "EL_iqr_db": g.get("EL_iqr_db", np.nan),
                        "min_bin_n": g.get("min_bin_n", 0),
                        "n_rows": g.get("n_rows", 0),
                    },
                    {
                        "component": "A2_isolation",
                        "status": a2i.get("status", ""),
                        "EL_std_db": a2i.get("EL_std_db", np.nan),
                        "n_rows": a2i.get("n_rows", 0),
                        "target": a2i.get("target", ""),
                    },
                    {
                        "component": "A5_isolation",
                        "status": a5i.get("status", ""),
                        "role": a5i.get("role", ""),
                        "delta_median_EL_stress_minus_base_db": a5i.get("delta_median_EL_stress_minus_base_db", np.nan),
                        "n_base": a5i.get("n_base", 0),
                        "n_stress": a5i.get("n_stress", 0),
                        "target": a5i.get("target", ""),
                    },
                ],
                ["component", "status", "role", "EL_iqr_db", "min_bin_n", "n_rows", "EL_std_db", "delta_median_EL_stress_minus_base_db", "n_base", "n_stress", "target"],
            )
        )
        lines.append("")

    d2 = d.get("D2", {})
    if isinstance(d2, dict):
        lines.append("- D2 split: material×angle coverage + stress×angle coverage + collinearity diagnostics")
        lines.append("")
        lines.append(
            report_md.md_table(
                [
                    {
                        "status": d2.get("status", ""),
                        "material_x_angle_status": d2.get("material_x_angle_status", ""),
                        "stress_x_angle_status": d2.get("stress_x_angle_status", ""),
                        "design_status": d2.get("design_status", ""),
                        "design_rank": d2.get("design_rank", 0),
                        "design_cols": d2.get("design_cols", 0),
                        "condition_number": d2.get("condition_number", np.nan),
                    }
                ],
                ["status", "material_x_angle_status", "stress_x_angle_status", "design_status", "design_rank", "design_cols", "condition_number"],
            )
        )
        lines.append("")
        st1 = d2.get("stage1", {})
        st2 = d2.get("stage2", {})
        if isinstance(st1, dict) or isinstance(st2, dict):
            lines.append(
                report_md.md_table(
                    [
                        {
                            "stage": "stage1_EL_identifying",
                            "status": st1.get("status", ""),
                            "n_rows": st1.get("n_rows", 0),
                            "design_rank": st1.get("design_rank", 0),
                            "design_cols": st1.get("design_cols", 0),
                            "condition_number": st1.get("condition_number", np.nan),
                        },
                        {
                            "stage": "stage2_effects_after_EL",
                            "status": st2.get("status", ""),
                            "n_rows": st2.get("n_rows", 0),
                            "design_rank": st2.get("design_rank", 0),
                            "design_cols": st2.get("design_cols", 0),
                            "condition_number": st2.get("condition_number", np.nan),
                        },
                    ],
                    ["stage", "status", "n_rows", "design_rank", "design_cols", "condition_number"],
                )
            )
            lines.append("")
        mat_cov = d2.get("material_x_angle_coverage", {})
        if isinstance(mat_cov, dict) and mat_cov:
            mrows = []
            for mat, cc in sorted(mat_cov.items()):
                if not isinstance(cc, dict):
                    continue
                mrows.append({"group": str(mat), "low": cc.get("low", 0), "mid": cc.get("mid", 0), "high": cc.get("high", 0), "NA": cc.get("NA", 0)})
            if mrows:
                lines.append(report_md.md_table(mrows, ["group", "low", "mid", "high", "NA"]))
                lines.append("")
        stress_cov = d2.get("stress_x_angle_coverage", {})
        if isinstance(stress_cov, dict) and stress_cov:
            srows2 = []
            for grp in ["base", "stress"]:
                cc = stress_cov.get(grp, {})
                if not isinstance(cc, dict):
                    continue
                srows2.append({"group": grp, "low": cc.get("low", 0), "mid": cc.get("mid", 0), "high": cc.get("high", 0), "NA": cc.get("NA", 0)})
            if srows2:
                lines.append(report_md.md_table(srows2, ["group", "low", "mid", "high", "NA"]))
                lines.append("")
        vif_warn = d2.get("vif_warnings", {})
        if isinstance(vif_warn, dict) and vif_warn:
            if "stage1" in vif_warn or "stage2" in vif_warn:
                vrows = []
                s1 = vif_warn.get("stage1", {})
                s2 = vif_warn.get("stage2", {})
                if isinstance(s1, dict):
                    for k, v in sorted(s1.items()):
                        vrows.append({"stage": "stage1", "feature": k, "vif": v})
                if isinstance(s2, dict):
                    for k, v in sorted(s2.items()):
                        vrows.append({"stage": "stage2", "feature": k, "vif": v})
                if vrows:
                    lines.append(report_md.md_table(vrows, ["stage", "feature", "vif"]))
                    lines.append("")
            else:
                vrows = [{"feature": k, "vif": v} for k, v in sorted(vif_warn.items())]
                lines.append(report_md.md_table(vrows, ["feature", "vif"]))
                lines.append("")

    d3 = d.get("D3", {})
    if isinstance(d3, dict):
        lines.append("- D3 split: LOS/NLOS×EL-bin strata coverage (viable-subset aware)")
        lines.append("")
        lines.append(
            report_md.md_table(
                [
                    {
                        "status": d3.get("status", ""),
                        "n_rows": d3.get("n_rows", 0),
                        "min_strata_all_n": d3.get("min_strata_all_n", 0),
                        "min_strata_viable_n": d3.get("min_strata_viable_n", 0),
                        "qna_total": d3.get("qna_total", 0),
                        "selected_rows_n": d3.get("selected_rows_n", 0),
                    }
                ],
                ["status", "n_rows", "min_strata_all_n", "min_strata_viable_n", "qna_total", "selected_rows_n"],
            )
        )
        lines.append("")
        strata3 = d3.get("strata_counts", {})
        if isinstance(strata3, dict):
            srows3 = [{"strata": k, "n": v} for k, v in sorted(strata3.items())]
            lines.append(report_md.md_table(srows3, ["strata", "n"]))
            lines.append("")
        hole_rows = d3.get("hole_analysis", [])
        if isinstance(hole_rows, list) and hole_rows:
            lines.append("- D3 hole diagnosis (structural vs sampling)")
            lines.append("")
            lines.append(report_md.md_table(hole_rows, ["strata", "pool_n", "selected_n", "hole_type", "status"]))
            lines.append("")
        per_sid_rows = d3.get("per_scenario_summary", [])
        if isinstance(per_sid_rows, list) and per_sid_rows:
            lines.append("- D3 per-scenario summary (B1/B2/B3)")
            lines.append("")
            lines.append(
                report_md.md_table(
                    per_sid_rows,
                    ["scenario_id", "status", "n_rows", "q1_db", "q2_db", "min_strata_all_n", "min_strata_viable_n", "qna_total"],
                )
            )
            lines.append("")

    strata = d.get("strata_counts", {})
    if isinstance(strata, dict):
        srows = [{"strata": k, "n": v} for k, v in sorted(strata.items())]
        lines.append("- Legacy D strata view")
        lines.append("")
        lines.append(report_md.md_table(srows, ["strata", "n"]))
        lines.append("")

    # E
    e = checks.get("E_power_based", {})
    lines.append("### E) Power-based Pipeline")
    lines.append("")
    lines.append(f"- Status: **{e.get('status', 'PASS')}**")
    lines.append(f"- Note: {e.get('note', '')}")
    lines.append(f"- basis values: `{e.get('basis_values', [])}`")
    lines.append(f"- convention values: `{e.get('convention_values', [])}`")
    lines.append(f"- matrix_source values: `{e.get('matrix_source_values', [])}`")
    lines.append(f"- force_cp_swap_on_odd_reflection(any): `{e.get('force_cp_swap_any_true', False)}`")
    if str(e.get("matrix_source_rule", "")).strip():
        lines.append(f"- matrix_source rule: {e.get('matrix_source_rule', '')}")
    if str(e.get("main_result_rule", "")).strip():
        lines.append(f"- main-result rule: {e.get('main_result_rule', '')}")
    used = e.get("used_metrics", [])
    if isinstance(used, list):
        lines.append("- Used metrics: " + ", ".join(str(x) for x in used))
    cfc = e.get("coupling_floor_check", {})
    if isinstance(cfc, dict):
        lines.append(
            f"- C0 coupling-floor sanity: status={cfc.get('status', 'INCONCLUSIVE')}, "
            f"C0 floor={_num(cfc.get('c0_xpd_floor_median_db', np.nan)):.3f} dB, "
            f"nominal={_num(cfc.get('coupling_floor_nominal_db', np.nan)):.3f} dB, "
            f"delta={_num(cfc.get('delta_c0_minus_nominal_db', np.nan)):.3f} dB"
        )
        if str(cfc.get("note", "")).strip():
            lines.append(f"- coupling-floor note: {cfc.get('note', '')}")
    lines.append(f"- A5 stress semantics: `{e.get('a5_stress_semantics', 'none')}`")
    lines.append(f"- A5 contamination-response ready: `{e.get('a5_contamination_response_ready', False)}`")
    if str(e.get("a5_semantics_note", "")).strip():
        lines.append(f"- A5 semantics note: {e.get('a5_semantics_note', '')}")
    lines.append("- Figure metadata table: [figure_metadata.csv](tables/figure_metadata.csv)")
    lines.append("")

    lines.append("## Scenario Sections")
    lines.append("")
    for s in sorted(scenario_scene.keys()):
        lines.append(f"### {s}")
        lines.append("")
        lines.append(f"- 의미: {report_md.scenario_meaning(s)}")
        if s == "A5":
            a5s2 = checks.get("A5_stress_semantics", {})
            if isinstance(a5s2, dict):
                lines.append(
                    f"- stress_semantics: `{a5s2.get('dominant_semantics', 'none')}` "
                    f"(response={a5s2.get('n_response', 0)}, polarization_only={a5s2.get('n_polarization_only', 0)})"
                )
                lines.append(f"- contamination-response ready: `{a5s2.get('contamination_response_ready', False)}`")
                if str(a5s2.get("note", "")).strip():
                    lines.append(f"- note: {a5s2.get('note', '')}")
        lines.append("")
        scene_png = scenario_scene[s]
        scene_rel = report_md.relpath(scene_png, out_root)
        lines.append(f"![{s} scene]({scene_rel})")
        lines.append("")
        for k, v in sorted(global_plots.items()):
            if s in k or k.startswith("ALL") or k.startswith("A2_A3") or k.startswith("C0"):
                rel = report_md.relpath(v, out_root)
                lines.append(f"- [{Path(v).name}]({rel})")
        lines.append("")

    if warns:
        lines.append("## WARN")
        lines.append("")
        for w in warns:
            lines.append(f"- {w}")
        lines.append("")

    if warn_cases:
        lines.append("## Warning Case Drilldown")
        lines.append("")
        lines.append(
            report_md.md_table(
                warn_cases,
                [
                    "scenario_id",
                    "case_id",
                    "case_label",
                    "warning",
                    "link_id",
                    "XPD_early_excess_db",
                    "XPD_late_excess_db",
                    "L_pol_db",
                    "EL_proxy_db",
                    "LOSflag",
                ],
            )
        )
        lines.append("")
        for wc in warn_cases:
            sid = str(wc.get("scenario_id", "NA"))
            cid = str(wc.get("case_id", ""))
            lines.append(f"### WARN Case {sid}/{cid}")
            lines.append("")
            lines.append(f"- reason: {wc.get('warning', '')}")
            sp = str(wc.get("scene_png_path", ""))
            if sp:
                lines.append(f"![warn-{sid}-{cid}]({report_md.relpath(sp, out_root)})")
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = io_lib.load_config(args.config)
    run_group = str(cfg.get("run_group", "run_group"))
    out_root = Path("analysis_report") / "out" / run_group
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    payload = io_lib.collect_all(cfg)
    runs = payload["runs"]
    link_rows = payload["link_rows"]
    ray_rows = payload["ray_rows"]
    scene_map = payload["scene_map"]

    if not link_rows:
        raise SystemExit("No link_metrics rows found from input_runs")

    floor_policy = dict(cfg.get("floor_policy", {}))
    mode = str(floor_policy.get("mode", "from_C0"))
    if mode == "from_calibration_json":
        calib_path = str(floor_policy.get("calibration_json", "")).strip()
        if not calib_path:
            raise SystemExit("floor_policy.mode=from_calibration_json but calibration_json is empty")
        calib = json.loads(Path(calib_path).read_text(encoding="utf-8"))
        floor_ref = metrics_lib.estimate_floor_from_calibration_json(calib, delta_method=str(floor_policy.get("delta_method", "p5_p95")))
    elif mode == "none":
        floor_ref = {"xpd_floor_db": 0.0, "delta_floor_db": 0.0, "method": "none", "count": 0, "p5_db": np.nan, "p95_db": np.nan}
    else:
        floor_ref = metrics_lib.estimate_floor_from_c0(link_rows, delta_method=str(floor_policy.get("delta_method", "p5_p95")))

    link_rows = metrics_lib.apply_floor_excess(
        link_rows,
        floor_db=float(floor_ref.get("xpd_floor_db", np.nan)),
        delta_db=float(floor_ref.get("delta_floor_db", np.nan)),
    )
    link_rows, el_impute_info = _impute_missing_el_proxy(link_rows, ray_rows)
    selected_rows, selected_meta = _load_selected_subset(cfg, link_rows)

    # plots
    global_plots = _make_diagnostic_plots(fig_dir, link_rows, ray_rows)

    # scene plots and index seed rows
    idx_rows, first_scene_by_scenario, scene_warns, warn_cases = _make_scene_plots(
        cfg,
        out_fig_dir=fig_dir,
        link_rows=link_rows,
        scene_map=scene_map,
    )
    a3_review_rows = _build_a3_geometry_review_rows(idx_rows, scene_map=scene_map, ray_rows=ray_rows)
    figure_meta_rows = _build_figure_metadata_rows(link_rows, idx_rows, runs=runs)

    # checks
    checks = _diagnostic_checks(
        link_rows,
        ray_rows,
        floor_ref=floor_ref,
        cfg=cfg,
        runs=runs,
        selected_rows=selected_rows,
    )

    # requested analysis tables (target/case/sensitivity levels)
    bt = dict(checks.get("B_time_resolution", {}))
    ws_cfg = dict(cfg.get("windows", {})) if isinstance(cfg.get("windows", {}), dict) else {}
    te_default_s = _num(bt.get("Te_s", np.nan))
    if not np.isfinite(te_default_s):
        te_default_s = float(_num(ws_cfg.get("Te_ns", 10.0))) * 1e-9
    dt_res_s = _num(bt.get("dt_res_s", np.nan))
    tmax_s = _num(bt.get("Tmax_s", np.nan))
    if not np.isfinite(tmax_s):
        tmax_s = float(_num(ws_cfg.get("Tmax_ns", 200.0))) * 1e-9
    floor_db = _num(floor_ref.get("xpd_floor_db", np.nan))
    w_target_s_by_sid_raw = bt.get("W_target_s_by_scenario", {})
    w_target_s_by_sid: dict[str, float] = {}
    if isinstance(w_target_s_by_sid_raw, dict):
        for k, v in w_target_s_by_sid_raw.items():
            vv = _num(v)
            if np.isfinite(vv) and vv > 0:
                w_target_s_by_sid[str(k)] = float(vv)
    target_map = {"A2": 1, "A3": 2, "A4": 1, "A5": 2}
    target_level_rows = _build_target_level_rows(
        link_rows=link_rows,
        ray_rows=ray_rows,
        runs=runs,
        floor_db=float(floor_db),
        te_default_s=float(te_default_s),
        dt_res_s=float(dt_res_s),
        w_target_s_by_sid=w_target_s_by_sid,
        target_map=target_map,
    )
    case_level_rows = _build_case_level_rows(link_rows)
    te_sweep_ns = _as_float_list(ws_cfg.get("te_sweep_ns", None), [2.0, 3.0, 5.0])
    noise_tail_sweep_ns = _as_float_list(ws_cfg.get("noise_tail_sweep_ns", None), [30.0, 50.0, 80.0])
    threshold_sweep_db = _as_float_list(ws_cfg.get("threshold_sweep_db", None), [4.0, 6.0, 8.0])
    sensitivity_level_rows = _build_sensitivity_level_rows(
        link_rows=link_rows,
        runs=runs,
        floor_db=float(floor_db),
        te_ns_list=te_sweep_ns,
        noise_tail_ns_list=noise_tail_sweep_ns,
        threshold_db_list=threshold_sweep_db,
        tmax_s=float(tmax_s),
    )

    # save tables/json
    _write_rows_csv(tab_dir / "diagnostic_link_rows.csv", link_rows)
    _write_rows_csv(tab_dir / "diagnostic_ray_rows.csv", ray_rows)
    _write_rows_csv(tab_dir / "figure_metadata.csv", figure_meta_rows)
    _write_rows_csv_with_columns(
        tab_dir / "target_level.csv",
        target_level_rows,
        [
            "scenario_id",
            "case_id",
            "link_id",
            "target_tau_ns",
            "target_rank",
            "target_in_Wearly",
            "target_is_first",
            "bounce_count",
            "parity",
            "incidence_angle_deg",
            "Pco_target",
            "Pcross_target",
            "xpd_target_raw_db",
            "xpd_target_ex_db",
            "C_target_db",
        ],
    )
    _write_rows_csv_with_columns(
        tab_dir / "case_level.csv",
        case_level_rows,
        [
            "scenario_id",
            "case_id",
            "link_id",
            "xpd_early_ex_db",
            "xpd_late_ex_db",
            "l_pol_db",
            "rho_early_linear",
            "rho_early_db",
            "ds_ns",
            "early_energy_fraction",
            "EL_proxy_db",
            "LOSflag",
            "material",
            "stress_flag",
            "claim_caution_early",
            "claim_caution_late",
        ],
    )
    _write_rows_csv_with_columns(
        tab_dir / "sensitivity_level.csv",
        sensitivity_level_rows,
        [
            "scenario_id",
            "case_id",
            "link_id",
            "Te_ns",
            "noise_tail_ns",
            "threshold_db",
            "xpd_early_ex_db",
            "xpd_late_ex_db",
            "l_pol_db",
            "rho_early_linear",
            "rho_early_db",
            "ds_ns",
            "early_energy_fraction",
            "delta_xpd_early_ex_db",
            "delta_xpd_late_ex_db",
            "delta_l_pol_db",
            "delta_rho_early_db",
            "delta_ds_ns",
            "delta_early_energy_fraction",
        ],
    )
    _write_rows_csv(tab_dir / "el_proxy_imputation_rows.csv", list(el_impute_info.get("rows", [])))
    _write_rows_csv(tab_dir / "A3_geometry_manual_review.csv", a3_review_rows)
    report_md.write_json(tab_dir / "diagnostic_checks.json", checks)
    report_md.write_json(tab_dir / "floor_reference_used.json", floor_ref)
    report_md.write_json(tab_dir / "selected_subset_info.json", selected_meta)
    report_md.write_json(tab_dir / "el_proxy_imputation_info.json", el_impute_info)
    d3 = dict(dict(checks.get("D_identifiability", {})).get("D3", {}))
    _write_rows_csv(tab_dir / "D3_hole_analysis.csv", list(d3.get("hole_analysis", [])))
    _write_rows_csv(tab_dir / "B_per_scenario_summary.csv", list(d3.get("per_scenario_summary", [])))
    _write_rows_csv(
        tab_dir / "A3_target_window_sign.csv",
        [
            row for row in [
                dict(dict(checks.get("B_time_resolution", {})).get("A2_target_window_sign", {})),
                dict(dict(checks.get("B_time_resolution", {})).get("A3_target_window_sign", {})),
                dict(dict(dict(checks.get("B_time_resolution", {})).get("A6_parity_benchmark", {})).get("odd", {})),
                dict(dict(dict(checks.get("B_time_resolution", {})).get("A6_parity_benchmark", {})).get("even", {})),
            ]
            if (np.isfinite(_num(row.get("n_eval", np.nan))) and _num(row.get("n_eval", np.nan)) > 0)
            or np.isfinite(_num(row.get("expected_sign_hit_rate", np.nan)))
        ],
    )
    a6_cmp = dict(dict(dict(checks.get("B_time_resolution", {})).get("A6_parity_benchmark", {})).get("case_set_compare", {}))
    a6_rows: list[dict[str, Any]] = []
    per_set = dict(a6_cmp.get("per_set", {}))
    for ss in sorted(per_set.keys()):
        ss_d = dict(per_set.get(ss, {}))
        for mode in ["odd", "even"]:
            mm = dict(ss_d.get(mode, {}))
            if not mm:
                continue
            a6_rows.append(
                {
                    "case_set": str(ss),
                    "mode": str(mode),
                    "status": str(ss_d.get("status", "")),
                    "hit_rate_min_set": _num(ss_d.get("hit_rate_min", np.nan)),
                    "n_eval": int(_num(mm.get("n_eval", np.nan))) if np.isfinite(_num(mm.get("n_eval", np.nan))) else 0,
                    "median_xpd_target_raw_db": _num(mm.get("median_xpd_target_raw_db", np.nan)),
                    "median_xpd_target_ex_db": _num(mm.get("median_xpd_target_ex_db", np.nan)),
                    "expected_sign_hit_rate": _num(mm.get("expected_sign_hit_rate", np.nan)),
                    "sign_metric_for_status": str(mm.get("sign_metric_for_status", "")),
                }
            )
    fvm = dict(a6_cmp.get("full_vs_minimal", {}))
    for mode in ["odd", "even"]:
        mm = dict(fvm.get(mode, {}))
        if not mm:
            continue
        a6_rows.append(
            {
                "case_set": "full_vs_minimal",
                "mode": str(mode),
                "status": "",
                "hit_rate_min_set": np.nan,
                "n_eval": np.nan,
                "median_xpd_target_raw_db": _num(mm.get("delta_raw_minimal_minus_full_db", np.nan)),
                "median_xpd_target_ex_db": _num(mm.get("delta_ex_minimal_minus_full_db", np.nan)),
                "expected_sign_hit_rate": _num(mm.get("delta_hit_rate_minimal_minus_full", np.nan)),
                "sign_metric_for_status": "delta(minimal-full)",
            }
        )
    _write_rows_csv(tab_dir / "A6_case_set_sign_compare.csv", a6_rows)

    a3_lines = ["# A3 Geometry Manual Review", ""]
    if a3_review_rows:
        a3_lines.append(
            report_md.md_table(
                a3_review_rows,
                [
                    "scenario_id",
                    "case_id",
                    "review_status",
                    "scene_debug_valid",
                    "los_rays",
                    "has_target_bounce_n2",
                    "rays_topk_n",
                    "scene_debug_issues",
                ],
            )
        )
        a3_lines.append("")
        for rr in a3_review_rows:
            sp = str(rr.get("scene_png_path", ""))
            if not sp:
                continue
            a3_lines.append(f"## Case {rr.get('case_id', '')}")
            a3_lines.append("")
            a3_lines.append(f"- review_status: **{rr.get('review_status', '')}**")
            a3_lines.append(f"- scene_debug_json: `{rr.get('scene_debug_json', '')}`")
            a3_lines.append(f"![A3-case-{rr.get('case_id','')}]({report_md.relpath(sp, out_root)})")
            a3_lines.append("")
    report_md.write_text(out_root / "A3_geometry_manual_review.md", "\n".join(a3_lines) + "\n")

    # complete index rows with run file refs
    run_by_scenario = {str(r.get("scenario_id", "")): r for r in runs}
    for r in idx_rows:
        sc = str(r.get("scenario_id", ""))
        rr = run_by_scenario.get(sc)
        if rr is not None:
            r["input_run_dir"] = str(rr.get("run_dir", ""))
            r["link_metrics_csv"] = str(rr.get("link_metrics_csv", ""))
            r["rays_csv"] = str(rr.get("rays_csv", ""))
            r["report_refs"] = {"diagnostic": f"scenario-{sc.lower()}"}

    index_path = out_root / "index.csv"
    indexer.update_index(index_path, idx_rows)
    idx_loaded = indexer.load_index(index_path)
    indexer.write_index_md(out_root / "index.md", idx_loaded)

    # markdown
    md = _build_markdown(
        out_root=out_root,
        run_group=run_group,
        link_rows=link_rows,
        checks=checks,
        floor_ref=floor_ref,
        scenario_scene=first_scene_by_scenario,
        global_plots=global_plots,
        warns=scene_warns,
        warn_cases=warn_cases,
        a3_review_rows=a3_review_rows,
    )
    out_md = out_root / "diagnostic_report.md"
    report_md.write_text(out_md, md)
    print(str(out_md))


if __name__ == "__main__":
    main()

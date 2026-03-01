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


def _diagnostic_checks(
    link_rows: list[dict[str, Any]],
    ray_rows: list[dict[str, Any]],
    floor_ref: dict[str, Any],
    cfg: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    scenario_rows = metrics_lib.split_by_scenario(link_rows)

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

    # B: time-resolution checks with purpose-specific windows
    ws = dict(cfg.get("windows", {}))
    te_s = float(ws.get("Te_ns", 10.0)) * 1e-9
    tmax_s = float(ws.get("Tmax_ns", 200.0)) * 1e-9
    w_floor_s = float(ws.get("floor_window_ns", max(1.0, 0.5 * float(ws.get("Te_ns", 10.0))))) * 1e-9
    w_target_s = float(ws.get("target_window_ns", max(2.0, 0.8 * float(ws.get("Te_ns", 10.0))))) * 1e-9
    bw, df, freq_src = _estimate_freq_grid(link_rows, cfg)
    dt_res = 1.0 / bw if bw > 0 else float("nan")
    tau_max = 1.0 / df if df > 0 else float("nan")

    floor_stats = _floor_window_contamination(ray_rows, w_floor_s=w_floor_s)
    target_map = {"A2": 1, "A3": 2, "A4": 1, "A5": 2}
    target_stats = {
        sid: _target_window_stats(
            ray_rows,
            scenario_id=sid,
            target_n=tn,
            w_target_s=w_target_s,
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

    checks["B_time_resolution"] = {
        "freq_source": str(freq_src),
        "BW_Hz": float(bw),
        "df_Hz": float(df),
        "dt_res_s": float(dt_res),
        "tau_max_s": float(tau_max),
        "Te_s": float(te_s),
        "Tmax_s": float(tmax_s),
        "W_floor_s": float(w_floor_s),
        "W_target_s": float(w_target_s),
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
    }

    # C effect vs floor
    floor_delta = float(floor_ref.get("delta_floor_db", np.nan))
    a2 = scenario_rows.get("A2", [])
    a3 = scenario_rows.get("A3", [])
    dmed = metrics_lib.delta_median(a3, a2, "XPD_early_excess_db")
    ratio = abs(float(dmed)) / abs(float(floor_delta)) if np.isfinite(dmed) and np.isfinite(floor_delta) and abs(floor_delta) > 0 else float("nan")
    c1_status = "PASS" if np.isfinite(ratio) and ratio >= 2.0 else ("WARN" if np.isfinite(ratio) and ratio >= 1.0 else "FAIL")

    a4 = scenario_rows.get("A4", [])
    by_mat: dict[str, list[dict[str, Any]]] = {}
    for r in a4:
        by_mat.setdefault(str(r.get("material_class", "NA")), []).append(r)
    mat_meds = [
        _median(v, "XPD_late_excess_db")
        for _, v in sorted(by_mat.items())
        if np.isfinite(_median(v, "XPD_late_excess_db"))
    ]
    mat_shift = float(np.max(mat_meds) - np.min(mat_meds)) if mat_meds else float("nan")

    a5 = scenario_rows.get("A5", [])
    base = [r for r in a5 if int(_num(r.get("roughness_flag", 0))) == 0 and int(_num(r.get("human_flag", 0))) == 0]
    stress = [r for r in a5 if int(_num(r.get("roughness_flag", 0))) == 1 or int(_num(r.get("human_flag", 0))) == 1]
    d_stress = metrics_lib.delta_median(stress, base, "XPD_late_excess_db")
    v_base = metrics_lib.tail_stats(base, "XPD_late_excess_db").get("std", np.nan)
    v_stress = metrics_lib.tail_stats(stress, "XPD_late_excess_db").get("std", np.nan)
    var_ratio = float(v_stress / v_base) if np.isfinite(v_stress) and np.isfinite(v_base) and v_base > 0 else float("nan")
    c2_status = "PASS" if (
        (np.isfinite(mat_shift) and np.isfinite(floor_delta) and abs(mat_shift) > abs(floor_delta))
        or (np.isfinite(d_stress) and np.isfinite(floor_delta) and abs(d_stress) > abs(floor_delta))
        or (np.isfinite(var_ratio) and var_ratio > 1.25)
    ) else "WARN"
    checks["C_effect_vs_floor"] = {
        "floor_delta_db": floor_delta,
        "A3_minus_A2_delta_median_db": dmed,
        "ratio_to_floor": ratio,
        "C1_status": c1_status,
        "A4_material_shift_late_excess_db": mat_shift,
        "A5_stress_delta_late_excess_db": d_stress,
        "A5_stress_var_ratio": var_ratio,
        "C2_status": c2_status,
    }

    # D identifiability
    x_el = np.asarray([_num(r.get("EL_proxy_db", np.nan)) for r in link_rows], dtype=float)
    x_el = x_el[np.isfinite(x_el)]
    el_iqr = float(np.percentile(x_el, 75.0) - np.percentile(x_el, 25.0)) if len(x_el) else float("nan")

    los = np.asarray([_num(r.get("LOSflag", np.nan)) for r in link_rows], dtype=float)
    d_m = np.asarray([_num(r.get("d_m", np.nan)) for r in link_rows], dtype=float)
    m = np.isfinite(los) & np.isfinite(d_m)
    corr_d_los = float(np.corrcoef(los[m], d_m[m])[0, 1]) if int(np.sum(m)) > 3 else float("nan")

    # strata LOS/NLOS x EL tertile
    strata_counts: dict[str, int] = {}
    if len(x_el) >= 3:
        q1, q2 = np.percentile(x_el, [33.3, 66.7])
    else:
        q1, q2 = (float("nan"), float("nan"))
    for r in link_rows:
        el = _num(r.get("EL_proxy_db", np.nan))
        lf = int(round(_num(r.get("LOSflag", np.nan)))) if np.isfinite(_num(r.get("LOSflag", np.nan))) else -1
        if not np.isfinite(el) or lf not in {0, 1} or not np.isfinite(q1) or not np.isfinite(q2):
            continue
        if el <= q1:
            eb = "q1"
        elif el <= q2:
            eb = "q2"
        else:
            eb = "q3"
        key = f"LOS{lf}_{eb}"
        strata_counts[key] = int(strata_counts.get(key, 0) + 1)
    min_strata = min(strata_counts.values()) if strata_counts else 0

    d_status = "PASS" if (np.isfinite(el_iqr) and el_iqr >= 3.0 and min_strata >= 2) else "WARN"
    checks["D_identifiability"] = {
        "EL_iqr_db": float(el_iqr),
        "corr_d_vs_LOS": float(corr_d_los),
        "strata_counts": strata_counts,
        "min_strata_n": int(min_strata),
        "status": d_status,
    }

    # E power-based only
    checks["E_power_based"] = {
        "status": "PASS",
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
        "note": "No complex-phase fields are used by this report pipeline.",
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

    # C
    c = checks.get("C_effect_vs_floor", {})
    lines.append("### C) Effect Size vs Floor Uncertainty")
    lines.append("")
    lines.append(
        report_md.md_table(
            [
                {
                    "floor_delta_db": c.get("floor_delta_db", np.nan),
                    "A3_minus_A2_delta_median_db": c.get("A3_minus_A2_delta_median_db", np.nan),
                    "ratio_to_floor": c.get("ratio_to_floor", np.nan),
                    "C1_status": c.get("C1_status", ""),
                    "A4_material_shift_late_excess_db": c.get("A4_material_shift_late_excess_db", np.nan),
                    "A5_stress_delta_late_excess_db": c.get("A5_stress_delta_late_excess_db", np.nan),
                    "A5_stress_var_ratio": c.get("A5_stress_var_ratio", np.nan),
                    "C2_status": c.get("C2_status", ""),
                }
            ],
            [
                "floor_delta_db",
                "A3_minus_A2_delta_median_db",
                "ratio_to_floor",
                "C1_status",
                "A4_material_shift_late_excess_db",
                "A5_stress_delta_late_excess_db",
                "A5_stress_var_ratio",
                "C2_status",
            ],
        )
    )
    lines.append("")

    # D
    d = checks.get("D_identifiability", {})
    lines.append("### D) Identifiability")
    lines.append("")
    lines.append(
        report_md.md_table(
            [
                {
                    "EL_iqr_db": d.get("EL_iqr_db", np.nan),
                    "corr_d_vs_LOS": d.get("corr_d_vs_LOS", np.nan),
                    "min_strata_n": d.get("min_strata_n", 0),
                    "status": d.get("status", ""),
                }
            ],
            ["EL_iqr_db", "corr_d_vs_LOS", "min_strata_n", "status"],
        )
    )
    lines.append("")
    strata = d.get("strata_counts", {})
    if isinstance(strata, dict):
        srows = [{"strata": k, "n": v} for k, v in sorted(strata.items())]
        lines.append(report_md.md_table(srows, ["strata", "n"]))
        lines.append("")

    # E
    e = checks.get("E_power_based", {})
    lines.append("### E) Power-based Pipeline")
    lines.append("")
    lines.append(f"- Status: **{e.get('status', 'PASS')}**")
    lines.append(f"- Note: {e.get('note', '')}")
    used = e.get("used_metrics", [])
    if isinstance(used, list):
        lines.append("- Used metrics: " + ", ".join(str(x) for x in used))
    lines.append("")

    lines.append("## Scenario Sections")
    lines.append("")
    for s in sorted(scenario_scene.keys()):
        lines.append(f"### {s}")
        lines.append("")
        lines.append(f"- 의미: {report_md.scenario_meaning(s)}")
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

    # plots
    global_plots = _make_diagnostic_plots(fig_dir, link_rows, ray_rows)

    # scene plots and index seed rows
    idx_rows, first_scene_by_scenario, scene_warns, warn_cases = _make_scene_plots(
        cfg,
        out_fig_dir=fig_dir,
        link_rows=link_rows,
        scene_map=scene_map,
    )

    # checks
    checks = _diagnostic_checks(link_rows, ray_rows, floor_ref=floor_ref, cfg=cfg, runs=runs)

    # save tables/json
    _write_rows_csv(tab_dir / "diagnostic_link_rows.csv", link_rows)
    _write_rows_csv(tab_dir / "diagnostic_ray_rows.csv", ray_rows)
    report_md.write_json(tab_dir / "diagnostic_checks.json", checks)
    report_md.write_json(tab_dir / "floor_reference_used.json", floor_ref)

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
    )
    out_md = out_root / "diagnostic_report.md"
    report_md.write_text(out_md, md)
    print(str(out_md))


if __name__ == "__main__":
    main()

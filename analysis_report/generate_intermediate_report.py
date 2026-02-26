"""Generate intermediate proposition report (M/G/L/R/P) from standard outputs."""

from __future__ import annotations

import argparse
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


def _num(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


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


def _status_support(cond: bool, cond_partial: bool = False) -> str:
    if cond:
        return "SUPPORTED"
    if cond_partial:
        return "PARTIAL"
    return "UNSUPPORTED"


def _status_data(n: int, min_n: int = 8) -> str:
    return "INCONCLUSIVE" if int(n) < int(min_n) else "OK"


def _ensure_scene_plots(
    out_fig_dir: Path,
    scene_map: dict[tuple[str, str], dict[str, Any]],
    link_rows: list[dict[str, Any]],
    figure_size: tuple[float, float],
) -> tuple[dict[tuple[str, str], str], dict[str, str], list[str]]:
    out_case: dict[tuple[str, str], str] = {}
    out_global: dict[str, str] = {}
    warns: list[str] = []

    for k, scene_obj in sorted(scene_map.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        sid, cid = k
        ok, probs = scene_lib.validate_scene_debug(scene_obj)
        if not ok:
            warns.append(f"scene_debug invalid: {sid}/{cid}: {';'.join(probs)}")
        out_png = out_fig_dir / f"{sid}__{cid}__scene.png"
        out_case[k] = scene_lib.plot_scene(scene_obj, out_png=out_png, figure_size=figure_size)

    # Global layout for B scenarios
    by_s: dict[str, list[dict[str, Any]]] = {}
    for r in link_rows:
        by_s.setdefault(str(r.get("scenario_id", "NA")), []).append(r)
    for sid in ["B1", "B2", "B3"]:
        rows = by_s.get(sid, [])
        if not rows:
            continue
        cand = [k for k in scene_map.keys() if k[0] == sid]
        if not cand:
            continue
        base = scene_map[cand[0]]
        rx_points = [(_num(r.get("rx_x", np.nan)), _num(r.get("rx_y", np.nan))) for r in rows]
        out_png = out_fig_dir / f"{sid}__GLOBAL__scene.png"
        out_global[sid] = scene_lib.plot_scene_global(base, rx_points=rx_points, out_png=out_png, figure_size=figure_size)
    return out_case, out_global, warns


def _median(rows: list[dict[str, Any]], key: str) -> float:
    x = np.asarray([_num(r.get(key, np.nan)) for r in rows], dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def _arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    x = np.asarray([_num(r.get(key, np.nan)) for r in rows], dtype=float)
    return x[np.isfinite(x)]


def _fit_linear(rows: list[dict[str, Any]], y_key: str, x_keys: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    y = np.asarray([_num(r.get(y_key, np.nan)) for r in rows], dtype=float)
    Xcols = [np.asarray([_num(r.get(k, np.nan)) for r in rows], dtype=float) for k in x_keys]
    if not Xcols:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), []
    M = np.vstack(Xcols).T
    mask = np.isfinite(y)
    for i in range(M.shape[1]):
        mask &= np.isfinite(M[:, i])
    y = y[mask]
    X = M[mask]
    if len(y) < max(4, len(x_keys) + 1):
        return np.asarray([], dtype=float), np.asarray([], dtype=float), []
    Xa = np.column_stack([np.ones(len(X), dtype=float), X])
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    return beta, Xa, [r for i, r in enumerate(rows) if mask[i]]


def _predict(beta: np.ndarray, Xa: np.ndarray) -> np.ndarray:
    if len(beta) == 0 or len(Xa) == 0:
        return np.asarray([], dtype=float)
    return Xa @ beta


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    if len(y) == 0 or len(yhat) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _gauss_nll(y: np.ndarray, yhat: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    e = y - yhat
    sig = float(np.std(e, ddof=1)) if len(e) > 1 else 1.0
    sig = max(sig, 1e-6)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * sig * sig) + (e * e) / (sig * sig)))


def _kfold_indices(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n, dtype=int)
    rng.shuffle(idx)
    k = max(2, min(int(k), int(n)))
    return [idx[i::k] for i in range(k)]


def _cv_compare(rows: list[dict[str, Any]], y_key: str, x_keys: list[str], k: int = 5, seed: int = 0) -> dict[str, float]:
    # dataset extraction
    y = np.asarray([_num(r.get(y_key, np.nan)) for r in rows], dtype=float)
    Xcols = [np.asarray([_num(r.get(kx, np.nan)) for r in rows], dtype=float) for kx in x_keys]
    if not Xcols:
        return {"n": 0, "rmse_const": np.nan, "rmse_lin": np.nan, "nll_const": np.nan, "nll_lin": np.nan}
    X = np.vstack(Xcols).T
    mask = np.isfinite(y)
    for i in range(X.shape[1]):
        mask &= np.isfinite(X[:, i])
    y = y[mask]
    X = X[mask]
    n = len(y)
    if n < max(8, len(x_keys) + 2):
        return {"n": int(n), "rmse_const": np.nan, "rmse_lin": np.nan, "nll_const": np.nan, "nll_lin": np.nan}

    folds = _kfold_indices(n, k=k, seed=seed)
    errs_const = []
    errs_lin = []
    nll_const = []
    nll_lin = []
    for test_idx in folds:
        tr_mask = np.ones(n, dtype=bool)
        tr_mask[test_idx] = False
        y_tr = y[tr_mask]
        y_te = y[test_idx]
        X_tr = X[tr_mask]
        X_te = X[test_idx]

        # constant
        mu0 = float(np.mean(y_tr))
        yhat0 = np.full_like(y_te, mu0)
        errs_const.append(_rmse(y_te, yhat0))
        nll_const.append(_gauss_nll(y_te, yhat0))

        # linear
        Xa_tr = np.column_stack([np.ones(len(X_tr), dtype=float), X_tr])
        Xa_te = np.column_stack([np.ones(len(X_te), dtype=float), X_te])
        beta, *_ = np.linalg.lstsq(Xa_tr, y_tr, rcond=None)
        yhat1 = Xa_te @ beta
        errs_lin.append(_rmse(y_te, yhat1))
        nll_lin.append(_gauss_nll(y_te, yhat1))

    return {
        "n": int(n),
        "rmse_const": float(np.nanmean(np.asarray(errs_const, dtype=float))),
        "rmse_lin": float(np.nanmean(np.asarray(errs_lin, dtype=float))),
        "nll_const": float(np.nanmean(np.asarray(nll_const, dtype=float))),
        "nll_lin": float(np.nanmean(np.asarray(nll_lin, dtype=float))),
    }


def _subsample_sign_stability(rows: list[dict[str, Any]], y_key: str, x_key: str, n_rep: int = 200, seed: int = 0) -> dict[str, float]:
    y = np.asarray([_num(r.get(y_key, np.nan)) for r in rows], dtype=float)
    x = np.asarray([_num(r.get(x_key, np.nan)) for r in rows], dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    n = len(y)
    if n < 10:
        return {"n": int(n), "base_sign": np.nan, "sign_keep_rate": np.nan}

    Xa = np.column_stack([np.ones(n, dtype=float), x])
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    base_sign = float(np.sign(beta[1]))
    rng = np.random.default_rng(int(seed))
    keep = 0
    valid = 0
    idx = np.arange(n, dtype=int)
    size = max(6, n // 2)
    for _ in range(max(20, int(n_rep))):
        ii = rng.choice(idx, size=size, replace=False)
        Xa_i = np.column_stack([np.ones(len(ii), dtype=float), x[ii]])
        beta_i, *_ = np.linalg.lstsq(Xa_i, y[ii], rcond=None)
        s = float(np.sign(beta_i[1]))
        if not np.isfinite(s) or s == 0:
            continue
        valid += 1
        if s == base_sign:
            keep += 1
    return {
        "n": int(n),
        "base_sign": float(base_sign),
        "sign_keep_rate": float(keep / valid) if valid else np.nan,
    }


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
        floor_ref = {"xpd_floor_db": 0.0, "delta_floor_db": 0.0, "method": "none", "count": 0}
    else:
        floor_ref = metrics_lib.estimate_floor_from_c0(link_rows, delta_method=str(floor_policy.get("delta_method", "p5_p95")))

    link_rows = metrics_lib.apply_floor_excess(
        link_rows,
        floor_db=float(floor_ref.get("xpd_floor_db", np.nan)),
        delta_db=float(floor_ref.get("delta_floor_db", np.nan)),
    )

    by_s = metrics_lib.split_by_scenario(link_rows)
    alpha = float(dict(cfg.get("stats", {})).get("alpha", 0.05))
    bootstrap_n = int(dict(cfg.get("stats", {})).get("bootstrap_n", 1000))

    # Ensure scene plots exist for index/report sections
    scene_cfg = dict(cfg.get("scene_plots", {}))
    fig_size = tuple(float(x) for x in scene_cfg.get("figure_size", [10, 6]))
    scene_case, scene_global, scene_warns = _ensure_scene_plots(fig_dir, scene_map, link_rows, fig_size)

    # Key plots for propositions
    fig_paths: dict[str, str] = {}

    c0 = by_s.get("C0", [])
    if c0:
        fig_paths["M1_c0_floor_vs_yaw"] = plot_lib.plot_scatter(
            x=[_num(r.get("yaw_deg", np.nan)) for r in c0],
            y=[_num(r.get("XPD_early_db", np.nan)) for r in c0],
            c=[_num(r.get("d_m", np.nan)) for r in c0],
            out_png=fig_dir / "C0__ALL__xpd_floor_vs_yaw.png",
            title="C0 floor vs yaw",
            xlabel="yaw (deg)",
            ylabel="XPD_early (dB)",
            add_fit=False,
        )
        fig_paths["M1_c0_floor_vs_logd"] = plot_lib.plot_scatter(
            x=np.log10(np.maximum(np.asarray([_num(r.get("d_m", np.nan)) for r in c0], dtype=float), 1e-6)),
            y=[_num(r.get("XPD_early_db", np.nan)) for r in c0],
            out_png=fig_dir / "C0__ALL__xpd_floor_vs_logd.png",
            title="C0 floor vs log10(d)",
            xlabel="log10(distance)",
            ylabel="XPD_early (dB)",
            add_fit=True,
        )

    a2 = by_s.get("A2", [])
    a3 = by_s.get("A3", [])
    fig_paths["G1G2_xpd_early_ex_cdf"] = plot_lib.plot_multi_cdf(
        {
            "A2": [_num(r.get("XPD_early_excess_db", np.nan)) for r in a2],
            "A3": [_num(r.get("XPD_early_excess_db", np.nan)) for r in a3],
            "C0": [_num(r.get("XPD_early_excess_db", np.nan)) for r in c0],
        },
        out_png=fig_dir / "A2A3C0__ALL__xpd_early_ex_cdf.png",
        title="A2/A3/C0 XPD_early_excess CDF",
        xlabel="XPD_early_excess (dB)",
    )
    fig_paths["G1G2_lpol_box"] = plot_lib.plot_box_by_group(
        rows=[*a2, *a3],
        group_key="scenario_id",
        value_key="L_pol_db",
        out_png=fig_dir / "A2A3__ALL__lpol_box.png",
        title="A2/A3 L_pol",
        ylabel="L_pol (dB)",
    )

    # L and R related
    all_non_c0 = [r for s in ["A2", "A3", "A4", "A5", "B1", "B2", "B3"] for r in by_s.get(s, [])]
    fig_paths["L1_early_late_box"] = plot_lib.plot_box_by_group(
        rows=[{**r, "metric": "early", "val": _num(r.get("XPD_early_excess_db", np.nan))} for r in all_non_c0]
        + [{**r, "metric": "late", "val": _num(r.get("XPD_late_excess_db", np.nan))} for r in all_non_c0],
        group_key="metric",
        value_key="val",
        out_png=fig_dir / "ALL__early_late_ex_box.png",
        title="XPD early/late excess",
        ylabel="dB",
    )
    fig_paths["L3_el_vs_xpd"] = plot_lib.plot_scatter(
        x=[_num(r.get("EL_proxy_db", np.nan)) for r in all_non_c0],
        y=[_num(r.get("XPD_early_excess_db", np.nan)) for r in all_non_c0],
        out_png=fig_dir / "ALL__xpd_early_ex_vs_el_proxy.png",
        title="XPD_early_excess vs EL_proxy",
        xlabel="EL_proxy (dB)",
        ylabel="XPD_early_excess (dB)",
        add_fit=True,
    )
    fig_paths["R2_ds_vs_xpd"] = plot_lib.plot_scatter(
        x=[_num(r.get("delay_spread_rms_s", np.nan)) for r in all_non_c0],
        y=[_num(r.get("XPD_early_excess_db", np.nan)) for r in all_non_c0],
        out_png=fig_dir / "ALL__ds_vs_xpd_early_ex.png",
        title="Delay spread vs XPD_early_excess",
        xlabel="delay_spread_rms_s",
        ylabel="XPD_early_excess (dB)",
        add_fit=True,
    )
    fig_paths["R2_earlyfrac_vs_rho"] = plot_lib.plot_scatter(
        x=[_num(r.get("rho_early_lin", np.nan)) for r in all_non_c0],
        y=[_num(r.get("early_energy_fraction", np.nan)) for r in all_non_c0],
        out_png=fig_dir / "ALL__early_fraction_vs_rho.png",
        title="Early fraction vs rho_early",
        xlabel="rho_early_lin",
        ylabel="early_energy_fraction",
        add_fit=True,
    )

    # B heatmaps and LOS/NLOS CDF
    b_rows = [r for s in ["B1", "B2", "B3"] for r in by_s.get(s, [])]
    if b_rows:
        fig_paths["R1_heatmap_xpd"] = plot_lib.plot_heatmap_xy(
            b_rows,
            value_key="XPD_early_excess_db",
            out_png=fig_dir / "B__ALL__heatmap_xpd_early_ex.png",
            title="B scenarios heatmap: XPD_early_excess",
        )
        fig_paths["R1_heatmap_lpol"] = plot_lib.plot_heatmap_xy(
            b_rows,
            value_key="L_pol_db",
            out_png=fig_dir / "B__ALL__heatmap_lpol.png",
            title="B scenarios heatmap: L_pol",
        )
        los = [r for r in b_rows if int(_num(r.get("LOSflag", np.nan))) == 1]
        nlos = [r for r in b_rows if int(_num(r.get("LOSflag", np.nan))) == 0]
        fig_paths["R1_los_nlos_cdf"] = plot_lib.plot_multi_cdf(
            {
                "LOS": [_num(r.get("XPD_early_excess_db", np.nan)) for r in los],
                "NLOS": [_num(r.get("XPD_early_excess_db", np.nan)) for r in nlos],
            },
            out_png=fig_dir / "B__ALL__los_nlos_xpd_ex_cdf.png",
            title="B LOS/NLOS XPD_early_excess CDF",
            xlabel="XPD_early_excess (dB)",
        )

    # A5 baseline vs stress cdf
    a5 = by_s.get("A5", [])
    base = [r for r in a5 if int(_num(r.get("roughness_flag", 0))) == 0 and int(_num(r.get("human_flag", 0))) == 0]
    stress = [r for r in a5 if int(_num(r.get("roughness_flag", 0))) == 1 or int(_num(r.get("human_flag", 0))) == 1]
    if a5:
        fig_paths["L2_a5_stress_cdf"] = plot_lib.plot_multi_cdf(
            {
                "A5_base": [_num(r.get("XPD_early_excess_db", np.nan)) for r in base],
                "A5_stress": [_num(r.get("XPD_early_excess_db", np.nan)) for r in stress],
            },
            out_png=fig_dir / "A5__ALL__base_vs_stress_xpd_early_ex_cdf.png",
            title="A5 base vs stress XPD_early_excess",
            xlabel="XPD_early_excess (dB)",
        )

    # Proposition stats
    props: dict[str, dict[str, Any]] = {}

    # M1
    trend = stats_lib.linear_trend_test(
        x=np.log10(np.maximum(_arr(c0, "d_m"), 1e-6)),
        y=_arr(c0, "XPD_early_db"),
    ) if c0 else {"slope": np.nan, "p": np.nan, "n": 0}
    vard = stats_lib.simple_anova_or_variance_decomp(c0, value_key="XPD_early_db") if c0 else {}
    m1_support = bool(np.isfinite(trend.get("slope", np.nan)))
    props["M1"] = {
        "definition": "C0 floor distance/yaw sensitivity",
        "n": int(trend.get("n", 0) if isinstance(trend, dict) else 0),
        "trend": trend,
        "variance_decomp": vard,
        "status": _status_support(m1_support, cond_partial=bool(c0)),
    }

    # M2
    delta_floor = float(floor_ref.get("delta_floor_db", np.nan))
    xex = _arr(link_rows, "XPD_early_excess_db")
    inside = float(np.mean(np.abs(xex) <= abs(delta_floor))) if len(xex) and np.isfinite(delta_floor) else np.nan
    outside = float(np.mean(np.abs(xex) > abs(delta_floor))) if len(xex) and np.isfinite(delta_floor) else np.nan
    props["M2"] = {
        "definition": "Fraction inside/outside floor uncertainty band",
        "inside_ratio": inside,
        "outside_ratio": outside,
        "status": _status_support(bool(np.isfinite(inside)), cond_partial=False),
    }

    # G1/G2
    dmed_a2 = _median(a2, "XPD_early_excess_db")
    dmed_a3 = _median(a3, "XPD_early_excess_db")
    dmed_c0 = _median(c0, "XPD_early_excess_db")
    g1_shift = float(dmed_a2 - dmed_c0) if np.isfinite(dmed_a2) and np.isfinite(dmed_c0) else np.nan
    ks_a2_c0 = stats_lib.ks_wasserstein(_arr(a2, "XPD_early_excess_db"), _arr(c0, "XPD_early_excess_db"))
    g1_ok = bool(np.isfinite(g1_shift) and g1_shift < 0)
    props["G1"] = {
        "definition": "A2 odd-bounce increases cross dominance vs C0",
        "delta_median_db": g1_shift,
        "ks_wasserstein": ks_a2_c0,
        "status": _status_support(g1_ok, cond_partial=bool(np.isfinite(g1_shift))),
    }

    g2_shift = float(dmed_a3 - dmed_a2) if np.isfinite(dmed_a3) and np.isfinite(dmed_a2) else np.nan
    ks_a3_a2 = stats_lib.ks_wasserstein(_arr(a3, "XPD_early_excess_db"), _arr(a2, "XPD_early_excess_db"))
    g2_ok = bool(np.isfinite(g2_shift) and g2_shift > 0)
    props["G2"] = {
        "definition": "A3 even-bounce recovery vs A2",
        "delta_median_db": g2_shift,
        "ks_wasserstein": ks_a3_a2,
        "status": _status_support(g2_ok, cond_partial=bool(np.isfinite(g2_shift))),
    }

    # L1/L2/L3
    early_med = _median(all_non_c0, "XPD_early_excess_db")
    late_med = _median(all_non_c0, "XPD_late_excess_db")
    lpol_med = _median(all_non_c0, "L_pol_db")
    props["L1"] = {
        "definition": "Early/Late leakage separation via XPD_early_ex/XPD_late_ex/L_pol",
        "median_early_ex_db": early_med,
        "median_late_ex_db": late_med,
        "median_lpol_db": lpol_med,
        "status": _status_support(bool(np.isfinite(lpol_med)), cond_partial=bool(all_non_c0)),
    }

    base_stats = metrics_lib.tail_stats(base, "XPD_early_excess_db")
    stress_stats = metrics_lib.tail_stats(stress, "XPD_early_excess_db")
    l2_delta = float(stress_stats.get("mean", np.nan) - base_stats.get("mean", np.nan))
    l2_var_ratio = float(stress_stats.get("std", np.nan) / base_stats.get("std", np.nan)) if np.isfinite(base_stats.get("std", np.nan)) and base_stats.get("std", np.nan) > 0 else np.nan
    l2_ok = bool(np.isfinite(l2_delta) and np.isfinite(l2_var_ratio) and (abs(l2_delta) > 0.5 or l2_var_ratio > 1.2))
    props["L2"] = {
        "definition": "A5 stress impact on center/tails",
        "base": base_stats,
        "stress": stress_stats,
        "delta_mean_db": l2_delta,
        "var_ratio": l2_var_ratio,
        "status": _status_support(l2_ok, cond_partial=bool(a5)),
    }

    sp = stats_lib.spearman_with_bootstrap(
        x=[_num(r.get("XPD_early_excess_db", np.nan)) for r in all_non_c0],
        y=[-_num(r.get("EL_proxy_db", np.nan)) for r in all_non_c0],
        n=bootstrap_n,
        alpha=alpha,
        seed=0,
    )
    l3_ok = bool(np.isfinite(sp.get("rho", np.nan)) and abs(float(sp.get("rho", 0.0))) >= 0.3)
    props["L3"] = {
        "definition": "Correlation between leakage excess and -EL_proxy",
        "spearman": sp,
        "status": _status_support(l3_ok, cond_partial=bool(np.isfinite(sp.get("rho", np.nan)))),
    }

    # R1/R2
    los = [r for r in b_rows if int(_num(r.get("LOSflag", np.nan))) == 1]
    nlos = [r for r in b_rows if int(_num(r.get("LOSflag", np.nan))) == 0]
    ks_los = stats_lib.ks_wasserstein(_arr(los, "XPD_early_excess_db"), _arr(nlos, "XPD_early_excess_db"))
    r1_ok = bool(len(b_rows) > 0 and len(los) > 0 and len(nlos) > 0)
    props["R1"] = {
        "definition": "Room-space LOS/NLOS and heatmap consistency",
        "n_b_rows": int(len(b_rows)),
        "n_los": int(len(los)),
        "n_nlos": int(len(nlos)),
        "ks_wasserstein": ks_los,
        "status": _status_support(r1_ok, cond_partial=bool(len(b_rows) > 0)),
    }

    sp_ds = stats_lib.spearman_with_bootstrap(
        x=[_num(r.get("delay_spread_rms_s", np.nan)) for r in all_non_c0],
        y=[_num(r.get("XPD_early_excess_db", np.nan)) for r in all_non_c0],
        n=bootstrap_n,
        alpha=alpha,
        seed=1,
    )
    r2_ok = bool(np.isfinite(sp_ds.get("rho", np.nan)))
    props["R2"] = {
        "definition": "DS/early-fraction relation with leakage",
        "spearman_ds_vs_xpd": sp_ds,
        "status": _status_support(r2_ok, cond_partial=bool(all_non_c0)),
    }

    # P1/P2
    p1_cmp = _cv_compare(
        all_non_c0,
        y_key="XPD_early_excess_db",
        x_keys=["EL_proxy_db", "LOSflag", "roughness_flag", "human_flag", "obstacle_flag"],
        k=5,
        seed=42,
    )
    p1_ok = bool(
        np.isfinite(p1_cmp.get("rmse_const", np.nan))
        and np.isfinite(p1_cmp.get("rmse_lin", np.nan))
        and float(p1_cmp.get("rmse_lin", np.nan)) < float(p1_cmp.get("rmse_const", np.nan))
    )
    props["P1"] = {
        "definition": "Conditional model improvement vs constant baseline",
        "cv": p1_cmp,
        "status": _status_support(p1_ok, cond_partial=bool(np.isfinite(p1_cmp.get("rmse_lin", np.nan)))),
    }

    p2_sign = _subsample_sign_stability(all_non_c0, y_key="XPD_early_excess_db", x_key="EL_proxy_db", n_rep=200, seed=123)
    p2_ok = bool(np.isfinite(p2_sign.get("sign_keep_rate", np.nan)) and float(p2_sign.get("sign_keep_rate", 0.0)) >= 0.8)
    props["P2"] = {
        "definition": "Sign stability of EL coefficient under subsampling",
        "stability": p2_sign,
        "status": _status_support(p2_ok, cond_partial=bool(np.isfinite(p2_sign.get("sign_keep_rate", np.nan)))),
    }

    # Save proposition table/json
    prop_rows = []
    for k in ["M1", "M2", "G1", "G2", "L1", "L2", "L3", "R1", "R2", "P1", "P2"]:
        pp = props.get(k, {})
        prop_rows.append(
            {
                "proposition": k,
                "status": pp.get("status", "INCONCLUSIVE"),
                "data_status": _status_data(int(pp.get("n", len(all_non_c0)) if isinstance(pp.get("n", None), (int, float)) else len(all_non_c0))),
                "definition": pp.get("definition", ""),
            }
        )
    _write_rows_csv(tab_dir / "intermediate_proposition_status.csv", prop_rows)
    report_md.write_json(tab_dir / "intermediate_proposition_details.json", props)

    # Update index with report refs + key plots
    run_by_s = {str(r.get("scenario_id", "")): r for r in runs}
    idx_rows = []
    for r in link_rows:
        sid = str(r.get("scenario_id", "NA"))
        cid = str(r.get("case_id", ""))
        key = (sid, cid)
        if any((x.get("scenario_id"), x.get("case_id")) == key for x in idx_rows):
            continue
        run = run_by_s.get(sid)
        scene_png = scene_case.get(key, "")
        if sid in {"B1", "B2", "B3"} and sid in scene_global:
            scene_png = scene_global[sid]
        idx_rows.append(
            {
                "scenario_id": sid,
                "case_id": cid,
                "case_label": str(r.get("case_label", r.get("link_id", cid))),
                "input_run_dir": str(run.get("run_dir", "")) if run else "",
                "link_metrics_csv": str(run.get("link_metrics_csv", "")) if run else "",
                "rays_csv": str(run.get("rays_csv", "")) if run else "",
                "scene_debug_json": str(scene_map.get(key, {}).get("_path", "")) if key in scene_map else "",
                "scene_png_path": scene_png,
                "key_plots": list(fig_paths.values()),
                "report_refs": {"intermediate": f"scenario-{sid.lower()}"},
            }
        )

    index_path = out_root / "index.csv"
    indexer.update_index(index_path, idx_rows)
    indexer.write_index_md(out_root / "index.md", indexer.load_index(index_path))

    # Build markdown
    lines: list[str] = []
    lines.append(f"# Intermediate Report ({run_group})")
    lines.append("")
    lines.append("## Proposition Status")
    lines.append("")
    lines.append(report_md.md_table(prop_rows, ["proposition", "status", "data_status", "definition"]))
    lines.append("")

    for k in ["M1", "M2", "G1", "G2", "L1", "L2", "L3", "R1", "R2", "P1", "P2"]:
        pp = props.get(k, {})
        lines.append(f"## {k}")
        lines.append("")
        lines.append(f"- Definition: {pp.get('definition', '')}")
        lines.append(f"- Status: **{pp.get('status', 'INCONCLUSIVE')}**")
        lines.append(f"- Details: `{json.dumps(pp, ensure_ascii=False)}`")
        lines.append("")

    lines.append("## Scenario Sections")
    lines.append("")
    scenarios = sorted({str(r.get("scenario_id", "NA")) for r in link_rows})
    for sid in scenarios:
        lines.append(f"### {sid}")
        lines.append("")
        img = scene_global.get(sid, "")
        if not img:
            cand = sorted([k for k in scene_case.keys() if k[0] == sid], key=lambda x: x[1])
            if cand:
                img = scene_case[cand[0]]
        if img:
            lines.append(f"![{sid} scene]({report_md.relpath(img, out_root)})")
        else:
            lines.append("- WARN: scene plot missing")
        lines.append("")
        for name, p in sorted(fig_paths.items()):
            if sid in name or name.startswith("ALL") or sid.startswith("B"):
                lines.append(f"- [{Path(p).name}]({report_md.relpath(p, out_root)})")
        lines.append("")

    if scene_warns:
        lines.append("## WARN")
        lines.append("")
        for w in scene_warns:
            lines.append(f"- {w}")
        lines.append("")

    out_md = out_root / "intermediate_report.md"
    report_md.write_text(out_md, "\n".join(lines) + "\n")

    # Save working tables
    _write_rows_csv(tab_dir / "intermediate_link_rows.csv", link_rows)
    _write_rows_csv(tab_dir / "intermediate_ray_rows.csv", ray_rows)
    print(str(out_md))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate detailed proposition-experiment-data-plot mapping report.

This script follows the user's M/G/L/R/P detailed plot checklist and writes:
1) plot files (when data is available),
2) per-plot mapping CSV,
3) markdown report with proposition PASS/FAIL + plot readiness.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.ticker import MultipleLocator

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {}
            for k in fieldnames:
                v = r.get(k, "")
                if isinstance(v, (dict, list, tuple)):
                    out[k] = json.dumps(v, ensure_ascii=False)
                else:
                    out[k] = v
            w.writerow(out)


def _to_float(v: Any, default: float = float("nan")) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def _to_int(v: Any, default: int = 0) -> int:
    x = _to_float(v, float("nan"))
    if not np.isfinite(x):
        return default
    return int(round(x))


def _finite(a: list[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    return x[np.isfinite(x)]


def _parse_jsonish(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s:
        return s
    if s[0] in "[{":
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return s
    return s


def _safe_log10(x: np.ndarray, floor: float = 1e-20) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), floor))


def _style_db_axis(ax: Any, *, ymin: float | None = None, ymax: float | None = None) -> None:
    if ymin is not None and ymax is not None and np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        ax.set_ylim(float(ymin), float(ymax))
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.yaxis.set_minor_locator(MultipleLocator(1.0))
    ax.grid(True, which="major", axis="y", alpha=0.35)
    ax.grid(True, which="minor", axis="y", alpha=0.12)


def _pdp_db_window(co_db: np.ndarray, cr_db: np.ndarray) -> tuple[float, float]:
    vals = np.concatenate([_finite(co_db), _finite(cr_db)])
    if vals.size == 0:
        return (-100.0, -60.0)
    pmax = float(np.max(vals))
    # Common PDP visualization rule: [peak-40, peak-5] dB
    y_top = pmax - 5.0
    y_bot = pmax - 40.0
    # Clamp to a practical RT/PDP display range
    y_top = min(y_top, pmax + 3.0)
    y_bot = max(y_bot, -140.0)
    if y_top <= y_bot:
        y_top = y_bot + 10.0
    return (y_bot, y_top)


def _status_to_pass_fail(s: str) -> str:
    up = str(s).upper()
    if up in {"PASS", "FAIL", "PARTIAL"}:
        return up
    if up == "SUPPORTED":
        return "PASS"
    if up in {"WARN", "WARNING"}:
        return "PARTIAL"
    return "FAIL"


def _prep_fig(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _by_scenario(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        s = str(r.get("scenario_id", "NA"))
        out.setdefault(s, []).append(r)
    return out


def _first_index_run(index_rows: list[dict[str, str]], scenario_id: str) -> Path | None:
    for r in index_rows:
        if str(r.get("scenario_id", "")) == scenario_id:
            p = Path(str(r.get("input_run_dir", "")))
            if p.exists():
                return p
    return None


def _dominant_incidence_by_link(ray_rows: list[dict[str, str]], scenario_id: str) -> dict[str, float]:
    # dominant ray = max P_lin per link
    best: dict[str, tuple[float, float]] = {}
    for r in ray_rows:
        if str(r.get("scenario_id", "")) != scenario_id:
            continue
        link = str(r.get("link_id", ""))
        p = _to_float(r.get("P_lin"))
        ang = _to_float(r.get("incidence_deg"))
        if not np.isfinite(p):
            continue
        if link not in best or p > best[link][0]:
            best[link] = (p, ang)
    return {k: v[1] for k, v in best.items()}


def _sample_case_for_scenario(link_rows: list[dict[str, str]], scenario_id: str, case_id: str | None = None) -> dict[str, str] | None:
    cand = [r for r in link_rows if str(r.get("scenario_id", "")) == scenario_id]
    if case_id is not None:
        for r in cand:
            if str(r.get("case_id", "")) == str(case_id):
                return r
    return cand[0] if cand else None


def _load_pdp(run_dir: Path, scenario_id: str, case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
    npz = run_dir / f"pdp_{scenario_id}_{case_id}.npz"
    if not npz.exists():
        cand = sorted(run_dir.glob(f"pdp_{scenario_id}_*.npz"))
        if not cand:
            return None
        npz = cand[0]
    z = np.load(npz)
    d = np.asarray(z["delay_tau_s"], dtype=float)
    pco = np.asarray(z["P_co"], dtype=float)
    pcr = np.asarray(z["P_cross"], dtype=float)
    xt = np.asarray(z["XPD_tau_db"], dtype=float) if "XPD_tau_db" in z.files else None
    return d, pco, pcr, xt


def _plot_pdp_overlay_with_windows(
    out: Path,
    d_s: np.ndarray,
    pco: np.ndarray,
    pcr: np.ndarray,
    title: str,
    w_early: tuple[float, float] | None = None,
    w_target: tuple[float, float] | None = None,
) -> None:
    out = _prep_fig(out)
    d_ns = d_s * 1e9
    co_db = _safe_log10(pco)
    cr_db = _safe_log10(pcr)
    nz_co = pco > 0
    nz_cr = pcr > 0
    sparse = np.sum(nz_co) <= 16 and np.sum(nz_cr) <= 16
    xpd_tap = co_db - cr_db
    yb, yt = _pdp_db_window(co_db, cr_db)
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.4]})
    ax = axs[0]
    if sparse:
        floor_db = yb - 20.0
        if np.any(nz_co):
            ax.vlines(d_ns[nz_co], floor_db, co_db[nz_co], colors="tab:red", linewidth=2.0, alpha=0.9)
            ax.scatter(d_ns[nz_co], co_db[nz_co], color="tab:red", s=24, label="P_co")
        if np.any(nz_cr):
            ax.vlines(d_ns[nz_cr], floor_db, cr_db[nz_cr], colors="tab:blue", linewidth=2.0, linestyles="--", alpha=0.9)
            ax.scatter(d_ns[nz_cr], cr_db[nz_cr], color="tab:blue", s=24, label="P_cross")
    else:
        ax.plot(d_ns, co_db, color="tab:red", lw=1.8, label="P_co")
        ax.plot(d_ns, cr_db, color="tab:blue", lw=1.6, ls="--", label="P_cross")
    if w_early is not None:
        ax.axvspan(w_early[0] * 1e9, w_early[1] * 1e9, color="#98df8a", alpha=0.18, label="W_early")
    if w_target is not None:
        ax.axvspan(w_target[0] * 1e9, w_target[1] * 1e9, color="#ffbb78", alpha=0.20, label="W_target")
    ax.set_title(title)
    ax.set_ylabel("PDP power (dB)")
    _style_db_axis(ax, ymin=yb, ymax=yt)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax2 = axs[1]
    ax2.plot(d_ns, xpd_tap, color="#d95319", lw=1.5, label="XPD_tap = P_co - P_cross")
    if w_early is not None:
        ax2.axvspan(w_early[0] * 1e9, w_early[1] * 1e9, color="#98df8a", alpha=0.15)
    if w_target is not None:
        ax2.axvspan(w_target[0] * 1e9, w_target[1] * 1e9, color="#ffbb78", alpha=0.18)
    ax2.axhline(0.0, color="#666", lw=0.9, ls="--")
    ax2.set_xlabel("Delay (ns)")
    ax2.set_ylabel("XPD_tap (dB)")
    _style_db_axis(ax2)
    ax2.grid(True, axis="x", alpha=0.25)
    ax2.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_tap_xpd(out: Path, d_s: np.ndarray, xpd_tau: np.ndarray, title: str) -> None:
    out = _prep_fig(out)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(d_s * 1e9, xpd_tau, color="#d95319", lw=1.8)
    ax.set_title(title)
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("XPD_tau (dB)")
    _style_db_axis(ax)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _scenario_color(s: str) -> str:
    return {
        "C0": "#4c78a8",
        "A2": "#e45756",
        "A3": "#72b7b2",
        "A4": "#54a24b",
        "A5": "#f58518",
        "B1": "#b279a2",
        "B2": "#ff9da6",
        "B3": "#9d755d",
    }.get(s, "#7f7f7f")


def _make_m1_m2_plots(
    fig_dir: Path,
    link_rows: list[dict[str, str]],
    ray_rows: list[dict[str, str]],
    index_rows: list[dict[str, str]],
    diag: dict[str, Any],
    detail_rows: list[dict[str, Any]],
) -> None:
    by = _by_scenario(link_rows)
    c0 = by.get("C0", [])
    if not c0:
        return

    # M1-1: C0 PDP grid (distance x yaw, first available case for each pair)
    run_c0 = _first_index_run(index_rows, "C0")
    dvals = sorted({round(_to_float(r.get("d_m")), 3) for r in c0 if np.isfinite(_to_float(r.get("d_m")))})
    yvals = sorted({round(_to_float(r.get("yaw_deg")), 3) for r in c0 if np.isfinite(_to_float(r.get("yaw_deg")))})
    if run_c0 and dvals and yvals:
        # pick representative case for each (d,yaw): min rep_id
        pick: dict[tuple[float, float], dict[str, str]] = {}
        for r in c0:
            d = round(_to_float(r.get("d_m")), 3)
            y = round(_to_float(r.get("yaw_deg")), 3)
            if not np.isfinite(d) or not np.isfinite(y):
                continue
            rep = _to_int(r.get("rep_id"), 99999)
            k = (d, y)
            if k not in pick or rep < _to_int(pick[k].get("rep_id"), 99999):
                pick[k] = r
        out = _prep_fig(fig_dir / "M1-1__C0_raw_pdp_overlay_grid.png")
        nr = len(yvals)
        nc = len(dvals)
        y_bots: list[float] = []
        y_tops: list[float] = []
        for y in yvals:
            for d in dvals:
                r0 = pick.get((d, y))
                if r0 is None:
                    continue
                pdp0 = _load_pdp(run_c0, "C0", str(r0.get("case_id", "0")))
                if pdp0 is None:
                    continue
                tau0, pco0, pcr0, _ = pdp0
                _ = tau0
                yb0, yt0 = _pdp_db_window(_safe_log10(pco0), _safe_log10(pcr0))
                y_bots.append(float(yb0))
                y_tops.append(float(yt0))
        g_ymin = float(np.min(np.asarray(y_bots, dtype=float))) if y_bots else -95.0
        g_ymax = float(np.max(np.asarray(y_tops, dtype=float))) if y_tops else -55.0
        fig, axs = plt.subplots(nr, nc, figsize=(max(10, 2.8 * nc), max(5, 2.2 * nr)), sharex=True, sharey=True)
        if nr == 1 and nc == 1:
            axs = np.asarray([[axs]])
        elif nr == 1:
            axs = np.asarray([axs])
        elif nc == 1:
            axs = np.asarray([[a] for a in axs])
        w_floor_s = _to_float(diag.get("B_time_resolution", {}).get("W_floor_s"))
        for i, y in enumerate(yvals):
            for j, d in enumerate(dvals):
                ax = axs[i, j]
                r = pick.get((d, y))
                if r is None:
                    ax.set_axis_off()
                    continue
                pdp = _load_pdp(run_c0, "C0", str(r.get("case_id", "0")))
                if pdp is None:
                    ax.set_axis_off()
                    continue
                tau, pco, pcr, _ = pdp
                co_db = _safe_log10(pco)
                cr_db = _safe_log10(pcr)
                ax.plot(tau * 1e9, co_db, color="tab:red", lw=1.1)
                ax.plot(tau * 1e9, cr_db, color="tab:blue", lw=1.0, ls="--")
                kpeak = int(np.argmax(np.maximum(pco + pcr, 0.0)))
                tau0 = float(tau[kpeak])
                ax.axvline(tau0 * 1e9, color="k", lw=0.7, alpha=0.6)
                if np.isfinite(w_floor_s) and w_floor_s > 0:
                    ax.axvspan((tau0 - 0.5 * w_floor_s) * 1e9, (tau0 + 0.5 * w_floor_s) * 1e9, color="#98df8a", alpha=0.14)
                ax.set_title(f"d={d:g}m, yaw={y:g}")
                _style_db_axis(ax, ymin=g_ymin, ymax=g_ymax)
                ax.grid(True, axis="x", alpha=0.2)
        fig.supxlabel("Delay (ns)")
        fig.supylabel("PDP power (dB)")
        fig.suptitle("M1-1 C0 raw PDP overlay")
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "M1-1", "file": out.name, "note": "C0 facet raw PDP"})

    # M1-2: C_floor boxplot by yaw (computed from rays around LOS)
    w_floor_s = _to_float(diag.get("B_time_resolution", {}).get("W_floor_s"), 1.5e-9)
    ray_c0 = [r for r in ray_rows if str(r.get("scenario_id", "")) == "C0"]
    by_case: dict[str, list[dict[str, str]]] = {}
    for r in ray_c0:
        by_case.setdefault(str(r.get("case_id", "")), []).append(r)
    c_floor_rows: list[tuple[float, float, float]] = []
    c0_case_meta = {str(r.get("case_id", "")): r for r in c0}
    for cid, rr in by_case.items():
        los = [x for x in rr if _to_int(x.get("los_flag_ray")) == 1]
        if not los:
            continue
        los_sorted = sorted(los, key=lambda x: _to_float(x.get("P_lin")), reverse=True)
        p_los = max(_to_float(los_sorted[0].get("P_lin")), 1e-30)
        tau_los = _to_float(los_sorted[0].get("tau_s"))
        p_non = 0.0
        for x in rr:
            if _to_int(x.get("los_flag_ray")) == 1:
                continue
            tau = _to_float(x.get("tau_s"))
            if np.isfinite(tau) and abs(tau - tau_los) <= 0.5 * w_floor_s:
                p_non += max(_to_float(x.get("P_lin")), 0.0)
        cdb = 10.0 * np.log10(max(p_non, 1e-30) / p_los)
        meta = c0_case_meta.get(cid, {})
        yaw = _to_float(meta.get("yaw_deg"))
        d = _to_float(meta.get("d_m"))
        if np.isfinite(yaw) and np.isfinite(d):
            c_floor_rows.append((yaw, d, cdb))
    if c_floor_rows:
        out = _prep_fig(fig_dir / "M1-2__C_floor_box_by_yaw.png")
        yaws = sorted({x[0] for x in c_floor_rows})
        data = [_finite([v for yy, _d, v in c_floor_rows if yy == y]) for y in yaws]
        fig, ax = plt.subplots(figsize=(max(7.0, 1.2 + 0.8 * len(yaws)), 4.6))
        ax.boxplot(data, labels=[f"{y:g}" for y in yaws], showfliers=True)
        ax.axhline(-10.0, color="#ff7f0e", ls="--", lw=1.2, label="-10 dB")
        ax.axhline(-15.0, color="#2ca02c", ls="--", lw=1.2, label="-15 dB")
        ax.set_xlabel("Yaw (deg)")
        ax.set_ylabel("C_floor (dB)")
        ax.set_title("M1-2 C_floor contamination by yaw")
        _style_db_axis(ax)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "M1-2", "file": out.name, "note": "C_floor threshold lines"})

    # M1-5: XPD_floor vs frequency/subband
    curves = []
    freqs = None
    for r in c0:
        f = _parse_jsonish(r.get("xpd_floor_freq_hz"))
        x = _parse_jsonish(r.get("xpd_floor_curve_db"))
        if isinstance(f, list) and isinstance(x, list) and len(f) == len(x) and len(f) > 0:
            fa = np.asarray(f, dtype=float)
            xa = np.asarray(x, dtype=float)
            if np.all(np.isfinite(fa)) and np.all(np.isfinite(xa)):
                if freqs is None:
                    freqs = fa
                if len(fa) == len(freqs):
                    curves.append(xa)
    if curves and freqs is not None:
        arr = np.asarray(curves, dtype=float)
        med = np.nanmedian(arr, axis=0)
        p5 = np.nanpercentile(arr, 5, axis=0)
        p95 = np.nanpercentile(arr, 95, axis=0)
        out = _prep_fig(fig_dir / "M1-5__xpd_floor_vs_frequency.png")
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        xg = freqs * 1e-9
        ax.plot(xg, med, lw=1.8, color="#1f77b4", label="median")
        ax.fill_between(xg, p5, p95, color="#1f77b4", alpha=0.2, label="p5-p95")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("XPD_floor (dB)")
        ax.set_title("M1-5 XPD_floor vs frequency")
        _style_db_axis(ax)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "M1-5", "file": out.name, "note": "frequency-dependent floor"})

    # M1-6: repeatability strip + box
    conds: dict[str, list[float]] = {}
    for r in c0:
        d = _to_float(r.get("d_m"))
        y = _to_float(r.get("yaw_deg"))
        xpd = _to_float(r.get("XPD_early_db"))
        if np.isfinite(d) and np.isfinite(y) and np.isfinite(xpd):
            key = f"d={d:g},yaw={y:g}"
            conds.setdefault(key, []).append(xpd)
    keys = sorted(conds.keys())
    if keys:
        out = _prep_fig(fig_dir / "M1-6__repeatability_strip_box.png")
        fig, ax = plt.subplots(figsize=(max(8.0, 0.5 * len(keys) + 2.0), 4.8))
        data = [np.asarray(conds[k], dtype=float) for k in keys]
        ax.boxplot(data, labels=keys, showfliers=True)
        for i, k in enumerate(keys, start=1):
            vals = np.asarray(conds[k], dtype=float)
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=14, color="#555", alpha=0.8)
        ax.set_ylabel("XPD_floor (dB)")
        ax.set_title("M1-6 repeatability by condition")
        _style_db_axis(ax)
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "M1-6", "file": out.name, "note": "repeat strip+box"})

    # M1-7 uncertainty budget summary
    cfx = diag.get("C_effect_vs_floor", {})
    floor_delta = _to_float(cfx.get("floor_delta_db"))
    rep_delta = _to_float(cfx.get("repeat_delta_db"))
    ref_delta = _to_float(cfx.get("delta_ref_db"))
    floor_db = _to_float(c0[0].get("xpd_floor_db")) if c0 else float("nan")
    if np.isfinite(floor_delta) and np.isfinite(rep_delta) and np.isfinite(ref_delta):
        out = _prep_fig(fig_dir / "M1-7__uncertainty_budget_summary.png")
        fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.4), gridspec_kw={"width_ratios": [1.0, 1.5]})
        ax0 = axs[0]
        ax0.bar(["XPD_floor"], [floor_db], color=["#4c78a8"])
        ax0.set_ylabel("dB")
        ax0.set_title("XPD_floor (absolute)")
        _style_db_axis(ax0)
        ax0.grid(True, axis="x", alpha=0.2)

        ax1 = axs[1]
        labels = ["Delta_floor", "Delta_repeat", "Delta_ref"]
        vals = [floor_delta, rep_delta, ref_delta]
        colors = ["#f58518", "#e45756", "#72b7b2"]
        ax1.bar(labels, vals, color=colors)
        ax1.set_ylabel("dB")
        ax1.set_title("Uncertainty deltas")
        _style_db_axis(ax1)
        ax1.grid(True, axis="x", alpha=0.2)
        fig.suptitle("M1-7 uncertainty budget (absolute vs delta split)")
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "M1-7", "file": out.name, "note": "Delta_ref=max(Delta_floor,Delta_repeat)"})

    # M2-1 scenario-wise early/late box with threshold lines
    delta_ref = _to_float(cfx.get("delta_ref_db"), _to_float(c0[0].get("delta_floor_db")) if c0 else 0.0)
    scen_order = ["A2", "A3", "A4", "A5", "B1", "B2", "B3"]
    out = _prep_fig(fig_dir / "M2-1__scenario_early_late_ex_box.png")
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ai, key in enumerate(["XPD_early_excess_db", "XPD_late_excess_db"]):
        ax = axs[ai]
        data = []
        labels = []
        for s in scen_order:
            vals = _finite([_to_float(r.get(key)) for r in by.get(s, [])])
            if len(vals) == 0:
                continue
            data.append(vals)
            labels.append(s)
        if data:
            ax.boxplot(data, labels=labels, showfliers=True)
        if np.isfinite(delta_ref) and delta_ref > 0:
            ax.axhline(+delta_ref, color="#999", ls="--", lw=1.0)
            ax.axhline(-delta_ref, color="#999", ls="--", lw=1.0)
        ax.set_title("early excess" if "early" in key else "late excess")
        ax.set_xlabel("scenario")
        _style_db_axis(ax)
        ax.grid(True, axis="y", alpha=0.3)
    axs[0].set_ylabel("XPD_excess (dB)")
    fig.suptitle("M2-1 scenario-wise XPD excess vs threshold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "M2-1", "file": out.name, "note": "with ±Delta_ref lines"})

    # M2-2 exceedance rate
    out = _prep_fig(fig_dir / "M2-2__exceedance_rate_bar.png")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(scen_order), dtype=float)
    w = 0.36
    re = []
    rl = []
    for s in scen_order:
        rr = by.get(s, [])
        ve = _finite([_to_float(r.get("XPD_early_excess_db")) for r in rr])
        vl = _finite([_to_float(r.get("XPD_late_excess_db")) for r in rr])
        if len(ve):
            re.append(float(np.mean(np.abs(ve) > delta_ref)))
        else:
            re.append(np.nan)
        if len(vl):
            rl.append(float(np.mean(np.abs(vl) > delta_ref)))
        else:
            rl.append(np.nan)
    ax.bar(x - 0.5 * w, re, width=w, label="early")
    ax.bar(x + 0.5 * w, rl, width=w, label="late")
    ax.set_xticks(x)
    ax.set_xticklabels(scen_order)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Pr(|XPD_ex| > Delta_ref)")
    ax.set_title("M2-2 exceedance rate by scenario")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "M2-2", "file": out.name, "note": "early/late exceedance"})

    # M2-3 CDF overlay C0 vs indoor scenarios
    out = _prep_fig(fig_dir / "M2-3__cdf_floor_vs_indoor_overlay.png")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    series = {"C0": _finite([_to_float(r.get("XPD_early_db")) for r in c0])}
    for s in scen_order:
        vals = _finite([_to_float(r.get("XPD_early_excess_db")) for r in by.get(s, [])])
        if len(vals):
            series[s] = vals
    for name, vals in series.items():
        x = np.sort(vals)
        y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
        ax.plot(x, y, lw=1.5, label=name, color=_scenario_color(name))
    ax.set_xlabel("XPD value (dB): C0 uses floor, indoor uses excess")
    ax.set_ylabel("CDF")
    ax.set_title("M2-3 floor distribution vs indoor distributions")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "M2-3", "file": out.name, "note": "C0 floor vs indoor excess"})


def _make_g_plots(
    fig_dir: Path,
    link_rows: list[dict[str, str]],
    ray_rows: list[dict[str, str]],
    index_rows: list[dict[str, str]],
    tab_dir: Path,
    diag: dict[str, Any],
    detail_rows: list[dict[str, Any]],
) -> None:
    by = _by_scenario(link_rows)
    bt = diag.get("B_time_resolution", {})
    w_target_map = bt.get("W_target_s_by_scenario", {}) if isinstance(bt, dict) else {}
    w_target_s_default = _to_float(bt.get("W_target_s_default"), _to_float(bt.get("W_target_s"), 3e-9))
    w_target_a2 = _to_float(w_target_map.get("A2"), w_target_s_default) if isinstance(w_target_map, dict) else w_target_s_default
    w_target_a3 = _to_float(w_target_map.get("A3"), w_target_s_default) if isinstance(w_target_map, dict) else w_target_s_default
    # G1-1, G1-2, G1-3 (A2)
    a2row = _sample_case_for_scenario(link_rows, "A2", case_id="0") or _sample_case_for_scenario(link_rows, "A2")
    run_a2 = _first_index_run(index_rows, "A2")
    if a2row and run_a2:
        cid = str(a2row.get("case_id", "0"))
        pdp = _load_pdp(run_a2, "A2", cid)
        if pdp is not None:
            d, pco, pcr, xt = pdp
            win = _parse_jsonish(a2row.get("window"))
            tau0 = _to_float(win.get("tau0_s")) if isinstance(win, dict) else float("nan")
            te = _to_float(win.get("Te_s")) if isinstance(win, dict) else 3e-9
            w_early = (tau0, tau0 + te) if np.isfinite(tau0) else None
            # target center from rays: dominant n_bounce==1
            rr = [r for r in ray_rows if str(r.get("scenario_id", "")) == "A2" and str(r.get("case_id", "")) == cid and _to_int(r.get("n_bounce")) == 1]
            tau_t = _to_float(rr[0].get("tau_s")) if rr else float("nan")
            w_target = (tau_t - 0.5 * w_target_a2, tau_t + 0.5 * w_target_a2) if np.isfinite(tau_t) else None
            out = fig_dir / "G1-1__A2_pdp_overlay_target_early.png"
            _plot_pdp_overlay_with_windows(out, d, pco, pcr, "G1-1 A2 PDP overlay", w_early=w_early, w_target=w_target)
            detail_rows.append({"plot_id": "G1-1", "file": out.name, "note": "W_target/W_early shown"})
            xt_use = np.asarray(xt, dtype=float) if xt is not None else (_safe_log10(pco) - _safe_log10(pcr))
            out2 = fig_dir / "G1-1b__A2_tap_xpd_tau.png"
            _plot_tap_xpd(out2, d, xt_use, "G1 A2 tap-wise XPD(tau)")
            detail_rows.append({"plot_id": "G1-1b", "file": out2.name, "note": "tap-wise"})

    # G1-2: A2 XPD_early_ex by case_id
    a2 = by.get("A2", [])
    if a2:
        out = _prep_fig(fig_dir / "G1-2__A2_xpd_early_ex_by_case.png")
        x = np.arange(len(a2), dtype=float)
        y = np.asarray([_to_float(r.get("XPD_early_excess_db")) for r in a2], dtype=float)
        a3v = _finite([_to_float(r.get("XPD_early_excess_db")) for r in by.get("A3", [])])
        a4v = _finite([_to_float(r.get("XPD_early_excess_db")) for r in by.get("A4", [])])
        fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.6), gridspec_kw={"width_ratios": [2.0, 1.0]})
        ax = axs[0]
        ax.scatter(x, y, s=18, color=_scenario_color("A2"), alpha=0.88, label="A2 cases")
        if len(a3v):
            ax.axhline(float(np.median(a3v)), color=_scenario_color("A3"), lw=1.1, ls="--", label="A3 median")
        if len(a4v):
            ax.axhline(float(np.median(a4v)), color=_scenario_color("A4"), lw=1.1, ls=":", label="A4 median")
        ax.axhline(0.0, color="#666", lw=0.9)
        if np.any(np.isfinite(y)):
            ym = _finite(y)
            lo = float(np.nanmin(ym) - 2.0)
            hi = float(np.nanmax(ym) + 2.0)
            if hi - lo < 10.0:
                c = 0.5 * (hi + lo)
                lo = c - 5.0
                hi = c + 5.0
            _style_db_axis(ax, ymin=lo, ymax=hi)
        ax.set_xlabel("A2 case index")
        ax.set_ylabel("XPD_early_ex (dB)")
        ax.set_title("G1-2 A2 early excess (zoom)")
        ax.grid(True, axis="x", alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        ax2 = axs[1]
        data = []
        labels = []
        for sid in ["A2", "A3", "A4"]:
            vv = _finite([_to_float(r.get("XPD_early_excess_db")) for r in by.get(sid, [])])
            if len(vv):
                data.append(vv)
                labels.append(sid)
        if data:
            ax2.boxplot(data, labels=labels, showfliers=True)
        ax2.axhline(0.0, color="#666", lw=0.9)
        ax2.set_title("A2 vs A3/A4")
        _style_db_axis(ax2)
        ax2.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "G1-2", "file": out.name, "note": "A2 zoom + A3/A4 contrast"})

    # G1-3: A2 rho_early vs incidence
    dom_a2 = _dominant_incidence_by_link(ray_rows, "A2")
    pts = []
    for r in a2:
        lid = str(r.get("link_id", ""))
        ang = dom_a2.get(lid, float("nan"))
        rho = _to_float(r.get("rho_early_db"))
        if np.isfinite(ang) and np.isfinite(rho):
            pts.append((ang, rho))
    if pts:
        out = _prep_fig(fig_dir / "G1-3__A2_rho_vs_incidence.png")
        x = np.asarray([p[0] for p in pts], dtype=float)
        y = np.asarray([p[1] for p in pts], dtype=float)
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.scatter(x, y, s=18, color=_scenario_color("A2"), alpha=0.85)
        if len(np.unique(np.round(x, 6))) >= 2:
            z = np.polyfit(x, y, 1)
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(xx, z[0] * xx + z[1], "--", lw=1.4, color="#333")
        ax.set_xlabel("Dominant incidence angle (deg)")
        ax.set_ylabel("rho_early (dB)")
        ax.set_title("G1-3 A2 rho_early vs incidence")
        _style_db_axis(ax)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "G1-3", "file": out.name, "note": "dominant-incidence proxy"})

    # G2-1: A3 PDP with target window
    a3row = _sample_case_for_scenario(link_rows, "A3", case_id="0") or _sample_case_for_scenario(link_rows, "A3")
    run_a3 = _first_index_run(index_rows, "A3")
    if a3row and run_a3:
        cid = str(a3row.get("case_id", "0"))
        pdp = _load_pdp(run_a3, "A3", cid)
        if pdp is not None:
            d, pco, pcr, _xt = pdp
            win = _parse_jsonish(a3row.get("window"))
            tau0 = _to_float(win.get("tau0_s")) if isinstance(win, dict) else float("nan")
            te = _to_float(win.get("Te_s")) if isinstance(win, dict) else 3e-9
            w_early = (tau0, tau0 + te) if np.isfinite(tau0) else None
            rr = [r for r in ray_rows if str(r.get("scenario_id", "")) == "A3" and str(r.get("case_id", "")) == cid and _to_int(r.get("n_bounce")) == 2]
            tau_t = _to_float(rr[0].get("tau_s")) if rr else float("nan")
            w_target = (tau_t - 0.5 * w_target_a3, tau_t + 0.5 * w_target_a3) if np.isfinite(tau_t) else None
            out = fig_dir / "G2-1__A3_pdp_overlay_target_early.png"
            _plot_pdp_overlay_with_windows(out, d, pco, pcr, "G2-1 A3 PDP overlay", w_early=w_early, w_target=w_target)
            detail_rows.append({"plot_id": "G2-1", "file": out.name, "note": "A3 mechanism view"})

    # G2-2/G2-3 target-window from exported summary table
    tw_path = tab_dir / "A3_target_window_sign.csv"
    if tw_path.exists():
        rows = _read_csv(tw_path)
        # G2-2 A3 target distribution (summary bar with p10/p50/p90)
        a3 = None
        a2 = None
        for r in rows:
            if str(r.get("scenario", "")) == "A3":
                a3 = r
            if str(r.get("scenario", "")) == "A2":
                a2 = r
        if a3:
            out = _prep_fig(fig_dir / "G2-2__A3_xpd_target_ex_summary.png")
            metric = str(a3.get("sign_metric_for_status", "excess")).lower()
            use_raw = metric == "raw"
            p10 = _to_float(a3.get("p10_xpd_target_raw_db")) if use_raw else _to_float(a3.get("p10_xpd_target_ex_db"))
            p50 = _to_float(a3.get("median_xpd_target_raw_db")) if use_raw else _to_float(a3.get("median_xpd_target_ex_db"))
            p90 = _to_float(a3.get("p90_xpd_target_raw_db")) if use_raw else _to_float(a3.get("p90_xpd_target_ex_db"))
            fig, ax = plt.subplots(figsize=(5.6, 4.2))
            ax.bar(["p10", "median", "p90"], [p10, p50, p90], color=_scenario_color("A3"))
            ax.axhline(0.0, color="#666", lw=0.9)
            ax.set_ylabel("XPD_target_raw (dB)" if use_raw else "XPD_target_ex (dB)")
            ax.set_title(f"G2-2 A3 target-window summary ({metric})")
            _style_db_axis(ax)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out, dpi=140)
            plt.close(fig)
            detail_rows.append({"plot_id": "G2-2", "file": out.name, "note": f"from A3_target_window_sign.csv ({metric})"})
        if a2 and a3:
            out = _prep_fig(fig_dir / "G2-3__A2_vs_A3_target_window_compare.png")
            metric2 = str(a3.get("sign_metric_for_status", "excess")).lower()
            use_raw2 = metric2 == "raw"
            a2v = _to_float(a2.get("median_xpd_target_raw_db")) if use_raw2 else _to_float(a2.get("median_xpd_target_ex_db"))
            a3v = _to_float(a3.get("median_xpd_target_raw_db")) if use_raw2 else _to_float(a3.get("median_xpd_target_ex_db"))
            vals = [a2v, a3v]
            fig, ax = plt.subplots(figsize=(5.8, 4.2))
            ax.bar(["A2", "A3"], vals, color=[_scenario_color("A2"), _scenario_color("A3")])
            ax.axhline(0.0, color="#666", lw=0.9)
            ax.set_ylabel("median XPD_target_raw (dB)" if use_raw2 else "median XPD_target_ex (dB)")
            ax.set_title(f"G2-3 A2 vs A3 target-window comparison ({metric2})")
            _style_db_axis(ax)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out, dpi=140)
            plt.close(fig)
            detail_rows.append({"plot_id": "G2-3", "file": out.name, "note": f"target-window median compare ({metric2})"})

    # G2-4: target_in_Wearly / first-rate bars from diagnostics
    if bt:
        a3d = bt.get("W_target_detail", {}).get("A3", {})
        out = _prep_fig(fig_dir / "G2-4__A3_target_inearly_first_rate.png")
        vals = [
            _to_float(a3d.get("target_in_Wearly_rate"), _to_float(bt.get("A3_target_in_Wearly_rate"))),
            _to_float(a3d.get("target_is_first_rate")),
        ]
        labels = ["target_in_Wearly", "target_is_first"]
        fig, ax = plt.subplots(figsize=(6.2, 4.1))
        ax.bar(labels, vals, color=["#ff9da6", "#9d755d"])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Rate")
        ax.set_title("G2-4 A3 system-early suitability")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "G2-4", "file": out.name, "note": "diagnostic role split"})

    # G3-1/G3-2/G3-3
    a2 = by.get("A2", [])
    a3 = by.get("A3", [])
    if a2 and a3:
        out = _prep_fig(fig_dir / "G3-1__A2_A3_xpd_ex_violin.png")
        v2 = _finite([_to_float(r.get("XPD_early_excess_db")) for r in a2])
        v3 = _finite([_to_float(r.get("XPD_early_excess_db")) for r in a3])
        fig, ax = plt.subplots(figsize=(6.0, 4.4))
        parts = ax.violinplot([v2, v3], showmeans=True, showmedians=True)
        for pc, cc in zip(parts["bodies"], [_scenario_color("A2"), _scenario_color("A3")]):
            pc.set_facecolor(cc)
            pc.set_alpha(0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["A2", "A3"])
        ax.set_ylabel("XPD_early_ex (dB)")
        ax.set_title("G3-1 A2/A3 leakage dispersion")
        _style_db_axis(ax)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "G3-1", "file": out.name, "note": "A2/A3 variance view"})

    # |XPD_ex| vs U(EL)
    sc = [r for r in link_rows if str(r.get("scenario_id", "")) in {"A2", "A3", "A4", "A5"}]
    if sc:
        out = _prep_fig(fig_dir / "G3-2__abs_xpd_ex_vs_el_scatter.png")
        x = np.asarray([_to_float(r.get("EL_proxy_db")) for r in sc], dtype=float)
        y = np.asarray([abs(_to_float(r.get("XPD_early_excess_db"))) for r in sc], dtype=float)
        sid = np.asarray([str(r.get("scenario_id", "")) for r in sc], dtype=object)
        m = np.isfinite(x) & np.isfinite(y)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        marker_map = {"A2": "o", "A3": "s", "A4": "^", "A5": "D"}
        for s in ["A2", "A3", "A4", "A5"]:
            mm = m & (sid == s)
            if not np.any(mm):
                continue
            ax.scatter(
                x[mm],
                y[mm],
                s=22,
                alpha=0.85,
                color=_scenario_color(s),
                marker=marker_map.get(s, "o"),
                label=s,
            )
        ax.set_xlabel("EL_proxy (dB)")
        ax.set_ylabel("|XPD_early_ex| (dB)")
        ax.set_title("G3-2 |XPD_ex| vs U(EL)")
        _style_db_axis(ax)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2, title="scenario")
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "G3-2", "file": out.name, "note": "conditioned by EL"})

    # variance by scenario
    out = _prep_fig(fig_dir / "G3-3__variance_xpd_ex_by_scenario.png")
    scen = ["A2", "A3", "A4", "A5"]
    vals = []
    for s in scen:
        v = _finite([_to_float(r.get("XPD_early_excess_db")) for r in by.get(s, [])])
        vals.append(float(np.var(v, ddof=1)) if len(v) > 1 else np.nan)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(scen, vals, color=[_scenario_color(s) for s in scen])
    ax.set_ylabel("Var(XPD_early_ex) [dB^2]")
    ax.set_title("G3-3 leakage variance by scenario")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "G3-3", "file": out.name, "note": "variance summary"})


def _make_l_plots(
    fig_dir: Path,
    link_rows: list[dict[str, str]],
    ray_rows: list[dict[str, str]],
    detail_rows: list[dict[str, Any]],
) -> None:
    by = _by_scenario(link_rows)
    scen = ["A2", "A3", "A4", "A5", "B1", "B2", "B3"]

    # L1-1 paired early/late by sampled cases
    out = _prep_fig(fig_dir / "L1-1__early_late_paired_lines.png")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    rng = np.random.default_rng(0)
    xloc = {"early": 0.0, "late": 1.0}
    for s in scen:
        rr = by.get(s, [])
        if not rr:
            continue
        idx = np.arange(len(rr))
        if len(idx) > 60:
            idx = rng.choice(idx, size=60, replace=False)
        for i in idx:
            r = rr[int(i)]
            e = _to_float(r.get("XPD_early_excess_db"))
            l = _to_float(r.get("XPD_late_excess_db"))
            if not (np.isfinite(e) and np.isfinite(l)):
                continue
            ax.plot([xloc["early"], xloc["late"]], [e, l], color=_scenario_color(s), alpha=0.10, lw=0.8)
    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["early", "late"])
    ax.set_ylabel("XPD_ex (dB)")
    ax.set_title("L1-1 early vs late paired lines (sampled cases)")
    _style_db_axis(ax)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "L1-1", "file": out.name, "note": "paired sampled cases"})

    # L1-2 L_pol by scenario
    out = _prep_fig(fig_dir / "L1-2__lpol_box_by_scenario.png")
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    data = []
    labels = []
    for s in scen:
        vals = _finite([_to_float(r.get("L_pol_db")) for r in by.get(s, [])])
        if len(vals):
            data.append(vals)
            labels.append(s)
    if data:
        ax.boxplot(data, labels=labels, showfliers=True)
    ax.axhline(0.0, color="#666", ls="--", lw=1.0)
    ax.set_ylabel("L_pol (dB)")
    ax.set_title("L1-2 L_pol distribution by scenario")
    _style_db_axis(ax)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "L1-2", "file": out.name, "note": "zero line shown"})

    # L2-M branch (A4)
    a4 = by.get("A4", [])
    dom = _dominant_incidence_by_link(ray_rows, "A4")
    if a4:
        # angle bins from dominant incidence quantiles
        angs = np.asarray([dom.get(str(r.get("link_id", "")), np.nan) for r in a4], dtype=float)
        good = angs[np.isfinite(angs)]
        if len(good) >= 6:
            q1, q2 = np.percentile(good, [33, 66])
        else:
            q1, q2 = 30.0, 60.0

        def _abin(a: float) -> str:
            if not np.isfinite(a):
                return "NA"
            if a <= q1:
                return "low"
            if a <= q2:
                return "mid"
            return "high"

        # L2-M1: material x angle-bin scatter (XPD_early_ex)
        out = _prep_fig(fig_dir / "L2-M1__A4_material_angle_xpd_early_ex.png")
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        mats = ["glass", "wood", "gypsum"]
        xmap = {m: i for i, m in enumerate(mats)}
        cmap = {"low": "#4c78a8", "mid": "#f58518", "high": "#54a24b", "NA": "#999"}
        for r in a4:
            m = str(r.get("material_class", "NA")).lower()
            if m not in xmap:
                continue
            ang = dom.get(str(r.get("link_id", "")), np.nan)
            b = _abin(ang)
            y = _to_float(r.get("XPD_early_excess_db"))
            if not np.isfinite(y):
                continue
            ax.scatter(xmap[m] + {"low": -0.15, "mid": 0.0, "high": 0.15, "NA": 0.22}[b], y, s=18, color=cmap[b], alpha=0.75)
        ax.set_xticks(list(xmap.values()))
        ax.set_xticklabels(mats)
        ax.set_ylabel("XPD_early_ex (dB)")
        ax.set_title("L2-M1 A4 material effect with angle bins")
        _style_db_axis(ax)
        ax.grid(True, axis="y", alpha=0.3)
        handles = [plt.Line2D([], [], ls="", marker="o", color=cmap[k], label=f"angle={k}") for k in ["low", "mid", "high"]]
        ax.legend(handles=handles, loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "L2-M1", "file": out.name, "note": "material x angle-bin"})

        # L2-M2 abs(XPD_ex) by material
        out = _prep_fig(fig_dir / "L2-M2__A4_abs_xpd_ex_by_material.png")
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        data = []
        labels = []
        for m in mats:
            vals = _finite([abs(_to_float(r.get("XPD_early_excess_db"))) for r in a4 if str(r.get("material_class", "")).lower() == m])
            if len(vals):
                data.append(vals)
                labels.append(m)
        if data:
            ax.boxplot(data, labels=labels, showfliers=True)
        ax.set_ylabel("|XPD_early_ex| (dB)")
        ax.set_title("L2-M2 |XPD_ex| by material")
        _style_db_axis(ax)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "L2-M2", "file": out.name, "note": "toward 0 dB check"})

        # L2-M3 variance summary
        out = _prep_fig(fig_dir / "L2-M3__A4_variance_by_material.png")
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        vars_ = []
        labs = []
        for m in mats:
            vals = _finite([_to_float(r.get("XPD_early_excess_db")) for r in a4 if str(r.get("material_class", "")).lower() == m])
            if len(vals) > 1:
                vars_.append(float(np.var(vals, ddof=1)))
                labs.append(m)
        if labs:
            ax.bar(labs, vars_, color=[_scenario_color("A4")] * len(labs))
        ax.set_ylabel("Var(XPD_early_ex) [dB^2]")
        ax.set_title("L2-M3 variance by material")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "L2-M3", "file": out.name, "note": "variance/IQR proxy"})

    # L2-S branch (A5)
    a5 = by.get("A5", [])
    if a5:
        base_by_case: dict[str, dict[str, str]] = {}
        stress_by_case: dict[str, dict[str, str]] = {}
        for r in a5:
            cid = str(r.get("case_id", ""))
            s = str(r.get("stress_mode", "")).lower()
            rf = _to_float(r.get("roughness_flag"))
            hf = _to_float(r.get("human_flag"))
            is_stress = (s in {"hybrid", "synthetic", "geometry", "stress", "on"}) or (rf > 0.5) or (hf > 0.5)
            if is_stress:
                stress_by_case[cid] = r
            else:
                base_by_case[cid] = r
        pair_ids = sorted(set(base_by_case.keys()) & set(stress_by_case.keys()), key=lambda x: int(x) if x.isdigit() else x)
        # L2-S1 paired L_pol
        if pair_ids:
            out = _prep_fig(fig_dir / "L2-S1__A5_base_stress_lpol_paired.png")
            fig, ax = plt.subplots(figsize=(6.2, 4.5))
            for cid in pair_ids:
                b = _to_float(base_by_case[cid].get("L_pol_db"))
                s = _to_float(stress_by_case[cid].get("L_pol_db"))
                if np.isfinite(b) and np.isfinite(s):
                    ax.plot([0, 1], [b, s], color="#b05aa9", alpha=0.35, lw=1.0)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["base", "stress"])
            ax.set_ylabel("L_pol (dB)")
            ax.set_title("L2-S1 A5 base vs stress paired L_pol")
            _style_db_axis(ax)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out, dpi=140)
            plt.close(fig)
            detail_rows.append({"plot_id": "L2-S1", "file": out.name, "note": "paired by case_id"})

        # L2-S2 late excess box
        out = _prep_fig(fig_dir / "L2-S2__A5_base_stress_xpd_late_ex.png")
        vb = _finite([_to_float(base_by_case[c].get("XPD_late_excess_db")) for c in pair_ids])
        vs = _finite([_to_float(stress_by_case[c].get("XPD_late_excess_db")) for c in pair_ids])
        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        if len(vb) and len(vs):
            ax.boxplot([vb, vs], labels=["base", "stress"], showfliers=True)
        ax.set_ylabel("XPD_late_ex (dB)")
        ax.set_title("L2-S2 A5 late contamination")
        _style_db_axis(ax)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "L2-S2", "file": out.name, "note": "late-ex branch"})

        # L2-S3 DS summary
        out = _prep_fig(fig_dir / "L2-S3__A5_base_stress_ds_tail.png")
        db = _finite([_to_float(base_by_case[c].get("delay_spread_rms_s")) * 1e9 for c in pair_ids])
        ds = _finite([_to_float(stress_by_case[c].get("delay_spread_rms_s")) * 1e9 for c in pair_ids])
        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        if len(db) and len(ds):
            ax.boxplot([db, ds], labels=["base", "stress"], showfliers=True)
        ax.set_ylabel("Delay spread (ns)")
        ax.set_title("L2-S3 A5 DS/tail proxy")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "L2-S3", "file": out.name, "note": "stress tail widening proxy"})

    # L3 branch
    sset = {"A3", "A4", "B1", "B2", "B3"}
    ss = [r for r in link_rows if str(r.get("scenario_id", "")) in sset]
    if ss:
        # L3-1 scatter + trend
        out = _prep_fig(fig_dir / "L3-1__el_vs_xpd_ex_scatter_regression.png")
        x = np.asarray([_to_float(r.get("EL_proxy_db")) for r in ss], dtype=float)
        y = np.asarray([_to_float(r.get("XPD_early_excess_db")) for r in ss], dtype=float)
        scn = np.asarray([str(r.get("scenario_id", "")) for r in ss], dtype=object)
        m = np.isfinite(x) & np.isfinite(y)
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        marker_map = {"A3": "o", "A4": "s", "B1": "^", "B2": "D", "B3": "P"}
        for sid in ["A3", "A4", "B1", "B2", "B3"]:
            mm = m & (scn == sid)
            if not np.any(mm):
                continue
            ax.scatter(
                x[mm],
                y[mm],
                s=24,
                alpha=0.85,
                color=_scenario_color(sid),
                marker=marker_map.get(sid, "o"),
                label=sid,
            )
        if np.sum(m) > 4 and len(np.unique(np.round(x[m], 6))) >= 2:
            z = np.polyfit(x[m], y[m], 1)
            xx = np.linspace(float(np.min(x[m])), float(np.max(x[m])), 120)
            ax.plot(xx, z[0] * xx + z[1], "--", color="#222", lw=1.5)
        ax.set_xlabel("EL_proxy (dB)")
        ax.set_ylabel("XPD_early_ex (dB)")
        ax.set_title("L3-1 EL vs XPD_ex (EL-identifying subset)")
        _style_db_axis(ax)
        ax.grid(True, axis="x", alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncol=2, title="scenario")
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "L3-1", "file": out.name, "note": "subset=A3/A4/B"})

        # L3-2 EL bin conditional box
        out = _prep_fig(fig_dir / "L3-2__el_bin_conditional_box.png")
        xv = x[m]
        yv = y[m]
        q = np.percentile(xv, [33, 66]) if len(xv) >= 3 else [np.nan, np.nan]
        bins = []
        for xx in xv:
            if xx <= q[0]:
                bins.append("low")
            elif xx <= q[1]:
                bins.append("mid")
            else:
                bins.append("high")
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        data = [_finite([yv[i] for i, b in enumerate(bins) if b == k]) for k in ["low", "mid", "high"]]
        ax.boxplot(data, labels=["low", "mid", "high"], showfliers=True)
        ax.set_xlabel("EL bin")
        ax.set_ylabel("XPD_early_ex (dB)")
        ax.set_title("L3-2 conditional box by EL tertile")
        _style_db_axis(ax)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "L3-2", "file": out.name, "note": "nonparametric monotonicity"})

        # L3-3 fitted mean ± CI
        out = _prep_fig(fig_dir / "L3-3__residualized_effect_fit_ci.png")
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        if len(xv) >= 5 and len(np.unique(np.round(xv, 6))) >= 2:
            z = np.polyfit(xv, yv, 1)
            xx = np.linspace(float(np.min(xv)), float(np.max(xv)), 120)
            yhat = z[0] * xx + z[1]
            resid = yv - (z[0] * xv + z[1])
            s = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
            ax.plot(xx, yhat, color="#1f77b4", lw=1.8, label="fitted mean")
            ax.fill_between(xx, yhat - s, yhat + s, color="#1f77b4", alpha=0.2, label="±1σ residual")
            ax.scatter(xv, yv, s=12, alpha=0.22, color="#444")
        ax.set_xlabel("EL_proxy (dB)")
        ax.set_ylabel("XPD_early_ex (dB)")
        ax.set_title("L3-3 fitted mean μ(U) and residual band")
        _style_db_axis(ax)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        detail_rows.append({"plot_id": "L3-3", "file": out.name, "note": "stage1-style fit"})


def _make_r_plots(fig_dir: Path, link_rows: list[dict[str, str]], detail_rows: list[dict[str, Any]]) -> None:
    by = _by_scenario(link_rows)
    bset = ["B1", "B2", "B3"]
    pool = [r for r in link_rows if str(r.get("scenario_id", "")) in set(bset)]
    if not pool:
        return

    def _facet_heat(metric: str, out_name: str, title: str, *, vmin: float | None = None, vmax: float | None = None) -> None:
        out = _prep_fig(fig_dir / out_name)
        fig, axs = plt.subplots(1, 3, figsize=(13.0, 4.0), sharex=True, sharey=True)
        sc_last = None
        for i, s in enumerate(bset):
            rr = by.get(s, [])
            x = np.asarray([_to_float(r.get("rx_x")) for r in rr], dtype=float)
            y = np.asarray([_to_float(r.get("rx_y")) for r in rr], dtype=float)
            v = np.asarray([_to_float(r.get(metric)) for r in rr], dtype=float)
            m = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
            ax = axs[i]
            if np.any(m):
                sc_last = ax.scatter(
                    x[m],
                    y[m],
                    c=v[m],
                    s=52,
                    cmap="coolwarm",
                    vmin=vmin,
                    vmax=vmax,
                    edgecolor="k",
                    linewidth=0.2,
                )
            ax.set_title(s)
            ax.grid(True, alpha=0.2)
            ax.set_xlabel("x (m)")
            if i == 0:
                ax.set_ylabel("y (m)")
        if sc_last is not None:
            fig.colorbar(sc_last, ax=axs.ravel().tolist(), fraction=0.035, pad=0.02)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)

    _facet_heat(
        "XPD_early_excess_db",
        "R1-1__B123_heatmap_xpd_early_ex.png",
        "R1-1 B1/B2/B3 heatmap: XPD_early_ex",
        vmin=-35.0,
        vmax=-18.0,
    )
    detail_rows.append({"plot_id": "R1-1", "file": "R1-1__B123_heatmap_xpd_early_ex.png", "note": "facet B1/B2/B3"})
    _facet_heat("rho_early_db", "R1-2__B123_heatmap_rho_early.png", "R1-2 B1/B2/B3 heatmap: rho_early")
    detail_rows.append({"plot_id": "R1-2", "file": "R1-2__B123_heatmap_rho_early.png", "note": "facet B1/B2/B3"})
    _facet_heat("L_pol_db", "R1-3__B123_heatmap_lpol.png", "R1-3 B1/B2/B3 heatmap: L_pol")
    detail_rows.append({"plot_id": "R1-3", "file": "R1-3__B123_heatmap_lpol.png", "note": "facet B1/B2/B3"})

    # R1-4 LOS/NLOS grouped CDF for three metrics
    out = _prep_fig(fig_dir / "R1-4__los_nlos_grouped_metrics.png")
    fig, axs = plt.subplots(1, 3, figsize=(14.0, 4.2))
    metrics = [("XPD_early_excess_db", "XPD_early_ex"), ("rho_early_db", "rho_early_db"), ("L_pol_db", "L_pol")]
    los = [r for r in pool if _to_int(r.get("LOSflag")) == 1]
    nlos = [r for r in pool if _to_int(r.get("LOSflag")) == 0]
    for i, (k, label) in enumerate(metrics):
        ax = axs[i]
        for name, rr, cc in [("LOS", los, "#1f77b4"), ("NLOS", nlos, "#d62728")]:
            vals = np.sort(_finite([_to_float(r.get(k)) for r in rr]))
            if len(vals):
                cdf = np.arange(1, len(vals) + 1, dtype=float) / float(len(vals))
                ax.plot(vals, cdf, lw=1.6, label=name, color=cc)
        ax.set_title(label)
        ax.set_xlabel("value")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("CDF")
    axs[0].legend(loc="best", fontsize=8)
    fig.suptitle("R1-4 LOS vs NLOS grouped CDF")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "R1-4", "file": out.name, "note": "LOS/NLOS grouped CDFs"})

    # R2-1, R2-2
    out = _prep_fig(fig_dir / "R2-1__xpd_early_ex_vs_ds_B123.png")
    x = np.asarray([_to_float(r.get("XPD_early_excess_db")) for r in pool], dtype=float)
    y = np.asarray([_to_float(r.get("delay_spread_rms_s")) * 1e9 for r in pool], dtype=float)
    c = np.asarray([_to_int(r.get("LOSflag"), -1) for r in pool], dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sc = ax.scatter(x[m], y[m], c=c[m], cmap="coolwarm", s=24, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="LOSflag")
    ax.set_xlabel("XPD_early_ex (dB)")
    ax.set_ylabel("Delay spread (ns)")
    ax.set_title("R2-1 XPD_early_ex vs DS")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "R2-1", "file": out.name, "note": "B1/B2/B3 pooled"})

    out = _prep_fig(fig_dir / "R2-2__rho_early_vs_ds_B123.png")
    xr = np.asarray([_to_float(r.get("rho_early_db")) for r in pool], dtype=float)
    m = np.isfinite(xr) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sc = ax.scatter(xr[m], y[m], c=c[m], cmap="coolwarm", s=24, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="LOSflag")
    ax.set_xlabel("rho_early (dB)")
    ax.set_ylabel("Delay spread (ns)")
    ax.set_title("R2-2 rho_early vs DS")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "R2-2", "file": out.name, "note": "B1/B2/B3 pooled"})

    # R2-3 quadrant plot
    out = _prep_fig(fig_dir / "R2-3__quadrant_useful_vs_risky.png")
    x0 = np.nanmedian(x[m]) if np.any(m) else 0.0
    y0 = np.nanmedian(y[m]) if np.any(m) else 0.0
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(x[m], y[m], c=c[m], cmap="coolwarm", s=24, alpha=0.85)
    ax.axvline(x0, color="#333", ls="--", lw=1.0)
    ax.axhline(y0, color="#333", ls="--", lw=1.0)
    ax.text(x0 + 0.2, y0 - 0.2, "useful", color="#2ca02c", fontsize=9)
    ax.text(x0 - 0.2, y0 + 0.2, "risky", color="#d62728", fontsize=9, ha="right")
    ax.set_xlabel("XPD_early_ex (dB)")
    ax.set_ylabel("Delay spread (ns)")
    ax.set_title("R2-3 quadrant: useful vs risky")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "R2-3", "file": out.name, "note": "median-threshold quadrants"})

    # R2-4 early fraction vs XPD
    out = _prep_fig(fig_dir / "R2-4__early_fraction_vs_xpd_early_ex.png")
    ef = np.asarray([_to_float(r.get("early_energy_fraction")) for r in pool], dtype=float)
    m2 = np.isfinite(x) & np.isfinite(ef)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    sc = ax.scatter(x[m2], ef[m2], c=c[m2], cmap="coolwarm", s=24, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="LOSflag")
    ax.set_xlabel("XPD_early_ex (dB)")
    ax.set_ylabel("early_energy_fraction")
    ax.set_title("R2-4 early concentration vs XPD_early_ex")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "R2-4", "file": out.name, "note": "system-surrogate link"})


def _make_p_plots(fig_dir: Path, link_rows: list[dict[str, str]], detail_rows: list[dict[str, Any]]) -> None:
    # Build stage1 subset
    subset = [r for r in link_rows if str(r.get("scenario_id", "")) in {"A3", "A4", "B1", "B2", "B3"}]
    x = np.asarray([_to_float(r.get("EL_proxy_db")) for r in subset], dtype=float)
    y = np.asarray([_to_float(r.get("XPD_early_excess_db")) for r in subset], dtype=float)
    sid = np.asarray([str(r.get("scenario_id", "")) for r in subset], dtype=object)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    sid = sid[m]
    if len(x) < 8:
        return
    # simple linear and constant baselines
    b1, b0 = np.polyfit(x, y, 1)
    y_lin = b1 * x + b0
    y_const = np.full_like(y, float(np.mean(y)))

    # P1-1 observed vs predicted CDF
    out = _prep_fig(fig_dir / "P1-1__observed_vs_predicted_cdf_overlay.png")
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8), gridspec_kw={"width_ratios": [2.0, 1.0]})
    ax = axs[0]
    for name, vals, cc in [
        ("observed", y, "#000000"),
        ("conditional-linear", y_lin, "#1f77b4"),
        ("constant-baseline", y_const, "#d62728"),
    ]:
        xx = np.sort(_finite(vals))
        yy = np.arange(1, len(xx) + 1, dtype=float) / float(len(xx))
        ax.plot(xx, yy, lw=1.7, label=name, color=cc)
    ax.set_xlabel("Z = XPD_early_ex (dB)")
    ax.set_ylabel("CDF")
    ax.set_title("P1-1 observed vs predicted CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    # Tail-source panel: where observed upper tail comes from.
    ax2 = axs[1]
    q80 = float(np.percentile(y, 80.0))
    tail = y >= q80
    sid_tail = sid[tail]
    cats = ["A3", "A4", "B1", "B2", "B3"]
    counts = [int(np.sum(sid_tail == c)) for c in cats]
    nz_idx = [i for i, v in enumerate(counts) if v > 0]
    if nz_idx:
        cats_nz = [cats[i] for i in nz_idx]
        cnt_nz = [counts[i] for i in nz_idx]
        ax2.bar(cats_nz, cnt_nz, color=[_scenario_color(c) for c in cats_nz])
    ax2.set_title("Observed upper-tail source\n(top 20%)")
    ax2.set_ylabel("count")
    ax2.set_xlabel("scenario")
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "P1-1", "file": out.name, "note": "simple linear proxy"})

    # P1-2 bin median observed vs predicted
    out = _prep_fig(fig_dir / "P1-2__predicted_vs_observed_bin_median.png")
    q1, q2 = np.percentile(x, [33, 66])
    bins = np.where(x <= q1, "low", np.where(x <= q2, "mid", "high"))
    labs = ["low", "mid", "high"]
    obs = []
    lin = []
    con = []
    for b in labs:
        idx = bins == b
        obs.append(float(np.median(y[idx])) if np.any(idx) else np.nan)
        lin.append(float(np.median(y_lin[idx])) if np.any(idx) else np.nan)
        con.append(float(np.median(y_const[idx])) if np.any(idx) else np.nan)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    xx = np.arange(3, dtype=float)
    w = 0.25
    ax.bar(xx - w, obs, width=w, label="observed")
    ax.bar(xx, lin, width=w, label="conditional")
    ax.bar(xx + w, con, width=w, label="constant")
    ax.set_xticks(xx)
    ax.set_xticklabels(labs)
    ax.set_xlabel("EL bin")
    ax.set_ylabel("median Z (dB)")
    ax.set_title("P1-2 predicted vs observed bin medians")
    _style_db_axis(ax)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "P1-2", "file": out.name, "note": "EL tertiles"})

    # P1-3 rank agreement
    out = _prep_fig(fig_dir / "P1-3__rank_agreement_scatter.png")
    r_obs = stats.rankdata(y)
    r_lin = stats.rankdata(y_lin)
    r_con = stats.rankdata(y_const)
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    ax.scatter(r_obs, r_lin, s=16, alpha=0.6, label="conditional", color="#1f77b4")
    ax.scatter(r_obs, r_con, s=16, alpha=0.5, label="constant", color="#d62728")
    ax.set_xlabel("observed rank")
    ax.set_ylabel("predicted rank")
    ax.set_title("P1-3 rank agreement")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "P1-3", "file": out.name, "note": "Spearman-oriented"})

    # P1-4 residual vs EL
    out = _prep_fig(fig_dir / "P1-4__residual_vs_el.png")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.scatter(x, y - y_lin, s=16, alpha=0.55, label="conditional residual", color="#1f77b4")
    ax.scatter(x, y - y_const, s=16, alpha=0.45, label="constant residual", color="#d62728")
    ax.axhline(0.0, color="#333", lw=0.9)
    ax.set_xlabel("EL_proxy (dB)")
    ax.set_ylabel("residual (dB)")
    ax.set_title("P1-4 residual vs EL")
    _style_db_axis(ax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "P1-4", "file": out.name, "note": "EL bias check"})

    # P2 minimal protocol simulation
    rng = np.random.default_rng(42)
    # minimal subset: 50% per scenario
    by = _by_scenario(link_rows)
    minimal: list[dict[str, str]] = []
    for s, rr in by.items():
        if len(rr) <= 2:
            minimal.extend(rr)
            continue
        idx = np.arange(len(rr))
        take = max(1, len(rr) // 2)
        sel = rng.choice(idx, size=take, replace=False)
        minimal.extend([rr[int(i)] for i in sel])

    def _effect_dict(rows: list[dict[str, str]]) -> dict[str, float]:
        bys = _by_scenario(rows)
        c0 = _finite([_to_float(r.get("XPD_early_excess_db")) for r in bys.get("C0", [])])
        a2 = _finite([_to_float(r.get("XPD_early_excess_db")) for r in bys.get("A2", [])])
        a3 = _finite([_to_float(r.get("XPD_early_excess_db")) for r in bys.get("A3", [])])
        bpool = [r for r in rows if str(r.get("scenario_id", "")) in {"B1", "B2", "B3"}]
        los = _finite([_to_float(r.get("XPD_early_excess_db")) for r in bpool if _to_int(r.get("LOSflag")) == 1])
        nlos = _finite([_to_float(r.get("XPD_early_excess_db")) for r in bpool if _to_int(r.get("LOSflag")) == 0])
        sub = [r for r in rows if str(r.get("scenario_id", "")) in {"A3", "A4", "B1", "B2", "B3"}]
        x = np.asarray([_to_float(r.get("EL_proxy_db")) for r in sub], dtype=float)
        y = np.asarray([_to_float(r.get("XPD_early_excess_db")) for r in sub], dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        rho = float(stats.spearmanr(x[m], y[m]).correlation) if np.sum(m) >= 5 else np.nan
        return {
            "G1": float(np.nanmedian(a2) - np.nanmedian(c0)) if len(a2) and len(c0) else np.nan,
            "G2": float(np.nanmedian(a3) - np.nanmedian(a2)) if len(a3) and len(a2) else np.nan,
            "L3": rho,
            "R1": float(stats.wasserstein_distance(los, nlos)) if len(los) and len(nlos) else np.nan,
        }

    eff_full = _effect_dict(link_rows)
    eff_min = _effect_dict(minimal)

    out = _prep_fig(fig_dir / "P2-1__full_vs_minimal_effect_size.png")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    claims = ["G1", "G2", "L3", "R1"]
    xx = np.arange(len(claims), dtype=float)
    w = 0.35
    ax.bar(xx - 0.5 * w, [eff_full[k] for k in claims], width=w, label="full")
    ax.bar(xx + 0.5 * w, [eff_min[k] for k in claims], width=w, label="minimal")
    ax.set_xticks(xx)
    ax.set_xticklabels(claims)
    ax.set_title("P2-1 full vs minimal effect sizes")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "P2-1", "file": out.name, "note": "G1/G2/L3/R1 effect compare"})

    # P2-2 coefficient stability (bootstrap b1)
    out = _prep_fig(fig_dir / "P2-2__coefficient_stability_subsampling.png")
    b1s = []
    idx = np.arange(len(x), dtype=int)
    for _ in range(200):
        ii = rng.choice(idx, size=max(8, len(idx) // 2), replace=True)
        xx = x[ii]
        yy = y[ii]
        m = np.isfinite(xx) & np.isfinite(yy)
        if np.sum(m) >= 5 and len(np.unique(np.round(xx[m], 6))) >= 2:
            b1, _b0 = np.polyfit(xx[m], yy[m], 1)
            b1s.append(float(b1))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if b1s:
        ax.hist(b1s, bins=24, alpha=0.75, color="#1f77b4")
        ax.axvline(float(np.median(b1s)), color="#d62728", lw=1.4, ls="--", label=f"median={np.median(b1s):.3f}")
    ax.set_xlabel("Estimated coefficient b1 (EL)")
    ax.set_ylabel("Count")
    ax.set_title("P2-2 coefficient stability under subsampling")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "P2-2", "file": out.name, "note": "bootstrap slope distribution"})

    # P2-3 full vs minimal CDF overlays
    out = _prep_fig(fig_dir / "P2-3__full_vs_minimal_cdf_overlay.png")
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.4))
    for ax, key, ttl in [
        (axs[0], "XPD_early_excess_db", "XPD_early_ex"),
        (axs[1], "L_pol_db", "L_pol"),
    ]:
        f = np.sort(_finite([_to_float(r.get(key)) for r in link_rows]))
        m2 = np.sort(_finite([_to_float(r.get(key)) for r in minimal]))
        if len(f):
            ax.plot(f, np.arange(1, len(f) + 1) / len(f), lw=1.6, label="full")
        if len(m2):
            ax.plot(m2, np.arange(1, len(m2) + 1) / len(m2), lw=1.6, ls="--", label="minimal")
        ax.set_title(ttl)
        ax.set_xlabel("value")
        _style_db_axis(ax)
        ax.grid(True, alpha=0.3)
    axs[0].set_ylabel("CDF")
    axs[0].legend(loc="best", fontsize=8)
    fig.suptitle("P2-3 minimal vs full protocol CDF")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    detail_rows.append({"plot_id": "P2-3", "file": out.name, "note": "distribution-level reproducibility"})


@dataclass
class PlotSpec:
    plot_id: str
    proposition: str
    scenario: str
    needed_data: str
    plot_desc: str
    pass_rule: str
    expected_file: str


def _specs() -> list[PlotSpec]:
    return [
        # M
        PlotSpec("M1-1", "M1", "C0", "co/cross PDP, W_floor", "C0 raw PDP overlay facet(d,yaw)", "LOS-only contamination low", "M1-1__C0_raw_pdp_overlay_grid.png"),
        PlotSpec("M1-2", "M1", "C0", "rays(C_floor)", "C_floor boxplot", "C_floor below -10/-15 dB", "M1-2__C_floor_box_by_yaw.png"),
        PlotSpec("M1-3", "M1", "C0", "XPD_floor,d", "XPD_floor vs distance", "weak distance dependence", "C0__ALL__xpd_floor_vs_distance.png"),
        PlotSpec("M1-4", "M1", "C0", "XPD_floor,yaw", "XPD_floor vs yaw", "yaw-driven spread", "C0__ALL__xpd_floor_vs_yaw.png"),
        PlotSpec("M1-5", "M1", "C0", "xpd_floor_curve,freq", "XPD_floor vs subband/frequency", "frequency baseline decision", "M1-5__xpd_floor_vs_frequency.png"),
        PlotSpec("M1-6", "M1", "C0", "repeat_id", "repeatability strip+box", "Delta_repeat estimation", "M1-6__repeatability_strip_box.png"),
        PlotSpec("M1-7", "M1", "C0", "Delta_floor,Delta_repeat", "uncertainty budget summary", "Delta_ref=max()", "M1-7__uncertainty_budget_summary.png"),
        PlotSpec("M2-1", "M2", "A2/A3/A4/A5/B", "XPD_early_ex, XPD_late_ex", "scenario-wise excess box", "beyond ±Delta_ref", "M2-1__scenario_early_late_ex_box.png"),
        PlotSpec("M2-2", "M2", "A2/A3/A4/A5/B", "XPD_ex,Delta_ref", "exceedance rate", "channel-claim fraction", "M2-2__exceedance_rate_bar.png"),
        PlotSpec("M2-3", "M2", "C0 + indoor", "floor vs indoor distributions", "CDF overlay", "separable distributions", "M2-3__cdf_floor_vs_indoor_overlay.png"),
        # G
        PlotSpec("G1-1", "G1", "A2", "PDP, W_target/W_early", "A2 PDP overlay", "odd cross-dominant trend", "G1-1__A2_pdp_overlay_target_early.png"),
        PlotSpec("G1-2", "G1", "A2", "XPD_early_ex", "A2 XPD distribution by case", "sign stability", "G1-2__A2_xpd_early_ex_by_case.png"),
        PlotSpec("G1-3", "G1", "A2", "rho_early + incidence", "rho vs angle", "odd supports rho increase", "G1-3__A2_rho_vs_incidence.png"),
        PlotSpec("G2-1", "G2", "A3", "PDP + target window", "A3 target-window PDP", "even mechanism visibility", "G2-1__A3_pdp_overlay_target_early.png"),
        PlotSpec("G2-2", "G2", "A3", "target-window summary", "A3 XPD_target(raw/ex) summary", "co-dominant tendency", "G2-2__A3_xpd_target_ex_summary.png"),
        PlotSpec("G2-3", "G2", "A2/A3", "target-window summary", "A2 vs A3 target(raw/ex) compare", "sign reversal", "G2-3__A2_vs_A3_target_window_compare.png"),
        PlotSpec("G2-4", "G2", "A3", "target_in_early/first rates", "A3 system-early suitability", "role split", "G2-4__A3_target_inearly_first_rate.png"),
        PlotSpec("G3-1", "G3", "A2/A3", "XPD_ex", "A2/A3 variance violin", "not perfectly separable", "G3-1__A2_A3_xpd_ex_violin.png"),
        PlotSpec("G3-2", "G3", "A2/A3/A4/A5", "|XPD_ex| + U", "|XPD_ex| vs EL", "conditional variation", "G3-2__abs_xpd_ex_vs_el_scatter.png"),
        PlotSpec("G3-3", "G3", "A2/A3/A4/A5", "variance", "variance summary", "leakage spread quantification", "G3-3__variance_xpd_ex_by_scenario.png"),
        # L
        PlotSpec("L1-1", "L1", "A2-A5+B", "early/late excess", "paired early-late plot", "late dirtier trend", "L1-1__early_late_paired_lines.png"),
        PlotSpec("L1-2", "L1", "A2-A5+B", "L_pol", "L_pol by scenario", "L_pol > 0 tendency", "L1-2__lpol_box_by_scenario.png"),
        PlotSpec("L2-M1", "L2", "A4", "material,angle,XPD_ex", "material x angle branch", "material dependence", "L2-M1__A4_material_angle_xpd_early_ex.png"),
        PlotSpec("L2-M2", "L2", "A4", "|XPD_ex|", "abs XPD by material", "toward 0dB/dispersion", "L2-M2__A4_abs_xpd_ex_by_material.png"),
        PlotSpec("L2-M3", "L2", "A4", "variance", "variance by material", "dispersion growth", "L2-M3__A4_variance_by_material.png"),
        PlotSpec("L2-S1", "L2", "A5", "base/stress L_pol", "paired L_pol", "stress-response", "L2-S1__A5_base_stress_lpol_paired.png"),
        PlotSpec("L2-S2", "L2", "A5", "base/stress late_ex", "late-ex compare", "late contamination", "L2-S2__A5_base_stress_xpd_late_ex.png"),
        PlotSpec("L2-S3", "L2", "A5", "base/stress DS", "tail variance/DS", "tail widening", "L2-S3__A5_base_stress_ds_tail.png"),
        PlotSpec("L3-1", "L3", "A3/A4/B", "EL,XPD_ex", "scatter + regression", "a1<0 monotonic tendency", "L3-1__el_vs_xpd_ex_scatter_regression.png"),
        PlotSpec("L3-2", "L3", "A3/A4/B", "EL bins", "conditional boxplot", "nonparametric monotonicity", "L3-2__el_bin_conditional_box.png"),
        PlotSpec("L3-3", "L3", "stage1 fit", "EL,fitted mean", "residualized effect", "mu(U) visualization", "L3-3__residualized_effect_fit_ci.png"),
        # R
        PlotSpec("R1-1", "R1", "B1/B2/B3", "x,y,XPD_early_ex", "heatmap facet", "spatial pattern", "R1-1__B123_heatmap_xpd_early_ex.png"),
        PlotSpec("R1-2", "R1", "B1/B2/B3", "x,y,rho_early", "heatmap facet", "cross contamination map", "R1-2__B123_heatmap_rho_early.png"),
        PlotSpec("R1-3", "R1", "B1/B2/B3", "x,y,L_pol", "heatmap facet", "early-late structure map", "R1-3__B123_heatmap_lpol.png"),
        PlotSpec("R1-4", "R1", "B pooled", "LOS/NLOS grouped metrics", "grouped CDF/box", "group difference", "R1-4__los_nlos_grouped_metrics.png"),
        PlotSpec("R2-1", "R2", "B pooled", "XPD_early_ex,DS", "XPD vs DS", "useful/risky relation", "R2-1__xpd_early_ex_vs_ds_B123.png"),
        PlotSpec("R2-2", "R2", "B pooled", "rho_early,DS", "rho vs DS", "contamination relation", "R2-2__rho_early_vs_ds_B123.png"),
        PlotSpec("R2-3", "R2", "B pooled", "XPD,DS,L_pol", "quadrant plot", "useful vs risky regions", "R2-3__quadrant_useful_vs_risky.png"),
        PlotSpec("R2-4", "R2", "B pooled", "early_fraction,XPD", "early fraction vs XPD", "early concentration link", "R2-4__early_fraction_vs_xpd_early_ex.png"),
        # P
        PlotSpec("P1-1", "P1", "A3/A4/B", "observed,predicted", "CDF overlay", "conditional>constant", "P1-1__observed_vs_predicted_cdf_overlay.png"),
        PlotSpec("P1-2", "P1", "A3/A4/B", "bin medians", "pred vs obs bin medians", "condition-wise fit", "P1-2__predicted_vs_observed_bin_median.png"),
        PlotSpec("P1-3", "P1", "A3/A4/B", "ranks", "rank agreement scatter", "Spearman advantage", "P1-3__rank_agreement_scatter.png"),
        PlotSpec("P1-4", "P1", "A3/A4/B", "residual", "residual vs EL", "reduced EL bias", "P1-4__residual_vs_el.png"),
        PlotSpec("P2-1", "P2", "full vs minimal", "effect-size set", "effect-size comparison", "minimal reproducibility", "P2-1__full_vs_minimal_effect_size.png"),
        PlotSpec("P2-2", "P2", "subsampling", "coefficients", "coefficient stability", "sign/scale stability", "P2-2__coefficient_stability_subsampling.png"),
        PlotSpec("P2-3", "P2", "full vs minimal", "CDF", "CDF overlay", "distribution reproducibility", "P2-3__full_vs_minimal_cdf_overlay.png"),
    ]


def _as_float_list(v: Any) -> list[float]:
    obj = _parse_jsonish(v)
    if not isinstance(obj, (list, tuple, np.ndarray)):
        return []
    out: list[float] = []
    for it in obj:
        fv = _to_float(it)
        if np.isfinite(fv):
            out.append(float(fv))
    return out


def _ecdf_rows(
    values: list[float] | np.ndarray,
    *,
    plot_id: str,
    series: str,
    scenario_id: str = "",
    case_id: str = "",
    link_id: str = "",
) -> list[dict[str, Any]]:
    x = np.sort(_finite(values))
    if len(x) == 0:
        return []
    y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
    rows: list[dict[str, Any]] = []
    for xv, yv in zip(x.tolist(), y.tolist()):
        rows.append(
            {
                "plot_id": plot_id,
                "series": series,
                "scenario_id": scenario_id,
                "case_id": case_id,
                "link_id": link_id,
                "x": float(xv),
                "y": float(yv),
                "data": float(xv),
                "meta": "",
            }
        )
    return rows


def _plot_data_fieldnames() -> list[str]:
    return ["plot_id", "series", "scenario_id", "case_id", "link_id", "x", "y", "data", "meta"]


def _pdp_rows_from_run(run_dir: Path, scenario_id: str, plot_id: str, max_files: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not run_dir.exists():
        return rows
    files = sorted(run_dir.glob(f"pdp_{scenario_id}_*.npz"))[: max(1, int(max_files))]
    for npz in files:
        stem = npz.stem
        parts = stem.split("_", 2)
        case_id = parts[2] if len(parts) >= 3 else ""
        z = np.load(npz)
        d = np.asarray(z["delay_tau_s"], dtype=float) * 1e9
        pco = _safe_log10(np.asarray(z["P_co"], dtype=float))
        pcr = _safe_log10(np.asarray(z["P_cross"], dtype=float))
        for xv, yv in zip(d.tolist(), pco.tolist()):
            rows.append(
                {
                    "plot_id": plot_id,
                    "series": "P_co_db",
                    "scenario_id": scenario_id,
                    "case_id": case_id,
                    "link_id": f"{scenario_id}_{case_id}",
                    "x": float(xv),
                    "y": float(yv),
                    "data": float(yv),
                    "meta": "delay_ns",
                }
            )
        for xv, yv in zip(d.tolist(), pcr.tolist()):
            rows.append(
                {
                    "plot_id": plot_id,
                    "series": "P_cross_db",
                    "scenario_id": scenario_id,
                    "case_id": case_id,
                    "link_id": f"{scenario_id}_{case_id}",
                    "x": float(xv),
                    "y": float(yv),
                    "data": float(yv),
                    "meta": "delay_ns",
                }
            )
        if "XPD_tau_db" in z.files:
            xt = np.asarray(z["XPD_tau_db"], dtype=float)
            for xv, yv in zip(d.tolist(), xt.tolist()):
                rows.append(
                    {
                        "plot_id": f"{plot_id}b",
                        "series": "XPD_tau_db",
                        "scenario_id": scenario_id,
                        "case_id": case_id,
                        "link_id": f"{scenario_id}_{case_id}",
                        "x": float(xv),
                        "y": float(yv),
                        "data": float(yv),
                        "meta": "delay_ns",
                    }
                )
    return rows


def _build_plot_data_rows(
    plot_id: str,
    link_rows: list[dict[str, str]],
    ray_rows: list[dict[str, str]],
    index_rows: list[dict[str, str]],
    tab_dir: Path,
    diag: dict[str, Any],
) -> list[dict[str, Any]]:
    by = _by_scenario(link_rows)
    rows: list[dict[str, Any]] = []

    def add_row(
        series: str,
        x: Any,
        y: Any,
        data: Any,
        *,
        scenario_id: str = "",
        case_id: str = "",
        link_id: str = "",
        meta: str = "",
    ) -> None:
        rows.append(
            {
                "plot_id": plot_id,
                "series": series,
                "scenario_id": scenario_id,
                "case_id": case_id,
                "link_id": link_id,
                "x": x,
                "y": y,
                "data": data,
                "meta": meta,
            }
        )

    # M
    if plot_id == "M1-1":
        run = _first_index_run(index_rows, "C0")
        if run:
            rows.extend(_pdp_rows_from_run(run, "C0", plot_id, max_files=12))
        return rows

    if plot_id == "M1-2":
        c0_link = {str(r.get("link_id", "")): r for r in by.get("C0", [])}
        per_link: dict[str, dict[str, float]] = {}
        for r in ray_rows:
            if str(r.get("scenario_id", "")) != "C0":
                continue
            link = str(r.get("link_id", ""))
            p = _to_float(r.get("P_lin"))
            los = _to_int(r.get("los_flag_ray"))
            if not np.isfinite(p):
                continue
            agg = per_link.setdefault(link, {"los": 0.0, "nlos": 0.0})
            if los == 1:
                agg["los"] += float(max(p, 0.0))
            else:
                agg["nlos"] += float(max(p, 0.0))
        for link, agg in per_link.items():
            los = max(float(agg["los"]), 1e-30)
            nlos = max(float(agg["nlos"]), 1e-30)
            cf = 10.0 * np.log10(nlos / los)
            lr = c0_link.get(link, {})
            yaw = _to_float(lr.get("yaw_deg"))
            add_row("C_floor_db", yaw, cf, cf, scenario_id="C0", case_id=str(lr.get("case_id", "")), link_id=link)
        return rows

    if plot_id in {"M1-3", "M1-4", "M1-6"}:
        for r in by.get("C0", []):
            x = _to_float(r.get("d_m")) if plot_id in {"M1-3", "M1-6"} else _to_float(r.get("yaw_deg"))
            y = _to_float(r.get("XPD_early_db"))
            series = "XPD_floor_db"
            if plot_id == "M1-6":
                series = f"d={_to_float(r.get('d_m')):.2f},yaw={_to_float(r.get('yaw_deg')):.1f}"
            add_row(
                series,
                x,
                y,
                y,
                scenario_id="C0",
                case_id=str(r.get("case_id", "")),
                link_id=str(r.get("link_id", "")),
            )
        return rows

    if plot_id == "M1-5":
        c0 = by.get("C0", [])
        if c0:
            f = _as_float_list(c0[0].get("xpd_floor_freq_hz"))
            xpd = _as_float_list(c0[0].get("xpd_floor_curve_db"))
            n = min(len(f), len(xpd))
            for i in range(n):
                add_row("XPD_floor_curve_db", f[i], xpd[i], xpd[i], scenario_id="C0", meta="freq_hz")
        return rows

    if plot_id == "M1-7":
        c = dict(diag.get("C_effect_vs_floor", {}))
        for k in ["floor_delta_db", "repeat_delta_db", "delta_ref_db"]:
            v = _to_float(c.get(k))
            add_row("uncertainty_budget_db", k, v, v, scenario_id="C0")
        return rows

    if plot_id == "M2-1":
        for r in link_rows:
            sc = str(r.get("scenario_id", ""))
            if sc == "C0":
                continue
            e = _to_float(r.get("XPD_early_excess_db"))
            l = _to_float(r.get("XPD_late_excess_db"))
            add_row("XPD_early_excess_db", sc, e, e, scenario_id=sc, case_id=str(r.get("case_id", "")), link_id=str(r.get("link_id", "")))
            add_row("XPD_late_excess_db", sc, l, l, scenario_id=sc, case_id=str(r.get("case_id", "")), link_id=str(r.get("link_id", "")))
        return rows

    if plot_id == "M2-2":
        dref = abs(_to_float(dict(diag.get("C_effect_vs_floor", {})).get("delta_ref_db")))
        for sc, rr in by.items():
            if sc == "C0":
                continue
            ev = _finite([_to_float(r.get("XPD_early_excess_db")) for r in rr])
            lv = _finite([_to_float(r.get("XPD_late_excess_db")) for r in rr])
            er = float(np.mean(np.abs(ev) > dref)) if len(ev) else np.nan
            lr = float(np.mean(np.abs(lv) > dref)) if len(lv) else np.nan
            add_row("early_exceed_rate", sc, er, er, scenario_id=sc, meta=f"delta_ref_db={dref:.4f}")
            add_row("late_exceed_rate", sc, lr, lr, scenario_id=sc, meta=f"delta_ref_db={dref:.4f}")
        return rows

    if plot_id == "M2-3":
        c0 = _finite([_to_float(r.get("XPD_early_db")) for r in by.get("C0", [])])
        rows.extend(_ecdf_rows(c0, plot_id=plot_id, series="C0_floor", scenario_id="C0"))
        indoor = [r for r in link_rows if str(r.get("scenario_id", "")) != "C0"]
        xe = _finite([_to_float(r.get("XPD_early_excess_db")) for r in indoor])
        rows.extend(_ecdf_rows(xe, plot_id=plot_id, series="indoor_early_excess"))
        return rows

    # G
    if plot_id == "G1-1":
        run = _first_index_run(index_rows, "A2")
        if run:
            rows.extend(_pdp_rows_from_run(run, "A2", plot_id, max_files=1))
        return rows

    if plot_id == "G1-1b":
        run = _first_index_run(index_rows, "A2")
        if run:
            rr = _pdp_rows_from_run(run, "A2", "G1-1", max_files=1)
            rows.extend([r for r in rr if str(r.get("plot_id", "")) == "G1-1b"])
        return rows

    if plot_id == "G1-2":
        for r in by.get("A2", []):
            y = _to_float(r.get("XPD_early_excess_db"))
            add_row("XPD_early_excess_db", str(r.get("case_id", "")), y, y, scenario_id="A2", case_id=str(r.get("case_id", "")), link_id=str(r.get("link_id", "")))
        return rows

    if plot_id == "G1-3":
        dom = _dominant_incidence_by_link(ray_rows, "A2")
        for r in by.get("A2", []):
            lk = str(r.get("link_id", ""))
            ang = float(dom.get(lk, np.nan))
            y = _to_float(r.get("rho_early_db"))
            add_row("rho_early_db", ang, y, y, scenario_id="A2", case_id=str(r.get("case_id", "")), link_id=lk)
        return rows

    if plot_id == "G2-1":
        run = _first_index_run(index_rows, "A3")
        if run:
            rows.extend(_pdp_rows_from_run(run, "A3", plot_id, max_files=1))
        return rows

    if plot_id in {"G2-2", "G2-3"}:
        p = tab_dir / "A3_target_window_sign.csv"
        if p.exists():
            for r in _read_csv(p):
                sc = str(r.get("scenario", ""))
                if plot_id == "G2-2" and sc != "A3":
                    continue
                metric = str(r.get("sign_metric_for_status", "excess")).lower()
                if metric == "raw":
                    y = _to_float(r.get("median_xpd_target_raw_db"))
                else:
                    y = _to_float(r.get("median_xpd_target_ex_db"))
                hit = _to_float(r.get("expected_sign_hit_rate"))
                add_row(
                    "median_xpd_target_raw_db" if metric == "raw" else "median_xpd_target_ex_db",
                    sc,
                    y,
                    y,
                    scenario_id=sc,
                    meta=f"sign_metric={metric};expected_sign_hit_rate={hit:.4f}",
                )
        return rows

    if plot_id == "G2-4":
        b = dict(diag.get("B_time_resolution", {}))
        for k in ["A2_target_in_Wearly_rate", "A3_target_in_Wearly_rate"]:
            v = _to_float(b.get(k))
            add_row("target_in_Wearly_rate", k, v, v, scenario_id="A3")
        return rows

    if plot_id == "G3-1":
        for sc in ["A2", "A3"]:
            for r in by.get(sc, []):
                y = _to_float(r.get("XPD_early_excess_db"))
                add_row("XPD_early_excess_db", sc, y, y, scenario_id=sc, case_id=str(r.get("case_id", "")), link_id=str(r.get("link_id", "")))
        return rows

    if plot_id == "G3-2":
        for sc in ["A2", "A3", "A4", "A5"]:
            for r in by.get(sc, []):
                x = _to_float(r.get("EL_proxy_db"))
                y = abs(_to_float(r.get("XPD_early_excess_db")))
                add_row("abs_XPD_early_excess_db", x, y, y, scenario_id=sc, case_id=str(r.get("case_id", "")), link_id=str(r.get("link_id", "")))
        return rows

    if plot_id == "G3-3":
        for sc in ["A2", "A3", "A4", "A5"]:
            vals = _finite([_to_float(r.get("XPD_early_excess_db")) for r in by.get(sc, [])])
            if len(vals):
                v = float(np.var(vals, ddof=1)) if len(vals) > 1 else 0.0
                add_row("var_XPD_early_excess_db", sc, v, v, scenario_id=sc)
        return rows

    # L
    if plot_id == "L1-1":
        for r in link_rows:
            sc = str(r.get("scenario_id", ""))
            if sc == "C0":
                continue
            e = _to_float(r.get("XPD_early_excess_db"))
            l = _to_float(r.get("XPD_late_excess_db"))
            lk = str(r.get("link_id", ""))
            cid = str(r.get("case_id", ""))
            add_row("early", "early", e, e, scenario_id=sc, case_id=cid, link_id=lk)
            add_row("late", "late", l, l, scenario_id=sc, case_id=cid, link_id=lk)
        return rows

    if plot_id == "L1-2":
        for r in link_rows:
            sc = str(r.get("scenario_id", ""))
            if sc == "C0":
                continue
            y = _to_float(r.get("L_pol_db"))
            add_row("L_pol_db", sc, y, y, scenario_id=sc, case_id=str(r.get("case_id", "")), link_id=str(r.get("link_id", "")))
        return rows

    if plot_id in {"L2-M1", "L2-M2", "L2-M3"}:
        a4 = by.get("A4", [])
        if plot_id in {"L2-M1", "L2-M2"}:
            dom = _dominant_incidence_by_link(ray_rows, "A4")
            for r in a4:
                mat = str(r.get("material_class", "NA"))
                lk = str(r.get("link_id", ""))
                ang = float(dom.get(lk, np.nan))
                val = _to_float(r.get("XPD_early_excess_db"))
                if plot_id == "L2-M2":
                    val = abs(val)
                add_row(
                    "XPD_early_excess_db" if plot_id == "L2-M1" else "abs_XPD_early_excess_db",
                    mat,
                    val,
                    val,
                    scenario_id="A4",
                    case_id=str(r.get("case_id", "")),
                    link_id=lk,
                    meta=f"incidence_deg={ang:.4f}",
                )
        else:
            mats = sorted({str(r.get("material_class", "NA")) for r in a4})
            for m in mats:
                vals = _finite([_to_float(r.get("XPD_early_excess_db")) for r in a4 if str(r.get("material_class", "NA")) == m])
                vv = float(np.var(vals, ddof=1)) if len(vals) > 1 else (0.0 if len(vals) == 1 else np.nan)
                add_row("var_XPD_early_excess_db", m, vv, vv, scenario_id="A4")
        return rows

    if plot_id in {"L2-S1", "L2-S2", "L2-S3"}:
        a5 = by.get("A5", [])
        for r in a5:
            cond = "stress" if (_to_int(r.get("roughness_flag")) == 1 or _to_int(r.get("human_flag")) == 1 or str(r.get("stress_mode", "")).lower() not in {"", "none"}) else "base"
            if plot_id == "L2-S1":
                val = _to_float(r.get("L_pol_db"))
                series = "L_pol_db"
            elif plot_id == "L2-S2":
                val = _to_float(r.get("XPD_late_excess_db"))
                series = "XPD_late_excess_db"
            else:
                val = _to_float(r.get("delay_spread_rms_s")) * 1e9
                series = "delay_spread_rms_ns"
            add_row(series, cond, val, val, scenario_id="A5", case_id=str(r.get("case_id", "")), link_id=str(r.get("link_id", "")))
        return rows

    if plot_id in {"L3-1", "L3-2", "L3-3", "P1-1", "P1-2", "P1-3", "P1-4", "P2-1", "P2-2", "P2-3"}:
        subset = [r for r in link_rows if str(r.get("scenario_id", "")) in {"A3", "A4", "B1", "B2", "B3"}]
        x = np.asarray([_to_float(r.get("EL_proxy_db")) for r in subset], dtype=float)
        y = np.asarray([_to_float(r.get("XPD_early_excess_db")) for r in subset], dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if len(x) >= 2:
            b1, b0 = np.polyfit(x, y, 1)
            y_lin = b1 * x + b0
            y_const = np.full_like(y, float(np.mean(y)))
        else:
            y_lin = np.asarray([], dtype=float)
            y_const = np.asarray([], dtype=float)

        if plot_id == "L3-1":
            for r in subset:
                xx = _to_float(r.get("EL_proxy_db"))
                yy = _to_float(r.get("XPD_early_excess_db"))
                add_row("XPD_early_excess_db", xx, yy, yy, scenario_id=str(r.get("scenario_id", "")), case_id=str(r.get("case_id", "")), link_id=str(r.get("link_id", "")))
            return rows
        if plot_id == "L3-2":
            if len(x):
                q1, q2 = np.percentile(x, [33, 66])
                bins = np.where(x <= q1, "low", np.where(x <= q2, "mid", "high"))
                for b in ["low", "mid", "high"]:
                    vv = y[bins == b]
                    for v in vv.tolist():
                        add_row("XPD_early_excess_db", b, v, v)
            return rows
        if plot_id == "L3-3":
            if len(x):
                xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 64)
                yy = b1 * xx + b0 if len(x) >= 2 else np.full_like(xx, np.nan)
                for xv, yv in zip(xx.tolist(), yy.tolist()):
                    add_row("fitted_mean", xv, yv, yv)
            return rows
        if plot_id == "P1-1":
            rows.extend(_ecdf_rows(y, plot_id=plot_id, series="observed"))
            rows.extend(_ecdf_rows(y_lin, plot_id=plot_id, series="conditional-linear"))
            rows.extend(_ecdf_rows(y_const, plot_id=plot_id, series="constant-baseline"))
            return rows
        if plot_id == "P1-2":
            if len(x):
                q1, q2 = np.percentile(x, [33, 66])
                bins = np.where(x <= q1, "low", np.where(x <= q2, "mid", "high"))
                for b in ["low", "mid", "high"]:
                    idx = bins == b
                    if np.any(idx):
                        add_row("observed", b, float(np.median(y[idx])), float(np.median(y[idx])))
                        add_row("conditional", b, float(np.median(y_lin[idx])), float(np.median(y_lin[idx])))
                        add_row("constant", b, float(np.median(y_const[idx])), float(np.median(y_const[idx])))
            return rows
        if plot_id == "P1-3":
            if len(y):
                r_obs = stats.rankdata(y)
                r_lin = stats.rankdata(y_lin)
                r_con = stats.rankdata(y_const)
                for xo, yl in zip(r_obs.tolist(), r_lin.tolist()):
                    add_row("conditional", xo, yl, yl)
                for xo, yc in zip(r_obs.tolist(), r_con.tolist()):
                    add_row("constant", xo, yc, yc)
            return rows
        if plot_id == "P1-4":
            if len(x):
                for xv, rv in zip(x.tolist(), (y - y_lin).tolist()):
                    add_row("conditional_residual", xv, rv, rv)
                for xv, rv in zip(x.tolist(), (y - y_const).tolist()):
                    add_row("constant_residual", xv, rv, rv)
            return rows

        # P2 family
        rng = np.random.default_rng(42)
        by_sc = _by_scenario(link_rows)
        minimal: list[dict[str, str]] = []
        for _s, rr in by_sc.items():
            if len(rr) <= 2:
                minimal.extend(rr)
                continue
            idx = np.arange(len(rr))
            take = max(1, len(rr) // 2)
            sel = rng.choice(idx, size=take, replace=False)
            minimal.extend([rr[int(i)] for i in sel])

        def _effect_dict(rows_in: list[dict[str, str]]) -> dict[str, float]:
            bys = _by_scenario(rows_in)
            c0 = _finite([_to_float(r.get("XPD_early_excess_db")) for r in bys.get("C0", [])])
            a2 = _finite([_to_float(r.get("XPD_early_excess_db")) for r in bys.get("A2", [])])
            a3 = _finite([_to_float(r.get("XPD_early_excess_db")) for r in bys.get("A3", [])])
            bpool = [r for r in rows_in if str(r.get("scenario_id", "")) in {"B1", "B2", "B3"}]
            los = _finite([_to_float(r.get("XPD_early_excess_db")) for r in bpool if _to_int(r.get("LOSflag")) == 1])
            nlos = _finite([_to_float(r.get("XPD_early_excess_db")) for r in bpool if _to_int(r.get("LOSflag")) == 0])
            sub = [r for r in rows_in if str(r.get("scenario_id", "")) in {"A3", "A4", "B1", "B2", "B3"}]
            xx = np.asarray([_to_float(r.get("EL_proxy_db")) for r in sub], dtype=float)
            yy = np.asarray([_to_float(r.get("XPD_early_excess_db")) for r in sub], dtype=float)
            mm = np.isfinite(xx) & np.isfinite(yy)
            rho = float(stats.spearmanr(xx[mm], yy[mm]).correlation) if np.sum(mm) >= 5 else np.nan
            return {
                "G1": float(np.nanmedian(a2) - np.nanmedian(c0)) if len(a2) and len(c0) else np.nan,
                "G2": float(np.nanmedian(a3) - np.nanmedian(a2)) if len(a3) and len(a2) else np.nan,
                "L3": rho,
                "R1": float(stats.wasserstein_distance(los, nlos)) if len(los) and len(nlos) else np.nan,
            }

        if plot_id == "P2-1":
            eff_full = _effect_dict(link_rows)
            eff_min = _effect_dict(minimal)
            for k in ["G1", "G2", "L3", "R1"]:
                add_row("full", k, eff_full.get(k), eff_full.get(k))
                add_row("minimal", k, eff_min.get(k), eff_min.get(k))
            return rows

        if plot_id == "P2-2":
            if len(x):
                b1s = []
                idx = np.arange(len(x), dtype=int)
                for i in range(200):
                    ii = rng.choice(idx, size=max(8, len(idx) // 2), replace=True)
                    xx = x[ii]
                    yy = y[ii]
                    mm = np.isfinite(xx) & np.isfinite(yy)
                    if np.sum(mm) >= 5 and len(np.unique(np.round(xx[mm], 6))) >= 2:
                        bb1, _bb0 = np.polyfit(xx[mm], yy[mm], 1)
                        b1s.append(float(bb1))
                for i, v in enumerate(b1s):
                    add_row("bootstrap_b1", i, v, v)
            return rows

        if plot_id == "P2-3":
            for key, series in [("XPD_early_excess_db", "XPD_early_ex"), ("L_pol_db", "L_pol")]:
                f = _finite([_to_float(r.get(key)) for r in link_rows])
                m2 = _finite([_to_float(r.get(key)) for r in minimal])
                rows.extend(_ecdf_rows(f, plot_id=plot_id, series=f"{series}_full"))
                rows.extend(_ecdf_rows(m2, plot_id=plot_id, series=f"{series}_minimal"))
            return rows

    # R
    if plot_id in {"R1-1", "R1-2", "R1-3", "R2-1", "R2-2", "R2-3", "R2-4", "R1-4"}:
        bpool = [r for r in link_rows if str(r.get("scenario_id", "")) in {"B1", "B2", "B3"}]
        if plot_id == "R1-1":
            key = "XPD_early_excess_db"
        elif plot_id == "R1-2":
            key = "rho_early_lin"
        elif plot_id == "R1-3":
            key = "L_pol_db"
        else:
            key = ""
        if plot_id in {"R1-1", "R1-2", "R1-3"}:
            for r in bpool:
                xv = _to_float(r.get("rx_x"))
                yv = _to_float(r.get("rx_y"))
                dv = _to_float(r.get(key))
                add_row(key, xv, yv, dv, scenario_id=str(r.get("scenario_id", "")), case_id=str(r.get("case_id", "")), link_id=str(r.get("link_id", "")))
            return rows
        if plot_id == "R1-4":
            for metric in ["XPD_early_excess_db", "rho_early_lin", "L_pol_db"]:
                for losv in [0, 1]:
                    vals = _finite([_to_float(r.get(metric)) for r in bpool if _to_int(r.get("LOSflag")) == losv])
                    rows.extend(_ecdf_rows(vals, plot_id=plot_id, series=f"{metric}_LOS{losv}"))
            return rows
        if plot_id == "R2-1":
            for r in bpool:
                x = _to_float(r.get("XPD_early_excess_db"))
                y = _to_float(r.get("delay_spread_rms_s")) * 1e9
                add_row("XPD_vs_DS", x, y, y, scenario_id=str(r.get("scenario_id", "")), case_id=str(r.get("case_id", "")), link_id=str(r.get("link_id", "")))
            return rows
        if plot_id == "R2-2":
            for r in bpool:
                x = _to_float(r.get("rho_early_lin"))
                y = _to_float(r.get("delay_spread_rms_s")) * 1e9
                add_row("rho_vs_DS", x, y, y, scenario_id=str(r.get("scenario_id", "")), case_id=str(r.get("case_id", "")), link_id=str(r.get("link_id", "")))
            return rows
        if plot_id == "R2-3":
            for r in bpool:
                x = _to_float(r.get("XPD_early_excess_db"))
                y = _to_float(r.get("delay_spread_rms_s")) * 1e9
                d = _to_float(r.get("L_pol_db"))
                add_row("quadrant", x, y, d, scenario_id=str(r.get("scenario_id", "")), case_id=str(r.get("case_id", "")), link_id=str(r.get("link_id", "")))
            return rows
        if plot_id == "R2-4":
            for r in bpool:
                x = _to_float(r.get("XPD_early_excess_db"))
                y = _to_float(r.get("early_energy_fraction"))
                add_row("early_fraction_vs_xpd", x, y, y, scenario_id=str(r.get("scenario_id", "")), case_id=str(r.get("case_id", "")), link_id=str(r.get("link_id", "")))
            return rows

    return rows


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Detailed proposition-plot mapping report generator")
    ap.add_argument("--run-group", required=True, help="analysis_report/out/<run_group>")
    ap.add_argument("--out-root", default="analysis_report/out", help="output base path")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_root).resolve() / str(args.run_group)
    tab_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    if not tab_dir.exists():
        raise SystemExit(f"Missing directory: {tab_dir}")

    link_rows = _read_csv(tab_dir / "intermediate_link_rows.csv")
    ray_rows = _read_csv(tab_dir / "intermediate_ray_rows.csv")
    index_rows = _read_csv(out_dir / "index.csv")
    pstat_rows = _read_csv(tab_dir / "intermediate_proposition_status.csv")
    diag = json.loads((tab_dir / "diagnostic_checks.json").read_text(encoding="utf-8"))
    pstatus = {str(r.get("proposition", "")): _status_to_pass_fail(str(r.get("status", ""))) for r in pstat_rows}
    # Backfill propositions that are not part of intermediate_proposition_status.csv (e.g., G3)
    pmap_path = tab_dir / "proposition_plot_mapping.csv"
    if pmap_path.exists():
        for r in _read_csv(pmap_path):
            pid = str(r.get("proposition", "")).strip()
            st = _status_to_pass_fail(str(r.get("pass_fail", "")).strip())
            if pid and st in {"PASS", "PARTIAL", "FAIL"}:
                pstatus[pid] = st

    detail_rows: list[dict[str, Any]] = []
    _make_m1_m2_plots(fig_dir, link_rows, ray_rows, index_rows, diag, detail_rows)
    _make_g_plots(fig_dir, link_rows, ray_rows, index_rows, tab_dir, diag, detail_rows)
    _make_l_plots(fig_dir, link_rows, ray_rows, detail_rows)
    _make_r_plots(fig_dir, link_rows, detail_rows)
    _make_p_plots(fig_dir, link_rows, detail_rows)
    made_lookup = {str(r.get("plot_id", "")): str(r.get("file", "")) for r in detail_rows}
    note_lookup = {str(r.get("plot_id", "")): str(r.get("note", "")) for r in detail_rows}
    plot_data_dir = tab_dir / "plot_data"
    plot_data_dir.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, Any]] = []
    for sp in _specs():
        fn = made_lookup.get(sp.plot_id, sp.expected_file)
        p = fig_dir / fn
        ready = p.exists()
        data_rows = _build_plot_data_rows(sp.plot_id, link_rows, ray_rows, index_rows, tab_dir, diag)
        data_fn = f"{sp.plot_id}__data.csv"
        _write_csv(plot_data_dir / data_fn, data_rows, _plot_data_fieldnames())
        rows_out.append(
            {
                "plot_id": sp.plot_id,
                "proposition": sp.proposition,
                "scenario": sp.scenario,
                "needed_data": sp.needed_data,
                "plot_desc": sp.plot_desc,
                "pass_rule": sp.pass_rule,
                "expected_file": fn,
                "plot_ready": "READY" if ready else "MISSING",
                "proposition_status": pstatus.get(sp.proposition, "FAIL"),
                "data_csv": f"tables/plot_data/{data_fn}",
                "notes": note_lookup.get(sp.plot_id, ""),
            }
        )

    out_csv = tab_dir / "proposition_plot_mapping_detailed.csv"
    _write_csv(
        out_csv,
        rows_out,
        [
            "plot_id",
            "proposition",
            "scenario",
            "needed_data",
            "plot_desc",
            "pass_rule",
            "proposition_status",
            "plot_ready",
            "expected_file",
            "data_csv",
            "notes",
        ],
    )

    # markdown
    out_md = out_dir / "proposition_plot_mapping_detailed_report.md"
    lines: list[str] = []
    lines.append(f"# Detailed Proposition-Experiment-Data-Plot Mapping ({args.run_group})")
    lines.append("")
    lines.append("## 0) 공통 플롯 규칙 요약")
    lines.append("- 공통 지표: XPD_floor, XPD_target_ex, XPD_early_ex, XPD_late_ex, rho_early, L_pol, DS, EL_proxy")
    lines.append("- window 규칙: W_floor(C0), W_target(A2/A3/A4), W_early/B, W_late")
    lines.append("- 시나리오 역할: C0=floor, A2=odd, A3=even-mechanism, A4=material, A5=stress-response, B1/B2/B3=real-space")
    lines.append("")
    total = len(rows_out)
    ready = sum(1 for r in rows_out if r["plot_ready"] == "READY")
    lines.append("## Summary")
    lines.append(f"- Detailed plots ready: {ready}/{total}")
    lines.append(f"- Proposition PASS: {sum(1 for r in rows_out if r['proposition_status']=='PASS')}/{total} (row-level reference)")
    lines.append(f"- Proposition PARTIAL: {sum(1 for r in rows_out if r['proposition_status']=='PARTIAL')}/{total}")
    lines.append(f"- Proposition FAIL: {sum(1 for r in rows_out if r['proposition_status']=='FAIL')}/{total}")
    lines.append("")
    lines.append("## Mapping Table")
    lines.append("")
    lines.append("| plot_id | 명제 | 시나리오(실험) | 필요한 데이터 | 플롯 | 데이터 CSV(x,y,data) | 통과 기준 | 명제 PASS/FAIL | 플롯 상태 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows_out:
        lines.append(
            f"| {r['plot_id']} | {r['proposition']} | {r['scenario']} | {r['needed_data']} | "
            f"[{r['expected_file']}](figures/{r['expected_file']}) | "
            f"[{Path(str(r['data_csv'])).name}]({r['data_csv']}) | {r['pass_rule']} | "
            f"{r['proposition_status']} | {r['plot_ready']} |"
        )
    lines.append("")
    lines.append("## Notes")
    for r in rows_out:
        if r["notes"]:
            lines.append(f"- {r['plot_id']}: {r['notes']}")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {out_csv}")
    print(f"[OK] {out_md}")
    print(f"[INFO] ready {ready}/{total}")


if __name__ == "__main__":
    main()

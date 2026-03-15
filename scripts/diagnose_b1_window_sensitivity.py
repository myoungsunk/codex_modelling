"""B1 early-window sensitivity diagnostics for Te={2,3,5} ns (configurable)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-18


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _window_info(row: dict[str, Any]) -> dict[str, float]:
    try:
        w = json.loads(str(row.get("window", "{}") or "{}"))
    except Exception:
        w = {}
    return {
        "tau0_s": _safe_float(w.get("tau0_s", np.nan)),
        "Tmax_s": _safe_float(w.get("Tmax_s", np.nan)),
        "Te_s": _safe_float(w.get("Te_s", np.nan)),
    }


def _compute_metrics(tau_s: np.ndarray, pco: np.ndarray, pcr: np.ndarray, tau0_s: float, te_s: float, tmax_s: float) -> dict[str, float]:
    pco = np.asarray(pco, dtype=float)
    pcr = np.asarray(pcr, dtype=float)
    tau = np.asarray(tau_s, dtype=float)
    pt = pco + pcr
    early = (tau >= tau0_s) & (tau < tau0_s + te_s)
    late = (tau >= tau0_s + te_s) & (tau <= tau0_s + tmax_s)
    if not np.any(early) and len(tau) > 0:
        j = int(np.argmin(np.abs(tau - tau0_s)))
        early[j] = True
    e_co = float(np.sum(pco[early]))
    e_cr = float(np.sum(pcr[early]))
    e_tot = float(np.sum(pt[early | late]))
    e_early_tot = float(np.sum(pt[early]))
    xpd = float(10.0 * np.log10((e_co + EPS) / (e_cr + EPS)))
    rho_lin = float((e_cr + EPS) / (e_co + EPS))
    rho_db = float(10.0 * np.log10(rho_lin + EPS))
    ef = float((e_early_tot + EPS) / (e_tot + EPS))
    return {
        "XPD_early_db": xpd,
        "rho_early_lin": rho_lin,
        "rho_early_db": rho_db,
        "early_energy_fraction": ef,
        "early_count": int(np.sum(early)),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 3:
        return float("nan")
    if float(np.nanstd(a[m])) == 0.0 or float(np.nanstd(b[m])) == 0.0:
        return float("nan")
    return float(stats.spearmanr(a[m], b[m]).correlation)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--link-metrics-csv", required=True, type=str)
    ap.add_argument("--pdp-dir", required=True, type=str)
    ap.add_argument("--te-list-ns", type=str, default="2,3,5")
    ap.add_argument("--baseline-te-ns", type=float, default=3.0)
    ap.add_argument("--out-dir", required=True, type=str)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdp_dir = Path(args.pdp_dir)

    te_list = [float(x.strip()) for x in str(args.te_list_ns).split(",") if x.strip()]
    te_list = sorted(set(te_list))
    base_te = float(args.baseline_te_ns)
    if base_te not in te_list:
        te_list.append(base_te)
        te_list = sorted(set(te_list))

    rows = _read_csv(args.link_metrics_csv)
    by_link = {str(r.get("link_id", "")): r for r in rows}

    out_rows: list[dict[str, Any]] = []
    for lid, r in by_link.items():
        npz = pdp_dir / f"pdp_{lid}.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        tau = np.asarray(d["delay_tau_s"], dtype=float)
        pco = np.asarray(d["P_co"], dtype=float)
        pcr = np.asarray(d["P_cross"], dtype=float)
        w = _window_info(r)
        tau0 = w["tau0_s"]
        tmax = w["Tmax_s"]
        if not np.isfinite(tau0):
            j = int(np.argmax(pco + pcr)) if len(tau) else 0
            tau0 = float(tau[j]) if len(tau) else 0.0
        if not np.isfinite(tmax):
            tmax = float(np.nanmax(tau) - tau0) if len(tau) else 0.0
        for te_ns in te_list:
            m = _compute_metrics(tau, pco, pcr, tau0, float(te_ns) * 1e-9, tmax)
            out_rows.append(
                {
                    "link_id": lid,
                    "scenario_id": str(r.get("scenario_id", "")),
                    "d_m": _safe_float(r.get("d_m", np.nan)),
                    "Te_ns": float(te_ns),
                    **m,
                }
            )

    out_csv = out_dir / "b1_window_sensitivity_metrics.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        cols = list(out_rows[0].keys()) if out_rows else ["link_id", "Te_ns"]
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in out_rows:
            wr.writerow(r)

    # rank-stability summary against baseline Te
    metrics = ["XPD_early_db", "rho_early_db", "early_energy_fraction"]
    summary: dict[str, Any] = {
        "te_list_ns": te_list,
        "baseline_te_ns": base_te,
        "n_links": len(by_link),
        "rank_stability": {},
        "plots": {},
    }

    # reshape by link
    links = sorted(set(r["link_id"] for r in out_rows))
    for mk in metrics:
        summary["rank_stability"][mk] = {}
        base_vals = []
        idx = []
        for lid in links:
            rr = [x for x in out_rows if x["link_id"] == lid and abs(float(x["Te_ns"]) - base_te) < 1e-9]
            if rr:
                base_vals.append(_safe_float(rr[0].get(mk, np.nan)))
                idx.append(lid)
        base = np.asarray(base_vals, dtype=float)
        for te_ns in te_list:
            if abs(te_ns - base_te) < 1e-9:
                continue
            y = []
            x = []
            for lid, b in zip(idx, base):
                rr = [z for z in out_rows if z["link_id"] == lid and abs(float(z["Te_ns"]) - te_ns) < 1e-9]
                if not rr:
                    continue
                x.append(float(b))
                y.append(_safe_float(rr[0].get(mk, np.nan)))
            xa = np.asarray(x, dtype=float)
            ya = np.asarray(y, dtype=float)
            corr = _spearman(xa, ya)
            summary["rank_stability"][mk][f"{base_te:.1f}_vs_{te_ns:.1f}"] = corr

            fig, ax = plt.subplots(figsize=(6, 4))
            m = np.isfinite(xa) & np.isfinite(ya)
            ax.scatter(xa[m], ya[m], alpha=0.8)
            lo = float(np.nanmin(np.r_[xa[m], ya[m]])) if np.any(m) else 0.0
            hi = float(np.nanmax(np.r_[xa[m], ya[m]])) if np.any(m) else 1.0
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
            ax.set_title(f"{mk}: Te={base_te:.1f} vs {te_ns:.1f} ns")
            ax.set_xlabel(f"Te={base_te:.1f} ns")
            ax.set_ylabel(f"Te={te_ns:.1f} ns")
            ax.grid(True, alpha=0.3)
            p = out_dir / f"b1_{mk}_te_{int(base_te)}_vs_{int(te_ns)}.png"
            fig.tight_layout()
            fig.savefig(p, dpi=180)
            plt.close(fig)
            summary["plots"][f"{mk}_{base_te:.1f}_vs_{te_ns:.1f}"] = str(p)

    out_json = out_dir / "b1_window_sensitivity_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    out_md = out_dir / "b1_window_sensitivity_report.md"
    lines = [
        "# B1 Window Sensitivity Report",
        "",
        f"- input_csv: {args.link_metrics_csv}",
        f"- pdp_dir: {args.pdp_dir}",
        f"- Te_list_ns: {te_list}",
        f"- baseline_te_ns: {base_te}",
        f"- n_links: {len(by_link)}",
        "",
        "## Rank Stability (Spearman)",
        "",
    ]
    for mk in metrics:
        lines.append(f"- {mk}:")
        for k, v in summary["rank_stability"].get(mk, {}).items():
            lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append(f"- metrics_csv: {out_csv}")
    lines.append(f"- summary_json: {out_json}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out_md)
    print(out_json)


if __name__ == "__main__":
    main()

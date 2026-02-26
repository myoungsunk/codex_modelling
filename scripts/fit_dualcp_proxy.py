"""Fit conditional dual-CP proxy models and produce bridge report."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.conditional_proxy import fit_proxy_model, predict_distribution


def _maybe_num(v: str) -> Any:
    s = str(v).strip()
    if s == "":
        return ""
    try:
        x = float(s)
        return int(x) if x.is_integer() else x
    except Exception:
        return s


def _load_csv(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        rows = []
        for r in rd:
            rows.append({k: _maybe_num(v) for k, v in r.items()})
    return rows


def _merge_rows(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    keys: tuple[str, str] = ("scenario_id", "case_id"),
) -> list[dict[str, Any]]:
    idx: dict[tuple[str, str], dict[str, Any]] = {}
    for r in right:
        k = (str(r.get(keys[0], "")), str(r.get(keys[1], "")))
        idx[k] = r
    out = []
    for r in left:
        k = (str(r.get(keys[0], "")), str(r.get(keys[1], "")))
        rr = idx.get(k, {})
        merged = dict(r)
        for kk, vv in rr.items():
            if kk not in merged:
                merged[kk] = vv
        out.append(merged)
    return out


def _finite_vals(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    arr = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)
    return arr[np.isfinite(arr)]


def _quantile_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    aa = aa[np.isfinite(aa)]
    bb = bb[np.isfinite(bb)]
    if len(aa) == 0 or len(bb) == 0:
        return float("nan")
    n = int(min(len(aa), len(bb)))
    q = np.linspace(0.0, 1.0, n)
    qa = np.quantile(aa, q)
    qb = np.quantile(bb, q)
    corr = stats.spearmanr(qa, qb)
    return float(corr.correlation) if np.isfinite(float(corr.correlation)) else float("nan")


def _plot_z_summary(
    rows: list[dict[str, Any]],
    z_key: str,
    model: dict[str, Any],
    out_dir: Path,
    u_keys: list[str],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    z = _finite_vals(rows, z_key)
    mu = []
    sigma = []
    for r in rows:
        m, s, _ = predict_distribution(model, r)
        mu.append(float(m))
        sigma.append(float(s))
    mu_a = np.asarray(mu, dtype=float)
    sigma_a = np.asarray(sigma, dtype=float)
    mu_a = mu_a[np.isfinite(mu_a)]
    sigma_a = sigma_a[np.isfinite(sigma_a)]

    rng = np.random.default_rng(0)
    z_synth = rng.normal(mu_a, np.maximum(sigma_a, 1e-6)) if len(mu_a) else np.asarray([], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(z):
        xs = np.sort(z)
        ys = np.arange(1, len(xs) + 1, dtype=float) / float(len(xs))
        ax.plot(xs, ys, label="observed", linewidth=1.8)
    if len(z_synth):
        xs2 = np.sort(z_synth)
        ys2 = np.arange(1, len(xs2) + 1, dtype=float) / float(len(xs2))
        ax.plot(xs2, ys2, label="proxy-sampled", linewidth=1.8)
    ax.set_title(f"CDF: {z_key}")
    ax.set_xlabel(z_key)
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = out_dir / f"{z_key}_cdf.png"
    fig.savefig(p1, dpi=180)
    plt.close(fig)

    cat_key = None
    for uk in u_keys:
        vals = [str(r.get(uk, "NA")) for r in rows]
        uniq = sorted(set(vals))
        if 1 < len(uniq) <= 8:
            cat_key = uk
            break
    if cat_key is None:
        cat_key = "scenario_id"
    groups: dict[str, list[float]] = {}
    for r in rows:
        zv = float(r.get(z_key, np.nan))
        if not np.isfinite(zv):
            continue
        groups.setdefault(str(r.get(cat_key, "NA")), []).append(zv)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    labels = sorted(groups.keys())
    arr = [np.asarray(groups[k], dtype=float) for k in labels]
    if arr:
        ax2.boxplot(arr, labels=labels, showfliers=False)
    ax2.set_title(f"Boxplot: {z_key} by {cat_key}")
    ax2.set_xlabel(cat_key)
    ax2.set_ylabel(z_key)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = out_dir / f"{z_key}_boxplot.png"
    fig2.savefig(p2, dpi=180)
    plt.close(fig2)
    return {"cdf_png": str(p1), "boxplot_png": str(p2), "group_key": str(cat_key)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", required=True, type=str)
    parser.add_argument("--calibration-json", type=str, default=None)
    parser.add_argument("--rt-features-csv", type=str, default=None)
    parser.add_argument("--out-model-json", type=str, default="outputs/proxy_model.json")
    parser.add_argument("--out-report", type=str, default="outputs/proxy_report.md")
    parser.add_argument(
        "--z-keys",
        type=str,
        default="xpd_early_excess_db,xpd_late_excess_db,l_pol_db,rho_early_db",
    )
    parser.add_argument(
        "--u-keys",
        type=str,
        default=(
            "los_blocked,material,scatter_stress,distance_d_m,pathloss_proxy_db,"
            "delay_bin,parity,incidence_angle_bin,excess_loss_proxy_db,bounce_count"
        ),
    )
    args = parser.parse_args()

    rows = _load_csv(args.metrics_csv)
    if args.rt_features_csv:
        rt_rows = _load_csv(args.rt_features_csv)
        rows = _merge_rows(rows, rt_rows)
        bridge_rows = rt_rows
    else:
        bridge_rows = list(rows)

    z_keys = [x.strip() for x in str(args.z_keys).split(",") if x.strip()]
    u_keys = [x.strip() for x in str(args.u_keys).split(",") if x.strip()]

    # Provide delay_bin for early/late variables.
    for r in rows:
        if "delay_bin" not in r or str(r.get("delay_bin", "")).strip() == "":
            r["delay_bin"] = "NA"
    for r in bridge_rows:
        if "delay_bin" not in r or str(r.get("delay_bin", "")).strip() == "":
            r["delay_bin"] = "NA"

    models: dict[str, Any] = {}
    bridge_eval: dict[str, Any] = {}
    plots: dict[str, Any] = {}
    report_lines = ["# Dual-CP Proxy Report", ""]
    report_lines.append(f"- generated_at: {datetime.now(timezone.utc).isoformat()}")
    report_lines.append(f"- input_metrics_csv: {args.metrics_csv}")
    report_lines.append(f"- input_rt_features_csv: {args.rt_features_csv or ''}")
    report_lines.append(f"- calibration_json: {args.calibration_json or ''}")
    report_lines.append(f"- u_keys: {u_keys}")
    report_lines.append("")

    plot_dir = Path(args.out_report).with_suffix("").with_name(Path(args.out_report).stem + "_plots")
    for zk in z_keys:
        rows_z: list[dict[str, Any]] = []
        for r in rows:
            rv = dict(r)
            if "xpd_early" in zk:
                rv["delay_bin"] = "early"
            elif "xpd_late" in zk:
                rv["delay_bin"] = "late"
            rows_z.append(rv)
        zvals = _finite_vals(rows_z, zk)
        if len(zvals) == 0:
            continue
        model = fit_proxy_model(rows_z, z_key=zk, u_keys=u_keys, method="binned+regression", seed=0)
        models[zk] = model

        mu_pred = []
        sig_pred = []
        for r in bridge_rows:
            rr = dict(r)
            if "xpd_early" in zk:
                rr["delay_bin"] = "early"
            elif "xpd_late" in zk:
                rr["delay_bin"] = "late"
            mu, sig, _ = predict_distribution(model, rr)
            mu_pred.append(float(mu))
            sig_pred.append(float(sig))
        mu_pred_a = np.asarray(mu_pred, dtype=float)
        mu_pred_a = mu_pred_a[np.isfinite(mu_pred_a)]
        wd = float(stats.wasserstein_distance(zvals, mu_pred_a)) if len(mu_pred_a) else float("nan")
        rank_corr = _quantile_rank_corr(zvals, mu_pred_a)

        gof = model.get("gof", {}) or {}
        bridge_eval[zk] = {
            "n_observed": int(len(zvals)),
            "n_pred": int(len(mu_pred_a)),
            "ks_p": float(gof.get("ks_p", np.nan)),
            "qq_r": float(gof.get("qq_r", np.nan)),
            "wasserstein": wd,
            "rank_corr_quantile": rank_corr,
        }

        p = _plot_z_summary(rows_z, z_key=zk, model=model, out_dir=plot_dir, u_keys=u_keys)
        plots[zk] = p

        # Effect size: range of group mean by selected group key.
        gk = str(p.get("group_key", "scenario_id"))
        gm: dict[str, list[float]] = {}
        for r in rows_z:
            zf = float(r.get(zk, np.nan))
            if not np.isfinite(zf):
                continue
            gm.setdefault(str(r.get(gk, "NA")), []).append(zf)
        means = [float(np.mean(np.asarray(v, dtype=float))) for v in gm.values() if len(v)]
        effect = float(max(means) - min(means)) if len(means) >= 2 else float("nan")

        report_lines.append(f"## {zk}")
        report_lines.append("")
        report_lines.append(f"- n: {len(zvals)}")
        report_lines.append(f"- gof_ks_p: {float(gof.get('ks_p', np.nan)):.4f}")
        report_lines.append(f"- gof_qq_r: {float(gof.get('qq_r', np.nan)):.4f}")
        report_lines.append(f"- gof_wasserstein_stdresid: {float(gof.get('wasserstein', np.nan)):.4f}")
        report_lines.append(f"- bridge_wasserstein(observed_vs_pred_mu): {wd:.4f}")
        report_lines.append(f"- bridge_rank_corr_quantile: {rank_corr:.4f}")
        report_lines.append(f"- effect_size_range_by_{gk}: {effect:.4f}")
        report_lines.append(f"- cdf_plot: {p['cdf_png']}")
        report_lines.append(f"- boxplot: {p['boxplot_png']}")
        report_lines.append("")

    bundle = {
        "schema_version": "dualcp_proxy_bundle_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "metrics_csv": str(args.metrics_csv),
            "rt_features_csv": str(args.rt_features_csv or ""),
            "calibration_json": str(args.calibration_json or ""),
        },
        "u_keys": u_keys,
        "z_keys": list(models.keys()),
        "models": models,
        "bridge_eval": bridge_eval,
        "plots": plots,
    }

    out_model = Path(args.out_model_json)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(report_lines), encoding="utf-8")
    print(str(out_model))
    print(str(out_report))


if __name__ == "__main__":
    main()

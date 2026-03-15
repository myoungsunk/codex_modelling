"""Detailed C2 (effect size) + D2 (identifiability) diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _metric_vals(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    x = np.asarray([_safe_float(r.get(key, np.nan)) for r in rows], dtype=float)
    return x[np.isfinite(x)]


def _effect_summary(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if len(a) == 0 or len(b) == 0:
        return {"delta_median": float("nan"), "var_ratio": float("nan"), "q10_shift": float("nan"), "ks_p": float("nan")}
    return {
        "delta_median": float(np.median(b) - np.median(a)),
        "var_ratio": float(np.var(b, ddof=1) / np.var(a, ddof=1)) if len(a) > 1 and len(b) > 1 and np.var(a, ddof=1) > 0 else float("nan"),
        "q10_shift": float(np.percentile(b, 10) - np.percentile(a, 10)),
        "ks_p": float(stats.ks_2samp(a, b).pvalue),
    }


def _dominant_incidence_by_link(rays_csv: str | Path) -> dict[str, float]:
    rows = _read_csv(rays_csv)
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(str(r.get("link_id", "")), []).append(r)
    out: dict[str, float] = {}
    for lid, rr in by.items():
        best = None
        best_p = -np.inf
        for r in rr:
            p = _safe_float(r.get("P_lin", np.nan))
            inc = _safe_float(r.get("incidence_deg", np.nan))
            if np.isfinite(p) and np.isfinite(inc) and p > best_p:
                best_p = p
                best = inc
        if best is not None:
            out[lid] = float(best)
    return out


def _inc_bin(v: float) -> str:
    if not np.isfinite(v):
        return "NA"
    if v < 30.0:
        return "low"
    if v < 60.0:
        return "mid"
    return "high"


def _build_design_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    num_keys = ["d_m", "EL_proxy_db"]
    cat_keys = ["LOSflag", "material_class", "roughness_flag", "human_flag", "obstacle_flag", "dominant_parity_early", "incidence_bin"]

    num_cols: list[np.ndarray] = []
    names: list[str] = []
    N = len(rows)

    for k in num_keys:
        col = np.asarray([_safe_float(r.get(k, np.nan)) for r in rows], dtype=float)
        m = np.nanmedian(col) if np.any(np.isfinite(col)) else 0.0
        col = np.where(np.isfinite(col), col, m)
        num_cols.append(col)
        names.append(k)

    cat_map: dict[str, np.ndarray] = {}
    for k in cat_keys:
        vals = [str(r.get(k, "NA")) for r in rows]
        lv = sorted(set(vals))
        if len(lv) <= 1:
            continue
        for c in lv[1:]:
            col = np.asarray([1.0 if str(v) == c else 0.0 for v in vals], dtype=float)
            num_cols.append(col)
            cname = f"{k}={c}"
            names.append(cname)
            cat_map[cname] = col

    if not num_cols:
        X = np.ones((N, 1), dtype=float)
        names = ["const"]
    else:
        X = np.column_stack([np.ones(N, dtype=float)] + num_cols)
        names = ["const"] + names
    return X, names, {k: np.asarray([_safe_float(r.get(k, np.nan)) for r in rows], dtype=float) for k in num_keys}


def _vif_for_numeric(num_data: dict[str, np.ndarray], X_full: np.ndarray, names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    # names include const; numeric names are keys in num_data
    for k, y in num_data.items():
        m = np.isfinite(y)
        if np.sum(m) < 4:
            out[k] = float("nan")
            continue
        try:
            idx = names.index(k)
        except ValueError:
            out[k] = float("nan")
            continue
        Xm = X_full[m, :]
        ym = y[m]
        Xo = np.delete(Xm, idx, axis=1)
        beta, *_ = np.linalg.lstsq(Xo, ym, rcond=None)
        yh = Xo @ beta
        ssr = float(np.sum((ym - yh) ** 2))
        sst = float(np.sum((ym - float(np.mean(ym))) ** 2))
        if sst <= 0:
            out[k] = float("nan")
            continue
        r2 = max(0.0, min(1.0, 1.0 - ssr / sst))
        out[k] = float(1.0 / max(1e-6, 1.0 - r2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c0-link-metrics-csv", type=str, default="")
    ap.add_argument("--a4-link-metrics-csvs", type=str, required=True, help="comma-separated")
    ap.add_argument("--a4-labels", type=str, required=True, help="comma-separated")
    ap.add_argument("--a4-rays-csvs", type=str, default="", help="comma-separated, optional")
    ap.add_argument("--a5-base-link-metrics-csv", type=str, required=True)
    ap.add_argument("--a5-stress-link-metrics-csv", type=str, required=True)
    ap.add_argument("--vif-threshold", type=float, default=5.0)
    ap.add_argument("--out-report", type=str, required=True)
    ap.add_argument("--out-json", type=str, required=True)
    args = ap.parse_args()

    a4_csvs = [x.strip() for x in str(args.a4_link_metrics_csvs).split(",") if x.strip()]
    labels = [x.strip() for x in str(args.a4_labels).split(",") if x.strip()]
    a4_rays = [x.strip() for x in str(args.a4_rays_csvs).split(",") if x.strip()]
    if len(a4_csvs) != len(labels):
        raise SystemExit("a4-link-metrics-csvs and a4-labels length mismatch")

    c0_delta = float("nan")
    if str(args.c0_link_metrics_csv).strip():
        c0 = _read_csv(args.c0_link_metrics_csv)
        x = _metric_vals(c0, "XPD_early_db")
        if len(x):
            c0_delta = float(np.percentile(x, 95) - np.percentile(x, 5))

    a4_results: list[dict[str, Any]] = []
    all_d2_rows: list[dict[str, Any]] = []

    for i, (csv_path, label) in enumerate(zip(a4_csvs, labels)):
        rows = _read_csv(csv_path)
        all_d2_rows.extend(rows)
        inc_map: dict[str, float] = {}
        if i < len(a4_rays) and a4_rays[i]:
            inc_map = _dominant_incidence_by_link(a4_rays[i])
        for r in rows:
            r["incidence_deg_dom"] = inc_map.get(str(r.get("link_id", "")), float("nan"))
            r["incidence_bin"] = _inc_bin(_safe_float(r.get("incidence_deg_dom", np.nan)))

        mats = sorted(set(str(r.get("material_class", "NA")) for r in rows))
        by_mat: dict[str, dict[str, float]] = {}
        med_early: list[float] = []
        for m in mats:
            rr = [r for r in rows if str(r.get("material_class", "NA")) == m]
            xe = _metric_vals(rr, "XPD_early_db")
            xl = _metric_vals(rr, "XPD_late_db")
            lp = _metric_vals(rr, "L_pol_db")
            by_mat[m] = {
                "n": int(len(rr)),
                "median_XPD_early_db": float(np.median(xe)) if len(xe) else float("nan"),
                "median_XPD_late_db": float(np.median(xl)) if len(xl) else float("nan"),
                "median_L_pol_db": float(np.median(lp)) if len(lp) else float("nan"),
            }
            if len(xe):
                med_early.append(float(np.median(xe)))

        # material x incidence coverage
        cov: dict[str, dict[str, int]] = {}
        for m in mats:
            cov[m] = {"low": 0, "mid": 0, "high": 0, "NA": 0}
        for r in rows:
            m = str(r.get("material_class", "NA"))
            b = str(r.get("incidence_bin", "NA"))
            cov.setdefault(m, {"low": 0, "mid": 0, "high": 0, "NA": 0})
            cov[m][b] = int(cov[m].get(b, 0)) + 1

        a4_results.append(
            {
                "label": label,
                "csv": csv_path,
                "n_links": int(len(rows)),
                "material_shift_range_db": float(max(med_early) - min(med_early)) if med_early else float("nan"),
                "material_stats": by_mat,
                "coverage_material_x_incidence": cov,
                "exceeds_delta_floor": bool(np.isfinite(c0_delta) and len(med_early) > 0 and (max(med_early) - min(med_early) > c0_delta)),
            }
        )

    a5_base = _read_csv(args.a5_base_link_metrics_csv)
    a5_stress = _read_csv(args.a5_stress_link_metrics_csv)
    all_d2_rows.extend(a5_base)
    all_d2_rows.extend(a5_stress)

    for r in all_d2_rows:
        if "incidence_bin" not in r:
            r["incidence_bin"] = "NA"

    a5_effect = {
        "XPD_early_db": _effect_summary(_metric_vals(a5_base, "XPD_early_db"), _metric_vals(a5_stress, "XPD_early_db")),
        "XPD_late_db": _effect_summary(_metric_vals(a5_base, "XPD_late_db"), _metric_vals(a5_stress, "XPD_late_db")),
        "L_pol_db": _effect_summary(_metric_vals(a5_base, "L_pol_db"), _metric_vals(a5_stress, "L_pol_db")),
    }

    # D2 diagnostics
    X, names, num_data = _build_design_matrix(all_d2_rows)
    rank = int(np.linalg.matrix_rank(X))
    cond = float(np.linalg.cond(X)) if X.size else float("nan")
    vif = _vif_for_numeric(num_data, X, names)
    vif_warn = {k: float(v) for k, v in vif.items() if np.isfinite(v) and v > float(args.vif_threshold)}

    # stress x incidence coverage (for A5 only)
    stress_cov: dict[str, dict[str, int]] = {"base": {"low": 0, "mid": 0, "high": 0, "NA": 0}, "stress": {"low": 0, "mid": 0, "high": 0, "NA": 0}}
    for r in a5_base:
        b = _inc_bin(_safe_float(r.get("incidence_deg_dom", np.nan)))
        stress_cov["base"][b] = int(stress_cov["base"].get(b, 0)) + 1
    for r in a5_stress:
        b = _inc_bin(_safe_float(r.get("incidence_deg_dom", np.nan)))
        stress_cov["stress"][b] = int(stress_cov["stress"].get(b, 0)) + 1

    out = {
        "c0_delta_floor_db": c0_delta,
        "C2_A4": a4_results,
        "C2_A5_effect": a5_effect,
        "D2": {
            "n_rows": int(len(all_d2_rows)),
            "design_rank": rank,
            "design_cols": int(X.shape[1]),
            "condition_number": cond,
            "vif": vif,
            "vif_threshold": float(args.vif_threshold),
            "vif_warnings": vif_warn,
            "stress_x_incidence_coverage": stress_cov,
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# C2/D2 Detailed Diagnostic Report",
        "",
        f"- c0_delta_floor_db: {c0_delta}",
        "",
        "## C2 (A4 material / A5 stress)",
        "",
    ]
    for a4 in a4_results:
        lines.append(f"### A4 set: {a4['label']}")
        lines.append(f"- csv: {a4['csv']}")
        lines.append(f"- n_links: {a4['n_links']}")
        lines.append(f"- material_shift_range_db: {a4['material_shift_range_db']}")
        lines.append(f"- exceeds_delta_floor: {a4['exceeds_delta_floor']}")
        lines.append("- material medians:")
        for m, d in a4["material_stats"].items():
            lines.append(f"  - {m}: early={d['median_XPD_early_db']}, late={d['median_XPD_late_db']}, L_pol={d['median_L_pol_db']}")
        lines.append("- material×incidence coverage:")
        for m, d in a4["coverage_material_x_incidence"].items():
            lines.append(f"  - {m}: {d}")
        lines.append("")

    lines.append("### A5 base vs stress effect")
    for k, d in a5_effect.items():
        lines.append(f"- {k}: Δmedian={d['delta_median']}, var_ratio={d['var_ratio']}, q10_shift={d['q10_shift']}, ks_p={d['ks_p']}")

    lines.extend(
        [
            "",
            "## D2 Identifiability",
            "",
            f"- n_rows: {out['D2']['n_rows']}",
            f"- design_rank / cols: {out['D2']['design_rank']} / {out['D2']['design_cols']}",
            f"- condition_number: {out['D2']['condition_number']}",
            f"- vif_threshold: {out['D2']['vif_threshold']}",
            f"- vif: {out['D2']['vif']}",
            f"- vif_warnings: {out['D2']['vif_warnings']}",
            f"- stress_x_incidence_coverage: {out['D2']['stress_x_incidence_coverage']}",
        ]
    )

    out_md = Path(args.out_report)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out_md)
    print(out_json)


if __name__ == "__main__":
    main()

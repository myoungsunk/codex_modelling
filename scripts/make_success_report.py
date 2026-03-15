"""Generate markdown success report from standardized outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.success_checks import evaluate_success_criteria, load_link_metrics_csv
from plots.standard_plots import generate_standard_plots_from_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--link-metrics-csv", required=True, type=str)
    parser.add_argument("--out-report", type=str, default="outputs/success_report.md")
    parser.add_argument("--out-json", type=str, default="outputs/success_report.json")
    parser.add_argument("--plots-dir", type=str, default="outputs/plots_standard")
    args = parser.parse_args()

    csv_paths = [x.strip() for x in str(args.link_metrics_csv).split(",") if x.strip()]
    rows = []
    for p in csv_paths:
        rows.extend(load_link_metrics_csv(p))
    checks = evaluate_success_criteria(rows)
    plots = generate_standard_plots_from_rows(rows, out_dir=args.plots_dir)

    def _sem(r: dict) -> str:
        s = str(r.get("stress_semantics", "")).strip().lower()
        if s in {"off", "response", "polarization_only"}:
            return s
        sp = r.get("stress_path_structure_active", "")
        sm = r.get("stress_polarization_mixer_active", "")
        try:
            if float(sp) == 1.0:
                return "response"
            if float(sm) == 1.0:
                return "polarization_only"
        except Exception:
            pass
        mode = str(r.get("stress_mode", "")).strip().lower()
        if mode in {"geometry", "hybrid"}:
            return "response"
        if mode == "synthetic":
            return "polarization_only"
        return "off"

    a5 = [r for r in rows if str(r.get("scenario_id", "")).upper() == "A5"]
    a5_stress = [r for r in a5 if int(float(r.get("roughness_flag", 0) or 0)) == 1 or int(float(r.get("human_flag", 0) or 0)) == 1]
    cnt = {"response": 0, "polarization_only": 0, "off": 0, "unknown": 0}
    for r in a5_stress:
        k = _sem(r)
        cnt[k if k in cnt else "unknown"] += 1

    lines = ["# Success Report", ""]
    lines.append(f"- link_metrics_csv: {csv_paths}")
    lines.append(f"- n_links: {len(rows)}")
    lines.append("")

    c0 = checks.get("C0_floor", {})
    lines.append("## C0 Floor")
    lines.append("")
    for k in [
        "n",
        "metric_key",
        "xpd_floor_mean_db",
        "xpd_floor_std_db",
        "xpd_floor_p5_db",
        "xpd_floor_p95_db",
        "delta_floor_db",
        "distance_rank_corr",
        "yaw_rank_corr",
        "anova_distance_p",
        "anova_distance_eta2",
        "anova_yaw_p",
        "anova_yaw_eta2",
        "dominant_factor",
    ]:
        if k in c0:
            lines.append(f"- {k}: {c0.get(k)}")
    lines.append("")

    p = checks.get("A2_A3_parity_sign", {})
    lines.append("## A2/A3 Parity Sign")
    lines.append("")
    lines.append(f"- metric_key: {p.get('metric_key')}")
    lines.append(f"- median_A2_xpd_early_db: {p.get('median_A2_xpd_early_db')}")
    lines.append(f"- median_A3_xpd_early_db: {p.get('median_A3_xpd_early_db')}")
    lines.append(f"- delta_median_A3_minus_A2_db: {p.get('delta_median_A3_minus_A2_db')}")
    lines.append(f"- ks_p_A2_vs_A3: {p.get('ks_p_A2_vs_A3')}")
    lines.append(f"- wasserstein_A2_vs_A3: {p.get('wasserstein_A2_vs_A3')}")
    lines.append(f"- outlier_rate_A2_p10: {p.get('outlier_rate_A2_p10')}")
    lines.append(f"- outlier_rate_A3_p10: {p.get('outlier_rate_A3_p10')}")
    lines.append(f"- pass_A2_negative: {p.get('pass_A2_negative')}")
    lines.append(f"- pass_A3_positive: {p.get('pass_A3_positive')}")
    lines.append("")

    br = checks.get("A4_A5_breaking", {})
    lines.append("## A4/A5 Breaking Trend")
    lines.append("")
    lines.append(f"- metric_key_early: {br.get('metric_key_early')}")
    lines.append(f"- metric_key_late: {br.get('metric_key_late')}")
    lines.append(f"- median_abs_xpd_early_base: {br.get('median_abs_xpd_early_base')}")
    lines.append(f"- median_abs_xpd_early_stress: {br.get('median_abs_xpd_early_stress')}")
    lines.append(f"- median_xpd_late_base_db: {br.get('median_xpd_late_base_db')}")
    lines.append(f"- median_xpd_late_stress_db: {br.get('median_xpd_late_stress_db')}")
    lines.append(f"- var_xpd_early_base: {br.get('var_xpd_early_base')}")
    lines.append(f"- var_xpd_early_stress: {br.get('var_xpd_early_stress')}")
    lines.append(f"- var_ratio_stress_over_base: {br.get('var_ratio_stress_over_base')}")
    lines.append(f"- p10_xpd_early_base: {br.get('p10_xpd_early_base')}")
    lines.append(f"- p10_xpd_early_stress: {br.get('p10_xpd_early_stress')}")
    lines.append(f"- pass_breaking_trend: {br.get('pass_breaking_trend')}")
    lines.append("")
    lines.append("### A5 Stress Semantics")
    lines.append("")
    lines.append(f"- n_A5_stress_rows: {len(a5_stress)}")
    lines.append(f"- n_response: {cnt.get('response', 0)}")
    lines.append(f"- n_polarization_only: {cnt.get('polarization_only', 0)}")
    lines.append(f"- contamination_response_ready: {cnt.get('response', 0) > 0}")
    lines.append("- note: synthetic-only는 편파축 stress이며, delay/path contamination-response 해석은 geometry/hybrid(response)에서만 유효")
    lines.append("")

    b = checks.get("B_space", {})
    lines.append("## B-space Consistency")
    lines.append("")
    lines.append(f"- metric_key: {b.get('metric_key')}")
    lines.append(f"- spearman_xpd_early_vs_minus_el_proxy: {b.get('spearman_xpd_early_vs_minus_el_proxy')}")
    lines.append(f"- spearman_ci95_lo: {b.get('spearman_ci95_lo')}")
    lines.append(f"- spearman_ci95_hi: {b.get('spearman_ci95_hi')}")
    lines.append(f"- distance_vs_losflag_rank_corr: {b.get('distance_vs_losflag_rank_corr')}")
    lines.append(f"- ks_p_los_vs_nlos_xpd_early: {b.get('ks_p_los_vs_nlos_xpd_early')}")
    lines.append(f"- wasserstein_los_vs_nlos_xpd_early: {b.get('wasserstein_los_vs_nlos_xpd_early')}")
    lines.append("")

    mt = checks.get("multiple_testing", {})
    labels = list(mt.get("labels", []))
    p_raw = list(mt.get("p_raw", []))
    p_fdr = list(mt.get("p_fdr_bh", []))
    if labels:
        lines.append("## Multiple Testing (FDR-BH)")
        lines.append("")
        for lb, pr, qv in zip(labels, p_raw, p_fdr):
            lines.append(f"- {lb}: p_raw={pr}, p_fdr_bh={qv}")
        lines.append("")

    lines.append("## Generated Plots")
    lines.append("")
    for k, v in sorted(plots.items()):
        lines.append(f"- {k}: {v}")
    lines.append("")

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines), encoding="utf-8")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"checks": checks, "plots": plots}, indent=2), encoding="utf-8")
    print(str(out_report))
    print(str(out_json))


if __name__ == "__main__":
    main()

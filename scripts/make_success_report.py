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

    lines = ["# Success Report", ""]
    lines.append(f"- link_metrics_csv: {csv_paths}")
    lines.append(f"- n_links: {len(rows)}")
    lines.append("")

    c0 = checks.get("C0_floor", {})
    lines.append("## C0 Floor")
    lines.append("")
    for k in ["n", "xpd_floor_mean_db", "xpd_floor_std_db", "xpd_floor_p5_db", "xpd_floor_p95_db", "delta_floor_db"]:
        if k in c0:
            lines.append(f"- {k}: {c0.get(k)}")
    lines.append("")

    p = checks.get("A2_A3_parity_sign", {})
    lines.append("## A2/A3 Parity Sign")
    lines.append("")
    lines.append(f"- median_A2_xpd_early_db: {p.get('median_A2_xpd_early_db')}")
    lines.append(f"- median_A3_xpd_early_db: {p.get('median_A3_xpd_early_db')}")
    lines.append(f"- pass_A2_negative: {p.get('pass_A2_negative')}")
    lines.append(f"- pass_A3_positive: {p.get('pass_A3_positive')}")
    lines.append("")

    br = checks.get("A4_A5_breaking", {})
    lines.append("## A4/A5 Breaking Trend")
    lines.append("")
    lines.append(f"- median_abs_xpd_early_base: {br.get('median_abs_xpd_early_base')}")
    lines.append(f"- median_abs_xpd_early_stress: {br.get('median_abs_xpd_early_stress')}")
    lines.append(f"- median_xpd_late_base_db: {br.get('median_xpd_late_base_db')}")
    lines.append(f"- median_xpd_late_stress_db: {br.get('median_xpd_late_stress_db')}")
    lines.append(f"- pass_breaking_trend: {br.get('pass_breaking_trend')}")
    lines.append("")

    b = checks.get("B_space", {})
    lines.append("## B-space Consistency")
    lines.append("")
    lines.append(f"- spearman_xpd_early_vs_minus_el_proxy: {b.get('spearman_xpd_early_vs_minus_el_proxy')}")
    lines.append(f"- ks_p_los_vs_nlos_xpd_early: {b.get('ks_p_los_vs_nlos_xpd_early')}")
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

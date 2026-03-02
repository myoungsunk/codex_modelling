#!/usr/bin/env python3
"""Generate proposition-experiment-data-plot mapping report with PASS/FAIL.

This script reads an existing analysis_report run-group output and:
1) verifies whether proposition-level key plots exist,
2) generates missing supplementary plots for proposition coverage,
3) writes a mapping table (CSV + Markdown) with PASS/FAIL status.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis_report.lib.plots import (
    plot_box_by_group,
    plot_cdf,
    plot_multi_cdf,
    plot_pdp_overlay,
    plot_scatter,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _to_float(v: Any, default: float = float("nan")) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _finite(vals: list[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def _status_to_pass_fail(s: str) -> str:
    up = str(s).upper()
    if up in {"PASS", "FAIL", "PARTIAL"}:
        return up
    if up == "SUPPORTED":
        return "PASS"
    if up == "PARTIAL":
        return "PARTIAL"
    return "FAIL"


def _first_row(rows: list[dict[str, str]], **conds: str) -> dict[str, str] | None:
    for r in rows:
        ok = True
        for k, v in conds.items():
            if str(r.get(k, "")) != str(v):
                ok = False
                break
        if ok:
            return r
    return None


def _rows_by(rows: list[dict[str, str]], key: str, value: str) -> list[dict[str, str]]:
    return [r for r in rows if str(r.get(key, "")) == str(value)]


def _ensure_m2_ratio_plot(fig_dir: Path, details: dict[str, Any]) -> str | None:
    from matplotlib import pyplot as plt

    d = details.get("M2", {})
    inside = _to_float(d.get("inside_ratio"))
    outside = _to_float(d.get("outside_ratio"))
    if not (np.isfinite(inside) and np.isfinite(outside)):
        return None
    out = fig_dir / "M2__inside_outside_ratio.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    labels = ["|XPD_ex|<=Delta_floor", "|XPD_ex|>Delta_floor"]
    vals = [inside, outside]
    ax.bar(labels, vals, color=["#2f7ed8", "#f45b5b"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Ratio")
    ax.set_title("M2: Excess-vs-floor decision ratio")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out.name


def _ensure_g1_pdp_and_tap(fig_dir: Path, index_rows: list[dict[str, str]]) -> list[str]:
    from matplotlib import pyplot as plt

    row = _first_row(index_rows, scenario_id="A2")
    if row is None:
        return []
    run_dir = Path(str(row.get("input_run_dir", "")))
    case_id = str(row.get("case_id", "0"))
    pdp = run_dir / f"pdp_A2_{case_id}.npz"
    if not pdp.exists():
        cand = sorted(run_dir.glob("pdp_A2_*.npz"))
        if not cand:
            return []
        pdp = cand[0]
        case_id = pdp.stem.split("_")[-1]
    z = np.load(pdp)
    d = np.asarray(z["delay_tau_s"], dtype=float)
    pco = np.asarray(z["P_co"], dtype=float)
    pcross = np.asarray(z["P_cross"], dtype=float)
    xt = np.asarray(z["XPD_tau_db"], dtype=float) if "XPD_tau_db" in z.files else None

    out1 = fig_dir / f"G1__A2_case{case_id}__pdp_overlay.png"
    plot_pdp_overlay(d, pco, pcross, out1, title=f"G1 A2 case {case_id}: co/cross PDP")
    made = [out1.name]

    if xt is not None:
        out2 = fig_dir / f"G1__A2_case{case_id}__tap_xpd_tau.png"
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(d * 1e9, xt, lw=1.8, color="#d95319")
        ax.set_xlabel("Delay (ns)")
        ax.set_ylabel("XPD_tau (dB)")
        ax.set_title(f"G1 A2 case {case_id}: tap-wise XPD(tau)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out2, dpi=140)
        plt.close(fig)
        made.append(out2.name)
    return made


def _ensure_g3_plots(fig_dir: Path, link_rows: list[dict[str, str]], delta_floor_db: float) -> tuple[list[str], dict[str, Any]]:
    from matplotlib import pyplot as plt

    sset = {"A2", "A3", "A4", "A5"}
    rows = [r for r in link_rows if str(r.get("scenario_id", "")) in sset]
    by: dict[str, list[float]] = {}
    for r in rows:
        k = str(r.get("scenario_id", "NA"))
        by.setdefault(k, []).append(_to_float(r.get("XPD_early_excess_db")))
    clean = {k: _finite(v) for k, v in by.items() if len(_finite(v)) >= 3}
    made: list[str] = []
    if clean:
        out_cdf = fig_dir / "G3__xpd_early_ex__scenario_cdf.png"
        plot_multi_cdf({k: v for k, v in clean.items()}, out_cdf, "G3: XPD_early_ex by scenario", "XPD_early_ex (dB)")
        made.append(out_cdf.name)

        out_var = fig_dir / "G3__xpd_early_ex__std_bar.png"
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        labels = sorted(clean.keys())
        vals = [float(np.std(clean[k], ddof=1)) if len(clean[k]) > 1 else 0.0 for k in labels]
        ax.bar(labels, vals, color="#7cb5ec")
        ax.set_ylabel("Std(XPD_early_ex) [dB]")
        ax.set_title("G3: dispersion by condition")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_var, dpi=140)
        plt.close(fig)
        made.append(out_var.name)

    med = {k: float(np.median(v)) for k, v in clean.items()}
    iqr = {k: (float(np.percentile(v, 25.0)), float(np.percentile(v, 75.0))) for k, v in clean.items()}
    span = float(max(med.values()) - min(med.values())) if med else float("nan")
    overlap = False
    keys = sorted(iqr.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = iqr[keys[i]]
            b = iqr[keys[j]]
            if min(a[1], b[1]) >= max(a[0], b[0]):
                overlap = True
                break
        if overlap:
            break
    variation_ok = bool(np.isfinite(span) and np.isfinite(delta_floor_db) and span > float(delta_floor_db))
    nonperfect_sep = overlap
    status = "PASS" if (variation_ok and nonperfect_sep) else ("PARTIAL" if variation_ok else "FAIL")
    info = {
        "median_span_db": span,
        "delta_floor_db": float(delta_floor_db),
        "variation_ok": variation_ok,
        "iqr_overlap_exists": nonperfect_sep,
        "status": status,
    }
    return made, info


def _ensure_l1_scatter(fig_dir: Path, link_rows: list[dict[str, str]]) -> str | None:
    x = [_to_float(r.get("XPD_early_excess_db")) for r in link_rows]
    y = [_to_float(r.get("XPD_late_excess_db")) for r in link_rows]
    out = fig_dir / "L1__early_vs_late_ex_scatter.png"
    plot_scatter(
        x,
        y,
        out,
        title="L1: XPD_early_ex vs XPD_late_ex",
        xlabel="XPD_early_ex (dB)",
        ylabel="XPD_late_ex (dB)",
        add_fit=True,
    )
    return out.name


def _ensure_l2_plots(fig_dir: Path, link_rows: list[dict[str, str]]) -> list[str]:
    made: list[str] = []
    # A4 material-wise CDF
    a4 = _rows_by(link_rows, "scenario_id", "A4")
    by_mat: dict[str, list[float]] = {}
    for r in a4:
        m = str(r.get("material_class", "NA"))
        by_mat.setdefault(m, []).append(_to_float(r.get("XPD_early_excess_db")))
    by_mat = {k: _finite(v) for k, v in by_mat.items() if len(_finite(v)) >= 3}
    if by_mat:
        out1 = fig_dir / "L2__A4_material_xpd_early_ex_cdf.png"
        plot_multi_cdf(by_mat, out1, "L2-M: material effect (A4)", "XPD_early_ex (dB)")
        made.append(out1.name)
    # A5 base vs stress CDF (L_pol)
    a5 = _rows_by(link_rows, "scenario_id", "A5")
    base: list[float] = []
    stress: list[float] = []
    for r in a5:
        v = _to_float(r.get("L_pol_db"))
        s = str(r.get("stress_mode", "")).lower()
        rf = _to_float(r.get("roughness_flag"))
        hf = _to_float(r.get("human_flag"))
        is_stress = (s in {"hybrid", "synthetic", "geometry", "stress", "on"}) or (rf > 0.5) or (hf > 0.5)
        (stress if is_stress else base).append(v)
    if len(_finite(base)) >= 3 and len(_finite(stress)) >= 3:
        out2 = fig_dir / "L2__A5_base_vs_stress_lpol_cdf.png"
        plot_multi_cdf(
            {"base": base, "stress": stress},
            out2,
            "L2-S: stress effect (A5)",
            "L_pol (dB)",
        )
        made.append(out2.name)
    return made


def _ensure_r2_ds_vs_rho(fig_dir: Path, link_rows: list[dict[str, str]]) -> str | None:
    out = fig_dir / "R2__ds_vs_rho_early_db.png"
    plot_scatter(
        [_to_float(r.get("rho_early_db")) for r in link_rows],
        [_to_float(r.get("delay_spread_rms_s")) * 1e9 for r in link_rows],
        out,
        title="R2: DS vs rho_early",
        xlabel="rho_early (dB)",
        ylabel="Delay spread (ns)",
        add_fit=True,
    )
    return out.name


def _ensure_p1_p2_plots(fig_dir: Path, details: dict[str, Any]) -> list[str]:
    from matplotlib import pyplot as plt

    made: list[str] = []
    p1 = details.get("P1", {})
    c1 = p1.get("cv_two_stage", {}) if isinstance(p1, dict) else {}
    c0 = p1.get("cv_one_shot_reference", {}) if isinstance(p1, dict) else {}
    rmse_const = _to_float(c1.get("rmse_const"))
    rmse_lin = _to_float(c1.get("rmse_lin"))
    nll_const = _to_float(c1.get("nll_const"))
    nll_lin = _to_float(c1.get("nll_lin"))
    rmse_ref = _to_float(c0.get("rmse_lin"))
    if np.isfinite(rmse_const) and np.isfinite(rmse_lin):
        out1 = fig_dir / "P1__model_vs_constant_metrics.png"
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        labels = ["RMSE_const", "RMSE_two_stage", "NLL_const", "NLL_two_stage", "RMSE_one_shot"]
        vals = [rmse_const, rmse_lin, nll_const, nll_lin, rmse_ref]
        ax.bar(labels, vals, color=["#b2b2b2", "#1f77b4", "#b2b2b2", "#1f77b4", "#ff7f0e"])
        ax.set_title("P1: predictive metrics comparison")
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(out1, dpi=140)
        plt.close(fig)
        made.append(out1.name)
    p2 = details.get("P2", {})
    st = p2.get("stability", {}) if isinstance(p2, dict) else {}
    keep = _to_float(st.get("sign_keep_rate"))
    if np.isfinite(keep):
        out2 = fig_dir / "P2__subsampling_sign_keep_rate.png"
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.bar(["sign_keep_rate"], [keep], color="#55a868")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("P2: subsampling sign stability")
        ax.set_ylabel("Rate")
        ax.text(0, keep + 0.02, f"{keep:.2f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(out2, dpi=140)
        plt.close(fig)
        made.append(out2.name)
    return made


@dataclass
class PropositionSpec:
    pid: str
    experiment: str
    required_data: str
    key_tests: str
    criterion: str
    required_plots: list[str]


def _specs() -> list[PropositionSpec]:
    return [
        PropositionSpec("M1", "C0 거리x각도 sweep", "co/cross PDP, XPD(floor)", "XPD_floor vs 거리/각도 + box/CDF", "거리 의존 약함 + 각도 민감도", [
            "C0__ALL__xpd_floor_vs_distance.png",
            "C0__ALL__xpd_floor_vs_yaw.png",
            "C0__ALL__xpd_floor_cdf.png",
            "M1__C0_xpd_floor_box_by_yaw.png",
        ]),
        PropositionSpec("M2", "C0 + 전체 시나리오", "XPD_excess, Delta_floor", "inside/outside floor band ratio", "채널 주장 구간/불확실 구간 분리", [
            "M2__inside_outside_ratio.png",
        ]),
        PropositionSpec("G1", "A2 odd + LOS 차단", "co/cross PDP, XPD_early_ex, rho_early", "PDP overlay + tap-wise XPD + 분포", "odd에서 early cross 우세", [
            "A2A3C0__ALL__xpd_early_ex_cdf.png",
            "G1__A2_case0__pdp_overlay.png",
            "G1__A2_case0__tap_xpd_tau.png",
        ]),
        PropositionSpec("G2", "A3 even + LOS 차단", "co/cross PDP, XPD_early_ex, rho_early", "A2 대비 CDF/box", "A3가 A2 대비 반대방향", [
            "A2A3__ALL__xpd_early_ex_box.png",
            "A2A3__ALL__xpd_early_ex_cdf.png",
        ]),
        PropositionSpec("G3", "A2/A3 + A4/A5", "XPD_ex 분산/꼬리", "조건별 CDF + 분산 비교", "완전분리 아님 + 조건별 변동", [
            "G3__xpd_early_ex__scenario_cdf.png",
            "G3__xpd_early_ex__std_bar.png",
        ]),
        PropositionSpec("L1", "A2-A5 + room", "XPD_early_ex, XPD_late_ex, L_pol", "L_pol box + early-late scatter", "기본 L_pol>0, stress 예외", [
            "ALL__early_late_ex_box.png",
            "L1__early_vs_late_ex_scatter.png",
        ]),
        PropositionSpec("L2", "A4 material / A5 stress", "material/stress label + XPD_ex", "material/stress CDF + effect size", "재질/스트레스에 따라 평균/분산 이동", [
            "L2__A4_material_xpd_early_ex_cdf.png",
            "L2__A5_base_vs_stress_lpol_cdf.png",
        ]),
        PropositionSpec("L3", "통제+room + EL proxy", "EL, XPD_ex", "scatter + Spearman", "EL 증가에 따라 XPD/XPR 감소 단조", [
            "ALL__xpd_early_ex_vs_el_proxy.png",
        ]),
        PropositionSpec("R1", "room grid LOS/NLOS", "Z 맵(XPD_ex, rho, L_pol, DS)", "heatmap + LOS/NLOS CDF", "공간 분포 + LOS/NLOS 차이", [
            "B__ALL__heatmap_xpd_early_ex.png",
            "B__ALL__heatmap_lpol.png",
            "B__ALL__los_nlos_xpd_ex_cdf.png",
        ]),
        PropositionSpec("R2", "room grid + 지표 연결", "Z와 DS/early집중도", "DS vs XPD_ex, DS vs rho", "유효조건 영역 분리", [
            "ALL__ds_vs_xpd_early_ex.png",
            "R2__ds_vs_rho_early_db.png",
        ]),
        PropositionSpec("P1", "전체 데이터(통제+room)", "Z, U(EL/material/late)", "조건부모델 vs 상수모델", "조건부 모델 예측 우수", [
            "P1__model_vs_constant_metrics.png",
        ]),
        PropositionSpec("P2", "최소세트 subsampling", "부분 샘플 반복", "계수 부호 안정성", "표본 축소에도 결론 유지", [
            "P2__subsampling_sign_keep_rate.png",
        ]),
    ]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate proposition mapping matrix report.")
    ap.add_argument("--run-group", required=True, help="analysis_report/out/<run_group> folder name")
    ap.add_argument("--out-root", default="analysis_report/out", help="Base output root")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_root).resolve() / str(args.run_group)
    tables = out_dir / "tables"
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    if not tables.exists():
        raise SystemExit(f"Missing tables directory: {tables}")

    link_rows = _read_csv(tables / "intermediate_link_rows.csv")
    index_rows = _read_csv(out_dir / "index.csv")
    status_rows = _read_csv(tables / "intermediate_proposition_status.csv")
    details = json.loads((tables / "intermediate_proposition_details.json").read_text(encoding="utf-8"))
    floor_ref = json.loads((tables / "floor_reference_used.json").read_text(encoding="utf-8"))
    delta_floor_db = _to_float(floor_ref.get("delta_floor_db"), 0.0)

    # Generate missing supplementary plots
    c0_rows = _rows_by(link_rows, "scenario_id", "C0")
    if c0_rows:
        plot_box_by_group(
            c0_rows,
            "yaw_deg",
            "XPD_early_db",
            figs / "M1__C0_xpd_floor_box_by_yaw.png",
            title="M1: C0 XPD_floor grouped by yaw",
            ylabel="XPD_floor (dB)",
        )
    _ensure_m2_ratio_plot(figs, details)
    g1_created = _ensure_g1_pdp_and_tap(figs, index_rows)
    g3_created, g3_info = _ensure_g3_plots(figs, link_rows, delta_floor_db)
    _ensure_l1_scatter(figs, link_rows)
    l2_created = _ensure_l2_plots(figs, link_rows)
    _ensure_r2_ds_vs_rho(figs, link_rows)
    p_created = _ensure_p1_p2_plots(figs, details)

    # Build status map
    status_map = {str(r.get("proposition", "")): str(r.get("status", "")) for r in status_rows}
    status_map["G3"] = g3_info.get("status", "FAIL")

    # Force-resolve dynamic plot names for G1 generated case
    g1_required = [name for name in g1_created if name.endswith(".png")]
    if len(g1_required) >= 2:
        g1_required = g1_required
    else:
        g1_required = ["G1__A2_case0__pdp_overlay.png", "G1__A2_case0__tap_xpd_tau.png"]

    # Produce mapping rows
    rows_out: list[dict[str, Any]] = []
    for spec in _specs():
        req = list(spec.required_plots)
        if spec.pid == "G1":
            req = ["A2A3C0__ALL__xpd_early_ex_cdf.png"] + g1_required
        exists = []
        missing = []
        for fn in req:
            p = figs / fn
            if p.exists():
                exists.append(fn)
            else:
                # graceful fallback for generated case-id-dependent G1 names
                if spec.pid == "G1" and fn.startswith("G1__A2_case0__"):
                    alt = sorted(figs.glob("G1__A2_case*__" + fn.split("__", 2)[-1]))
                    if alt:
                        exists.append(alt[0].name)
                        continue
                missing.append(fn)
        pstatus = _status_to_pass_fail(status_map.get(spec.pid, "FAIL"))
        plot_status = "READY" if len(missing) == 0 else "MISSING"
        notes = ""
        if spec.pid == "G3":
            notes = (
                f"median_span={_to_float(g3_info.get('median_span_db')):.3f} dB, "
                f"delta_floor={_to_float(g3_info.get('delta_floor_db')):.3f} dB, "
                f"iqr_overlap={g3_info.get('iqr_overlap_exists')}"
            )
        rows_out.append(
            {
                "proposition": spec.pid,
                "experiment": spec.experiment,
                "required_data": spec.required_data,
                "key_tests": spec.key_tests,
                "criterion": spec.criterion,
                "pass_fail": pstatus,
                "plot_ready": plot_status,
                "plot_files_found": "; ".join(exists),
                "plot_files_missing": "; ".join(missing),
                "notes": notes,
            }
        )

    # Save csv
    out_csv = tables / "proposition_plot_mapping.csv"
    fields = [
        "proposition",
        "experiment",
        "required_data",
        "key_tests",
        "criterion",
        "pass_fail",
        "plot_ready",
        "plot_files_found",
        "plot_files_missing",
        "notes",
    ]
    _write_csv(out_csv, rows_out, fields)

    # Save markdown report
    md = out_dir / "proposition_plot_mapping_report.md"
    lines: list[str] = []
    lines.append(f"# Proposition Mapping Report ({args.run_group})")
    lines.append("")
    lines.append("명제-실험-데이터-플롯 매칭과 PASS/FAIL 결과를 정리한 표입니다.")
    lines.append("")
    lines.append("## Summary")
    total = len(rows_out)
    n_pass = sum(1 for r in rows_out if r["pass_fail"] == "PASS")
    n_partial = sum(1 for r in rows_out if r["pass_fail"] == "PARTIAL")
    n_fail = sum(1 for r in rows_out if r["pass_fail"] == "FAIL")
    n_plot_missing = sum(1 for r in rows_out if r["plot_ready"] != "READY")
    lines.append(f"- PASS: {n_pass}/{total}")
    lines.append(f"- PARTIAL: {n_partial}/{total}")
    lines.append(f"- FAIL: {n_fail}/{total}")
    lines.append(f"- Plot missing propositions: {n_plot_missing}/{total}")
    lines.append("")
    lines.append("## Proposition Table")
    lines.append("")
    lines.append("| 명제 | 실험 | 필요한 데이터 | 핵심 플롯/검정 | 통과 기준 | PASS/FAIL | 플롯 준비 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows_out:
        lines.append(
            f"| {r['proposition']} | {r['experiment']} | {r['required_data']} | {r['key_tests']} | "
            f"{r['criterion']} | {r['pass_fail']} | {r['plot_ready']} |"
        )
    lines.append("")
    lines.append("## Plot Files")
    lines.append("")
    for r in rows_out:
        lines.append(f"### {r['proposition']}")
        found = [x.strip() for x in str(r["plot_files_found"]).split(";") if x.strip()]
        miss = [x.strip() for x in str(r["plot_files_missing"]).split(";") if x.strip()]
        if found:
            lines.append("- Found:")
            for fn in found:
                lines.append(f"  - [figures/{fn}](figures/{fn})")
        else:
            lines.append("- Found: (none)")
        if miss:
            lines.append("- Missing:")
            for fn in miss:
                lines.append(f"  - `{fn}`")
        if r["notes"]:
            lines.append(f"- Notes: {r['notes']}")
        lines.append("")

    md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {md}")
    print(f"[INFO] Supplementary plots created: {len(g1_created) + len(g3_created) + len(l2_created) + len(p_created) + 3}")


if __name__ == "__main__":
    main()

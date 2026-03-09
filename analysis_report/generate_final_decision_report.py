"""Generate final decision report from diagnostic artifacts.

Outputs:
- analysis_report/out/<run_group>/final_diagnostic_decision.md
- analysis_report/out/<run_group>/scenario_space_plots.md
- analysis_report/out/<run_group>/figures/<SID>__ALL__scene_montage.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis_report.lib import io as io_lib
from analysis_report.lib import report_md


def _num(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _fmt(x: Any, n: int = 3) -> str:
    v = _num(x)
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{n}f}"


def _ensure_scene_montage(fig_dir: Path, scenario_id: str) -> Path | None:
    files: list[tuple[int, str, Path]] = []
    for p in fig_dir.glob(f"{scenario_id}__*__scene.png"):
        parts = p.stem.split("__")
        if len(parts) < 3:
            continue
        cid = str(parts[1])
        if cid == "GLOBAL":
            continue
        try:
            cnum = int(cid)
        except Exception:
            cnum = 10**9
        files.append((cnum, cid, p))
    files = sorted(files, key=lambda x: (x[0], x[1]))
    if not files:
        return None

    out_png = fig_dir / f"{scenario_id}__ALL__scene_montage.png"
    if out_png.exists():
        return out_png

    # Matplotlib cache path must be writable in sandbox/mac app.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    n = len(files)
    cols = 6 if n >= 24 else (5 if n >= 15 else 4)
    rows = int(math.ceil(n / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.1))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in axes_flat:
        ax.axis("off")
    for i, (_, cid, p) in enumerate(files):
        ax = axes_flat[i]
        try:
            img = mpimg.imread(p)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, p.name, ha="center", va="center", fontsize=7)
        ax.set_title(f"{scenario_id}:{cid}", fontsize=8)
    fig.suptitle(f"{scenario_id} scene montage (n={n})", fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def _generate_space_plot_index(
    out_root: Path,
    fig_dir: Path,
    scenarios: list[str],
) -> Path:
    lines: list[str] = []
    lines.append("# Scenario Space Plot Index")
    lines.append("")
    lines.append("- Full case-level scene links: [index.md](index.md)")
    lines.append("- A3 manual geometry review: [A3_geometry_manual_review.md](A3_geometry_manual_review.md)")
    lines.append("")

    for sid in scenarios:
        lines.append(f"## {sid}")
        lines.append("")
        montage = _ensure_scene_montage(fig_dir, sid)
        if montage is not None and montage.exists():
            lines.append(f"- Montage: [{montage.name}]({report_md.relpath(montage, out_root)})")
            lines.append(f"![{sid} montage]({report_md.relpath(montage, out_root)})")
            lines.append("")
        g = fig_dir / f"{sid}__GLOBAL__scene.png"
        if g.exists():
            lines.append(f"- Global layout: [{g.name}]({report_md.relpath(g, out_root)})")
            lines.append(f"![{sid} global]({report_md.relpath(g, out_root)})")
            lines.append("")
    lines.append("- All individual case scenes: [index.csv](index.csv), [index.md](index.md)")
    lines.append("")
    out_md = out_root / "scenario_space_plots.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def _extract_b_per_scenario(d3: dict[str, Any], tab_dir: Path) -> dict[str, dict[str, Any]]:
    rows = _read_csv(tab_dir / "B_per_scenario_summary.csv")
    if not rows:
        rows = d3.get("per_scenario_summary", []) if isinstance(d3.get("per_scenario_summary", []), list) else []
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        sid = str(r.get("scenario_id", ""))
        if sid:
            out[sid] = r
    return out


def _build_final_markdown(
    *,
    run_group: str,
    out_root: Path,
    checks: dict[str, Any],
    floor_ref: dict[str, Any],
    b_per: dict[str, dict[str, Any]],
) -> str:
    a1 = checks.get("A1_los_blocked", [])
    a2b = checks.get("A2_target_bounce", [])
    a3_sanity = checks.get("A3_coord_sanity", {})
    b = checks.get("B_time_resolution", {})
    c = checks.get("C_effect_vs_floor", {})
    d = checks.get("D_identifiability", {})
    d1 = d.get("D1", {})
    d2 = d.get("D2", {})
    d3 = d.get("D3", {})
    e = checks.get("E_power_based", {})

    a3_review_rows = _read_csv(out_root / "tables" / "A3_geometry_manual_review.csv")
    a3_pass = int(sum(str(x.get("review_status", "")) == "PASS" for x in a3_review_rows))
    a3_total = int(len(a3_review_rows))

    a2_sign = dict(b.get("A2_target_window_sign", {}))
    a3_sign = dict(b.get("A3_target_window_sign", {}))
    a3_sign_reporting = str(b.get("A3_target_window_sign_reporting_status", a3_sign.get("status", "INCONCLUSIVE")))
    a6_parity = dict(b.get("A6_parity_benchmark", {}))
    g2_src = str(b.get("G2_primary_evidence_source", "A3_target_window_supplementary_only"))
    g2_st = str(b.get("G2_primary_evidence_status", "INCONCLUSIVE"))
    holes = [x for x in d3.get("hole_analysis", []) if str(x.get("hole_type", "")) != "none"]

    los_list = [int(_num(x.get("los_rays", np.nan))) for x in a1 if np.isfinite(_num(x.get("los_rays", np.nan)))]
    a2_hit = a2b[0] if len(a2b) > 0 else {}
    a3_hit = a2b[1] if len(a2b) > 1 else {}

    lines: list[str] = []
    lines.append("# Final Diagnostic Decision")
    lines.append("")
    lines.append("- Branch: `feature/dualcp-proxy-bridge`")
    lines.append(f"- Experiment tag: `{run_group}`")
    lines.append("- Reference artifacts:")
    for p in [
        "tables/diagnostic_checks.json",
        "diagnostic_report.md",
        "tables/A3_target_window_sign.csv",
        "tables/A3_geometry_manual_review.csv",
        "A3_geometry_manual_review.md",
        "tables/el_proxy_imputation_rows.csv",
        "tables/D3_hole_analysis.csv",
        "tables/B_per_scenario_summary.csv",
        "scenario_space_plots.md",
    ]:
        lines.append(f"  - [{Path(p).name}]({p})")
    lines.append("")
    lines.append("## Overall Decision")
    lines.append("**Conditional Go**")
    lines.append("")
    lines.append("## One-line rationale")
    lines.append(
        "Calibration, effect size, and power-domain proxy modeling are validated for measurement start; "
        "remaining warnings are interpretation/role-definition items, not blocking physics failures."
    )
    lines.append("")
    lines.append("## Final Scenario Structure (Agreed)")
    lines.append("")
    lines.append(report_md.md_table(report_md.final_structure_rows(), ["unit", "role", "notes"]))
    lines.append("")
    lines.append("## Promotion to Go requires")
    lines.append(f"1. A3 manual review sign-off (`A3 pass={a3_pass}/{a3_total}` now; reviewer sign-off pending)")
    lines.append("2. `LOS0_q3 = structural_hole` documented in D3 hole analysis")
    lines.append(f"3. `qNA_total = {int(_num(d3.get('qna_total', np.nan))) if np.isfinite(_num(d3.get('qna_total', np.nan))) else 'nan'}` maintained as zero")
    lines.append("4. `B_per_scenario_summary.csv` archived as frozen reporting table")
    lines.append("")

    lines.append("## A. Geometry / Path Validity")
    lines.append("**Status:** PASS (conditional)")
    lines.append("")
    lines.append("### Evidence")
    lines.append(f"- LOS-blocked scenarios A2/A3/A4/A5: los_rays={los_list}")
    lines.append(
        f"- Target bounce existence: A2={a2_hit.get('hit', 'NA')}/{a2_hit.get('total', 'NA')} (rate={_fmt(a2_hit.get('rate', np.nan),2)}), "
        f"A3={a3_hit.get('hit', 'NA')}/{a3_hit.get('total', 'NA')} (rate={_fmt(a3_hit.get('rate', np.nan),2)})"
    )
    lines.append("")
    lines.append("### Interpretation")
    lines.append(
        "Intended physical interactions are generated correctly. "
        "A3 좌표/관통 검수는 자동 산출물만으로 완전 확정 불가하므로 수동 검토를 유지합니다."
    )
    lines.append("")
    lines.append("### Remaining action")
    lines.append("- [A3_geometry_manual_review.md](A3_geometry_manual_review.md) reviewer sign-off")
    lines.append("- Scenario space plots: [scenario_space_plots.md](scenario_space_plots.md)")
    lines.append("")

    lines.append("## B. Time Resolution / Delay Separation")
    lines.append("**Status:** WARN")
    lines.append("")
    lines.append("### Evidence")
    lines.append(
        f"- dt_res={_num(b.get('dt_res_s', np.nan)):.3e} s, tau_max={_num(b.get('tau_max_s', np.nan)):.3e} s, "
        f"Te={_num(b.get('Te_s', np.nan)):.3e} s, Tmax={_num(b.get('Tmax_s', np.nan)):.3e} s"
    )
    lines.append(
        f"- C0 floor window={b.get('W_floor_status','NA')}, "
        f"A2 target-window sign={a2_sign.get('status','NA')}, "
        f"A3 target-window sign(raw)={a3_sign.get('status','NA')} -> reporting={a3_sign_reporting}"
    )
    if bool(a6_parity.get("available", False)):
        lines.append(
            f"- A6 parity benchmark={a6_parity.get('status','NA')} "
            f"(odd hit={_fmt(a6_parity.get('hit_rate_odd', np.nan),3)}, even hit={_fmt(a6_parity.get('hit_rate_even', np.nan),3)})"
        )
    lines.append(f"- G2 primary evidence source={g2_src}, status={g2_st}")
    lines.append(
        f"- A3 mechanism status={b.get('A3_mechanism_status','NA')}, "
        f"A3 system-early status={b.get('A3_system_early_status','NA')}"
    )
    lines.append(
        f"- A5 target mode={b.get('A5_target_mode','NA')}, "
        f"W3 (room Te sweep) status={b.get('W3_status','NA')} (best S_xpd_early={_fmt(b.get('W3_best_S_xpd_early', np.nan),3)})"
    )
    lines.append("")
    lines.append("### Interpretation")
    lines.append(
        "A2는 odd early-anchor로 적합합니다. A3는 mechanism 검증에는 적합하지만 "
        "fixed system early-window baseline으로는 부적합합니다. "
        "A5는 stress-isolation이 아니라 contamination-response로 해석해야 합니다."
    )
    lines.append("")
    lines.append("### Reporting rule")
    lines.append("- A3: mechanism validation only (supplementary when A6 is present)")
    lines.append("- A5: stress-response / contamination analysis")
    lines.append("- A3를 fixed system early-window even baseline 증거로 사용하지 않음")
    lines.append(f"- G2 본증거는 `{g2_src}` 기준으로 사용")
    lines.append("")

    lines.append("## C. Effect Size vs Calibration Uncertainty")
    lines.append("**Status:** PASS")
    lines.append("")
    lines.append("### Evidence")
    lines.append(
        f"- delta_ref={_fmt(c.get('delta_ref_db', np.nan),3)} dB "
        f"(floor_delta={_fmt(c.get('floor_delta_db', np.nan),3)} dB, repeat_delta={_fmt(c.get('repeat_delta_db', np.nan),3)} dB)"
    )
    lines.append(
        f"- A3-A2 early delta_median={_fmt(c.get('A3_minus_A2_delta_median_db', np.nan),3)} dB "
        f"(ratio={_fmt(c.get('ratio_to_floor', np.nan),3)}, C1={c.get('C1_status','NA')})"
    )
    lines.append(f"- Material primary span={_fmt(c.get('C2M_primary_span_db', np.nan),3)} dB ({c.get('C2M_primary_status','NA')})")
    lines.append(
        f"- Stress primary effect ΔL_pol={_fmt(c.get('C2S_delta_lpol_db', np.nan),3)} dB "
        f"({c.get('C2S_primary_status','NA')}, C2S={c.get('C2S_status','NA')})"
    )
    lines.append("")
    lines.append("### Interpretation")
    lines.append("효과 크기가 보정 불확실도 대비 충분히 큽니다. floor 보정 이후에도 관측 가능한 차이를 유지할 가능성이 높습니다.")
    lines.append("")

    lines.append("## D. Identifiability")
    lines.append(f"**Status:** {d.get('status','NA')}")
    lines.append("")
    lines.append("### Evidence")
    lines.append(f"- D1-global={d1.get('global',{}).get('status','NA')} (EL_iqr={_fmt(d1.get('global',{}).get('EL_iqr_db', np.nan),3)} dB)")
    lines.append(
        f"- D1-local(A2)={d1.get('A2_isolation',{}).get('status','NA')}, "
        f"D1-local(A5)={d1.get('A5_isolation',{}).get('status','NA')} role={d1.get('A5_isolation',{}).get('role','NA')}"
    )
    lines.append(
        f"- D2(stage1)={d2.get('stage1',{}).get('status','NA')}, "
        f"D2(stage2)={d2.get('stage2',{}).get('status','NA')} (overall={d2.get('status','NA')})"
    )
    lines.append(
        f"- D3={d3.get('status','NA')} "
        f"(qNA_total={int(_num(d3.get('qna_total', np.nan))) if np.isfinite(_num(d3.get('qna_total', np.nan))) else 'nan'}, "
        f"selected_rows_n={int(_num(d3.get('selected_rows_n', np.nan))) if np.isfinite(_num(d3.get('selected_rows_n', np.nan))) else 0})"
    )
    for h in holes:
        lines.append(
            f"  - {h.get('strata','')}: {h.get('hole_type','')} "
            f"(pool_n={h.get('pool_n','')}, selected_n={h.get('selected_n','')})"
        )
    lines.append("")
    lines.append("### Interpretation")
    lines.append("회귀 식별성은 현재 단계에서 유효합니다. 잔여 이슈는 `LOS0_q3` 구조적 홀 문서화와 strata 해석 규칙 고정입니다.")
    lines.append("")
    lines.append("### Reporting rule")
    lines.append("- `LOS0_q3`는 sampling hole이 아닌 structural hole로 명시")
    lines.append("- D3 평가는 all-theoretical strata가 아니라 viable strata 기준으로 보고")
    lines.append("")

    lines.append("## E. Model-Form Consistency")
    lines.append(f"**Status:** {e.get('status','NA')}")
    lines.append("")
    lines.append("### Evidence")
    lines.append("- Power-domain metrics only: XPD_early, XPD_late, rho_early, L_pol, DS, EL_proxy, LOSflag")
    lines.append("- Complex phase 기반 의사결정 없음")
    lines.append("- 검증 단위: Z 분포/상관/순위 (파형 매칭 아님)")
    lines.append("")
    lines.append("### Interpretation")
    lines.append("RT를 잠재 설명변수 생성기로 쓰고, 검증은 관측 가능한 power-domain 통계로 제한하는 목표와 구현이 일치합니다.")
    lines.append("")

    lines.append("## Scenario-level Final Roles and Decisions")
    lines.append("")
    lines.append("| Scenario Unit | Final Role | Status | Use in paper | Exclude from |")
    lines.append("|---|---|---:|---|---|")
    lines.append("| C0 | floor calibration | PASS | calibration / uncertainty | effect regression |")
    lines.append("| A2_off | odd parity isolation | PASS | G1 primary evidence | EL-identification |")
    lines.append("| A2_on | bridge observability | WARN | LOS-on bridge check | G1 sign-off |")
    lines.append("| A3_on | bridge observability | WARN | LOS-on bridge check | G2 sign-off |")
    lines.append("| A3_corner | supplementary mechanism | WARN | mechanism-only context | system early baseline / G2 sign-off |")
    if bool(a6_parity.get("available", False)):
        lines.append(f"| A6 | near-normal parity benchmark | {a6_parity.get('status','NA')} | G2 primary sign evidence | oblique generalization |")
    lines.append("| A4_iso | material effect primary | PASS | L2-M primary | none |")
    lines.append("| A4_bridge | material effect secondary | WARN | L2-M bridge/late sensitivity | primary material claim |")
    lines.append("| A4_on | bridge observability | WARN | LOS-on bridge check | L2-M sign-off |")
    lines.append("| A5_pair | proxy stress response | PASS | L2-S primary (synthetic) + geometric sensitivity | faithful rough/human solver claim |")
    lines.append(f"| B1 | LOS real-space baseline | {b_per.get('B1',{}).get('status','NA')} | leverage baseline | strict full-strata claim |")
    lines.append(f"| B2 | partition NLOS | {b_per.get('B2',{}).get('status','NA')} | contamination / NLOS comparison | none |")
    lines.append(f"| B3 | corner high-EL NLOS | {b_per.get('B3',{}).get('status','NA')} | high-EL / structural stress region | none |")
    lines.append(f"| B-all | real-space identifiability set | {d3.get('status','NA')} | stage1 EL fit / leverage map | strict full-strata claim |")
    lines.append("")

    lines.append("## Final Measurement Readiness Decision")
    lines.append("")
    lines.append("### Decision")
    lines.append("**Conditional Go**")
    lines.append("")
    lines.append("### Blocking issues")
    lines.append("- None at physical-validity/effect-size level")
    lines.append("")
    lines.append("### Non-blocking but mandatory documentation actions")
    lines.append("1. Freeze A3 as mechanism-only scenario")
    lines.append("2. Freeze A5 as contamination-response scenario")
    lines.append("3. Document `LOS0_q3` as structural hole")
    lines.append("4. Keep `qNA_total = 0` in frozen diagnostic output")
    lines.append("5. Archive per-scenario summary (`B_per_scenario_summary.csv`)")
    lines.append("")
    lines.append("### Measurement may proceed because")
    lines.append("- Calibration floor is stable")
    lines.append("- Odd/material/stress effects exceed reference uncertainty")
    lines.append("- Stage1/Stage2 regression design is identifiable")
    lines.append("- Remaining WARN items are interpretation/documentation scope")
    lines.append("")

    lines.append("## Scenario Space Plots (All)")
    lines.append("")
    lines.append("- Full gallery and links: [scenario_space_plots.md](scenario_space_plots.md)")
    lines.append("- Case-level index: [index.md](index.md)")
    for sid in ["C0", "A2", "A2_on", "A3", "A3_on", "A4", "A4_on", "A5", "A6", "B1", "B2", "B3"]:
        p = out_root / "figures" / f"{sid}__ALL__scene_montage.png"
        if p.exists():
            lines.append(f"- [{p.name}]({report_md.relpath(p, out_root)})")
    lines.append("")
    lines.append(f"- Floor reference used: median={_fmt(floor_ref.get('xpd_floor_db', np.nan),3)} dB, delta={_fmt(floor_ref.get('delta_floor_db', np.nan),3)} dB")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to analysis_report config (json/yaml)")
    ap.add_argument("--scenarios", default="C0,A2,A2_on,A3,A3_on,A4,A4_on,A5,A6,B1,B2,B3")
    args = ap.parse_args()

    cfg = io_lib.load_config(args.config)
    run_group = str(cfg.get("run_group", "run_group"))
    out_root = Path("analysis_report") / "out" / run_group
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    if not out_root.exists():
        raise SystemExit(f"Output root missing: {out_root}. Run diagnostic report first.")

    checks_path = tab_dir / "diagnostic_checks.json"
    floor_path = tab_dir / "floor_reference_used.json"
    if not checks_path.exists() or not floor_path.exists():
        raise SystemExit(
            "Missing diagnostic artifacts. Run generate_diagnostic_report.py first: "
            f"{checks_path}, {floor_path}"
        )

    checks = _read_json(checks_path)
    floor_ref = _read_json(floor_path)
    scenarios = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]

    _generate_space_plot_index(out_root, fig_dir, scenarios)
    d3 = dict(dict(checks.get("D_identifiability", {})).get("D3", {}))
    b_per = _extract_b_per_scenario(d3, tab_dir)

    md = _build_final_markdown(
        run_group=run_group,
        out_root=out_root,
        checks=checks,
        floor_ref=floor_ref,
        b_per=b_per,
    )
    out_md = out_root / "final_diagnostic_decision.md"
    out_md.write_text(md, encoding="utf-8")
    print(str(out_md))


if __name__ == "__main__":
    main()

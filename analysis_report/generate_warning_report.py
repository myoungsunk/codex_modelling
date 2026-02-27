"""Generate warning-focused report for diagnostic WARN/FAIL items (case-by-case)."""

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

from analysis_report.generate_diagnostic_report import _diagnostic_checks
from analysis_report.lib import io as io_lib
from analysis_report.lib import metrics as metrics_lib
from analysis_report.lib import plots as plot_lib
from analysis_report.lib import report_md
from analysis_report.lib import scene as scene_lib


def _num(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _json_parse(v: Any) -> Any:
    if isinstance(v, dict):
        return v
    if not isinstance(v, str):
        return {}
    s = v.strip()
    if not s:
        return {}
    if not (s.startswith("{") and s.endswith("}")):
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


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


def _case_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("scenario_id", "NA")), str(row.get("case_id", ""))


def _rows_by_case(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(_case_key(r), []).append(r)
    return out


def _top_rays(case_rays: list[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
    if not case_rays:
        return []
    sr = sorted(case_rays, key=lambda x: _num(x.get("P_lin", np.nan)), reverse=True)
    out: list[dict[str, Any]] = []
    for i, r in enumerate(sr[: max(1, int(k))]):
        out.append(
            {
                "rank": int(i + 1),
                "ray_index": int(_num(r.get("ray_index", np.nan))) if np.isfinite(_num(r.get("ray_index", np.nan))) else -1,
                "n_bounce": int(_num(r.get("n_bounce", np.nan))) if np.isfinite(_num(r.get("n_bounce", np.nan))) else -1,
                "P_lin": _num(r.get("P_lin", np.nan)),
                "tau_ns": _num(r.get("tau_s", np.nan)) * 1e9,
                "los_flag_ray": int(_num(r.get("los_flag_ray", np.nan))) if np.isfinite(_num(r.get("los_flag_ray", np.nan))) else -1,
                "material_class": str(r.get("material_class", "")),
            }
        )
    return out


def _target_bounce_expected(sid: str) -> int | None:
    if sid == "A2":
        return 1
    if sid == "A3":
        return 2
    return None


def _find_target_tau_ns(case_rays: list[dict[str, Any]], target_n: int) -> float:
    cand = [r for r in case_rays if int(_num(r.get("n_bounce", np.nan))) == int(target_n)]
    if not cand:
        return float("nan")
    c0 = sorted(cand, key=lambda x: _num(x.get("P_lin", np.nan)), reverse=True)[0]
    return _num(c0.get("tau_s", np.nan)) * 1e9


def _collect_diag_alerts(checks: dict[str, Any]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []

    for r in checks.get("A1_los_blocked", []):
        st = str(r.get("status", ""))
        if st not in {"PASS"}:
            alerts.append({"item": "A1_los_blocked", "status": st, "detail": r})

    for r in checks.get("A2_target_bounce", []):
        st = str(r.get("status", ""))
        if st not in {"PASS"}:
            alerts.append({"item": "A2_target_bounce", "status": st, "detail": r})

    a3 = checks.get("A3_coord_sanity", {})
    if isinstance(a3, dict):
        st = str(a3.get("status", ""))
        if st not in {"PASS"}:
            alerts.append({"item": "A3_coord_sanity", "status": st, "detail": a3})

    b = checks.get("B_time_resolution", {})
    if isinstance(b, dict):
        for k in ["B2_status", "B3_status"]:
            st = str(b.get(k, ""))
            if st and st not in {"PASS"}:
                alerts.append({"item": f"B_time_resolution::{k}", "status": st, "detail": b})

    c = checks.get("C_effect_vs_floor", {})
    if isinstance(c, dict):
        for k in ["C1_status", "C2_status"]:
            st = str(c.get(k, ""))
            if st and st not in {"PASS"}:
                alerts.append({"item": f"C_effect_vs_floor::{k}", "status": st, "detail": c})

    d = checks.get("D_identifiability", {})
    if isinstance(d, dict):
        st = str(d.get("status", ""))
        if st and st not in {"PASS"}:
            alerts.append({"item": "D_identifiability", "status": st, "detail": d})

    e = checks.get("E_power_based", {})
    if isinstance(e, dict):
        st = str(e.get("status", ""))
        if st and st not in {"PASS"}:
            alerts.append({"item": "E_power_based", "status": st, "detail": e})

    return alerts


def _severity_from_flags(flags: dict[str, bool]) -> str:
    fail_keys = {"los_block_violation", "target_bounce_missing"}
    if any(bool(flags.get(k, False)) for k in fail_keys):
        return "FAIL"
    if any(bool(v) for v in flags.values()):
        return "WARN"
    return "PASS"


def _scene_for_case(
    row: dict[str, Any],
    scene_map: dict[tuple[str, str], dict[str, Any]],
    out_fig_dir: Path,
    figure_size: tuple[float, float],
) -> tuple[str, str, list[str]]:
    sid, cid = _case_key(row)
    key = (sid, cid)
    out_png = out_fig_dir / f"{sid}__{cid}__scene.png"
    warns: list[str] = []
    if key in scene_map:
        scene_obj = scene_map[key]
        ok, probs = scene_lib.validate_scene_debug(scene_obj)
        if not ok:
            warns.append("scene_debug_invalid")
            warns.extend(probs)
        p = scene_lib.plot_scene(scene_obj, out_png=out_png, figure_size=figure_size)
        return p, "scene_debug_json", warns
    fb_scene, fb_warns = scene_lib.build_fallback_scene_from_link_row(row)
    warns.append("scene_debug_missing_fallback_used")
    warns.extend(fb_warns)
    p = scene_lib.plot_scene(fb_scene, out_png=out_png, figure_size=figure_size)
    return p, "fallback_layout", warns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--top-k-rays", type=int, default=5)
    ap.add_argument("--max-cases", type=int, default=120)
    ap.add_argument("--include-pdp", action="store_true")
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
    link_rows = payload["link_rows"]
    ray_rows = payload["ray_rows"]
    scene_map = payload["scene_map"]
    runs = payload["runs"]

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

    checks = _diagnostic_checks(link_rows, ray_rows, floor_ref=floor_ref, cfg=cfg)
    diag_alerts = _collect_diag_alerts(checks)

    ws = dict(cfg.get("windows", {}))
    te_ns = float(ws.get("Te_ns", 10.0))

    by_case_link = _rows_by_case(link_rows)
    by_case_ray = _rows_by_case(ray_rows)

    scene_cfg = dict(cfg.get("scene_plots", {}))
    figure_size = tuple(float(x) for x in scene_cfg.get("figure_size", [10, 6]))

    warning_cases: list[dict[str, Any]] = []
    max_cases = max(1, int(args.max_cases))

    for key in sorted(by_case_link.keys(), key=lambda k: (k[0], k[1])):
        sid, cid = key
        lrows = by_case_link[key]
        row = lrows[0]
        rays = by_case_ray.get(key, [])
        top = _top_rays(rays, k=int(args.top_k_rays))

        flags = {
            "scene_fallback": False,
            "los_block_violation": False,
            "target_bounce_missing": False,
            "target_tau_outside_early": False,
            "effect_within_floor_uncert": False,
            "metric_missing": False,
        }
        reasons: list[str] = []

        # scene
        scene_png, scene_source, scene_warns = _scene_for_case(row, scene_map=scene_map, out_fig_dir=fig_dir, figure_size=figure_size)
        if scene_source != "scene_debug_json":
            flags["scene_fallback"] = True
            reasons.append("scene_debug missing -> fallback spatial layout")
        for w in scene_warns:
            if w not in {"scene_debug_missing_fallback_used"}:
                reasons.append(f"scene_warn:{w}")

        # LOS violation
        if sid in {"A2", "A3", "A4", "A5"}:
            los_hits = int(sum(int(_num(r.get("los_flag_ray", 0))) == 1 for r in rays))
            if los_hits > 0:
                flags["los_block_violation"] = True
                reasons.append(f"LOS ray exists in blocked scenario (count={los_hits})")

        # target bounce and target tau in early
        tbn = _target_bounce_expected(sid)
        if tbn is not None:
            top_hit = any(int(_num(r.get("n_bounce", np.nan))) == int(tbn) for r in top)
            if not top_hit:
                flags["target_bounce_missing"] = True
                reasons.append(f"target bounce n={tbn} missing in top-{len(top)} rays")
            tau_ns = _find_target_tau_ns(rays, target_n=tbn)
            if np.isfinite(tau_ns) and tau_ns >= te_ns:
                flags["target_tau_outside_early"] = True
                reasons.append(f"target tau {tau_ns:.3f}ns >= Te {te_ns:.3f}ns")

        # effect size below floor uncertainty
        xex = _num(row.get("XPD_early_excess_db", np.nan))
        delta = _num(row.get("delta_floor_db", floor_ref.get("delta_floor_db", np.nan)))
        if np.isfinite(xex) and np.isfinite(delta) and abs(xex) <= abs(delta):
            flags["effect_within_floor_uncert"] = True
            reasons.append("|XPD_early_excess| <= delta_floor (effect claim caution)")

        # missing metrics
        for kk in ["XPD_early_excess_db", "XPD_late_excess_db", "L_pol_db", "EL_proxy_db", "LOSflag"]:
            if not np.isfinite(_num(row.get(kk, np.nan))):
                flags["metric_missing"] = True
        if flags["metric_missing"]:
            reasons.append("missing/NaN in key metrics")

        sev = _severity_from_flags(flags)
        if sev == "PASS":
            continue

        link_id = str(row.get("link_id", ""))

        pdp_png = ""
        if bool(args.include_pdp):
            run_match = None
            for rr in runs:
                if str(rr.get("scenario_id", "")) == sid:
                    run_match = rr
                    break
            if run_match is not None:
                pdp = io_lib.load_pdp_npz(run_match, link_id=link_id)
                if pdp is not None and len(np.asarray(pdp.get("delay_tau_s", []), dtype=float)) > 0:
                    pdp_png = plot_lib.plot_pdp_overlay(
                        delay_s=np.asarray(pdp["delay_tau_s"], dtype=float),
                        p_co=np.asarray(pdp["P_co"], dtype=float),
                        p_cross=np.asarray(pdp["P_cross"], dtype=float),
                        out_png=fig_dir / f"{sid}__{cid}__warn_pdp.png",
                        title=f"PDP overlay {sid}/{cid} ({link_id})",
                    )

        warning_cases.append(
            {
                "severity": sev,
                "scenario_id": sid,
                "case_id": cid,
                "case_label": str(row.get("case_label", link_id or cid)),
                "link_id": link_id,
                "scene_source": scene_source,
                "scene_png": scene_png,
                "pdp_png": pdp_png,
                "reasons": reasons,
                "flags": flags,
                "XPD_early_excess_db": xex,
                "XPD_late_excess_db": _num(row.get("XPD_late_excess_db", np.nan)),
                "L_pol_db": _num(row.get("L_pol_db", np.nan)),
                "EL_proxy_db": _num(row.get("EL_proxy_db", np.nan)),
                "LOSflag": _num(row.get("LOSflag", np.nan)),
                "top_rays": top,
            }
        )

    # prioritize FAIL, then WARN
    order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    warning_cases = sorted(warning_cases, key=lambda r: (order.get(str(r.get("severity", "WARN")), 9), str(r.get("scenario_id", "")), str(r.get("case_id", ""))))
    warning_cases = warning_cases[:max_cases]

    # output tables
    rows_csv = []
    for r in warning_cases:
        rows_csv.append(
            {
                "severity": r["severity"],
                "scenario_id": r["scenario_id"],
                "case_id": r["case_id"],
                "case_label": r["case_label"],
                "link_id": r["link_id"],
                "scene_source": r["scene_source"],
                "reasons": json.dumps(r["reasons"], ensure_ascii=False),
                "flags": json.dumps(r["flags"], ensure_ascii=False),
                "XPD_early_excess_db": r["XPD_early_excess_db"],
                "XPD_late_excess_db": r["XPD_late_excess_db"],
                "L_pol_db": r["L_pol_db"],
                "EL_proxy_db": r["EL_proxy_db"],
                "LOSflag": r["LOSflag"],
            }
        )
    _write_rows_csv(tab_dir / "warning_cases.csv", rows_csv)
    report_md.write_json(tab_dir / "warning_cases.json", warning_cases)
    report_md.write_json(tab_dir / "warning_diagnostic_alerts.json", diag_alerts)

    # markdown
    lines: list[str] = []
    lines.append(f"# Warning Report ({run_group})")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    n_fail = int(sum(str(r.get("severity", "")) == "FAIL" for r in warning_cases))
    n_warn = int(sum(str(r.get("severity", "")) == "WARN" for r in warning_cases))
    lines.append(f"- total warning cases: **{len(warning_cases)}** (FAIL={n_fail}, WARN={n_warn})")
    lines.append(f"- diagnostic alerts (A-E status WARN/FAIL): **{len(diag_alerts)}**")
    lines.append("")

    if diag_alerts:
        lines.append("## Diagnostic Alerts (from A-E)")
        lines.append("")
        da_rows = []
        for a in diag_alerts:
            da_rows.append(
                {
                    "item": a.get("item", ""),
                    "status": a.get("status", ""),
                    "detail": json.dumps(a.get("detail", {}), ensure_ascii=False),
                }
            )
        lines.append(report_md.md_table(da_rows, ["item", "status", "detail"]))
        lines.append("")

    lines.append("## Warning Cases Table")
    lines.append("")
    lines.append(
        report_md.md_table(
            rows_csv,
            [
                "severity",
                "scenario_id",
                "case_id",
                "case_label",
                "link_id",
                "scene_source",
                "XPD_early_excess_db",
                "XPD_late_excess_db",
                "L_pol_db",
                "EL_proxy_db",
                "LOSflag",
            ],
        )
    )
    lines.append("")

    lines.append("## Case-by-case Review")
    lines.append("")
    for r in warning_cases:
        sid = str(r.get("scenario_id", "NA"))
        cid = str(r.get("case_id", ""))
        lines.append(f"### [{r.get('severity', 'WARN')}] {sid}/{cid} ({r.get('link_id', '')})")
        lines.append("")
        lines.append(f"- case_label: {r.get('case_label', '')}")
        lines.append(f"- scene_source: {r.get('scene_source', '')}")
        lines.append("- reasons:")
        for reason in r.get("reasons", []):
            lines.append(f"  - {reason}")
        lines.append("")

        sp = str(r.get("scene_png", ""))
        if sp:
            lines.append(f"![scene-{sid}-{cid}]({report_md.relpath(sp, out_root)})")
            lines.append("")

        pp = str(r.get("pdp_png", ""))
        if pp:
            lines.append(f"![pdp-{sid}-{cid}]({report_md.relpath(pp, out_root)})")
            lines.append("")

        lines.append(
            report_md.md_table(
                [
                    {
                        "XPD_early_excess_db": r.get("XPD_early_excess_db", np.nan),
                        "XPD_late_excess_db": r.get("XPD_late_excess_db", np.nan),
                        "L_pol_db": r.get("L_pol_db", np.nan),
                        "EL_proxy_db": r.get("EL_proxy_db", np.nan),
                        "LOSflag": r.get("LOSflag", np.nan),
                    }
                ],
                ["XPD_early_excess_db", "XPD_late_excess_db", "L_pol_db", "EL_proxy_db", "LOSflag"],
            )
        )
        lines.append("")

        top = r.get("top_rays", [])
        if isinstance(top, list) and top:
            lines.append("Top rays")
            lines.append("")
            lines.append(
                report_md.md_table(
                    top,
                    ["rank", "ray_index", "n_bounce", "P_lin", "tau_ns", "los_flag_ray", "material_class"],
                )
            )
            lines.append("")

    out_md = out_root / "warning_report.md"
    report_md.write_text(out_md, "\n".join(lines) + "\n")
    print(str(out_md))


if __name__ == "__main__":
    main()

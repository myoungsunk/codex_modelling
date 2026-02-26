"""Run standardized proxy simulations and export schema v1 outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.el_proxy import compute_el_proxy
from analysis.excess_loss import add_el_db
from analysis.link_conditions import build_link_U_from_scenario
from analysis.link_metrics import compute_link_metrics
from analysis.pdp_synthesis import synthesize_dualcp_pdp
from analysis.ray_table import build_ray_table_from_rt
from analysis.windowing import estimate_tau0, make_early_late_masks
from calibration.floor_model import AngleSensitiveFloorXPD, ConstantFloorXPD, FloorXPDModel, FreqDependentFloorXPD
from polarization.xpr_models import BaseXPRModel, BinnedXPR, ConditionalLinearXPR, ConstantXPR
from rt_core.geometry import Material, Plane
from rt_core.tracer import trace_paths
from rt_io.standard_outputs_hdf5 import export_csv, save_run
from rt_types.standard_outputs import RayTable, SCHEMA_VERSION, StandardOutputBundle
from scenarios import A2_pec_plane, A3_corner_2bounce, A4_dielectric_plane, A5_depol_stress, B0_room_box, C0_free_space
from scenarios.common import default_antennas, paths_to_records, uwb_frequency


def _parse_float_list(raw: str, default: list[float]) -> list[float]:
    out = []
    for tok in str(raw).split(","):
        s = tok.strip()
        if not s:
            continue
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out or list(default)


def _git_meta() -> tuple[str, str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = "unknown"
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        branch = "unknown"
    return commit, branch


def _load_json(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise ValueError(f"JSON not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _make_xpr_model(scenario: str, cfg: dict[str, Any]) -> BaseXPRModel:
    if cfg:
        t = str(cfg.get("type", "constant")).lower()
        if t == "linear":
            return ConditionalLinearXPR(
                a0=float(cfg.get("a0", 8.0)),
                a_el=float(cfg.get("a_el", 0.0)),
                a_late=float(cfg.get("a_late", 0.0)),
                a_incidence=float(cfg.get("a_incidence", 0.0)),
                a_rough=float(cfg.get("a_rough", -3.0)),
                a_human=float(cfg.get("a_human", -3.0)),
                sigma_db=float(cfg.get("sigma_db", 0.0)),
                material_bias=dict(cfg.get("material_bias", {})),
            )
        if t == "binned":
            return BinnedXPR(default_xpr_db=float(cfg.get("default_xpr_db", 8.0)), bins=list(cfg.get("bins", [])))
        return ConstantXPR(float(cfg.get("xpr_db", 8.0)))

    s = str(scenario).upper()
    if s == "A4":
        return ConditionalLinearXPR(a0=3.0, a_el=-0.03, sigma_db=0.6)
    if s == "A5":
        return ConditionalLinearXPR(a0=6.0, a_rough=-4.0, a_human=-4.0, sigma_db=0.8)
    if s.startswith("B"):
        return ConditionalLinearXPR(a0=7.0, a_el=-0.02, a_late=-1.2, sigma_db=1.0)
    return ConstantXPR(8.0)


def _make_floor_model(cfg: dict[str, Any]) -> FloorXPDModel:
    if not cfg:
        return ConstantFloorXPD(25.0, sigma_db=0.5)
    t = str(cfg.get("type", "constant")).lower()
    if t == "freq":
        return FreqDependentFloorXPD(
            freq_hz=np.asarray(cfg.get("freq_hz", [6e9, 8e9, 10e9]), dtype=float),
            xpd_floor_db=np.asarray(cfg.get("xpd_floor_db", [23.0, 25.0, 27.0]), dtype=float),
            sigma_db=float(cfg.get("sigma_db", 0.0)),
        )
    if t == "angle":
        return AngleSensitiveFloorXPD(
            base_db=float(cfg.get("base_db", 25.0)),
            yaw_slope_db_per_deg=float(cfg.get("yaw_slope_db_per_deg", -0.05)),
            pitch_slope_db_per_deg=float(cfg.get("pitch_slope_db_per_deg", -0.02)),
            sigma_db=float(cfg.get("sigma_db", 0.0)),
        )
    return ConstantFloorXPD(float(cfg.get("xpd_floor_db", 25.0)), sigma_db=float(cfg.get("sigma_db", 0.0)))


def _run_room_case(kind: str, rx_x: float, rx_y: float, f_hz: np.ndarray) -> list[dict[str, Any]]:
    tx, rx = default_antennas(basis="circular")
    tx.position[:] = [2.0, 0.0, 1.5]
    rx.position[:] = [rx_x, rx_y, 1.5]
    scene = B0_room_box.build_scene()
    if kind == "B2":
        scene.append(
            Plane(
                id=201,
                p0=np.array([5.0, 0.0, 1.5]),
                normal=np.array([-1.0, 0.0, 0.0]),
                material=Material.pec(),
                u_axis=np.array([0.0, 1.0, 0.0]),
                v_axis=np.array([0.0, 0.0, 1.0]),
                half_extent_u=1.8,
                half_extent_v=1.5,
            )
        )
    if kind == "B3":
        scene.extend(
            [
                Plane(
                    id=301,
                    p0=np.array([5.5, 1.5, 1.5]),
                    normal=np.array([-1.0, 0.0, 0.0]),
                    material=Material.pec(),
                    u_axis=np.array([0.0, 1.0, 0.0]),
                    v_axis=np.array([0.0, 0.0, 1.0]),
                    half_extent_u=1.5,
                    half_extent_v=1.5,
                ),
                Plane(
                    id=302,
                    p0=np.array([5.5, 1.5, 1.5]),
                    normal=np.array([0.0, -1.0, 0.0]),
                    material=Material.pec(),
                    u_axis=np.array([1.0, 0.0, 0.0]),
                    v_axis=np.array([0.0, 0.0, 1.0]),
                    half_extent_u=1.5,
                    half_extent_v=1.5,
                ),
            ]
        )
    paths = trace_paths(scene, tx, rx, f_hz, max_bounce=2, los_enabled=True)
    return paths_to_records(paths)


def _build_case_records(args: argparse.Namespace, f_hz: np.ndarray) -> list[dict[str, Any]]:
    s = str(args.scenario).upper()
    out: list[dict[str, Any]] = []
    if s == "C0":
        dlist = _parse_float_list(args.dist_list, [3.0, 6.0, 9.0])
        yaw_list = _parse_float_list(args.yaw_list, [0.0])
        pitch_list = _parse_float_list(args.pitch_list, [0.0])
        cid = 0
        for d in dlist:
            for yaw in yaw_list:
                for pit in pitch_list:
                    paths = C0_free_space.run_case({"distance_m": d}, f_hz, basis="circular")
                    out.append(
                        {
                            "case_id": str(cid),
                            "scenario_id": "C0",
                            "link_id": f"C0_{cid}",
                            "params": {"distance_m": d, "yaw_deg": yaw, "pitch_deg": pit},
                            "paths": paths_to_records(paths),
                            "meta": {
                                "d_m": d,
                                "LOSflag": 1,
                                "yaw_deg": yaw,
                                "pitch_deg": pit,
                                "material_class": "free_space",
                            },
                        }
                    )
                    cid += 1
        return out

    if s == "A2":
        dlist = _parse_float_list(args.dist_list, [4.0, 6.0, 8.0])
        cid = 0
        for d in dlist:
            p = {"y_plane": 2.0, "distance_m": d}
            paths = A2_pec_plane.run_case(p, f_hz, basis="circular")
            out.append(
                {
                    "case_id": str(cid),
                    "scenario_id": "A2",
                    "link_id": f"A2_{cid}",
                    "params": p,
                    "paths": paths_to_records(paths),
                    "meta": {"d_m": d, "LOSflag": 0, "material_class": "PEC", "obstacle_flag": 1},
                }
            )
            cid += 1
        return out

    if s == "A3":
        base = A3_corner_2bounce.build_sweep_params()
        cid = 0
        for p in base:
            paths = A3_corner_2bounce.run_case(p, f_hz, basis="circular")
            d = float(np.linalg.norm(np.asarray([p["rx_x"], p["rx_y"], 1.5]) - np.asarray([0.0, 0.0, 1.5])))
            out.append(
                {
                    "case_id": str(cid),
                    "scenario_id": "A3",
                    "link_id": f"A3_{cid}",
                    "params": p,
                    "paths": paths_to_records(paths),
                    "meta": {"d_m": d, "LOSflag": 0, "material_class": "PEC", "obstacle_flag": 1},
                }
            )
            cid += 1
        return out

    if s == "A4":
        mats = [m.strip() for m in str(args.material_list).split(",") if m.strip()] or ["concrete", "glass"]
        cid = 0
        for m in mats:
            for y in [1.5, 2.5]:
                p = {"material": m, "y_plane": y, "distance_m": 6.0}
                paths = A4_dielectric_plane.run_case(p, f_hz, basis="circular")
                out.append(
                    {
                        "case_id": str(cid),
                        "scenario_id": "A4",
                        "link_id": f"A4_{cid}",
                        "params": p,
                        "paths": paths_to_records(paths),
                        "meta": {"d_m": 6.0, "LOSflag": 0, "material_class": m, "obstacle_flag": 1},
                    }
                )
                cid += 1
        return out

    if s == "A5":
        cid = 0
        params = A5_depol_stress.build_sweep_params()[:8]
        for p in params:
            paths = A5_depol_stress.run_case(p, f_hz, basis="circular")
            d = float(np.linalg.norm(np.asarray([p["rx_x"], p["rx_y"], 1.5]) - np.asarray([0.0, 0.0, 1.5])))
            out.append(
                {
                    "case_id": str(cid),
                    "scenario_id": "A5",
                    "link_id": f"A5_{cid}",
                    "params": p,
                    "paths": paths_to_records(paths),
                    "meta": {
                        "d_m": d,
                        "LOSflag": 0,
                        "material_class": "PEC",
                        "roughness_flag": int(bool(args.stress_flag)),
                        "human_flag": int(bool(args.stress_flag)),
                        "obstacle_flag": 1,
                    },
                }
            )
            cid += 1
        return out

    # B1/B2/B3 room-grid style
    kind = s
    x_vals = np.arange(float(args.grid_x_min), float(args.grid_x_max) + 1e-9, float(args.grid_step_m))
    y_vals = np.arange(float(args.grid_y_min), float(args.grid_y_max) + 1e-9, float(args.grid_step_m))
    cid = 0
    for x in x_vals:
        for y in y_vals:
            paths = _run_room_case(kind=kind, rx_x=float(x), rx_y=float(y), f_hz=f_hz)
            out.append(
                {
                    "case_id": str(cid),
                    "scenario_id": kind,
                    "link_id": f"{kind}_{cid}",
                    "params": {"rx_x": float(x), "rx_y": float(y)},
                    "paths": paths,
                    "meta": {
                        "d_m": float(np.linalg.norm(np.asarray([x, y, 1.5]) - np.asarray([2.0, 0.0, 1.5]))),
                        "LOSflag": int(1),
                        "material_class": "PEC",
                        "obstacle_flag": int(kind in {"B2", "B3"}),
                    },
                }
            )
            cid += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["C0", "A2", "A3", "A4", "A5", "B1", "B2", "B3"])
    parser.add_argument("--out-h5", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs/standard")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nf", type=int, default=128)
    parser.add_argument("--f-center-hz", type=float, default=8e9)
    parser.add_argument("--delay-bin-ns", type=float, default=0.25)
    parser.add_argument("--Te-ns", type=float, default=3.0)
    parser.add_argument("--Tmax-ns", type=float, default=30.0)
    parser.add_argument("--xpr-model-config", type=str, default=None)
    parser.add_argument("--floor-model-config", type=str, default=None)
    parser.add_argument("--dist-list", type=str, default="")
    parser.add_argument("--yaw-list", type=str, default="0")
    parser.add_argument("--pitch-list", type=str, default="0")
    parser.add_argument("--material-list", type=str, default="")
    parser.add_argument("--stress-flag", action="store_true")
    parser.add_argument("--strict-los-blocked", action="store_true")
    parser.add_argument("--max-links", type=int, default=0)
    parser.add_argument("--ds-reference", type=str, default="total", choices=["total", "co"])
    parser.add_argument("--el-proxy-mode", type=str, default="early_sum", choices=["early_sum", "dominant_early_ray"])
    parser.add_argument("--grid-x-min", type=float, default=3.0)
    parser.add_argument("--grid-x-max", type=float, default=7.0)
    parser.add_argument("--grid-y-min", type=float, default=-2.0)
    parser.add_argument("--grid-y-max", type=float, default=2.0)
    parser.add_argument("--grid-step-m", type=float, default=2.0)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    commit, branch = _git_meta()
    run_id = str(args.run_id or f"{args.scenario}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    f_hz = uwb_frequency(nf=int(args.nf))
    xpr_cfg = _load_json(args.xpr_model_config)
    floor_cfg = _load_json(args.floor_model_config)
    xpr_model = _make_xpr_model(str(args.scenario), xpr_cfg)
    floor_model = _make_floor_model(floor_cfg)

    case_records = _build_case_records(args, f_hz=f_hz)
    if int(args.max_links) > 0:
        case_records = case_records[: int(args.max_links)]

    bundles: list[StandardOutputBundle] = []
    for c in case_records:
        scenario_id = str(c["scenario_id"])
        link_id = str(c["link_id"])
        case_id = str(c["case_id"])
        paths = list(c.get("paths", []))
        ray_rows = build_ray_table_from_rt(paths, matrix_source="A", include_material=True, include_angles=True)
        ray_rows = add_el_db(ray_rows, f_center_hz=float(args.f_center_hz), method="fspl")

        if scenario_id in {"A2", "A3", "A4", "A5"} and bool(args.strict_los_blocked):
            if any(int(r.get("los_flag_ray", 0)) == 1 for r in ray_rows):
                raise SystemExit(f"LOS blocked check failed: scenario={scenario_id}, link_id={link_id}")

        dt = max(float(args.delay_bin_ns), 1e-6) * 1e-9
        delay_tau_s = np.arange(0.0, max(float(args.Tmax_ns), float(args.Te_ns)) * 1e-9 + dt, dt, dtype=float)
        synth_U = {
            "scenario_id": scenario_id,
            "f_center_hz": float(args.f_center_hz),
            "yaw_deg": float(c.get("meta", {}).get("yaw_deg", 0.0)),
            "pitch_deg": float(c.get("meta", {}).get("pitch_deg", 0.0)),
            "roughness_flag": int(c.get("meta", {}).get("roughness_flag", 0)),
            "human_flag": int(c.get("meta", {}).get("human_flag", 0)),
            "material_class": str(c.get("meta", {}).get("material_class", "NA")),
            "EL_proxy_db": float(np.nanmedian(np.asarray([r.get("EL_db", np.nan) for r in ray_rows], dtype=float))),
        }
        pdp_obj = synthesize_dualcp_pdp(
            ray_rows,
            delay_tau_s,
            xpr_model=xpr_model,
            link_U=synth_U,
            rng=rng,
            include_xpd_tau=True,
            floor_model=floor_model if scenario_id == "C0" else None,
        )
        pdp = {
            "delay_tau_s": np.asarray(pdp_obj.delay_tau_s, dtype=float),
            "P_co": np.asarray(pdp_obj.P_co, dtype=float),
            "P_cross": np.asarray(pdp_obj.P_cross, dtype=float),
        }
        p_total = pdp["P_co"] + pdp["P_cross"]
        tau0 = estimate_tau0(
            pdp["delay_tau_s"],
            p_total,
            method="threshold",
            noise_tail_s=8e-9,
            margin_db=6.0,
        )
        early_mask, late_mask = make_early_late_masks(
            pdp["delay_tau_s"],
            tau0_s=float(tau0["tau0_s"]),
            Te_s=float(args.Te_ns) * 1e-9,
            Tmax_s=float(args.Tmax_ns) * 1e-9,
        )
        metrics = compute_link_metrics(
            pdp=pdp,
            delay_tau_s=pdp["delay_tau_s"],
            masks=(early_mask, late_mask),
            ds_reference=str(args.ds_reference),
            window_params={
                "tau0_s": float(tau0["tau0_s"]),
                "Te_s": float(args.Te_ns) * 1e-9,
                "Tmax_s": float(args.Tmax_ns) * 1e-9,
                "tau0_method": str(tau0["method"]),
                "noise_floor_def": "tail_median+margin",
            },
        )
        el_proxy_db = compute_el_proxy(
            ray_rows,
            pdp=pdp,
            mode=str(args.el_proxy_mode),
            early_mask=early_mask,
            L_ref_mode="los|minL",
            f_center_hz=float(args.f_center_hz),
        )
        link_meta = dict(c.get("meta", {}))
        link_meta["delay_tau_s"] = pdp["delay_tau_s"].tolist()
        U = build_link_U_from_scenario(
            link_meta,
            ray_rows=ray_rows,
            pdp=pdp,
            masks=(early_mask, late_mask),
            el_proxy_db=float(el_proxy_db),
        )

        bundle = StandardOutputBundle(
            link_id=link_id,
            scenario_id=scenario_id,
            case_id=case_id,
            rays=RayTable(rows=ray_rows),
            pdp=pdp_obj,
            metrics=metrics,
            conditions=U,
            provenance={
                "git_commit": commit,
                "git_branch": branch,
                "seed": int(args.seed),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "scenario_id": scenario_id,
                "case_id": case_id,
                "link_params": c.get("params", {}),
                "cmdline": " ".join(shlex.quote(x) for x in sys.argv),
            },
        )
        bundles.append(bundle)

    run_meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "scenario_id": str(args.scenario),
        "seed": int(args.seed),
        "nf": int(args.nf),
        "f_center_hz": float(args.f_center_hz),
        "Te_s": float(args.Te_ns) * 1e-9,
        "Tmax_s": float(args.Tmax_ns) * 1e-9,
        "delay_bin_s": float(args.delay_bin_ns) * 1e-9,
        "cmdline": " ".join(shlex.quote(x) for x in sys.argv),
        "git_commit": commit,
        "git_branch": branch,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_run(run_meta, bundles, out_h5=args.out_h5, run_id=run_id)
    csv_out = export_csv(bundles, out_dir=args.out_dir)
    summary = {
        "run_id": run_id,
        "out_h5": str(args.out_h5),
        "out_dir": str(args.out_dir),
        "n_links": len(bundles),
        "link_metrics_csv": csv_out["link_metrics_csv"],
        "rays_csv": csv_out["rays_csv"],
    }
    out_summary = Path(args.out_dir) / "run_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(args.out_h5))
    print(csv_out["link_metrics_csv"])
    print(csv_out["rays_csv"])
    print(str(out_summary))


if __name__ == "__main__":
    main()

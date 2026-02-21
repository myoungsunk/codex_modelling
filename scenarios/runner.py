"""Scenario sweep runner with HDF5 export, plots, and validation reports.

Example:
    python -m scenarios.runner --output outputs/rt_dataset.h5
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import numpy as np

from analysis.reciprocity import reciprocity_sanity
from analysis.tap_path_consistency import evaluate_dataset_tap_path_consistency, write_outlier_csv
from analysis.ctf_cir import ctf_to_cir, detect_cir_wrap, first_peak_tau_s, pdp, synthesize_ctf_with_source, tau_resolution_s
from analysis.run_model_fit import fit_and_generate
from analysis.xpd_stats import (
    conditional_fit,
    estimate_leakage_floor_from_antenna_config,
    gof_model_selection_db,
    incidence_angle_bin_label,
    make_subbands,
    leakage_limited_summary,
    pathwise_xpd,
)
from plots.model_compare import generate_rt_vs_synth_plots
from plots.plot_config import PlotConfig
from plots.p0_p13 import generate_all_plots
from rt_core.antenna import Antenna
from rt_core.geometry import normalize
from rt_io.hdf5_io import save_rt_dataset, self_test_meta_roundtrip
from scenarios import (
    A1_los_only,
    A2_pec_plane,
    A2_rotated_plane,
    A3_corner_2bounce,
    A3_rotated_dihedral,
    A4_dielectric_plane,
    A5_depol_stress,
    A6_cp_parity_benchmark,
    B0_room_box,
    C0_free_space,
)
from scenarios.common import paths_to_records, uwb_frequency


SCENARIOS = {
    "C0": C0_free_space,
    "A1": A1_los_only,
    "A2": A2_pec_plane,
    "A2R": A2_rotated_plane,
    "A3": A3_corner_2bounce,
    "A3R": A3_rotated_dihedral,
    "A4": A4_dielectric_plane,
    "A5": A5_depol_stress,
    "A6": A6_cp_parity_benchmark,
    "B0": B0_room_box,
}

DEFAULT_EXACT_BOUNCE = {"A2": 1, "A2R": 1, "A3": 2, "A3R": 2, "A4": 1}


def _parse_bases(basis: str | None, bases: str | None) -> list[str]:
    if basis:
        return [basis]
    if not bases:
        return ["linear", "circular"]
    parsed = [x.strip() for x in bases.split(",") if x.strip()]
    allowed = {"linear", "circular"}
    out = [x for x in parsed if x in allowed]
    return out or ["linear", "circular"]


def _basis_output_path(path: str | Path, basis: str, multi: bool) -> Path:
    p = Path(path)
    if not multi:
        return p
    if p.suffix:
        return p.with_name(f"{p.stem}_{basis}{p.suffix}")
    return p / basis


def _basis_output_dir(path: str | Path, basis: str, multi: bool) -> Path:
    p = Path(path)
    if not multi:
        return p
    return p / basis


def _path_power(path: dict[str, Any], matrix_source: str) -> float:
    use_j = str(matrix_source).upper() == "J" and "J_f" in path
    m = np.asarray(path["J_f"] if use_j else path["A_f"], dtype=np.complex128)
    return float(np.mean(np.abs(m) ** 2))


def _circular_delay_error(a_s: float, b_s: float, period_s: float) -> float:
    if period_s <= 0.0:
        return abs(float(a_s) - float(b_s))
    d = abs(float(a_s) - float(b_s)) % period_s
    return min(d, period_s - d)


def _gof_by_bucket(
    samples: list[dict[str, Any]],
    key: str,
    min_n: int = 20,
    bootstrap_B: int = 200,
    seed: int = 0,
    floor_db: float | None = None,
    pinned_tol_db: float = 0.5,
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[float]] = {}
    for s in samples:
        k = str(s.get(key, "NA"))
        buckets.setdefault(k, []).append(float(s.get("xpd_db", np.nan)))
    out: dict[str, dict[str, Any]] = {}
    for i, (k, vals) in enumerate(sorted(buckets.items(), key=lambda x: x[0])):
        out[k] = gof_model_selection_db(
            vals,
            min_n=min_n,
            bootstrap_B=bootstrap_B,
            seed=seed + i,
            floor_db=floor_db,
            pinned_tol_db=pinned_tol_db,
        )
    return out


def _git_meta() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = "unknown"
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True)
        dirty = bool(status.strip())
    except Exception:
        dirty = True
    return commit, dirty


def _projected_hv(boresight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = normalize(np.asarray(boresight, dtype=float))
    h = np.array([0.0, 1.0, 0.0], dtype=float)
    h = h - float(np.dot(h, b)) * b
    if np.linalg.norm(h) < 1e-9:
        h = np.array([0.0, 0.0, 1.0], dtype=float) - float(np.dot(np.array([0.0, 0.0, 1.0]), b)) * b
    h = normalize(h)
    v = normalize(np.cross(b, h))
    return h, v


def _antennas_from_points(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    basis: str,
    convention: str,
    antenna_config: dict[str, Any],
) -> tuple[Antenna, Antenna]:
    b_tx = normalize(np.asarray(rx_pos, dtype=float) - np.asarray(tx_pos, dtype=float))
    h_tx, v_tx = _projected_hv(b_tx)
    b_rx = normalize(np.asarray(tx_pos, dtype=float) - np.asarray(rx_pos, dtype=float))
    h_rx, v_rx = _projected_hv(b_rx)
    tx = Antenna(
        position=np.asarray(tx_pos, dtype=float),
        boresight=b_tx,
        h_axis=h_tx,
        v_axis=v_tx,
        basis=basis,
        convention=convention,
        cross_pol_leakage_db=float(antenna_config.get("tx_cross_pol_leakage_db", 35.0)),
        axial_ratio_db=float(antenna_config.get("tx_axial_ratio_db", 0.0)),
        enable_coupling=bool(antenna_config.get("enable_coupling", True)),
    )
    rx = Antenna(
        position=np.asarray(rx_pos, dtype=float),
        boresight=b_rx,
        h_axis=h_rx,
        v_axis=v_rx,
        basis=basis,
        convention=convention,
        cross_pol_leakage_db=float(antenna_config.get("rx_cross_pol_leakage_db", 35.0)),
        axial_ratio_db=float(antenna_config.get("rx_axial_ratio_db", 0.0)),
        enable_coupling=bool(antenna_config.get("enable_coupling", True)),
    )
    return tx, rx


def _build_scene_from_module(mod: Any, params: dict[str, Any]) -> list[Any]:
    if not hasattr(mod, "build_scene"):
        return []
    sig = inspect.signature(mod.build_scene)
    p = dict(params)
    if "material_name" in sig.parameters and "material" in p:
        p["material_name"] = p["material"]
    kwargs = {k: p[k] for k in sig.parameters if k in p}
    return mod.build_scene(**kwargs)


def compute_reciprocity_checks(
    data: dict[str, Any],
    matrix_source: str,
    scenario_ids: list[str] | None = None,
    tau_tol_s: float = 1e-12,
    sigma_tol_db: float = 1e-6,
    require_bidirectional_paths: bool = True,
) -> dict[str, Any]:
    if scenario_ids is None or len(scenario_ids) == 0:
        targets = [sid for sid in data.get("scenarios", {}).keys() if sid in SCENARIOS]
    else:
        targets = [sid for sid in scenario_ids if sid in SCENARIOS]
    freq = np.asarray(data["frequency"], dtype=float)
    basis = str(data.get("meta", {}).get("basis", "linear"))
    conv = str(data.get("meta", {}).get("convention", "IEEE-RHCP"))
    ant_cfg = data.get("meta", {}).get("antenna_config", {})
    force_cp = bool(data.get("meta", {}).get("force_cp_swap_on_odd_reflection", False))

    entries = []
    for sid in targets:
        if sid not in data["scenarios"] or sid not in SCENARIOS:
            continue
        mod = SCENARIOS[sid]
        for cid, case in data["scenarios"][sid]["cases"].items():
            paths = case.get("paths", [])
            if not paths or "points" not in paths[0] or len(paths[0]["points"]) < 2:
                continue
            tx_pos = np.asarray(paths[0]["points"][0], dtype=float)
            rx_pos = np.asarray(paths[0]["points"][-1], dtype=float)
            tx, rx = _antennas_from_points(tx_pos, rx_pos, basis=basis, convention=conv, antenna_config=ant_cfg)
            scene = _build_scene_from_module(mod, case.get("params", {}))
            max_bounce = int(max(int(p["meta"]["bounce_count"]) for p in paths))
            los_enabled = bool(any(int(p["meta"]["bounce_count"]) == 0 for p in paths))
            out = reciprocity_sanity(
                scene=scene,
                tx=tx,
                rx=rx,
                f_hz=freq,
                max_bounce=max_bounce,
                los_enabled=los_enabled,
                use_fspl=True,
                force_cp_swap_on_odd_reflection=force_cp,
                matrix_source=matrix_source,
                tau_tol_s=float(tau_tol_s),
                sigma_tol_db=float(sigma_tol_db),
            )
            out["scenario_id"] = sid
            out["case_id"] = str(cid)
            entries.append(out)

    if not entries:
        return {"entries": []}
    covered_scenarios = sorted({str(e.get("scenario_id", "NA")) for e in entries})
    if require_bidirectional_paths:
        checked_entries = [e for e in entries if int(e.get("n_forward", 0)) > 0 and int(e.get("n_reverse", 0)) > 0]
    else:
        checked_entries = list(entries)
    reverse_empty_entries = [e for e in entries if int(e.get("n_forward", 0)) > 0 and int(e.get("n_reverse", 0)) == 0]

    n_fwd = np.asarray([float(e.get("n_forward", 0)) for e in checked_entries], dtype=float)
    n_m = np.asarray([float(e.get("n_matched", 0)) for e in checked_entries], dtype=float)
    dt = np.asarray([float(e.get("delta_tau_max_s", np.nan)) for e in checked_entries], dtype=float)
    ds = np.asarray([float(e.get("delta_sigma_max_db", np.nan)) for e in checked_entries], dtype=float)
    unmatched = np.asarray([int(e.get("unmatched_count", 0)) for e in checked_entries], dtype=int)
    tau_mm = np.asarray([int(e.get("tau_mismatch_count", 0)) for e in checked_entries], dtype=int)
    mat_mm = np.asarray([int(e.get("matrix_mismatch_count", 0)) for e in checked_entries], dtype=int)
    checked_scenarios = sorted({str(e.get("scenario_id", "NA")) for e in checked_entries})

    by_scenario: dict[str, dict[str, int]] = {}
    worst: list[dict[str, Any]] = []
    for e in entries:
        sid = str(e.get("scenario_id", "NA"))
        by_scenario.setdefault(
            sid,
            {
                "covered_cases": 0,
                "checked_cases": 0,
                "reverse_empty_cases": 0,
                "reverse_trace_empty": 0,
                "unmatched_forward": 0,
                "unmatched_reverse": 0,
                "tau_mismatch_count": 0,
                "matrix_mismatch_count": 0,
            },
        )
        by_scenario[sid]["covered_cases"] += 1
        n_f = int(e.get("n_forward", 0))
        n_r = int(e.get("n_reverse", 0))
        if (not require_bidirectional_paths) or (n_f > 0 and n_r > 0):
            by_scenario[sid]["checked_cases"] += 1
            by_scenario[sid]["unmatched_forward"] += int(e.get("unmatched_count", 0))
            by_scenario[sid]["unmatched_reverse"] += int(e.get("unmatched_reverse_count", 0))
            by_scenario[sid]["tau_mismatch_count"] += int(e.get("tau_mismatch_count", 0))
            by_scenario[sid]["matrix_mismatch_count"] += int(e.get("matrix_mismatch_count", 0))
            for v in e.get("violations", []):
                w = dict(v)
                w["scenario_id"] = sid
                w["case_id"] = str(e.get("case_id", "NA"))
                worst.append(w)
        elif n_f > 0 and n_r == 0:
            by_scenario[sid]["reverse_empty_cases"] += 1
            by_scenario[sid]["reverse_trace_empty"] += 1
            worst.append(
                {
                    "scenario_id": sid,
                    "case_id": str(e.get("case_id", "NA")),
                    "type": "reverse_trace_empty",
                    "path_index_forward": -1,
                    "path_index_reverse": -1,
                    "bounce_count": -1,
                    "surface_pattern": "NA",
                    "delta_tau_s": np.nan,
                    "delta_sigma_max_db": np.nan,
                }
            )

    def _sev(v: dict[str, Any]) -> tuple[float, float, float]:
        t = str(v.get("type", ""))
        pri = (
            3.0
            if t in {"unmatched_forward", "unmatched_reverse", "reverse_trace_empty"}
            else (2.0 if t == "matrix_invariant_mismatch" else 1.0)
        )
        ds = float(v.get("delta_sigma_max_db", np.nan))
        dt = float(v.get("delta_tau_s", np.nan))
        return (
            pri,
            ds if np.isfinite(ds) else -1.0,
            dt if np.isfinite(dt) else -1.0,
        )

    worst_sorted = sorted(worst, key=_sev, reverse=True)[:5]
    requirement_flags: dict[str, bool] = {}
    for sid, cnt in by_scenario.items():
        if require_bidirectional_paths:
            requirement_flags[sid] = bool(
                int(cnt.get("covered_cases", 0)) == int(cnt.get("checked_cases", 0))
                and int(cnt.get("reverse_empty_cases", 0)) == 0
            )
        else:
            requirement_flags[sid] = True

    reverse_trace_empty_total = int(sum(int(c.get("reverse_trace_empty", 0)) for c in by_scenario.values()))
    unmatched_forward_total = int(sum(int(c.get("unmatched_forward", 0)) for c in by_scenario.values()))
    unmatched_reverse_total = int(sum(int(c.get("unmatched_reverse", 0)) for c in by_scenario.values()))
    tau_mismatch_total = int(sum(int(c.get("tau_mismatch_count", 0)) for c in by_scenario.values()))
    matrix_mismatch_total = int(sum(int(c.get("matrix_mismatch_count", 0)) for c in by_scenario.values()))
    coverage_pass = bool(
        (not require_bidirectional_paths)
        or (int(len(checked_entries)) == int(len(entries)) and int(len(reverse_empty_entries)) == 0)
    )

    return {
        "entries": entries,
        "covered_scenarios": covered_scenarios,
        "covered_cases": int(len(entries)),
        "checked_scenarios": checked_scenarios,
        "checked_cases": int(len(checked_entries)),
        "matched_ratio_global": float(np.sum(n_m) / max(np.sum(n_fwd), 1.0)) if len(checked_entries) else np.nan,
        "delta_tau_max_s_global": float(np.nanmax(dt)) if len(checked_entries) else np.nan,
        "delta_sigma_max_db_global": float(np.nanmax(ds)) if len(checked_entries) else np.nan,
        "delta_fro_max_db_global": float(
            np.nanmax(np.asarray([float(e.get("delta_fro_max_db", np.nan)) for e in checked_entries], dtype=float))
        )
        if len(checked_entries)
        else np.nan,
        "require_bidirectional_paths": bool(require_bidirectional_paths),
        "reverse_empty_cases": int(len(reverse_empty_entries)),
        "unmatched_count_total": int(np.sum(unmatched)),
        "unmatched_reverse_count_total": int(
            np.sum(np.asarray([int(e.get("unmatched_reverse_count", 0)) for e in checked_entries], dtype=int))
        ),
        "tau_mismatch_count_total": int(np.sum(tau_mm)),
        "matrix_mismatch_count_total": int(np.sum(mat_mm)),
        "type_counts_total": {
            "reverse_trace_empty": reverse_trace_empty_total,
            "unmatched_forward": unmatched_forward_total,
            "unmatched_reverse": unmatched_reverse_total,
            "tau_mismatch": tau_mismatch_total,
            "matrix_mismatch": matrix_mismatch_total,
        },
        "tau_tol_s": float(tau_tol_s),
        "sigma_tol_db": float(sigma_tol_db),
        "counts_by_scenario": by_scenario,
        "requirement_flags_by_scenario": requirement_flags,
        "coverage_pass": coverage_pass,
        "worst_violations": worst_sorted,
        "matrix_source": matrix_source,
    }


def build_dataset(
    basis: str = "linear",
    convention: str = "IEEE-RHCP",
    nf: int = 256,
    antenna_config: dict[str, Any] | None = None,
    force_cp_swap_on_odd_reflection: bool = False,
) -> dict[str, Any]:
    freq = uwb_frequency(nf=nf)
    git_commit, git_dirty = _git_meta()
    data: dict[str, Any] = {
        "meta": {
            "basis": basis,
            "convention": convention,
            "antenna_config": antenna_config or {},
            "force_cp_swap_on_odd_reflection": force_cp_swap_on_odd_reflection,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
        "frequency": freq,
        "scenarios": {},
    }

    for sid, mod in SCENARIOS.items():
        cases: dict[str, Any] = {}
        for idx, params in enumerate(mod.build_sweep_params()):
            paths = mod.run_case(
                params,
                freq,
                basis=basis,
                antenna_config=antenna_config or {},
                force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
            )
            cases[str(idx)] = {"params": params, "paths": paths_to_records(paths)}
        data["scenarios"][sid] = {"cases": cases}
    return data


def build_quality_report(
    data: dict[str, Any],
    out_md: str | Path,
    matrix_source: str = "A",
    model_metrics: dict[str, Any] | None = None,
    reciprocity_metrics: dict[str, Any] | None = None,
    tap_path_metrics: dict[str, Any] | None = None,
) -> Path:
    lines = ["# RT Validation Report", ""]
    basis_now = str(data.get("meta", {}).get("basis", "NA"))
    convention_now = str(data.get("meta", {}).get("convention", "NA"))
    git_dirty_now = bool(data.get("meta", {}).get("git_dirty", True))
    release_mode_now = bool(data.get("meta", {}).get("release_mode", False))
    lines.append(f"- basis: {basis_now}")
    lines.append(f"- convention: {convention_now}")
    lines.append(f"- git_commit: {data.get('meta', {}).get('git_commit', 'unknown')}")
    lines.append(f"- git_dirty: {git_dirty_now}")
    lines.append(f"- release_mode: {release_mode_now}")
    lines.append(f"- cmdline: {data.get('meta', {}).get('cmdline', '')}")
    lines.append(f"- seed: {data.get('meta', {}).get('seed', '')}")
    if git_dirty_now:
        lines.append("- WARNING: git_dirty=True (provenance risk for paper/release artifacts).")
    else:
        lines.append("- git_clean_check: PASS (git_dirty=False)")
    if release_mode_now and git_dirty_now:
        lines.append("- FAIL: release_mode requires git_dirty=False.")
    xpd_src_now = str(data.get("meta", {}).get("xpd_matrix_source", matrix_source))
    exact_bounce_now = data.get("meta", {}).get("exact_bounce_defaults", DEFAULT_EXACT_BOUNCE)
    lines.append(f"- xpd_matrix_source: {xpd_src_now}")
    lines.append(f"- exact_bounce_defaults: {exact_bounce_now}")
    lines.append("- report_exact_bounce_applied: True (scenario-specific default map)")
    antenna_config = data.get("meta", {}).get("antenna_config", {})
    lines.append(f"- antenna_config: {antenna_config}")
    physics_mode = bool(
        data.get("meta", {}).get(
            "physics_validation_mode",
            not bool(antenna_config.get("enable_coupling", True)),
        )
    )
    lines.append(f"- physics_validation_mode: {physics_mode}")
    floor_info = estimate_leakage_floor_from_antenna_config(antenna_config)
    lines.append(
        f"- predicted_leakage_floor_db: {floor_info['xpd_floor_db']:.3f} "
        f"(eps_tx={floor_info['eps_tx']:.5f}, eps_rx={floor_info['eps_rx']:.5f})"
    )
    lines.append(f"- model_compare_enabled: {model_metrics is not None}")
    lines.append("")

    freq = np.asarray(data["frequency"], dtype=float)
    res_s = tau_resolution_s(freq, nfft=2048)
    delay_period_s = float(1.0 / (freq[1] - freq[0])) if len(freq) > 1 else 0.0
    scenario_medians: list[float] = []

    for sid, sc in data["scenarios"].items():
        lines += [f"## {sid}", ""]
        all_paths = []
        all_ctx: list[dict[str, Any]] = []
        path_counts_per_case: list[int] = []
        d3_ok = 0
        d3_n = 0
        d3_skipped_overlap = 0
        wrap_count = 0
        d5_max_abs = 0.0
        exact_bounce = DEFAULT_EXACT_BOUNCE.get(sid)
        for cid, case in sc["cases"].items():
            paths = case["paths"]
            path_counts_per_case.append(len(paths))
            stat_paths = [p for p in paths if exact_bounce is None or int(p["meta"]["bounce_count"]) == exact_bounce]
            for p in paths:
                all_paths.append(p)
                all_ctx.append({"params": case.get("params", {}), "scenario_id": sid, "case_id": cid})
            counts = [p["meta"]["bounce_count"] for p in paths]
            lines.append(f"- case {cid}: paths={len(paths)}, bounce_dist={dict((c, counts.count(c)) for c in sorted(set(counts)))}")
            if paths:
                powers = [_path_power(p, matrix_source=matrix_source) for p in paths]
                j = int(np.argmax(powers))
                lines.append(f"- strongest path: tau={paths[j]['tau_s']:.3e}s, power={powers[j]:.3e}")
                has_los = any(p["meta"]["bounce_count"] == 0 for p in paths)
                lines.append(f"- LOS exists: {has_los}")

                # D3/D4/D5: CIR consistency and wrap detection.
                H = synthesize_ctf_with_source(paths, freq, matrix_source=matrix_source)
                h_tau, tau_s = ctf_to_cir(H, freq, nfft=2048)
                peak_tau = first_peak_tau_s(h_tau, tau_s)
                strongest_tau = float(paths[j]["tau_s"])
                alias_tau = strongest_tau % delay_period_s if delay_period_s > 0.0 else strongest_tau
                p_sort = np.sort(np.asarray(powers, dtype=float))[::-1]
                if len(p_sort) <= 1:
                    dominant = True
                else:
                    dom_db = 10.0 * np.log10((p_sort[0] + 1e-18) / (p_sort[1] + 1e-18))
                    dominant = bool(dom_db >= 3.0)
                if dominant:
                    d3_n += 1
                    if _circular_delay_error(peak_tau, alias_tau, delay_period_s) <= 2.0 * res_s:
                        d3_ok += 1
                else:
                    d3_skipped_overlap += 1
                if strongest_tau > delay_period_s and _circular_delay_error(peak_tau, alias_tau, delay_period_s) <= 2.0 * res_s:
                    wrap_count += 1
                elif detect_cir_wrap(h_tau, tau_s, expected_first_tau_s=strongest_tau, resolution_s=res_s):
                    wrap_count += 1
                p = pdp(h_tau)
                d5_max_abs = max(d5_max_abs, float(np.max(np.abs((p["co"] + p["cross"]) - p["sum"]))))

                if sid.startswith("A2") and has_los:
                    lines.append(f"- WARNING: {sid} should block LOS but LOS path is present")
                if sid.startswith("A2") and not any(p["meta"]["bounce_count"] == 1 for p in paths):
                    lines.append(f"- WARNING: {sid} missing 1-bounce path")
                if sid.startswith("A3") and not any(p["meta"]["bounce_count"] == 2 for p in paths):
                    lines.append(f"- WARNING: {sid} missing 2-bounce path")
            if exact_bounce is not None and len(stat_paths) == 0:
                lines.append(f"- WARNING: no paths matched exact_bounce={exact_bounce} for stats")

        n_cases = len(path_counts_per_case)
        avg_paths_per_case = float(np.mean(path_counts_per_case)) if n_cases > 0 else 0.0
        lines.append(f"- avg_paths_per_case: {avg_paths_per_case:.2f} (cases={n_cases})")
        if n_cases > 0 and avg_paths_per_case < 2.0:
            lines.append(
                "- WARNING: low path count per case; statistics may be unstable "
                "(single-path dominance likely)."
            )
        lines.append(f"- cir_peak_match_ratio: {d3_ok}/{d3_n}")
        lines.append(f"- cir_peak_match_skipped_overlap_cases: {d3_skipped_overlap}")
        lines.append(f"- delay_ambiguity_period_ns: {delay_period_s * 1e9:.3f}")
        lines.append(f"- wrap_detected_cases: {wrap_count}")
        lines.append(f"- pdp_sum_consistency_max_abs: {d5_max_abs:.3e}")
        if wrap_count > 0:
            lines.append(
                "- WARNING: CIR delay ambiguity/wrap detected (tau > 1/df aliasing likely); "
                "use denser frequency grid, delay unwrapping, or path-domain validation."
            )

        samples = pathwise_xpd(all_paths, exact_bounce=exact_bounce, matrix_source=matrix_source)
        par_stats = conditional_fit(samples, keys=["parity"])
        lines.append(f"- parity XPD stats (exact_bounce={exact_bounce}): {par_stats}")
        xpd_vals = [float(s["xpd_db"]) for s in samples]
        leak_chk = leakage_limited_summary(xpd_vals, xpd_floor_db=float(floor_info["xpd_floor_db"]))
        lines.append(
            "- leakage-limited check: "
            f"median_xpd_db={leak_chk['median_xpd_db']:.3f}, "
            f"sigma_db={leak_chk['sigma_xpd_db']:.3f}, "
            f"delta_to_floor_db={leak_chk['delta_floor_db']:.3f}, "
            f"floor_db={floor_info['xpd_floor_db']:.3f}"
        )
        if bool(leak_chk["is_leakage_limited"]):
            warn = (
                "XPD appears leakage-limited; use --physics-validation-mode "
                "and/or --xpd-matrix-source J for propagation-only analysis."
            )
            lines.append(f"- WARNING: {warn}")
            print(
                f"[WARNING][{sid}] {warn} "
                f"(median={leak_chk['median_xpd_db']:.3f} dB, "
                f"sigma={leak_chk['sigma_xpd_db']:.3f} dB, "
                f"floor={floor_info['xpd_floor_db']:.3f} dB, "
                f"antenna_config={antenna_config})"
            )
        if np.isfinite(leak_chk["median_xpd_db"]):
            scenario_medians.append(float(leak_chk["median_xpd_db"]))

        # E8: GOF diagnostics for major conditional buckets.
        gof_min_n = 20
        gof_B = 200
        if len(samples) > 0:
            taus = np.asarray([float(s["tau_s"]) for s in samples], dtype=float)
            split_tau = float(np.median(taus))
            enriched: list[dict[str, Any]] = []
            inc_bins = [0.0, 20.0, 40.0, 60.0, 90.0]
            for s in samples:
                i = int(s["path_index"])
                p = all_paths[i]
                meta = p.get("meta", {})
                ctx = all_ctx[i] if i < len(all_ctx) else {"params": {}}
                e = dict(s)
                e["material"] = str(ctx.get("params", {}).get("material", "NA"))
                e["incidence_angle_bin"] = incidence_angle_bin_label(meta.get("incidence_angles", []), bins_deg=inc_bins)
                e["delay_bin"] = "early" if float(s["tau_s"]) <= split_tau else "late"
                enriched.append(e)

            gof_keys = ["parity", "material", "incidence_angle_bin", "delay_bin"]
            for gk in gof_keys:
                lines.append(f"- GOF[{gk}] (min_n={gof_min_n}, bootstrap_B={gof_B}):")
                table = _gof_by_bucket(
                    enriched,
                    key=gk,
                    min_n=gof_min_n,
                    bootstrap_B=gof_B,
                    seed=0,
                    floor_db=float(floor_info["xpd_floor_db"]),
                    pinned_tol_db=0.5,
                )
                for bk, gr in table.items():
                    if gr.get("status") == "INSUFFICIENT":
                        lines.append(
                            f"-   {bk}: INSUFFICIENT (n={int(gr.get('n', 0))}, n_fit={int(gr.get('n_fit', 0))})"
                        )
                        lines.append(f"-   WARNING: GOF skipped for {gk}={bk} due to insufficient samples.")
                        continue
                    bm = str(gr.get("best_model", "NA"))
                    bmm = (gr.get("best_metrics", {}) or {})
                    pre = (gr.get("pre_normal", {}) or {})
                    post = (gr.get("post_normal", {}) or {})
                    lines.append(
                        f"-   {bk}: n={int(gr.get('n', 0))}, n_fit={int(gr.get('n_fit', 0))}, "
                        f"excluded={int(gr.get('n_excluded', 0))} "
                        f"(floor={int(gr.get('excluded_floor_count', 0))}, pinned={int(gr.get('excluded_pinned_count', 0))})"
                    )
                    lines.append(
                        f"-   best_model={bm}, AIC={float(bmm.get('aic', np.nan)):.3f}, "
                        f"BIC={float(bmm.get('bic', np.nan)):.3f}, "
                        f"qq_r={float(bmm.get('qq_r', np.nan)):.4f}, "
                        f"ks_p_boot={float(bmm.get('ks_p_boot', np.nan)):.4f}"
                    )
                    lines.append(
                        f"-   normal_pre: qq_r={float(pre.get('qq_r', np.nan)):.4f}, "
                        f"ks_p_boot={float(pre.get('ks_p_boot', np.nan)):.4f}; "
                        f"normal_post: qq_r={float(post.get('qq_r', np.nan)):.4f}, "
                        f"ks_p_boot={float(post.get('ks_p_boot', np.nan)):.4f}"
                    )
                    lines.append(
                        f"-   model_decision: single_normal_fail={bool(gr.get('single_normal_fail', False))}, "
                        f"alternative_improved={bool(gr.get('alternative_improved', False))}, "
                        f"reason={str(gr.get('model_reason', 'NA'))}"
                    )
                    if bool(gr.get("single_normal_fail", False)) and bool(gr.get("alternative_improved", False)):
                        lines.append("-   PASS: single-normal failed but alternative model improved fit.")
                    elif bool(gr.get("warning", False)):
                        lines.append("-   WARNING: best-model GOF remains borderline (qq_r<0.98 or ks_p_boot<0.05).")

        # E4: early/late trend diagnostic.
        if len(samples) >= 2:
            taus = np.asarray([float(s["tau_s"]) for s in samples], dtype=float)
            split = float(np.median(taus))
            early = np.asarray([float(s["xpd_db"]) for s in samples if float(s["tau_s"]) <= split], dtype=float)
            late = np.asarray([float(s["xpd_db"]) for s in samples if float(s["tau_s"]) > split], dtype=float)
            if len(early) > 0 and len(late) > 0:
                early_mu = float(np.mean(early))
                late_mu = float(np.mean(late))
                lines.append(f"- early_late_xpd_mu_db: early={early_mu:.3f}, late={late_mu:.3f}, split_tau_s={split:.3e}")
                if early_mu <= late_mu:
                    lines.append("- NOTE: early<=late observed (environment-dependent); do not over-claim early-tap advantage.")

        # E3: subband variability diagnostic with leakage context.
        if len(all_paths) > 0:
            subbands = make_subbands(len(freq), 3)
            sb = pathwise_xpd(all_paths, subbands=subbands, exact_bounce=exact_bounce, matrix_source=matrix_source)
            sb_mu = []
            for bidx in range(len(subbands)):
                vals = [float(s["xpd_db"]) for s in sb if int(s.get("subband", -1)) == bidx]
                if vals:
                    sb_mu.append(float(np.mean(vals)))
            if len(sb_mu) >= 2:
                sb_span = float(max(sb_mu) - min(sb_mu))
                lines.append(f"- subband_mu_span_db: {sb_span:.3f}")
                if sb_span < 0.5 and bool(leak_chk["is_leakage_limited"]):
                    lines.append("- NOTE: weak subband variation likely leakage-floor dominated.")
                elif sb_span < 0.5:
                    lines.append("- NOTE: weak subband variation observed (may be physically weak frequency dependence).")

        if sid == "A4":
            with_mat = []
            for _, case in sc["cases"].items():
                mat = str(case["params"].get("material", "NA"))
                for s in pathwise_xpd(case["paths"], exact_bounce=1, matrix_source=matrix_source):
                    s["material"] = mat
                    with_mat.append(s)
            mat_stats = conditional_fit(with_mat, keys=["material"])
            lines.append(f"- material sub-summary: {mat_stats}")
        if sid == "A6":
            a6_odd_ang = []
            a6_even_ang = []
            for p in all_paths:
                angs = [float(a) for a in p.get("meta", {}).get("incidence_angles", [])]
                if not angs:
                    continue
                if int(p.get("meta", {}).get("bounce_count", 0)) == 1:
                    a6_odd_ang.append(float(np.rad2deg(max(angs))))
                elif int(p.get("meta", {}).get("bounce_count", 0)) == 2:
                    a6_even_ang.append(float(np.rad2deg(max(angs))))
            if a6_odd_ang:
                lines.append(f"- A6_near_normal_odd_max_deg: {max(a6_odd_ang):.3f}")
            if a6_even_ang:
                lines.append(f"- A6_near_normal_even_max_deg: {max(a6_even_ang):.3f}")
            lines.append(
                "- note: A6 is a near-normal-incidence CP benchmark; odd/even trends "
                "from this setup should not be generalized to arbitrary oblique incidence."
            )
        lines.append("")

    # C4: check pinned behavior under physics validation mode.
    if physics_mode and len(scenario_medians) >= 2:
        std_med = float(np.std(np.asarray(scenario_medians, dtype=float), ddof=1))
        lines.append("## Physics Mode Check")
        lines.append("")
        lines.append(f"- scenario_median_xpd_std_db: {std_med:.3f}")
        if std_med < 1.0:
            lines.append("- WARNING: medians appear pinned across scenarios even in physics mode.")
        else:
            lines.append("- non-pinned behavior observed across scenarios in physics mode.")
        lines.append("")

    if model_metrics is not None:
        lines.append("## Model Compare (F2-F6)")
        lines.append("")
        for k in sorted(model_metrics.keys()):
            lines.append(f"- {k}: {model_metrics[k]}")
        f3_delta = float(model_metrics.get("f3_xpd_mu_delta_abs_db", np.nan))
        f3_sigma_delta = float(model_metrics.get("f3_xpd_sigma_delta_abs_db", np.nan))
        f3_ks2_p = float(model_metrics.get("f3_xpd_ks2_p", np.nan))
        f3_ks2_status = str(model_metrics.get("f3_xpd_ks2_status", "NA"))
        f3_ks2_reason = str(model_metrics.get("f3_xpd_ks2_reason", "NA"))
        clamp_rate = float(model_metrics.get("synthetic_kappa_clamp_rate", np.nan))
        trunc_rate = float(model_metrics.get("synthetic_kappa_truncation_rate", np.nan))
        if np.isfinite(f3_delta):
            lines.append(f"- f3_xpd_mu_delta_abs_db: {f3_delta:.3f}")
            if np.isfinite(f3_sigma_delta):
                lines.append(f"- f3_xpd_sigma_delta_abs_db: {f3_sigma_delta:.3f}")
            if np.isfinite(f3_ks2_p):
                lines.append(f"- f3_xpd_ks2_p: {f3_ks2_p:.4f}")
            if f3_ks2_status != "NA":
                lines.append(f"- f3_xpd_ks2_status: {f3_ks2_status} ({f3_ks2_reason})")
            if np.isfinite(clamp_rate):
                lines.append(f"- synthetic_kappa_clamp_rate: {clamp_rate:.4f}")
            if np.isfinite(trunc_rate):
                lines.append(f"- synthetic_kappa_truncation_rate: {trunc_rate:.4f}")
            if f3_delta > 3.0 or (np.isfinite(f3_sigma_delta) and f3_sigma_delta > 6.0):
                reasons = []
                if np.isfinite(clamp_rate) and clamp_rate > 0.01:
                    reasons.append(f"kappa_clamp_rate={clamp_rate:.3f}")
                if np.isfinite(trunc_rate) and trunc_rate > 0.20:
                    reasons.append(f"kappa_truncation_rate={trunc_rate:.3f}")
                noise_sig = float(model_metrics.get("synthetic_xpd_freq_noise_sigma_db", np.nan))
                if np.isfinite(noise_sig) and noise_sig > 0.0:
                    reasons.append(f"xpd_freq_noise_sigma_db={noise_sig:.3f}")
                sb_rmse = float(model_metrics.get("f5_subband_mu_rmse_db", np.nan))
                if np.isfinite(sb_rmse) and sb_rmse > 10.0:
                    reasons.append(f"subband_mu_rmse_db={sb_rmse:.3f}")
                if not reasons:
                    reasons.append("sampling mismatch or sigma over-dispersion")
                lines.append(
                    "- WARNING: F3 distribution mismatch above target (|mu|>3 dB or |sigma|>6 dB); "
                    "check synthetic sampling settings. Suspected causes: "
                    + ", ".join(reasons)
                )
            if f3_ks2_status == "FAIL":
                lines.append("- WARNING: F3 KS 2-sample test rejects RT vs Synthetic XPD distribution (FAIL).")
            elif f3_ks2_status.startswith("SKIPPED"):
                lines.append(f"- F3 KS 2-sample check: {f3_ks2_status} ({f3_ks2_reason}).")
            elif np.isfinite(f3_ks2_p):
                lines.append("- F3 KS 2-sample check: PASS (p>=0.05).")
            else:
                lines.append("- F3 KS 2-sample check: SKIPPED (insufficient samples).")
        if not bool(model_metrics.get("f4_parity_direction_match", False)):
            lines.append("- WARNING: synthetic parity direction does not match RT.")
        phase_basis = str(model_metrics.get("phase_test_basis", model_metrics.get("f6_phase_test_basis", "unknown")))
        phase_common = bool(model_metrics.get("common_phase_removed", model_metrics.get("f6_common_phase_removed", True)))
        phase_per_ray = bool(model_metrics.get("per_ray_sampling", model_metrics.get("f6_per_ray_sampling", True)))
        phase_rt_status = str(model_metrics.get("phase_uniformity_rt_status", "INFO_DETERMINISTIC"))
        phase_sy_status = str(model_metrics.get("phase_uniformity_synth_status", "NA"))
        lines.append(
            f"- phase_test_config: basis={phase_basis}, common_phase_removed={phase_common}, per_ray_sampling={phase_per_ray}"
        )
        lines.append(f"- phase_test_rt_status: {phase_rt_status}")
        lines.append(f"- phase_test_synth_status: {phase_sy_status}")
        lines.append("")

    if reciprocity_metrics is not None and reciprocity_metrics.get("entries"):
        lines.append("## Reciprocity Sanity (C10)")
        lines.append("")
        lines.append(f"- matrix_source: {reciprocity_metrics.get('matrix_source', matrix_source)}")
        lines.append(f"- covered_scenarios: {reciprocity_metrics.get('covered_scenarios', [])}")
        lines.append(f"- covered_cases: {int(reciprocity_metrics.get('covered_cases', 0))}")
        lines.append(f"- checked_scenarios: {reciprocity_metrics.get('checked_scenarios', [])}")
        lines.append(f"- checked_cases: {int(reciprocity_metrics.get('checked_cases', 0))}")
        lines.append(
            f"- require_bidirectional_paths: {bool(reciprocity_metrics.get('require_bidirectional_paths', True))}, "
            f"reverse_empty_cases={int(reciprocity_metrics.get('reverse_empty_cases', 0))}"
        )
        lines.append(
            f"- tolerance: tau_tol_s={float(reciprocity_metrics.get('tau_tol_s', np.nan)):.3e}, "
            f"sigma_tol_db={float(reciprocity_metrics.get('sigma_tol_db', np.nan)):.3e}"
        )
        lines.append(f"- matched_ratio_global: {float(reciprocity_metrics.get('matched_ratio_global', np.nan)):.6f}")
        lines.append(f"- delta_tau_max_s_global: {float(reciprocity_metrics.get('delta_tau_max_s_global', np.nan)):.3e}")
        lines.append(f"- delta_sigma_max_db_global: {float(reciprocity_metrics.get('delta_sigma_max_db_global', np.nan)):.3e}")
        lines.append(f"- delta_fro_max_db_global: {float(reciprocity_metrics.get('delta_fro_max_db_global', np.nan)):.3e}")
        lines.append(f"- c10_coverage_pass: {bool(reciprocity_metrics.get('coverage_pass', False))}")
        if bool(reciprocity_metrics.get("require_bidirectional_paths", True)):
            cc = int(reciprocity_metrics.get("checked_cases", 0))
            cv = int(reciprocity_metrics.get("covered_cases", 0))
            re = int(reciprocity_metrics.get("reverse_empty_cases", 0))
            lines.append(
                f"- c10_hard_gate: checked_cases==covered_cases ({cc}=={cv}) "
                f"AND reverse_empty_cases==0 ({re}==0)"
            )
            if not (cc == cv and re == 0):
                lines.append("- FAIL: C10 bidirectional coverage gate not satisfied.")
        lines.append(
            "- type_counts: "
            f"unmatched_count_total={int(reciprocity_metrics.get('unmatched_count_total', 0))}, "
            f"unmatched_reverse_count_total={int(reciprocity_metrics.get('unmatched_reverse_count_total', 0))}, "
            f"tau_mismatch_count_total={int(reciprocity_metrics.get('tau_mismatch_count_total', 0))}, "
            f"matrix_mismatch_count_total={int(reciprocity_metrics.get('matrix_mismatch_count_total', 0))}"
        )
        tct = reciprocity_metrics.get("type_counts_total", {}) or {}
        lines.append(
            "- type_counts_total: "
            f"reverse_trace_empty={int(tct.get('reverse_trace_empty', 0))}, "
            f"unmatched_forward={int(tct.get('unmatched_forward', 0))}, "
            f"unmatched_reverse={int(tct.get('unmatched_reverse', 0))}, "
            f"tau_mismatch={int(tct.get('tau_mismatch', 0))}, "
            f"matrix_mismatch={int(tct.get('matrix_mismatch', 0))}"
        )
        lines.append("- counts_by_scenario:")
        req_flags = reciprocity_metrics.get("requirement_flags_by_scenario", {}) or {}
        for sid, c in sorted((reciprocity_metrics.get("counts_by_scenario", {}) or {}).items()):
            lines.append(
                f"-   {sid}: covered_cases={int(c.get('covered_cases', 0))}, "
                f"checked_cases={int(c.get('checked_cases', 0))}, "
                f"reverse_empty_cases={int(c.get('reverse_empty_cases', 0))}, "
                f"reverse_trace_empty={int(c.get('reverse_trace_empty', 0))}, "
                f"unmatched_forward={int(c.get('unmatched_forward', 0))}, "
                f"unmatched_reverse={int(c.get('unmatched_reverse', 0))}, "
                f"tau_mismatch={int(c.get('tau_mismatch_count', 0))}, "
                f"matrix_mismatch={int(c.get('matrix_mismatch_count', 0))}, "
                f"requirement_met={bool(req_flags.get(sid, False))}"
            )
        lines.append("- worst_5_violations:")
        for v in reciprocity_metrics.get("worst_violations", []) or []:
            lines.append(
                f"-   {v.get('scenario_id', 'NA')}/{v.get('case_id', 'NA')}: "
                f"type={v.get('type', 'NA')}, path_id={int(v.get('path_index_forward', -1))}, "
                f"bounce={int(v.get('bounce_count', 0))}, "
                f"delta_tau_s={float(v.get('delta_tau_s', np.nan)):.3e}, "
                f"delta_sigma_max_db={float(v.get('delta_sigma_max_db', np.nan)):.3e}, "
                f"surface={v.get('surface_pattern', 'none')}"
            )
        for e in reciprocity_metrics.get("entries", []):
            sid = e.get("scenario_id", "NA")
            cid = e.get("case_id", "NA")
            lines.append(
                f"- {sid}/{cid}: matched_ratio={float(e.get('matched_ratio', np.nan)):.3f}, "
                f"delta_tau_max_s={float(e.get('delta_tau_max_s', np.nan)):.3e}, "
                f"delta_sigma_max_db={float(e.get('delta_sigma_max_db', np.nan)):.3e}, "
                f"n_forward={int(e.get('n_forward', 0))}, n_reverse={int(e.get('n_reverse', 0))}"
            )
            if e.get("unmatched_forward"):
                lines.append(f"- WARNING: reciprocity unmatched paths in {sid}/{cid}: {len(e.get('unmatched_forward', []))}")
        lines.append("")

    if tap_path_metrics is not None and tap_path_metrics.get("entries"):
        lines.append("## Tap-vs-Path Consistency (E12)")
        lines.append("")
        lines.append(f"- matrix_source: {tap_path_metrics.get('matrix_source', matrix_source)}")
        lines.append(
            f"- window_config: half_window_bins={int(tap_path_metrics.get('half_window_bins', 0))}, "
            f"overlap_policy={tap_path_metrics.get('overlap_policy', 'NA')}"
        )
        lines.append(f"- n_cases: {int(tap_path_metrics.get('n_cases', 0))}")
        lines.append(f"- delta_tau_median_s: {float(tap_path_metrics.get('delta_tau_median_s', np.nan)):.3e}")
        lines.append(f"- delta_tau_max_s: {float(tap_path_metrics.get('delta_tau_max_s', np.nan)):.3e}")
        lines.append(f"- delta_xpd_median_db: {float(tap_path_metrics.get('delta_xpd_median_db', np.nan)):.3f}")
        lines.append(f"- delta_xpd_max_db: {float(tap_path_metrics.get('delta_xpd_max_db', np.nan)):.3f}")
        lines.append(
            f"- delta_xpd_non_overlap_median_db: {float(tap_path_metrics.get('delta_xpd_non_overlap_median_db', np.nan)):.3f}"
        )
        lines.append(
            f"- delta_xpd_non_overlap_max_db: {float(tap_path_metrics.get('delta_xpd_non_overlap_max_db', np.nan)):.3f}"
        )
        lines.append(f"- wrap_detected_cases: {int(tap_path_metrics.get('wrap_detected_cases', 0))}")
        lines.append(f"- overlap_cases: {int(tap_path_metrics.get('overlap_cases', 0))}")
        lines.append(f"- overlap_labeled_cases: {int(tap_path_metrics.get('overlap_labeled_cases', 0))}")
        lines.append(f"- outlier_cases: {int(tap_path_metrics.get('outlier_cases', 0))}")
        lines.append(f"- outlier_reason_counts: {tap_path_metrics.get('outlier_reason_counts', {})}")
        if tap_path_metrics.get("outlier_csv"):
            lines.append(f"- outlier_csv: {tap_path_metrics.get('outlier_csv')}")
        lines.append("- per-case (scenario/case): Δtau_s, ΔXPD_dB, overlap_count, reason, wrap, n_paths")

        single_path_warn_thresh_db = 10.0
        for e in tap_path_metrics.get("entries", []):
            sid = str(e.get("scenario_id", "NA"))
            cid = str(e.get("case_id", "NA"))
            dt = float(e.get("delta_tau_s", np.nan))
            dx = float(e.get("delta_xpd_db", np.nan))
            wr = bool(e.get("wrap_detected", False))
            npth = int(e.get("n_paths", 0))
            ov = int(e.get("overlap_count", 0))
            rs = str(e.get("outlier_reason", "NONE"))
            lines.append(
                f"- {sid}/{cid}: delta_tau_s={dt:.3e}, delta_xpd_db={dx:.3f}, "
                f"overlap_count={ov}, reason={rs}, wrap={wr}, n_paths={npth}"
            )
            if wr:
                lines.append(f"-   NOTE: wrap_detected=True for {sid}/{cid} (interpret tap/path mismatch cautiously).")
            if rs == "OVERLAP":
                lines.append(f"-   NOTE: overlap-labeled case ({sid}/{cid}); tap/path direct comparison limited.")
            if npth <= 1 and sid in {"C0", "A1", "A2"} and np.isfinite(dx) and dx > single_path_warn_thresh_db:
                lines.append(
                    f"-   WARNING: large tap/path XPD mismatch in near-single-path case {sid}/{cid} "
                    f"(delta_xpd_db={dx:.3f} > {single_path_warn_thresh_db:.1f} dB)."
                )
        lines.append("")

    lines.append("## Measurement Bridge (G2)")
    lines.append("")
    lines.append(f"- current_basis: {basis_now}")
    lines.append(f"- current_convention: {convention_now}")
    lines.append(f"- current_matrix_source: {matrix_source}")
    enable_coupling = bool(antenna_config.get("enable_coupling", True))
    lines.append(f"- current_antenna_coupling_enabled: {enable_coupling}")
    lines.append(
        "- comparison_rule: "
        "antenna-included S21/S12 -> A_f (embedded), "
        "antenna de-embedded propagation comparison -> J_f (propagation-only)"
    )
    if str(matrix_source).upper() == "A":
        lines.append(
            "- recommendation_for_this_run: compare to antenna-included measurements "
            "(e.g., raw S21/S12 with antenna effects)."
        )
        lines.append(
            "- if_target_is_deembedded: rerun with --xpd-matrix-source J "
            "and keep basis/convention identical."
        )
    else:
        lines.append(
            "- recommendation_for_this_run: compare to antenna de-embedded / propagation-only references."
        )
        lines.append(
            "- if_target_is_embedded_s21: rerun with --xpd-matrix-source A "
            "and realistic coupling/leakage settings."
        )
    lines.append(
        "- basis_convention_rule: measurement post-processing must match "
        f"basis={basis_now}, convention={convention_now}; do not mix linear CP interpretation without explicit conversion."
    )
    lines.append("- mismatch_diagnosis_order:")
    lines.append("-   1) geometry/delay/occlusion")
    lines.append("-   2) FSPL/scalar_gain")
    lines.append("-   3) Fresnel Gamma_s,Gamma_p")
    lines.append("-   4) polarization basis/parity interpretation")
    lines.append("-   5) antenna coupling/leakage/AR")
    lines.append("")

    p = Path(out_md)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="outputs/rt_dataset.h5")
    parser.add_argument("--plots-dir", type=str, default="outputs/plots")
    parser.add_argument("--report", type=str, default="outputs/validation_report.md")
    parser.add_argument("--basis", type=str, default=None, choices=["linear", "circular"])
    parser.add_argument("--bases", type=str, default="linear,circular")
    parser.add_argument("--xpd-matrix-source", type=str, default="A", choices=["A", "J"])
    parser.add_argument("--plot-scenario", type=str, default=None)
    parser.add_argument("--plot-case", type=str, default=None)
    parser.add_argument("--plot-matrix-source", type=str, default=None, choices=["A", "J"])
    parser.add_argument("--plot-topk-paths", type=int, default=5)
    parser.add_argument("--tau-plot-max-ns", type=float, default=None)
    parser.add_argument("--plot-power-floor-db", type=float, default=-120.0)
    parser.add_argument("--plot-apply-exact-bounce", dest="plot_apply_exact_bounce", action="store_true")
    parser.add_argument("--no-plot-exact-bounce", dest="plot_apply_exact_bounce", action="store_false")
    parser.add_argument("--convention", type=str, default="IEEE-RHCP")
    parser.add_argument("--nf", type=int, default=256)
    parser.add_argument("--tx-cross-pol-leakage-db", type=float, default=35.0)
    parser.add_argument("--rx-cross-pol-leakage-db", type=float, default=35.0)
    parser.add_argument("--tx-axial-ratio-db", type=float, default=0.0)
    parser.add_argument("--rx-axial-ratio-db", type=float, default=0.0)
    parser.add_argument("--disable-antenna-coupling", action="store_true")
    parser.add_argument("--physics-validation-mode", action="store_true")
    parser.add_argument("--force-cp-swap-on-odd-reflection", action="store_true")
    parser.add_argument("--model-compare", dest="model_compare", action="store_true")
    parser.add_argument("--no-model-compare", dest="model_compare", action="store_false")
    parser.add_argument("--model-num-subbands", type=int, default=4)
    parser.add_argument("--model-num-synth-rays", type=int, default=128)
    parser.add_argument("--model-num-paths-mode", type=str, default="match_rt_total", choices=["fixed", "match_rt_total", "sample_case_hist"])
    parser.add_argument("--model-disable-incidence-conditioning", action="store_true")
    parser.add_argument("--model-phase-keep-common", action="store_true")
    parser.add_argument("--model-phase-all-freq-samples", action="store_true")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--release-mode", action="store_true")
    parser.add_argument("--reciprocity-scenarios", type=str, default="all")
    parser.add_argument("--reciprocity-tau-tol-s", type=float, default=1e-12)
    parser.add_argument("--reciprocity-sigma-tol-db", type=float, default=1e-6)
    parser.add_argument("--no-reciprocity-require-bidirectional", action="store_true")
    parser.add_argument("--tap-path-half-window-bins", type=int, default=2)
    parser.add_argument("--tap-path-overlap-policy", type=str, default="skip", choices=["skip", "merge"])
    parser.add_argument("--tap-path-low-snr-rel-threshold", type=float, default=1e-6)
    parser.add_argument("--tap-path-outlier-xpd-db", type=float, default=10.0)
    parser.set_defaults(plot_apply_exact_bounce=True)
    parser.set_defaults(model_compare=True)
    args = parser.parse_args()

    if bool(args.release_mode):
        _, dirty_now = _git_meta()
        if dirty_now:
            raise SystemExit("release-mode failed: git working tree is dirty. Commit/stash changes and rerun.")

    antenna_config = {
        "convention": args.convention,
        "tx_cross_pol_leakage_db": args.tx_cross_pol_leakage_db,
        "rx_cross_pol_leakage_db": args.rx_cross_pol_leakage_db,
        "tx_axial_ratio_db": args.tx_axial_ratio_db,
        "rx_axial_ratio_db": args.rx_axial_ratio_db,
        "enable_coupling": not args.disable_antenna_coupling,
    }
    if args.physics_validation_mode:
        antenna_config["tx_cross_pol_leakage_db"] = 120.0
        antenna_config["rx_cross_pol_leakage_db"] = 120.0
        antenna_config["tx_axial_ratio_db"] = 0.0
        antenna_config["rx_axial_ratio_db"] = 0.0
        antenna_config["enable_coupling"] = False

    bases = _parse_bases(args.basis, args.bases)
    multi = len(bases) > 1

    for b in bases:
        data = build_dataset(
            basis=b,
            convention=args.convention,
            nf=args.nf,
            antenna_config=antenna_config,
            force_cp_swap_on_odd_reflection=args.force_cp_swap_on_odd_reflection,
        )
        data.setdefault("meta", {})["cmdline"] = " ".join(shlex.quote(a) for a in sys.argv)
        data.setdefault("meta", {})["seed"] = {"model_seed": int(args.model_seed)}
        data.setdefault("meta", {})["seed_json"] = "{\"model_seed\":%d}" % int(args.model_seed)
        data.setdefault("meta", {})["release_mode"] = bool(args.release_mode)
        data.setdefault("meta", {})["xpd_matrix_source"] = str(args.xpd_matrix_source)
        data.setdefault("meta", {})["exact_bounce_defaults"] = dict(DEFAULT_EXACT_BOUNCE)
        data.setdefault("meta", {})["physics_validation_mode"] = not bool(antenna_config.get("enable_coupling", True))
        data.setdefault("meta", {})["antenna_config"] = dict(antenna_config)
        out_h5 = _basis_output_path(args.output, b, multi)
        out_plot = _basis_output_dir(args.plots_dir, b, multi)
        out_rep = _basis_output_path(args.report, b, multi)
        if args.plot_matrix_source is not None:
            plot_matrix_source = args.plot_matrix_source
        elif args.plot_scenario == "A6":
            plot_matrix_source = "J"
        else:
            plot_matrix_source = args.xpd_matrix_source
        plot_config = PlotConfig(
            scenario_id=args.plot_scenario,
            case_id=args.plot_case,
            top_k_paths=max(0, int(args.plot_topk_paths)),
            matrix_source=plot_matrix_source,
            apply_exact_bounce=bool(args.plot_apply_exact_bounce),
            cp_eval_basis="circular",
            convention=args.convention,
            tau_plot_max_ns=args.tau_plot_max_ns,
            power_floor_db=float(args.plot_power_floor_db),
        )

        rec_scenarios = None
        if str(args.reciprocity_scenarios).strip().lower() != "all":
            rec_scenarios = [s.strip() for s in str(args.reciprocity_scenarios).split(",") if s.strip()]
        reciprocity_metrics = compute_reciprocity_checks(
            data,
            matrix_source=args.xpd_matrix_source,
            scenario_ids=rec_scenarios,
            tau_tol_s=float(args.reciprocity_tau_tol_s),
            sigma_tol_db=float(args.reciprocity_sigma_tol_db),
            require_bidirectional_paths=not bool(args.no_reciprocity_require_bidirectional),
        )
        data.setdefault("meta", {})["reciprocity_sanity"] = reciprocity_metrics
        tap_path_metrics = evaluate_dataset_tap_path_consistency(
            data,
            matrix_source=args.xpd_matrix_source,
            nfft=2048,
            window="hann",
            power_floor=1e-12,
            half_window_bins=int(max(0, args.tap_path_half_window_bins)),
            overlap_policy=str(args.tap_path_overlap_policy),
            low_snr_rel_threshold=float(max(0.0, args.tap_path_low_snr_rel_threshold)),
            outlier_tau_factor=2.0,
            outlier_xpd_db=float(args.tap_path_outlier_xpd_db),
        )
        outlier_csv = out_rep.with_name(f"{out_rep.stem}_tap_path_outliers.csv")
        write_outlier_csv(outlier_csv, tap_path_metrics.get("entries", []))
        tap_path_metrics["outlier_csv"] = str(outlier_csv)
        data.setdefault("meta", {})["tap_path_consistency"] = tap_path_metrics

        save_rt_dataset(out_h5, data)
        if not self_test_meta_roundtrip(out_h5, expected_meta=data.get("meta", {})):
            raise SystemExit("HDF5 meta roundtrip self-test failed: saved artifact is not reproducible.")
        generate_all_plots(data, out_dir=out_plot, config=plot_config, exact_bounce_map=DEFAULT_EXACT_BOUNCE)
        model_metrics = None
        if args.model_compare:
            model_json = out_h5.with_name(f"{out_h5.stem}_model_params.json")
            synth_json = out_h5.with_name(f"{out_h5.stem}_synthetic_compare.json")
            model_res = fit_and_generate(
                input_h5=out_h5,
                output_json=model_json,
                synthetic_compare_json=synth_json,
                matrix_source=args.xpd_matrix_source,
                num_subbands=int(args.model_num_subbands),
                num_synth_rays=int(args.model_num_synth_rays),
                num_paths_mode=str(args.model_num_paths_mode),
                use_incidence_conditioning=not bool(args.model_disable_incidence_conditioning),
                phase_common_removed=not bool(args.model_phase_keep_common),
                phase_per_ray_sampling=not bool(args.model_phase_all_freq_samples),
                seed=int(args.model_seed),
                return_paths=True,
            )
            plot_metrics = generate_rt_vs_synth_plots(
                rt_paths=model_res.get("rt_paths", []),
                synth_paths=model_res.get("synthetic_paths", []),
                f_hz=np.asarray(data["frequency"], dtype=float),
                out_dir=out_plot,
                matrix_source=args.xpd_matrix_source,
                num_subbands=int(args.model_num_subbands),
                phase_common_removed=not bool(args.model_phase_keep_common),
                phase_per_ray_sampling=not bool(args.model_phase_all_freq_samples),
            )
            compare_metrics = model_res.get("comparison", {}).get("comparison", {})
            model_metrics = dict(compare_metrics)
            model_metrics.update(plot_metrics)

        build_quality_report(
            data,
            out_rep,
            matrix_source=args.xpd_matrix_source,
            model_metrics=model_metrics,
            reciprocity_metrics=reciprocity_metrics,
            tap_path_metrics=tap_path_metrics,
        )


if __name__ == "__main__":
    main()

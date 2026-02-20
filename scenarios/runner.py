"""Scenario sweep runner with HDF5 export, plots, and validation reports.

Example:
    python -m scenarios.runner --output outputs/rt_dataset.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from analysis.ctf_cir import ctf_to_cir, detect_cir_wrap, first_peak_tau_s, pdp, synthesize_ctf_with_source, tau_resolution_s
from analysis.run_model_fit import fit_and_generate
from analysis.xpd_stats import (
    conditional_fit,
    estimate_leakage_floor_from_antenna_config,
    make_subbands,
    leakage_limited_summary,
    pathwise_xpd,
)
from plots.model_compare import generate_rt_vs_synth_plots
from plots.plot_config import PlotConfig
from plots.p0_p13 import generate_all_plots
from rt_io.hdf5_io import save_rt_dataset
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


def build_dataset(
    basis: str = "linear",
    convention: str = "IEEE-RHCP",
    nf: int = 256,
    antenna_config: dict[str, Any] | None = None,
    force_cp_swap_on_odd_reflection: bool = False,
) -> dict[str, Any]:
    freq = uwb_frequency(nf=nf)
    data: dict[str, Any] = {
        "meta": {
            "basis": basis,
            "convention": convention,
            "antenna_config": antenna_config or {},
            "force_cp_swap_on_odd_reflection": force_cp_swap_on_odd_reflection,
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
) -> Path:
    lines = ["# RT Validation Report", ""]
    lines.append(f"- basis: {data.get('meta', {}).get('basis', 'NA')}")
    lines.append(f"- convention: {data.get('meta', {}).get('convention', 'NA')}")
    lines.append(f"- xpd_matrix_source: {matrix_source}")
    lines.append(f"- exact_bounce_defaults: {DEFAULT_EXACT_BOUNCE}")
    lines.append("- report_exact_bounce_applied: True (scenario-specific default map)")
    antenna_config = data.get("meta", {}).get("antenna_config", {})
    lines.append(f"- antenna_config: {antenna_config}")
    physics_mode = not bool(antenna_config.get("enable_coupling", True))
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
            all_paths.extend(paths)
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
        lines.append("## Model Compare (F2-F4)")
        lines.append("")
        for k in sorted(model_metrics.keys()):
            lines.append(f"- {k}: {model_metrics[k]}")
        if not bool(model_metrics.get("f4_parity_direction_match", False)):
            lines.append("- WARNING: synthetic parity direction does not match RT.")
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
    parser.add_argument("--model-seed", type=int, default=0)
    parser.set_defaults(plot_apply_exact_bounce=True)
    parser.set_defaults(model_compare=True)
    args = parser.parse_args()

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

        save_rt_dataset(out_h5, data)
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
                seed=int(args.model_seed),
                return_paths=True,
            )
            model_metrics = generate_rt_vs_synth_plots(
                rt_paths=model_res.get("rt_paths", []),
                synth_paths=model_res.get("synthetic_paths", []),
                f_hz=np.asarray(data["frequency"], dtype=float),
                out_dir=out_plot,
                matrix_source=args.xpd_matrix_source,
            )
            model_metrics["model_params_json"] = str(model_json)
            model_metrics["synthetic_compare_json"] = str(synth_json)

        build_quality_report(data, out_rep, matrix_source=args.xpd_matrix_source, model_metrics=model_metrics)


if __name__ == "__main__":
    main()

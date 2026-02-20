"""Scenario sweep runner with HDF5 export, plots, and validation reports.

Example:
    python -m scenarios.runner --output outputs/rt_dataset.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from analysis.xpd_stats import conditional_fit, pathwise_xpd
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


def build_quality_report(data: dict[str, Any], out_md: str | Path, matrix_source: str = "A") -> Path:
    lines = ["# RT Validation Report", ""]
    lines.append(f"- basis: {data.get('meta', {}).get('basis', 'NA')}")
    lines.append(f"- xpd_matrix_source: {matrix_source}")
    lines.append(f"- antenna_config: {data.get('meta', {}).get('antenna_config', {})}")
    lines.append("")

    for sid, sc in data["scenarios"].items():
        lines += [f"## {sid}", ""]
        all_paths = []
        exact_bounce = DEFAULT_EXACT_BOUNCE.get(sid)
        for cid, case in sc["cases"].items():
            paths = case["paths"]
            stat_paths = [p for p in paths if exact_bounce is None or int(p["meta"]["bounce_count"]) == exact_bounce]
            all_paths.extend(paths)
            counts = [p["meta"]["bounce_count"] for p in paths]
            lines.append(f"- case {cid}: paths={len(paths)}, bounce_dist={dict((c, counts.count(c)) for c in sorted(set(counts)))}")
            if paths:
                powers = [float(np.mean(np.abs(np.asarray(p["A_f"])) ** 2)) for p in paths]
                j = int(np.argmax(powers))
                lines.append(f"- strongest path: tau={paths[j]['tau_s']:.3e}s, power={powers[j]:.3e}")
                has_los = any(p["meta"]["bounce_count"] == 0 for p in paths)
                lines.append(f"- LOS exists: {has_los}")
                if sid.startswith("A2") and has_los:
                    lines.append(f"- WARNING: {sid} should block LOS but LOS path is present")
                if sid.startswith("A2") and not any(p["meta"]["bounce_count"] == 1 for p in paths):
                    lines.append(f"- WARNING: {sid} missing 1-bounce path")
                if sid.startswith("A3") and not any(p["meta"]["bounce_count"] == 2 for p in paths):
                    lines.append(f"- WARNING: {sid} missing 2-bounce path")
            if exact_bounce is not None and len(stat_paths) == 0:
                lines.append(f"- WARNING: no paths matched exact_bounce={exact_bounce} for stats")

        samples = pathwise_xpd(all_paths, exact_bounce=exact_bounce, matrix_source=matrix_source)
        par_stats = conditional_fit(samples, keys=["parity"])
        lines.append(f"- parity XPD stats (exact_bounce={exact_bounce}): {par_stats}")

        if sid == "A4":
            with_mat = []
            for _, case in sc["cases"].items():
                mat = str(case["params"].get("material", "NA"))
                for s in pathwise_xpd(case["paths"], exact_bounce=1, matrix_source=matrix_source):
                    s["material"] = mat
                    with_mat.append(s)
            mat_stats = conditional_fit(with_mat, keys=["material"])
            lines.append(f"- material sub-summary: {mat_stats}")
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
    parser.add_argument("--convention", type=str, default="IEEE-RHCP")
    parser.add_argument("--nf", type=int, default=256)
    parser.add_argument("--tx-cross-pol-leakage-db", type=float, default=35.0)
    parser.add_argument("--rx-cross-pol-leakage-db", type=float, default=35.0)
    parser.add_argument("--tx-axial-ratio-db", type=float, default=0.0)
    parser.add_argument("--rx-axial-ratio-db", type=float, default=0.0)
    parser.add_argument("--disable-antenna-coupling", action="store_true")
    parser.add_argument("--physics-validation-mode", action="store_true")
    parser.add_argument("--force-cp-swap-on-odd-reflection", action="store_true")
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

        save_rt_dataset(out_h5, data)
        generate_all_plots(data, out_dir=out_plot)
        build_quality_report(data, out_rep, matrix_source=args.xpd_matrix_source)


if __name__ == "__main__":
    main()

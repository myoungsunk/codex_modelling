"""Scenario sweep runner with HDF5 export, plots, and validation report.

Example:
    python -m scenarios.runner --output outputs/rt_dataset.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from analysis.ctf_cir import ctf_to_cir, pdp, synthesize_ctf
from analysis.xpd_stats import conditional_fit, pathwise_xpd
from plots.p0_p13 import generate_all_plots
from rt_io.hdf5_io import save_rt_dataset
from scenarios import A1_los_only, A2_pec_plane, A3_corner_2bounce, A4_dielectric_plane, A5_depol_stress, C0_free_space
from scenarios.common import paths_to_records, uwb_frequency


SCENARIOS = {
    "C0": C0_free_space,
    "A1": A1_los_only,
    "A2": A2_pec_plane,
    "A3": A3_corner_2bounce,
    "A4": A4_dielectric_plane,
    "A5": A5_depol_stress,
}

DEFAULT_EXACT_BOUNCE = {"A2": 1, "A3": 2, "A4": 1}


def build_dataset(basis: str = "linear", convention: str = "IEEE-RHCP", nf: int = 256) -> dict[str, Any]:
    freq = uwb_frequency(nf=nf)
    data: dict[str, Any] = {
        "meta": {"basis": basis, "convention": convention},
        "frequency": freq,
        "scenarios": {},
    }

    for sid, mod in SCENARIOS.items():
        cases: dict[str, Any] = {}
        for idx, params in enumerate(mod.build_sweep_params()):
            paths = mod.run_case(params, freq, basis=basis)
            cases[str(idx)] = {"params": params, "paths": paths_to_records(paths)}
        data["scenarios"][sid] = {"cases": cases}
    return data


def build_quality_report(data: dict[str, Any], out_md: str | Path) -> Path:
    freq = np.asarray(data["frequency"], dtype=float)
    lines = ["# RT Validation Report", ""]

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
                if sid == "A2" and has_los:
                    lines.append("- WARNING: A2 should block LOS but LOS path is present")
                if sid == "A2" and not any(p["meta"]["bounce_count"] == 1 for p in paths):
                    lines.append("- WARNING: A2 missing 1-bounce path")
                if sid == "A3" and not any(p["meta"]["bounce_count"] == 2 for p in paths):
                    lines.append("- WARNING: A3 missing 2-bounce path")
            if exact_bounce is not None and len(stat_paths) == 0:
                lines.append(f"- WARNING: no paths matched exact_bounce={exact_bounce} for stats")

        samples = pathwise_xpd(all_paths, exact_bounce=exact_bounce)
        par_stats = conditional_fit(samples, keys=["parity"])
        lines.append(f"- parity XPD stats (exact_bounce={exact_bounce}): {par_stats}")

        if sid == "A4":
            with_mat = []
            for cid, case in sc["cases"].items():
                mat = str(case["params"].get("material", "NA"))
                for s in pathwise_xpd(case["paths"], exact_bounce=1):
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
    parser.add_argument("--basis", type=str, default="linear", choices=["linear", "circular"])
    parser.add_argument("--nf", type=int, default=256)
    args = parser.parse_args()

    data = build_dataset(basis=args.basis, nf=args.nf)
    save_rt_dataset(args.output, data)
    generate_all_plots(data, out_dir=args.plots_dir)
    build_quality_report(data, args.report)


if __name__ == "__main__":
    main()

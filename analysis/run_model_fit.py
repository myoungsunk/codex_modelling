"""Run Step 3 model fitting and synthetic generation from RT HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from analysis.sv_polarimetric_model import generate_synthetic_paths, summarize_rt_vs_synth
from analysis.xpd_stats import (
    conditional_fit,
    fit_linear_mu_frequency,
    incidence_angle_bin_label,
    make_subbands,
    pathwise_xpd,
    save_stats_json,
    subband_centers_hz,
)
from rt_io.hdf5_io import load_rt_dataset


def _iter_case_paths(dataset: dict[str, Any]):
    for sid, sc in dataset["scenarios"].items():
        for cid, case in sc["cases"].items():
            yield sid, cid, case["params"], case["paths"]


def _surface_pattern(surface_ids: list[int]) -> str:
    if not surface_ids:
        return "none"
    return "-".join(str(x) for x in surface_ids)


def _enrich_sample(sample: dict[str, Any], path: dict[str, Any], params: dict[str, Any], sid: str, cid: str, bins_deg: list[float]) -> dict[str, Any]:
    meta = path.get("meta", {})
    out = dict(sample)
    out["scenario_id"] = sid
    out["case_id"] = cid
    out["material"] = str(params.get("material", "NA"))
    out["incidence_angle_bin"] = incidence_angle_bin_label(meta.get("incidence_angles", []), bins_deg=bins_deg)
    out["surface_id_pattern"] = _surface_pattern(meta.get("surface_ids", []))
    return out


def fit_and_generate(
    input_h5: str | Path,
    output_json: str | Path,
    synthetic_compare_json: str | Path,
    matrix_source: str = "A",
    num_subbands: int = 4,
    num_synth_rays: int = 64,
    incidence_bins_deg: list[float] | None = None,
    seed: int = 0,
    return_paths: bool = False,
) -> dict[str, Any]:
    ds = load_rt_dataset(input_h5)
    freq = np.asarray(ds["frequency"], dtype=float)
    subbands = make_subbands(len(freq), num_subbands)
    centers = subband_centers_hz(freq, subbands)
    bins = incidence_bins_deg or [0.0, 20.0, 40.0, 60.0, 90.0]

    full_samples: list[dict[str, Any]] = []
    sb_samples: list[dict[str, Any]] = []
    all_paths: list[dict[str, Any]] = []

    for sid, cid, params, paths in _iter_case_paths(ds):
        all_paths.extend(paths)
        for p in paths:
            s_full = pathwise_xpd([p], matrix_source=matrix_source)
            if s_full:
                full_samples.append(_enrich_sample(s_full[0], p, params, sid, cid, bins))
            s_sb = pathwise_xpd([p], subbands=subbands, matrix_source=matrix_source)
            for x in s_sb:
                y = _enrich_sample(x, p, params, sid, cid, bins)
                y["subband_center_hz"] = float(centers[int(y["subband"])])
                sb_samples.append(y)

    condition_fits = {
        "parity": conditional_fit(full_samples, ["parity"]),
        "bounce_count": conditional_fit(full_samples, ["bounce_count"]),
        "material": conditional_fit(full_samples, ["material"]),
        "incidence_angle_bin": conditional_fit(full_samples, ["incidence_angle_bin"]),
        "surface_id_pattern": conditional_fit(full_samples, ["surface_id_pattern"]),
    }

    slope_models = {
        "parity": fit_linear_mu_frequency(sb_samples, centers, ["parity"]),
        "material": fit_linear_mu_frequency(sb_samples, centers, ["material"]),
    }

    rt_parity_counts = {
        "odd": int(sum(1 for s in full_samples if s["parity"] == "odd")),
        "even": int(sum(1 for s in full_samples if s["parity"] == "even")),
    }
    total = max(rt_parity_counts["odd"] + rt_parity_counts["even"], 1)
    parity_probs = {"odd": rt_parity_counts["odd"] / total, "even": rt_parity_counts["even"] / total}

    delays = np.asarray([float(p["tau_s"]) for p in all_paths], dtype=float)
    if len(full_samples) > 0:
        powers = np.asarray([float(s["co_power"] + s["cross_power"]) for s in full_samples], dtype=float)
    else:
        powers = np.ones((max(len(all_paths), 1),), dtype=float)

    synth_paths = generate_synthetic_paths(
        f_hz=freq,
        num_rays=num_synth_rays,
        delay_samples_s=delays,
        power_samples=powers,
        parity_probs=parity_probs,
        parity_fit=condition_fits["parity"],
        parity_slope_model=slope_models["parity"],
        seed=seed,
    )

    comparison = summarize_rt_vs_synth(all_paths, synth_paths, subbands=subbands, rt_matrix_source=matrix_source)

    model_obj = {
        "matrix_source": matrix_source,
        "num_paths_rt": len(all_paths),
        "num_samples_rt": len(full_samples),
        "subbands": subbands,
        "subband_centers_hz": centers.tolist(),
        "condition_fits": condition_fits,
        "frequency_slope_models": slope_models,
        "parity_probs": parity_probs,
    }
    save_stats_json(output_json, model_obj)

    synth_obj = {
        "comparison": comparison,
        "synthetic_preview": [
            {
                "tau_s": float(p["tau_s"]),
                "parity": p.get("meta", {}).get("parity", "NA"),
                "power": float(np.mean(np.abs(np.asarray(p["A_f"], dtype=np.complex128)) ** 2)),
            }
            for p in synth_paths[: min(32, len(synth_paths))]
        ],
    }
    save_stats_json(synthetic_compare_json, synth_obj)
    out = {"model": model_obj, "comparison": synth_obj}
    if return_paths:
        out["rt_paths"] = all_paths
        out["synthetic_paths"] = synth_paths
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/rt_dataset.h5")
    parser.add_argument("--output", type=str, default="outputs/model_params.json")
    parser.add_argument("--synthetic-output", type=str, default="outputs/synthetic_compare.json")
    parser.add_argument("--matrix-source", type=str, default="A", choices=["A", "J"])
    parser.add_argument("--num-subbands", type=int, default=4)
    parser.add_argument("--num-synth-rays", type=int, default=64)
    parser.add_argument("--incidence-bins-deg", type=str, default="0,20,40,60,90")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bins = [float(x) for x in args.incidence_bins_deg.split(",") if x.strip()]
    fit_and_generate(
        input_h5=args.input,
        output_json=args.output,
        synthetic_compare_json=args.synthetic_output,
        matrix_source=args.matrix_source,
        num_subbands=args.num_subbands,
        num_synth_rays=args.num_synth_rays,
        incidence_bins_deg=bins,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

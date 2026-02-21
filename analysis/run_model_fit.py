"""Run Step 3 model fitting and synthetic generation from RT HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from analysis.sv_polarimetric_model import generate_synthetic_paths, summarize_rt_vs_synth
from analysis.xpd_stats import (
    censoring_profile_by_bucket,
    conditional_fit,
    estimate_leakage_floor_from_antenna_config,
    fit_linear_mu_frequency,
    floor_pinned_exclusion_mask,
    gof_model_selection_db,
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


def _align_synth_continuous_xpd(
    rt_samples: list[dict[str, Any]],
    synth_paths: list[dict[str, Any]],
    matrix_source: str,
    floor_db: float | None,
    pinned_tol_db: float = 0.5,
) -> dict[str, Any]:
    """Quantile-align synthetic continuous XPD to RT continuous XPD."""

    rt_vals = np.asarray([float(s.get("xpd_db", np.nan)) for s in rt_samples], dtype=float)
    rt_vals = rt_vals[np.isfinite(rt_vals)]
    if len(rt_vals) < 8 or len(synth_paths) == 0:
        return {"applied": False, "reason": "insufficient_samples"}

    rt_ex = floor_pinned_exclusion_mask(rt_vals, floor_db=floor_db, pinned_tol_db=pinned_tol_db)
    rt_cont = np.asarray(rt_ex["values"], dtype=float)[~np.asarray(rt_ex["mask_excluded"], dtype=bool)]
    if len(rt_cont) < 8:
        return {"applied": False, "reason": "rt_continuous_insufficient"}

    sy_samples = pathwise_xpd(synth_paths, matrix_source=matrix_source)
    sy_vals = np.asarray([float(s.get("xpd_db", np.nan)) for s in sy_samples], dtype=float)
    sy_ex = floor_pinned_exclusion_mask(sy_vals, floor_db=floor_db, pinned_tol_db=pinned_tol_db)
    sy_cont_mask = ~np.asarray(sy_ex["mask_excluded"], dtype=bool)
    cont_idx = np.asarray([int(sy_samples[i].get("path_index", -1)) for i in range(len(sy_samples)) if sy_cont_mask[i]], dtype=int)
    sy_cont = sy_vals[sy_cont_mask]
    if len(sy_cont) < 8 or len(cont_idx) != len(sy_cont):
        return {"applied": False, "reason": "synth_continuous_insufficient"}

    ord_sy = np.argsort(sy_cont)
    sy_sorted = sy_cont[ord_sy]
    rt_sorted = np.sort(rt_cont)
    n_sy = len(sy_sorted)
    n_rt = len(rt_sorted)
    tgt_sorted = np.asarray(
        [rt_sorted[int(round(i * (n_rt - 1) / max(n_sy - 1, 1)))] for i in range(n_sy)],
        dtype=float,
    )
    target_vals = np.empty_like(sy_cont)
    target_vals[ord_sy] = tgt_sorted

    adjusted = 0
    for j, pidx in enumerate(cont_idx.tolist()):
        if pidx < 0 or pidx >= len(synth_paths):
            continue
        p = synth_paths[pidx]
        M = np.asarray((p.get("J_f") if str(matrix_source).upper() == "J" and "J_f" in p else p.get("A_f")), dtype=np.complex128)
        if M.ndim != 3 or M.shape[1:] != (2, 2):
            continue
        cur = float(sy_cont[j])
        tgt = float(target_vals[j])
        g = float(10.0 ** ((cur - tgt) / 20.0))
        if not np.isfinite(g) or g <= 0.0:
            continue
        M_new = M.copy()
        M_new[:, 0, 1] *= g
        M_new[:, 1, 0] *= g
        p["A_f"] = np.asarray(M_new, dtype=np.complex128)
        p["J_f"] = np.asarray(M_new, dtype=np.complex128)
        adjusted += 1
    return {"applied": True, "adjusted_paths": int(adjusted), "n_rt_cont": int(len(rt_cont)), "n_synth_cont": int(len(sy_cont))}


def fit_and_generate(
    input_h5: str | Path,
    output_json: str | Path,
    synthetic_compare_json: str | Path,
    matrix_source: str = "A",
    num_subbands: int = 4,
    num_synth_rays: int = 64,
    num_paths_mode: str = "fixed",
    incidence_bins_deg: list[float] | None = None,
    use_incidence_conditioning: bool = True,
    phase_common_removed: bool = True,
    phase_per_ray_sampling: bool = True,
    seed: int = 0,
    xpd_freq_noise_sigma_db: float = 0.0,
    sample_slope: bool = False,
    slope_sigma_db_per_hz: float = 0.0,
    kappa_min: float = 1e-6,
    kappa_max: float = 1e12,
    kappa_freq_mode: str = "piecewise_constant",
    return_paths: bool = False,
) -> dict[str, Any]:
    ds = load_rt_dataset(input_h5)
    freq = np.asarray(ds["frequency"], dtype=float)
    subbands = make_subbands(len(freq), num_subbands)
    centers = subband_centers_hz(freq, subbands)
    bins = incidence_bins_deg or [0.0, 20.0, 40.0, 60.0, 90.0]
    ds_meta = ds.get("meta", {})
    basis_in = str(ds_meta.get("basis", "linear"))
    conv = str(ds_meta.get("convention", "IEEE-RHCP"))

    full_samples: list[dict[str, Any]] = []
    sb_samples: list[dict[str, Any]] = []
    all_paths: list[dict[str, Any]] = []
    case_path_counts: list[int] = []
    samples_by_scenario: dict[str, list[dict[str, Any]]] = {}
    path_scenarios: list[str] = []
    path_cases: list[str] = []

    for sid, cid, params, paths in _iter_case_paths(ds):
        for p in paths:
            meta = dict(p.get("meta", {}))
            meta["scenario_id"] = str(sid)
            meta["case_id"] = str(cid)
            p2 = dict(p)
            p2["meta"] = meta
            all_paths.append(p2)
            path_scenarios.append(str(sid))
            path_cases.append(str(cid))
        case_path_counts.append(len(paths))
        for p in paths:
            s_full = pathwise_xpd(
                [p],
                matrix_source=matrix_source,
                input_basis=basis_in,
                eval_basis=basis_in,
                convention=conv,
            )
            if s_full:
                fs = _enrich_sample(s_full[0], p, params, sid, cid, bins)
                full_samples.append(fs)
                samples_by_scenario.setdefault(str(sid), []).append(fs)
            s_sb = pathwise_xpd(
                [p],
                subbands=subbands,
                matrix_source=matrix_source,
                input_basis=basis_in,
                eval_basis=basis_in,
                convention=conv,
            )
            for x in s_sb:
                y = _enrich_sample(x, p, params, sid, cid, bins)
                y["subband_center_hz"] = float(centers[int(y["subband"])])
                sb_samples.append(y)

    condition_fits = {
        "parity": conditional_fit(full_samples, ["parity"]),
        "bounce_count": conditional_fit(full_samples, ["bounce_count"]),
        "material": conditional_fit(full_samples, ["material"]),
        "incidence_angle_bin": conditional_fit(full_samples, ["incidence_angle_bin"]),
        "parity_incidence_angle_bin": conditional_fit(full_samples, ["parity", "incidence_angle_bin"]),
        "surface_id_pattern": conditional_fit(full_samples, ["surface_id_pattern"]),
    }

    slope_models = {
        "parity": fit_linear_mu_frequency(sb_samples, centers, ["parity"]),
        "material": fit_linear_mu_frequency(sb_samples, centers, ["material"]),
        "parity_incidence_angle_bin": fit_linear_mu_frequency(sb_samples, centers, ["parity", "incidence_angle_bin"]),
    }
    parity_subband_fit: dict[str, dict[str, dict[str, float]]] = {"odd": {}, "even": {}}
    for parity in ("odd", "even"):
        for bidx in range(len(subbands)):
            vals = np.asarray(
                [
                    float(s["xpd_db"])
                    for s in sb_samples
                    if str(s.get("parity", "NA")) == parity and int(s.get("subband", -1)) == bidx
                ],
                dtype=float,
            )
            if len(vals) == 0:
                continue
            if len(vals) == 1:
                mu = float(vals[0])
                sg = 0.0
            else:
                mu = float(np.mean(vals))
                sg = float(np.std(vals, ddof=1))
            parity_subband_fit[parity][str(bidx)] = {"mu": mu, "sigma": sg, "n": int(len(vals))}

    floor_db = float(
        estimate_leakage_floor_from_antenna_config(ds_meta.get("antenna_config", {})).get("xpd_floor_db", np.nan)
    )
    gof_global = gof_model_selection_db(
        [float(s["xpd_db"]) for s in full_samples],
        min_n=20,
        bootstrap_B=200,
        seed=seed,
        floor_db=floor_db if np.isfinite(floor_db) else None,
        pinned_tol_db=0.5,
    )
    gof_global_by_parity: dict[str, dict[str, Any]] = {}
    for parity in ("odd", "even"):
        vals = [float(s["xpd_db"]) for s in full_samples if str(s.get("parity", "NA")) == parity]
        gof_global_by_parity[parity] = gof_model_selection_db(
            vals,
            min_n=20,
            bootstrap_B=200,
            seed=seed + (0 if parity == "odd" else 1),
            floor_db=floor_db if np.isfinite(floor_db) else None,
            pinned_tol_db=0.5,
        )
    gof_by_scenario: dict[str, Any] = {}
    for sid, samples_sid in sorted(samples_by_scenario.items()):
        vals_sid = [float(s["xpd_db"]) for s in samples_sid]
        gof_all = gof_model_selection_db(
            vals_sid,
            min_n=20,
            bootstrap_B=200,
            seed=seed + 101,
            floor_db=floor_db if np.isfinite(floor_db) else None,
            pinned_tol_db=0.5,
        )
        gof_par: dict[str, Any] = {}
        for parity in ("odd", "even"):
            vals = [float(s["xpd_db"]) for s in samples_sid if str(s.get("parity", "NA")) == parity]
            gof_par[parity] = gof_model_selection_db(
                vals,
                min_n=20,
                bootstrap_B=200,
                seed=seed + 211 + (0 if parity == "odd" else 1),
                floor_db=floor_db if np.isfinite(floor_db) else None,
                pinned_tol_db=0.5,
            )
        gof_by_scenario[sid] = {"all": gof_all, "parity": gof_par}
    gof_by_subband: dict[str, Any] = {}
    for bidx in range(len(subbands)):
        vals = [float(s["xpd_db"]) for s in sb_samples if int(s.get("subband", -1)) == bidx]
        gof_by_subband[str(bidx)] = gof_model_selection_db(
            vals,
            min_n=20,
            bootstrap_B=120,
            seed=seed + 401 + bidx,
            floor_db=floor_db if np.isfinite(floor_db) else None,
            pinned_tol_db=0.5,
        )
    parity_censoring = censoring_profile_by_bucket(
        full_samples,
        key_fields=["parity"],
        value_key="xpd_db",
        floor_db=floor_db if np.isfinite(floor_db) else None,
        pinned_tol_db=0.5,
    )
    parity_subband_censoring = censoring_profile_by_bucket(
        sb_samples,
        key_fields=["parity", "subband"],
        value_key="xpd_db",
        floor_db=floor_db if np.isfinite(floor_db) else None,
        pinned_tol_db=0.5,
    )
    sampling_parity_fit: dict[str, dict[str, float]] = {}
    for parity in ("odd", "even"):
        vals = np.asarray([float(s["xpd_db"]) for s in full_samples if str(s.get("parity", "NA")) == parity], dtype=float)
        ex = floor_pinned_exclusion_mask(
            vals,
            floor_db=floor_db if np.isfinite(floor_db) else None,
            pinned_tol_db=0.5,
        )
        vv = np.asarray(ex["values"], dtype=float)
        mm = np.asarray(ex["mask_excluded"], dtype=bool)
        vfit = vv[~mm]
        vuse = vfit if len(vfit) >= 5 else vv
        if len(vuse) == 0:
            sampling_parity_fit[parity] = {"mu": 10.0, "sigma": 3.0, "n": 0}
        elif len(vuse) == 1:
            sampling_parity_fit[parity] = {"mu": float(vuse[0]), "sigma": 0.0, "n": int(len(vuse))}
        else:
            sampling_parity_fit[parity] = {
                "mu": float(np.mean(vuse)),
                "sigma": float(np.std(vuse, ddof=1)),
                "n": int(len(vuse)),
            }
    sampling_parity_subband_fit: dict[str, dict[str, dict[str, float]]] = {"odd": {}, "even": {}}
    for parity in ("odd", "even"):
        for bidx in range(len(subbands)):
            vals = np.asarray(
                [
                    float(s["xpd_db"])
                    for s in sb_samples
                    if str(s.get("parity", "NA")) == parity and int(s.get("subband", -1)) == bidx
                ],
                dtype=float,
            )
            ex = floor_pinned_exclusion_mask(
                vals,
                floor_db=floor_db if np.isfinite(floor_db) else None,
                pinned_tol_db=0.5,
            )
            vv = np.asarray(ex["values"], dtype=float)
            mm = np.asarray(ex["mask_excluded"], dtype=bool)
            vfit = vv[~mm]
            vuse = vfit if len(vfit) >= 5 else vv
            if len(vuse) == 0:
                continue
            if len(vuse) == 1:
                mu = float(vuse[0])
                sg = 0.0
            else:
                mu = float(np.mean(vuse))
                sg = float(np.std(vuse, ddof=1))
            sampling_parity_subband_fit[parity][str(bidx)] = {"mu": mu, "sigma": sg, "n": int(len(vuse))}

    rt_parity_counts = {
        "odd": int(sum(1 for s in full_samples if s["parity"] == "odd")),
        "even": int(sum(1 for s in full_samples if s["parity"] == "even")),
    }
    total = max(rt_parity_counts["odd"] + rt_parity_counts["even"], 1)
    parity_probs = {"odd": rt_parity_counts["odd"] / total, "even": rt_parity_counts["even"] / total}
    incidence_counts: dict[str, int] = {}
    incidence_by_parity: dict[str, dict[str, int]] = {"odd": {}, "even": {}}
    for s in full_samples:
        inc = str(s.get("incidence_angle_bin", "NA"))
        pr = str(s.get("parity", "NA"))
        incidence_counts[inc] = incidence_counts.get(inc, 0) + 1
        if pr in incidence_by_parity:
            incidence_by_parity[pr][inc] = incidence_by_parity[pr].get(inc, 0) + 1
    incidence_probs = {
        k: float(v / max(sum(incidence_counts.values()), 1))
        for k, v in incidence_counts.items()
    }
    incidence_probs_by_parity: dict[str, dict[str, float]] = {}
    for pr, cmap in incidence_by_parity.items():
        den = max(sum(cmap.values()), 1)
        incidence_probs_by_parity[pr] = {k: float(v / den) for k, v in cmap.items()}
    def _continuous_pool(vals: list[float]) -> list[float]:
        arr = np.asarray(vals, dtype=float)
        ex = floor_pinned_exclusion_mask(
            arr,
            floor_db=floor_db if np.isfinite(floor_db) else None,
            pinned_tol_db=0.5,
        )
        vv = np.asarray(ex["values"], dtype=float)
        mm = np.asarray(ex["mask_excluded"], dtype=bool)
        out = vv[~mm]
        if len(out) >= 5:
            return [float(x) for x in out]
        return [float(x) for x in vv]

    grouped_emp: dict[str, list[float]] = {"ALL": [float(s["xpd_db"]) for s in full_samples]}
    for s in full_samples:
        pr = str(s.get("parity", "NA"))
        inc = str(s.get("incidence_angle_bin", "NA"))
        grouped_emp.setdefault(pr, []).append(float(s["xpd_db"]))
        grouped_emp.setdefault(f"{pr}|{inc}", []).append(float(s["xpd_db"]))
    empirical_xpd_by_condition = {k: _continuous_pool(v) for k, v in grouped_emp.items()}

    delays = np.asarray([float(p["tau_s"]) for p in all_paths], dtype=float)
    if len(full_samples) > 0:
        powers = np.asarray([float(s["co_power"] + s["cross_power"]) for s in full_samples], dtype=float)
    else:
        powers = np.ones((max(len(all_paths), 1),), dtype=float)

    synth_out = generate_synthetic_paths(
        f_hz=freq,
        num_rays=num_synth_rays,
        delay_samples_s=delays,
        power_samples=powers,
        parity_probs=parity_probs,
        parity_fit=sampling_parity_fit,
        parity_slope_model=slope_models["parity"],
        parity_subband_fit=sampling_parity_subband_fit,
        parity_censoring=parity_censoring,
        parity_subband_censoring=parity_subband_censoring,
        subbands=subbands,
        kappa_freq_mode=str(kappa_freq_mode),
        incidence_probs=incidence_probs if use_incidence_conditioning else None,
        incidence_probs_by_parity=incidence_probs_by_parity if use_incidence_conditioning else None,
        parity_incidence_fit=condition_fits["parity_incidence_angle_bin"] if use_incidence_conditioning else None,
        parity_incidence_slope_model=slope_models["parity_incidence_angle_bin"] if use_incidence_conditioning else None,
        empirical_xpd_by_condition=empirical_xpd_by_condition,
        matrix_source=matrix_source,
        xpd_freq_noise_sigma_db=float(xpd_freq_noise_sigma_db),
        sample_slope=bool(sample_slope),
        slope_sigma_db_per_hz=float(max(slope_sigma_db_per_hz, 0.0)),
        kappa_min=float(kappa_min),
        kappa_max=float(kappa_max),
        num_paths_mode=str(num_paths_mode),
        rt_case_path_counts=np.asarray(case_path_counts, dtype=np.int64),
        rt_path_scenarios=path_scenarios,
        rt_path_cases=path_cases,
        return_diagnostics=True,
        seed=seed,
    )
    synth_paths, synth_diag = synth_out
    align_diag = _align_synth_continuous_xpd(
        rt_samples=full_samples,
        synth_paths=synth_paths,
        matrix_source=matrix_source,
        floor_db=floor_db if np.isfinite(floor_db) else None,
        pinned_tol_db=0.5,
    )
    synth_diag["continuous_quantile_alignment"] = align_diag

    comparison = summarize_rt_vs_synth(
        all_paths,
        synth_paths,
        subbands=subbands,
        rt_matrix_source=matrix_source,
        synth_matrix_source=matrix_source,
        phase_bootstrap_B=500,
        phase_test_basis=str(ds.get("meta", {}).get("basis", "unknown")),
        common_phase_removed=bool(phase_common_removed),
        per_ray_phase_sampling=bool(phase_per_ray_sampling),
        floor_db=floor_db if np.isfinite(floor_db) else None,
        pinned_tol_db=0.5,
        input_basis=basis_in,
        eval_basis=basis_in,
        convention=conv,
        seed=seed,
    )
    comparison.update(
        {
            "synthetic_kappa_clamp_rate": float(synth_diag.get("kappa_clamp_rate", np.nan)),
            "synthetic_kappa_clamp_count": int(synth_diag.get("kappa_clamp_count", 0)),
            "synthetic_kappa_total": int(synth_diag.get("kappa_total", 0)),
            "synthetic_xpd_freq_noise_sigma_db": float(synth_diag.get("xpd_freq_noise_sigma_db", 0.0)),
            "synthetic_sample_slope": bool(synth_diag.get("sample_slope", False)),
            "synthetic_kappa_truncation_rate": float(synth_diag.get("kappa_truncation_rate", np.nan)),
            "synthetic_kappa_truncation_count": int(synth_diag.get("kappa_truncation_count", 0)),
            "synthetic_num_paths_mode": str(synth_diag.get("num_paths_mode", "fixed")),
            "synthetic_resolved_num_paths": int(synth_diag.get("resolved_num_paths", len(synth_paths))),
        }
    )

    model_obj = {
        "matrix_source": matrix_source,
        "num_paths_rt": len(all_paths),
        "num_samples_rt": len(full_samples),
        "xpd_random_variables": {
            "pathwise": "per-path mean-power ratio over frequency",
            "tapwise": "CIR tap/window energy ratio",
            "basis_input": basis_in,
            "basis_eval": basis_in,
            "convention": conv,
        },
        "subbands": subbands,
        "subband_centers_hz": centers.tolist(),
        "condition_fits": condition_fits,
        "frequency_slope_models": slope_models,
        "parity_subband_fit": parity_subband_fit,
        "sampling_parity_fit": sampling_parity_fit,
        "sampling_parity_subband_fit": sampling_parity_subband_fit,
        "parity_censoring": parity_censoring,
        "parity_subband_censoring": parity_subband_censoring,
        "gof_global": gof_global,
        "gof_global_by_parity": gof_global_by_parity,
        "gof_by_scenario": gof_by_scenario,
        "gof_by_subband": gof_by_subband,
        "parity_probs": parity_probs,
        "incidence_probs": incidence_probs,
        "incidence_probs_by_parity": incidence_probs_by_parity,
        "num_paths_mode": str(num_paths_mode),
        "kappa_freq_mode": str(kappa_freq_mode),
        "use_incidence_conditioning": bool(use_incidence_conditioning),
        "phase_common_removed": bool(phase_common_removed),
        "phase_per_ray_sampling": bool(phase_per_ray_sampling),
        "synthetic_generation": synth_diag,
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
    parser.add_argument(
        "--num-paths-mode",
        type=str,
        default="match_rt_per_scenario",
        choices=["fixed", "match_rt_total", "match_rt_per_scenario", "match_rt_per_case", "sample_case_hist"],
    )
    parser.add_argument("--kappa-freq-mode", type=str, default="piecewise_constant", choices=["piecewise_constant", "linear"])
    parser.add_argument("--incidence-bins-deg", type=str, default="0,20,40,60,90")
    parser.add_argument("--disable-incidence-conditioning", action="store_true")
    parser.add_argument("--phase-keep-common", action="store_true")
    parser.add_argument("--phase-all-freq-samples", action="store_true")
    parser.add_argument("--xpd-freq-noise-sigma-db", type=float, default=0.0)
    parser.add_argument("--sample-slope", action="store_true")
    parser.add_argument("--slope-sigma-db-per-hz", type=float, default=0.0)
    parser.add_argument("--kappa-min", type=float, default=1e-6)
    parser.add_argument("--kappa-max", type=float, default=1e12)
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
        num_paths_mode=args.num_paths_mode,
        kappa_freq_mode=args.kappa_freq_mode,
        incidence_bins_deg=bins,
        use_incidence_conditioning=not bool(args.disable_incidence_conditioning),
        phase_common_removed=not bool(args.phase_keep_common),
        phase_per_ray_sampling=not bool(args.phase_all_freq_samples),
        xpd_freq_noise_sigma_db=args.xpd_freq_noise_sigma_db,
        sample_slope=args.sample_slope,
        slope_sigma_db_per_hz=args.slope_sigma_db_per_hz,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

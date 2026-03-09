"""Build link-level condition variables U from scenario/rays."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_types.standard_outputs import LinkConditionsU


def _dominant_parity_early(ray_rows: list[dict[str, Any]], tau_limit_s: float | None) -> str:
    rows = list(ray_rows)
    if tau_limit_s is not None and np.isfinite(float(tau_limit_s)):
        rows = [r for r in rows if float(r.get("tau_s", np.inf)) <= float(tau_limit_s)]
    if not rows:
        return "NA"
    p = np.asarray([float(r.get("P_lin", 0.0)) for r in rows], dtype=float)
    if len(p) == 0:
        return "NA"
    j = int(np.argmax(p))
    par = str(rows[j].get("parity", "NA")).lower()
    if par in {"odd", "even"}:
        return par
    n = int(rows[j].get("n_bounce", 0))
    return "even" if (n % 2 == 0) else "odd"


def build_link_U_from_scenario(
    meta: dict[str, Any],
    ray_rows: list[dict[str, Any]],
    pdp: dict[str, np.ndarray],
    masks: tuple[np.ndarray, np.ndarray] | None,
    el_proxy_db: float,
) -> LinkConditionsU:
    _ = pdp
    mm = dict(meta or {})
    d_m = float(mm.get("d_m", mm.get("distance_m", mm.get("distance_d_m", np.nan))))
    if not np.isfinite(d_m):
        d_m = float(mm.get("link_distance_m", np.nan))

    losflag = mm.get("LOSflag", None)
    if losflag is None:
        losflag = int(any(int(r.get("los_flag_ray", 0)) == 1 or int(r.get("n_bounce", 1)) == 0 for r in ray_rows))

    material = str(mm.get("material_class", mm.get("material", "NA")))
    rough = int(mm.get("roughness_flag", 0))
    human = int(mm.get("human_flag", 0))
    obst = int(mm.get("obstacle_flag", 0))
    extras: dict[str, Any] = {}
    for k in [
        "rx_x",
        "rx_y",
        "rx_z",
        "scenario_id",
        "case_id",
        "yaw_deg",
        "pitch_deg",
        "basis",
        "convention",
        "matrix_source",
        "force_cp_swap_on_odd_reflection",
        "los_block_method",
        "tx_cross_pol_leakage_db",
        "rx_cross_pol_leakage_db",
        "tx_cross_pol_leakage_db_slope_per_ghz",
        "rx_cross_pol_leakage_db_slope_per_ghz",
        "stress_mode",
        "stress_semantics",
        "scatterer_count",
        "diffuse_enabled",
        "diffuse_factor",
        "diffuse_lobe_alpha",
        "diffuse_rays_per_hit",
        "stress_path_structure_active",
        "stress_polarization_mixer_active",
        "a4_layout_mode",
        "a4_dispersion_mode",
        "material_dispersion",
        "include_late_panel",
        "a6_mode",
        "a6_case_set",
        "a6_even_path_policy",
        "a6_even_layout",
        "target_bounce",
        "incidence_max_deg",
        "near_normal_benchmark",
        "rep_id",
        "rep_outer",
    ]:
        if k in mm:
            extras[k] = mm.get(k)
    tau_lim = None
    if masks is not None and "delay_tau_s" in mm:
        tau = np.asarray(mm.get("delay_tau_s", []), dtype=float)
        early = np.asarray(masks[0], dtype=bool)
        if len(tau) == len(early) and np.any(early):
            tau_lim = float(np.nanmax(tau[early]))
    dom = _dominant_parity_early(ray_rows, tau_lim)
    return LinkConditionsU(
        d_m=float(d_m),
        LOSflag=int(losflag),
        EL_proxy_db=float(el_proxy_db),
        material_class=material,
        roughness_flag=rough,
        human_flag=human,
        obstacle_flag=obst,
        dominant_parity_early=dom,
        extras=extras,
    )

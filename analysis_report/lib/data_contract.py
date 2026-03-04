"""3-level data contract for dual-CP proxy reporting.

Layer 1 — target_level  : per-link target-path metrics (G1/G2 sign claims)
Layer 2 — case_level    : per-link system/excess metrics (L/M/R claims)
Layer 3 — sensitivity   : Te-sweep stability table (reproducibility check)

Sign claims (G1/G2) MUST use xpd_target_raw_db.
Excess claims (L/M/R) MUST use xpd_*_excess_db with claim_caution flags.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ── column contracts ──────────────────────────────────────────────────────────

TARGET_LEVEL_COLS: list[str] = [
    "scenario_id",
    "case_id",
    "link_id",
    # Target-path geometry
    "target_tau_ns",
    "target_rank",
    "target_in_Wearly",
    "target_is_first",
    "bounce_count",
    "parity",
    "incidence_angle_deg",
    # Raw power in target window (G1/G2 sign primary metric)
    "Pco_target_lin",
    "Pcross_target_lin",
    "xpd_target_raw_db",      # sign judged here: <0 → cross-dom, >0 → co-dom
    # Excess (supplementary — L/M/R use only)
    "xpd_target_ex_db",
    # Calibration context
    "floor_db",
    "delta_floor_db",
    "claim_caution_target",   # True if |xpd_target_ex| < delta_floor
]

CASE_LEVEL_COLS: list[str] = [
    "scenario_id",
    "case_id",
    "link_id",
    # Excess-domain system metrics (L/M/R claims)
    "xpd_early_ex_db",
    "xpd_late_ex_db",
    "l_pol_db",
    "rho_early_lin",          # linear scale [0,1]; preferred for mixing severity
    "rho_early_db",           # dB supplement
    "ds_ns",
    "early_energy_fraction",
    "EL_proxy_db",
    "LOSflag",
    "material",
    "stress_flag",
    "stress_mode",            # synthetic / geometric / hybrid / none
    # Claim caution flags (from metrics.apply_floor_excess)
    "claim_caution_early",
    "claim_caution_late",
    # Calibration context
    "floor_db",
    "delta_floor_db",
]

SENSITIVITY_LEVEL_COLS: list[str] = [
    "scenario_id",
    "Te_ns",
    "noise_tail_ns",
    "n_links",
    # Key metrics under this Te/noise_tail setting
    "median_xpd_early_raw_db",
    "median_xpd_early_ex_db",
    "sign_hit_rate",          # fraction of links with correct sign
    "S_xpd_early",            # scenario-separation score
    "S_rho_early",
    "S_l_pol",
    # Stability flags
    "sign_stable",            # True if sign_hit_rate >= 0.8 across all tested Te
]

# ── helper ────────────────────────────────────────────────────────────────────

def _num(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return default


def _bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    try:
        return bool(int(float(str(x))))
    except Exception:
        return default


def build_target_level_rows(
    link_rows: list[dict[str, Any]],
    ray_rows: list[dict[str, Any]],
    floor_db: float,
    delta_floor_db: float,
) -> list[dict[str, Any]]:
    """Build target_level rows from link_rows + ray_rows.

    xpd_target_raw_db is required for G1/G2 sign claims.
    If not pre-computed, callers should integrate the PDP over the target
    window before calling this function and pass the result via link_rows
    with key 'xpd_target_raw_db'.
    """
    delta_abs = abs(float(delta_floor_db)) if np.isfinite(delta_floor_db) else float("nan")

    # Index ray_rows by link_id for quick lookup
    ray_by_link: dict[str, list[dict[str, Any]]] = {}
    for r in ray_rows:
        lid = str(r.get("link_id", ""))
        ray_by_link.setdefault(lid, []).append(r)

    out: list[dict[str, Any]] = []
    for r in link_rows:
        lid = str(r.get("link_id", ""))
        rays = ray_by_link.get(lid, [])

        xpd_raw = _num(r.get("xpd_target_raw_db", r.get("XPD_target_raw_db")))
        xpd_ex = (float(xpd_raw - floor_db)
                  if np.isfinite(xpd_raw) and np.isfinite(floor_db)
                  else float("nan"))
        claim_caution_target = bool(
            np.isfinite(xpd_ex) and np.isfinite(delta_abs) and abs(xpd_ex) < delta_abs
        )

        # Best target ray (highest power among rays with correct bounce count)
        target_n = int(_num(r.get("target_bounce_n", r.get("target_n", 1))))
        target_rays = [rr for rr in rays if int(_num(rr.get("n_bounce", -1))) == target_n]
        best_ray = max(target_rays, key=lambda rr: _num(rr.get("P_lin", 0.0)), default=None)

        row: dict[str, Any] = {c: "" for c in TARGET_LEVEL_COLS}
        row["scenario_id"] = str(r.get("scenario_id", ""))
        row["case_id"] = str(r.get("case_id", ""))
        row["link_id"] = lid
        row["xpd_target_raw_db"] = xpd_raw
        row["xpd_target_ex_db"] = xpd_ex
        row["floor_db"] = float(floor_db)
        row["delta_floor_db"] = float(delta_floor_db)
        row["claim_caution_target"] = claim_caution_target
        if best_ray is not None:
            row["target_tau_ns"] = _num(best_ray.get("tau_s", float("nan"))) * 1e9
            row["bounce_count"] = int(_num(best_ray.get("n_bounce", target_n)))
            row["incidence_angle_deg"] = _num(best_ray.get("incidence_angle_deg"))
            parity_raw = str(best_ray.get("parity", ""))
            row["parity"] = parity_raw if parity_raw else ("odd" if target_n % 2 else "even")
        else:
            row["target_tau_ns"] = float("nan")
            row["bounce_count"] = target_n
            row["parity"] = "odd" if target_n % 2 else "even"
        row["target_in_Wearly"] = _bool(r.get("target_in_Wearly", r.get("target_in_Wearly_rate")))
        row["target_is_first"] = _bool(r.get("target_is_first", r.get("target_is_first_rate")))
        row["Pco_target_lin"] = _num(r.get("Pco_target_lin"))
        row["Pcross_target_lin"] = _num(r.get("Pcross_target_lin"))
        out.append(row)
    return out


def build_case_level_rows(link_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build case_level rows from floor-excess-applied link_rows.

    Expects link_rows to already have XPD_early_excess_db, claim_caution_early,
    claim_caution_late from metrics.apply_floor_excess().
    """
    out: list[dict[str, Any]] = []
    for r in link_rows:
        row: dict[str, Any] = {c: "" for c in CASE_LEVEL_COLS}
        row["scenario_id"] = str(r.get("scenario_id", ""))
        row["case_id"] = str(r.get("case_id", ""))
        row["link_id"] = str(r.get("link_id", ""))
        row["xpd_early_ex_db"] = _num(r.get("XPD_early_excess_db"))
        row["xpd_late_ex_db"] = _num(r.get("XPD_late_excess_db"))
        row["l_pol_db"] = _num(r.get("L_pol_db"))
        row["rho_early_lin"] = _num(r.get("rho_early_lin", r.get("rho_early")))
        row["rho_early_db"] = _num(r.get("rho_early_db"))
        row["ds_ns"] = _num(r.get("delay_spread_rms_ns", r.get("delay_spread_ns")))
        row["early_energy_fraction"] = _num(r.get("early_energy_fraction"))
        row["EL_proxy_db"] = _num(r.get("EL_proxy_db"))
        row["LOSflag"] = int(_num(r.get("LOSflag", r.get("los_flag", 0))))
        row["material"] = str(r.get("material", r.get("material_class", "")))
        row["stress_flag"] = int(_num(r.get("roughness_flag", r.get("human_flag", 0))))
        row["stress_mode"] = str(r.get("stress_mode", "none"))
        row["claim_caution_early"] = _bool(r.get("claim_caution_early", False))
        row["claim_caution_late"] = _bool(r.get("claim_caution_late", False))
        row["floor_db"] = _num(r.get("floor_db"))
        row["delta_floor_db"] = _num(r.get("delta_floor_db"))
        out.append(row)
    return out


def build_sensitivity_level_rows(
    te_sweep_results: list[dict[str, Any]],
    scenario_id: str,
) -> list[dict[str, Any]]:
    """Build sensitivity_level rows from Te-sweep diagnostic output.

    te_sweep_results: list of per_te dicts from _target_sign_stability_te_sweep.
    """
    out: list[dict[str, Any]] = []
    all_rates = [_num(x.get("expected_sign_hit_rate")) for x in te_sweep_results]
    all_rates_finite = [r for r in all_rates if np.isfinite(r)]
    sign_stable = bool(all_rates_finite and min(all_rates_finite) >= 0.8)
    for entry in te_sweep_results:
        row: dict[str, Any] = {c: "" for c in SENSITIVITY_LEVEL_COLS}
        row["scenario_id"] = scenario_id
        row["Te_ns"] = _num(entry.get("Te_ns"))
        row["noise_tail_ns"] = _num(entry.get("noise_tail_ns", float("nan")))
        row["n_links"] = int(_num(entry.get("n", 0)))
        row["median_xpd_early_raw_db"] = _num(entry.get("median_xpd_early_raw_db"))
        row["median_xpd_early_ex_db"] = _num(entry.get("median_xpd_early_ex_db"))
        row["sign_hit_rate"] = _num(entry.get("expected_sign_hit_rate"))
        row["S_xpd_early"] = _num(entry.get("S_xpd_early", float("nan")))
        row["S_rho_early"] = _num(entry.get("S_rho_early_db", float("nan")))
        row["S_l_pol"] = _num(entry.get("S_l_pol", float("nan")))
        row["sign_stable"] = sign_stable
        out.append(row)
    return out

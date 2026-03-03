"""Link-level excess-loss proxy derivations."""

from __future__ import annotations

from typing import Any

import numpy as np

from physics.fspl import fspl_db


EPS = 1e-15


def _choose_lref(ray_rows: list[dict[str, Any]], mode: str = "los|minL") -> float:
    rows = list(ray_rows)
    if not rows:
        return float("nan")
    m = str(mode).lower()
    if "los" in m:
        los = [r for r in rows if int(r.get("los_flag_ray", 0)) == 1]
        if los:
            return float(np.nanmin(np.asarray([float(r.get("L_m", np.nan)) for r in los], dtype=float)))
    return float(np.nanmin(np.asarray([float(r.get("L_m", np.nan)) for r in rows], dtype=float)))


def compute_el_proxy(
    ray_rows: list[dict[str, Any]],
    pdp: dict[str, np.ndarray] | None = None,
    mode: str = "early_sum",
    early_mask: np.ndarray | None = None,
    L_ref_mode: str = "los|minL",
    f_center_hz: float = 8e9,
) -> float:
    """Compute link-level EL proxy from ray table (and optional PDP)."""

    rows = list(ray_rows)
    if len(rows) == 0:
        return float("nan")

    mode_l = str(mode).lower()
    if mode_l == "dominant_early_ray":
        if early_mask is not None and pdp is not None and "delay_tau_s" in pdp:
            tau = np.asarray(pdp["delay_tau_s"], dtype=float)
            tmax = float(np.nanmax(tau[early_mask])) if np.any(early_mask) else float(np.nanmin(tau))
            cand = [r for r in rows if float(r.get("tau_s", np.inf)) <= tmax + 1e-15]
        else:
            cand = list(rows)
        if not cand:
            cand = list(rows)
        valid = [r for r in cand if np.isfinite(float(r.get("EL_db", np.nan)))]
        if valid:
            j = int(np.argmax(np.asarray([float(r.get("P_lin", 0.0)) for r in valid], dtype=float)))
            return float(valid[j].get("EL_db", np.nan))
        return float("nan")

    # early_sum default
    if pdp is None or early_mask is None:
        p_tot = float(np.sum(np.asarray([float(r.get("P_lin", 0.0)) for r in rows], dtype=float)))
    else:
        pco = np.asarray(pdp.get("P_co", []), dtype=float)
        pcx = np.asarray(pdp.get("P_cross", []), dtype=float)
        p_tot = float(np.sum((pco + pcx)[np.asarray(early_mask, dtype=bool)]))

    L_ref = _choose_lref(rows, mode=L_ref_mode)
    if (not np.isfinite(L_ref)) or p_tot <= 0.0:
        return float("nan")
    P_fs_db = -float(fspl_db(L_ref, f_center_hz))
    P_early_db = 10.0 * np.log10(max(float(p_tot), EPS))
    return float(P_fs_db - P_early_db)

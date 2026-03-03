"""Link-level summary metrics Z from dual-CP PDP."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_types.standard_outputs import LinkMetricsZ


EPS = 1e-15


def _xpd_db(sum_co: float, sum_cross: float) -> float:
    return float(10.0 * np.log10((float(sum_co) + EPS) / (float(sum_cross) + EPS)))


def _tau_rms(delay_tau_s: np.ndarray, weight: np.ndarray) -> float:
    tau = np.asarray(delay_tau_s, dtype=float)
    w = np.asarray(weight, dtype=float)
    s = float(np.sum(w))
    if s <= 0.0:
        return float("nan")
    mu = float(np.sum(w * tau) / s)
    m2 = float(np.sum(w * (tau ** 2)) / s)
    var = max(m2 - mu * mu, 0.0)
    return float(np.sqrt(var))


def compute_link_metrics(
    pdp: dict[str, Any],
    delay_tau_s: np.ndarray,
    masks: tuple[np.ndarray, np.ndarray],
    ds_reference: str = "total",
    window_params: dict[str, Any] | None = None,
) -> LinkMetricsZ:
    pco = np.asarray(pdp.get("P_co", []), dtype=float)
    pcx = np.asarray(pdp.get("P_cross", []), dtype=float)
    tau = np.asarray(delay_tau_s, dtype=float)
    if len(pco) != len(tau) or len(pcx) != len(tau):
        raise ValueError("PDP/tau length mismatch")
    early, late = masks
    e_co = float(np.sum(pco[early]))
    e_cx = float(np.sum(pcx[early]))
    l_co = float(np.sum(pco[late]))
    l_cx = float(np.sum(pcx[late]))
    xpd_e = _xpd_db(e_co, e_cx)
    xpd_l = _xpd_db(l_co, l_cx)
    rho_lin = float((e_cx + EPS) / (e_co + EPS))
    rho_db = float(10.0 * np.log10(rho_lin + EPS))
    p_total = pco + pcx
    ds_w = p_total if str(ds_reference).lower() == "total" else pco
    ds = _tau_rms(tau, ds_w)
    frac = float((np.sum(p_total[early]) + EPS) / (np.sum(p_total) + EPS))
    return LinkMetricsZ(
        XPD_early_db=xpd_e,
        XPD_late_db=xpd_l,
        rho_early_lin=rho_lin,
        rho_early_db=rho_db,
        L_pol_db=float(xpd_e - xpd_l),
        delay_spread_rms_s=float(ds),
        early_energy_fraction=float(frac),
        window=dict(window_params or {}),
    )

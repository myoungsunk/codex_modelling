"""Synthesize measurement-compatible dual-CP PDP from ray table."""

from __future__ import annotations

from typing import Any

import numpy as np

from calibration.floor_model import FloorXPDModel
from polarization.power_split import split_power, xpd_from_xpr
from polarization.xpr_models import BaseXPRModel, ConstantXPR
from rt_types.standard_outputs import DualCPPDP


EPS = 1e-15


def _bin_index(delay_tau_s: np.ndarray, tau_s: float) -> int:
    d = np.asarray(delay_tau_s, dtype=float)
    if len(d) == 0:
        return -1
    if len(d) == 1:
        return 0
    mids = 0.5 * (d[:-1] + d[1:])
    edges = np.concatenate([np.asarray([-np.inf]), mids, np.asarray([np.inf])])
    idx = int(np.searchsorted(edges, float(tau_s), side="right") - 1)
    return int(np.clip(idx, 0, len(d) - 1))


def synthesize_dualcp_pdp(
    ray_rows: list[dict[str, Any]],
    delay_tau_s: np.ndarray,
    xpr_model: BaseXPRModel | None = None,
    link_U: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
    include_xpd_tau: bool = True,
    floor_model: FloorXPDModel | None = None,
) -> DualCPPDP:
    """Synthesize P_co/P_cross delay-domain vectors from ray latent table."""

    d = np.asarray(delay_tau_s, dtype=float)
    p_co = np.zeros((len(d),), dtype=float)
    p_cross = np.zeros((len(d),), dtype=float)
    if len(d) == 0:
        return DualCPPDP(delay_tau_s=d, P_co=p_co, P_cross=p_cross, XPD_tau_db=None)

    rows = list(ray_rows)
    U = dict(link_U or {})
    rr = rng if rng is not None else np.random.default_rng()
    model = xpr_model if xpr_model is not None else ConstantXPR(10.0)

    for r in rows:
        tau = float(r.get("tau_s", np.nan))
        if not np.isfinite(tau):
            continue
        idx = _bin_index(d, tau)
        if idx < 0:
            continue
        P = float(r.get("P_lin", 0.0))
        if P <= 0.0:
            continue

        if floor_model is not None and str(U.get("scenario_id", "")).upper().startswith("C0"):
            xpd_db = float(
                floor_model.sample_floor_xpd_db(
                    f_hz=U.get("f_center_hz", None),
                    yaw_deg=float(U.get("yaw_deg", 0.0)),
                    pitch_deg=float(U.get("pitch_deg", 0.0)),
                    rng=rr,
                )
            )
        else:
            xpr_db = float(model.sample_xpr_db(r, U, rr))
            parity = r.get("parity", ("even" if int(r.get("n_bounce", 0)) % 2 == 0 else "odd"))
            xpd_db = float(xpd_from_xpr(parity, xpr_db))
        a, b = split_power(P, xpd_db)
        p_co[idx] += float(a)
        p_cross[idx] += float(b)

    xpd_tau = None
    if include_xpd_tau:
        xpd_tau = 10.0 * np.log10((p_co + EPS) / (p_cross + EPS))
    return DualCPPDP(delay_tau_s=d, P_co=p_co, P_cross=p_cross, XPD_tau_db=xpd_tau)

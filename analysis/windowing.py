"""Window helpers for tau0 detection and early/late masks."""

from __future__ import annotations

from typing import Any

import numpy as np


EPS = 1e-15


def estimate_tau0(
    delay_tau_s: np.ndarray,
    P_total: np.ndarray,
    method: str = "threshold",
    noise_tail_s: float = 8e-9,
    margin_db: float = 6.0,
) -> dict[str, Any]:
    tau = np.asarray(delay_tau_s, dtype=float)
    p = np.asarray(P_total, dtype=float)
    if len(tau) == 0:
        return {"tau0_s": 0.0, "index": 0, "noise_floor": 0.0, "threshold": 0.0, "method": str(method)}

    if str(method).lower() == "peak":
        i = int(np.argmax(p))
        return {"tau0_s": float(tau[i]), "index": i, "noise_floor": np.nan, "threshold": np.nan, "method": "peak"}

    t_tail_start = float(max(0.0, tau[-1] - max(noise_tail_s, 0.0)))
    tail = tau >= t_tail_start
    if not np.any(tail):
        tail = np.ones_like(tau, dtype=bool)
    noise = float(np.median(p[tail]))
    thr = float(max(noise * (10.0 ** (float(margin_db) / 10.0)), EPS))
    idxs = np.where(p >= thr)[0]
    if len(idxs):
        i = int(idxs[0])
    else:
        i = int(np.argmax(p))
    return {
        "tau0_s": float(tau[i]),
        "index": i,
        "noise_floor": noise,
        "threshold": thr,
        "method": "threshold",
    }


def make_early_late_masks(
    delay_tau_s: np.ndarray,
    tau0_s: float,
    Te_s: float,
    Tmax_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    tau = np.asarray(delay_tau_s, dtype=float)
    te = max(float(Te_s), 0.0)
    tmax = max(float(Tmax_s), te)
    early = (tau >= float(tau0_s)) & (tau < float(tau0_s) + te)
    late = (tau >= float(tau0_s) + te) & (tau <= float(tau0_s) + tmax)
    return early, late

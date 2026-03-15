"""Parity-aware XPD/XPR sign rule and power split."""

from __future__ import annotations

import numpy as np


EPS = 1e-15


def xpd_from_xpr(parity: str | int, xpr_db: float) -> float:
    p = str(parity).lower()
    if p in {"odd", "1", "-1"}:
        return float(-xpr_db)
    return float(xpr_db)


def split_power(P_lin: float, xpd_db: float, eps: float = EPS) -> tuple[float, float]:
    """Split total power into co/cross using XPD ratio."""

    P = max(float(P_lin), 0.0)
    r = float(10.0 ** (float(xpd_db) / 10.0))
    den = max(1.0 + r, float(eps))
    p_co = float(P * r / den)
    p_cross = float(P / den)
    # preserve sum numerically
    if P > 0.0:
        s = p_co + p_cross
        if s > 0.0:
            scale = P / s
            p_co *= scale
            p_cross *= scale
    return p_co, p_cross

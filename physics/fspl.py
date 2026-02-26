"""Free-space path-loss helpers."""

from __future__ import annotations

import numpy as np


C0_MPS = 299_792_458.0
EPS = 1e-15


def fspl_db(L_m: float | np.ndarray, f_hz: float | np.ndarray) -> np.ndarray:
    """FSPL in dB: 20log10(4*pi*L/lambda)."""

    L = np.asarray(L_m, dtype=float)
    f = np.asarray(f_hz, dtype=float)
    lam = C0_MPS / np.maximum(f, EPS)
    x = 4.0 * np.pi * np.maximum(L, EPS) / np.maximum(lam, EPS)
    return 20.0 * np.log10(np.maximum(x, EPS))


def fspl_linear(L_m: float | np.ndarray, f_hz: float | np.ndarray) -> np.ndarray:
    """Linear path gain equivalent to FSPL (1/FSPL_linear)."""

    db = fspl_db(L_m, f_hz)
    return 10.0 ** (-db / 10.0)

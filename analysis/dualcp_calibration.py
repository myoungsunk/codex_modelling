"""Calibration helpers for dual-CP sequential measurements."""

from __future__ import annotations

from typing import Any

import numpy as np


EPS = 1e-15


def dualcp_xpd_db_from_Hf(H_f: np.ndarray, power_floor: float = 1e-18) -> np.ndarray:
    """Return frequency-wise XPD in dB for dual-CP mapping.

    Uses RHCP->RHCP as co and RHCP->LHCP as cross:
      co = |H00|^2
      cross = |H10|^2
    """

    h = np.asarray(H_f, dtype=np.complex128)
    if h.ndim != 3 or h.shape[1:] != (2, 2):
        raise ValueError("H_f must have shape (Nf,2,2)")
    co = np.abs(h[:, 0, 0]) ** 2
    cross = np.maximum(np.abs(h[:, 1, 0]) ** 2, float(power_floor))
    return (10.0 * np.log10((co + EPS) / (cross + EPS))).astype(float)


def _align_curve(
    f_src: np.ndarray,
    y_src: np.ndarray,
    f_ref: np.ndarray,
) -> np.ndarray:
    f_src_a = np.asarray(f_src, dtype=float)
    y_src_a = np.asarray(y_src, dtype=float)
    f_ref_a = np.asarray(f_ref, dtype=float)
    if len(f_src_a) != len(y_src_a):
        raise ValueError("frequency/value length mismatch")
    return np.interp(
        f_ref_a,
        f_src_a,
        y_src_a,
        left=float(y_src_a[0]),
        right=float(y_src_a[-1]),
    )


def estimate_xpd_floor_from_cases(
    cases: list[dict[str, Any]],
    method: str = "median",
    subbands: list[tuple[int, int]] | None = None,
    percentiles: tuple[float, float] = (5.0, 95.0),
    alignment_sweep: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate frequency-wise XPD floor from LOS/free-space cases."""

    if not cases:
        raise ValueError("cases must be non-empty")
    p_lo, p_hi = float(percentiles[0]), float(percentiles[1])
    if not (0.0 <= p_lo < p_hi <= 100.0):
        raise ValueError("percentiles must satisfy 0<=lo<hi<=100")

    f_ref = np.asarray(cases[0]["frequency_hz"], dtype=float)
    if f_ref.ndim != 1 or len(f_ref) == 0:
        raise ValueError("invalid case frequency_hz")

    curves: list[np.ndarray] = []
    labels: list[str] = []
    for i, c in enumerate(cases):
        f = np.asarray(c["frequency_hz"], dtype=float)
        h = np.asarray(c["H_f"], dtype=np.complex128)
        xpd = dualcp_xpd_db_from_Hf(h)
        if len(f) == len(f_ref) and np.allclose(f, f_ref, rtol=0.0, atol=0.0):
            xpd_i = xpd
        else:
            xpd_i = _align_curve(f, xpd, f_ref)
        curves.append(np.asarray(xpd_i, dtype=float))
        labels.append(str(c.get("case_id", i)))

    X = np.asarray(curves, dtype=float)
    if str(method).lower() != "median":
        raise ValueError("unsupported method, currently only 'median'")
    floor_db = np.nanmedian(X, axis=0)
    p_low = np.nanpercentile(X, p_lo, axis=0)
    p_high = np.nanpercentile(X, p_hi, axis=0)
    uncert = 0.5 * (p_high - p_low)

    out: dict[str, Any] = {
        "version": "dualcp_floor_v1",
        "method": f"median+p{int(p_lo)}_p{int(p_hi)}",
        "n_cases": int(X.shape[0]),
        "case_ids": labels,
        "frequency_hz": f_ref.tolist(),
        "xpd_floor_db": np.asarray(floor_db, dtype=float).tolist(),
        "xpd_floor_uncert_db": np.asarray(uncert, dtype=float).tolist(),
        "xpd_floor_p_lo_db": np.asarray(p_low, dtype=float).tolist(),
        "xpd_floor_p_hi_db": np.asarray(p_high, dtype=float).tolist(),
        "percentiles": [p_lo, p_hi],
    }
    if alignment_sweep is not None:
        out["alignment_sweep"] = dict(alignment_sweep)

    if subbands:
        sb_out: list[dict[str, Any]] = []
        for idx, (s, e) in enumerate(subbands):
            si, ei = int(s), int(e)
            if ei <= si or si < 0 or ei > len(f_ref):
                continue
            sb_out.append(
                {
                    "index": idx,
                    "start_idx": si,
                    "end_idx": ei,
                    "f_lo_hz": float(f_ref[si]),
                    "f_hi_hz": float(f_ref[ei - 1]),
                    "xpd_floor_db": float(np.nanmedian(floor_db[si:ei])),
                    "xpd_floor_uncert_db": float(np.nanmedian(uncert[si:ei])),
                }
            )
        out["subbands"] = sb_out
    return out


def apply_floor_excess(xpd_db: np.ndarray | list[float], floor_db: np.ndarray | list[float] | float) -> np.ndarray:
    """Compute excess XPD over floor in dB."""

    x = np.asarray(xpd_db, dtype=float)
    f = np.asarray(floor_db, dtype=float)
    if f.ndim == 0:
        return x - float(f)
    if len(f) != len(x):
        raise ValueError("xpd_db and floor_db length mismatch")
    return x - f

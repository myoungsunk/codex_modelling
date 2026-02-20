"""Tap-wise vs path-wise consistency diagnostics (E12).

This module compares:
1) strongest path delay/XPD in path domain, and
2) peak tap delay/XPD in CIR domain.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis.ctf_cir import ctf_to_cir, detect_cir_wrap, synthesize_ctf_with_source, tau_resolution_s
from analysis.xpd_stats import tapwise_xpd


def _path_power(path: dict[str, Any], matrix_source: str) -> float:
    use_j = str(matrix_source).upper() == "J" and "J_f" in path
    m = np.asarray(path["J_f"] if use_j else path["A_f"], dtype=np.complex128)
    return float(np.mean(np.abs(m) ** 2))


def _xpd_db_with_relative_floor(co_power: float, cross_power: float, rel_floor: float = 1e-9) -> float:
    co = float(max(co_power, 0.0))
    cr = float(max(cross_power, 0.0))
    floor = max(rel_floor * max(co, 1e-30), 1e-30)
    return float(10.0 * np.log10((co + 1e-30) / (max(cr, floor) + 1e-30)))


def evaluate_case_tap_path_consistency(
    paths: list[dict[str, Any]],
    f_hz: np.ndarray,
    matrix_source: str = "A",
    nfft: int = 2048,
    window: str = "hann",
    power_floor: float = 1e-12,
    outlier_tau_factor: float = 2.0,
    outlier_xpd_db: float = 10.0,
) -> dict[str, Any]:
    """Return per-case tap/path consistency metrics."""

    if len(paths) == 0:
        return {
            "n_paths": 0,
            "status": "EMPTY",
            "peak_tau_s": np.nan,
            "strongest_tau_s": np.nan,
            "delta_tau_s": np.nan,
            "xpd_tap_peak_db": np.nan,
            "xpd_path_strongest_db": np.nan,
            "delta_xpd_db": np.nan,
            "wrap_detected": False,
            "is_outlier": False,
        }

    freq = np.asarray(f_hz, dtype=float)
    powers = np.asarray([_path_power(p, matrix_source=matrix_source) for p in paths], dtype=float)
    i_strong = int(np.argmax(powers))
    p_strong = paths[i_strong]
    strongest_tau_s = float(p_strong["tau_s"])

    use_j = str(matrix_source).upper() == "J" and "J_f" in p_strong
    m_f = np.asarray(p_strong["J_f"] if use_j else p_strong["A_f"], dtype=np.complex128)
    co_path = float(np.mean(np.abs(m_f[:, 0, 0]) ** 2 + np.abs(m_f[:, 1, 1]) ** 2))
    cr_path = float(np.mean(np.abs(m_f[:, 0, 1]) ** 2 + np.abs(m_f[:, 1, 0]) ** 2))
    # Relative floor avoids artificial path-vs-tap offset from absolute FFT scaling.
    rel_floor = float(max(power_floor, 1e-30))
    xpd_path_strongest_db = _xpd_db_with_relative_floor(co_path, cr_path, rel_floor=rel_floor)

    H_f = synthesize_ctf_with_source(paths, freq, matrix_source=matrix_source)
    h_tau, tau_s = ctf_to_cir(H_f, freq, nfft=nfft, window=window)
    res_s = tau_resolution_s(freq, nfft=nfft)

    p_total = np.sum(np.abs(h_tau) ** 2, axis=(1, 2))
    i_peak = int(np.argmax(p_total)) if len(p_total) else 0
    peak_tau_s = float(tau_s[i_peak]) if len(tau_s) else np.nan

    tap_stats = tapwise_xpd(h_tau, tau_s, win_s=None, power_floor=1e-30)
    if len(tap_stats["co"]) > i_peak and len(tap_stats["cross"]) > i_peak:
        co_tap = float(tap_stats["co"][i_peak])
        cr_tap = float(tap_stats["cross"][i_peak])
        xpd_tap_peak_db = _xpd_db_with_relative_floor(co_tap, cr_tap, rel_floor=rel_floor)
    else:
        xpd_tap_peak_db = np.nan

    delay_period_s = float(1.0 / (freq[1] - freq[0])) if len(freq) > 1 else 0.0
    alias_tau = strongest_tau_s % delay_period_s if delay_period_s > 0.0 else strongest_tau_s
    # Use alias-aware delay error because CIR is periodic in 1/df.
    delta_tau_s = float(abs(peak_tau_s - alias_tau))

    wrap_detected = bool(detect_cir_wrap(h_tau, tau_s, expected_first_tau_s=strongest_tau_s, resolution_s=res_s))
    delta_xpd_db = float(abs(xpd_tap_peak_db - xpd_path_strongest_db))

    is_outlier = bool((delta_tau_s > outlier_tau_factor * max(res_s, 1e-15)) or (delta_xpd_db > outlier_xpd_db))
    return {
        "n_paths": int(len(paths)),
        "status": "OK",
        "peak_tau_s": peak_tau_s,
        "strongest_tau_s": strongest_tau_s,
        "delta_tau_s": delta_tau_s,
        "xpd_tap_peak_db": xpd_tap_peak_db,
        "xpd_path_strongest_db": xpd_path_strongest_db,
        "delta_xpd_db": delta_xpd_db,
        "resolution_s": float(res_s),
        "delay_period_s": float(delay_period_s),
        "wrap_detected": wrap_detected,
        "is_outlier": is_outlier,
    }


def evaluate_dataset_tap_path_consistency(
    data: dict[str, Any],
    matrix_source: str = "A",
    nfft: int = 2048,
    window: str = "hann",
    power_floor: float = 1e-12,
    outlier_tau_factor: float = 2.0,
    outlier_xpd_db: float = 10.0,
) -> dict[str, Any]:
    """Evaluate tap/path consistency for all scenario/cases in the dataset."""

    freq = np.asarray(data["frequency"], dtype=float)
    entries: list[dict[str, Any]] = []
    for sid, sc in data.get("scenarios", {}).items():
        for cid, case in sc.get("cases", {}).items():
            r = evaluate_case_tap_path_consistency(
                case.get("paths", []),
                f_hz=freq,
                matrix_source=matrix_source,
                nfft=nfft,
                window=window,
                power_floor=power_floor,
                outlier_tau_factor=outlier_tau_factor,
                outlier_xpd_db=outlier_xpd_db,
            )
            r["scenario_id"] = str(sid)
            r["case_id"] = str(cid)
            entries.append(r)

    if len(entries) == 0:
        return {"entries": []}

    d_tau = np.asarray([float(e.get("delta_tau_s", np.nan)) for e in entries], dtype=float)
    d_xpd = np.asarray([float(e.get("delta_xpd_db", np.nan)) for e in entries], dtype=float)
    wraps = np.asarray([bool(e.get("wrap_detected", False)) for e in entries], dtype=bool)
    outliers = np.asarray([bool(e.get("is_outlier", False)) for e in entries], dtype=bool)
    return {
        "entries": entries,
        "matrix_source": matrix_source,
        "n_cases": int(len(entries)),
        "delta_tau_median_s": float(np.nanmedian(d_tau)),
        "delta_tau_max_s": float(np.nanmax(d_tau)),
        "delta_xpd_median_db": float(np.nanmedian(d_xpd)),
        "delta_xpd_max_db": float(np.nanmax(d_xpd)),
        "wrap_detected_cases": int(np.sum(wraps)),
        "outlier_cases": int(np.sum(outliers)),
    }

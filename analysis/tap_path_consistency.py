"""Tap-wise vs path-wise consistency diagnostics (E12).

This module compares:
1) strongest path delay/XPD in path domain, and
2) tap-window XPD around path delay in CIR domain.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from analysis.ctf_cir import ctf_to_cir, detect_cir_wrap, synthesize_ctf_with_source, tau_resolution_s


def _path_power(path: dict[str, Any], matrix_source: str) -> float:
    use_j = str(matrix_source).upper() == "J" and "J_f" in path
    m = np.asarray(path["J_f"] if use_j else path["A_f"], dtype=np.complex128)
    return float(np.mean(np.abs(m) ** 2))


def _xpd_db_with_relative_floor(co_power: float, cross_power: float, rel_floor: float = 1e-9) -> float:
    co = float(max(co_power, 0.0))
    cr = float(max(cross_power, 0.0))
    floor = max(rel_floor * max(co, 1e-30), 1e-30)
    return float(10.0 * np.log10((co + 1e-30) / (max(cr, floor) + 1e-30)))


def _window_bounds(center_idx: int, half_window_bins: int, n: int) -> tuple[int, int]:
    s = max(0, int(center_idx) - int(half_window_bins))
    e = min(int(n), int(center_idx) + int(half_window_bins) + 1)
    return s, e


def _co_cross_from_window(h_tau: np.ndarray, s: int, e: int) -> tuple[float, float]:
    p = np.abs(h_tau[s:e]) ** 2
    co = float(np.sum(p[:, 0, 0] + p[:, 1, 1]))
    cross = float(np.sum(p[:, 0, 1] + p[:, 1, 0]))
    return co, cross


def _path_xpd_db(path: dict[str, Any], matrix_source: str, rel_floor: float) -> float:
    use_j = str(matrix_source).upper() == "J" and "J_f" in path
    m_f = np.asarray(path["J_f"] if use_j else path["A_f"], dtype=np.complex128)
    co_path = float(np.mean(np.abs(m_f[:, 0, 0]) ** 2 + np.abs(m_f[:, 1, 1]) ** 2))
    cr_path = float(np.mean(np.abs(m_f[:, 0, 1]) ** 2 + np.abs(m_f[:, 1, 0]) ** 2))
    return _xpd_db_with_relative_floor(co_path, cr_path, rel_floor=rel_floor)


def evaluate_case_tap_path_consistency(
    paths: list[dict[str, Any]],
    f_hz: np.ndarray,
    matrix_source: str = "A",
    nfft: int = 2048,
    window: str = "hann",
    power_floor: float = 1e-12,
    half_window_bins: int = 2,
    overlap_policy: str = "skip",
    low_snr_rel_threshold: float = 1e-6,
    outlier_tau_factor: float = 2.0,
    outlier_xpd_db: float = 10.0,
) -> dict[str, Any]:
    """Return per-case tap/path consistency metrics."""

    if len(paths) == 0:
        return {
            "n_paths": 0,
            "status": "EMPTY",
            "outlier_reason": "EMPTY",
            "peak_tau_s": np.nan,
            "strongest_tau_s": np.nan,
            "delta_tau_s": np.nan,
            "xpd_tap_window_db": np.nan,
            "xpd_tap_peak_db": np.nan,
            "xpd_path_strongest_db": np.nan,
            "delta_xpd_db": np.nan,
            "wrap_detected": False,
            "is_outlier": False,
            "overlap": False,
            "overlap_count": 0,
            "window_bins": int(2 * max(0, half_window_bins) + 1),
        }

    freq = np.asarray(f_hz, dtype=float)
    powers = np.asarray([_path_power(p, matrix_source=matrix_source) for p in paths], dtype=float)
    i_strong = int(np.argmax(powers))
    p_strong = paths[i_strong]
    strongest_tau_s = float(p_strong["tau_s"])
    strongest_meta = p_strong.get("meta", {})
    strongest_bounce = int(strongest_meta.get("bounce_count", 0))
    inc = [float(a) for a in strongest_meta.get("incidence_angles", [])]
    strongest_inc_deg = float(np.rad2deg(np.nanmean(np.asarray(inc, dtype=float)))) if inc else np.nan

    rel_floor = float(max(power_floor, 1e-30))
    xpd_path_strongest_db = _path_xpd_db(p_strong, matrix_source=matrix_source, rel_floor=rel_floor)

    H_f = synthesize_ctf_with_source(paths, freq, matrix_source=matrix_source)
    h_tau, tau_s = ctf_to_cir(H_f, freq, nfft=nfft, window=window)
    res_s = tau_resolution_s(freq, nfft=nfft)
    n_tau = len(tau_s)

    p_total = np.sum(np.abs(h_tau) ** 2, axis=(1, 2))
    i_peak = int(np.argmax(p_total)) if len(p_total) else 0
    peak_tau_s = float(tau_s[i_peak]) if len(tau_s) else np.nan

    delay_period_s = float(1.0 / (freq[1] - freq[0])) if len(freq) > 1 else 0.0
    alias_tau = strongest_tau_s % delay_period_s if delay_period_s > 0.0 else strongest_tau_s
    if n_tau > 0:
        i_ref = int(np.argmin(np.abs(tau_s - alias_tau)))
    else:
        i_ref = 0

    half_w = int(max(0, half_window_bins))
    s, e = _window_bounds(i_ref, half_w, n_tau)

    # Path-overlap: multiple path delays landing in same analysis window.
    idx_all = []
    for p in paths:
        t = float(p.get("tau_s", 0.0))
        ta = t % delay_period_s if delay_period_s > 0.0 else t
        idx_all.append(int(np.argmin(np.abs(tau_s - ta))) if n_tau > 0 else 0)
    idx_all = np.asarray(idx_all, dtype=int)
    overlap_count = int(np.sum((idx_all >= s) & (idx_all < e))) if n_tau > 0 else 0
    overlap = bool(overlap_count > 1)

    # Window-center tap for delay mismatch metric.
    if e > s and n_tau > 0:
        local = p_total[s:e]
        i_loc = int(np.argmax(local))
        i_tap = int(s + i_loc)
        tau_tap_s = float(tau_s[i_tap])
        co_tap, cr_tap = _co_cross_from_window(h_tau, s, e)
        xpd_tap_window_db = _xpd_db_with_relative_floor(co_tap, cr_tap, rel_floor=rel_floor)
        win_pwr = float(np.sum(local))
        tot_pwr = float(np.sum(p_total) + 1e-30)
        low_snr = bool((win_pwr / tot_pwr) < float(max(low_snr_rel_threshold, 0.0)))
    else:
        i_tap = 0
        tau_tap_s = np.nan
        xpd_tap_window_db = np.nan
        low_snr = True

    wrap_detected = bool(detect_cir_wrap(h_tau, tau_s, expected_first_tau_s=strongest_tau_s, resolution_s=res_s))
    delta_tau_s = float(abs(tau_tap_s - alias_tau)) if np.isfinite(tau_tap_s) else np.nan

    if overlap and str(overlap_policy).lower() == "skip":
        xpd_tap_eval_db = np.nan
        status = "SKIPPED_OVERLAP"
        outlier_reason = "OVERLAP"
    else:
        xpd_tap_eval_db = xpd_tap_window_db
        status = "OK"
        outlier_reason = "NONE"

    delta_xpd_db = (
        float(abs(xpd_tap_eval_db - xpd_path_strongest_db))
        if np.isfinite(xpd_tap_eval_db) and np.isfinite(xpd_path_strongest_db)
        else np.nan
    )

    if status == "OK":
        if low_snr and (not np.isfinite(delta_xpd_db) or delta_xpd_db > outlier_xpd_db):
            status = "LOW_SNR"
            outlier_reason = "LOW_SNR"
        elif wrap_detected and np.isfinite(delta_tau_s) and delta_tau_s > outlier_tau_factor * max(res_s, 1e-15):
            status = "ALIAS"
            outlier_reason = "ALIAS"
        elif np.isfinite(delta_xpd_db) and delta_xpd_db > outlier_xpd_db:
            status = "OUTLIER"
            outlier_reason = "MISMATCH"

    is_outlier = bool(status in {"OUTLIER", "LOW_SNR", "ALIAS", "SKIPPED_OVERLAP"})
    return {
        "n_paths": int(len(paths)),
        "status": status,
        "outlier_reason": outlier_reason,
        "peak_tau_s": peak_tau_s,
        "tap_window_tau_s": tau_tap_s,
        "strongest_tau_s": strongest_tau_s,
        "delta_tau_s": delta_tau_s,
        "xpd_tap_window_db": xpd_tap_window_db,
        "xpd_tap_peak_db": xpd_tap_window_db,
        "xpd_path_strongest_db": xpd_path_strongest_db,
        "delta_xpd_db": delta_xpd_db,
        "resolution_s": float(res_s),
        "delay_period_s": float(delay_period_s),
        "window_start_idx": int(s),
        "window_end_idx": int(max(s, e - 1)),
        "window_bins": int(max(0, e - s)),
        "overlap": overlap,
        "overlap_count": int(overlap_count),
        "overlap_policy": str(overlap_policy),
        "low_snr": bool(low_snr),
        "strongest_path_index": int(i_strong),
        "strongest_bounce_count": int(strongest_bounce),
        "strongest_incidence_angle_deg": float(strongest_inc_deg),
        "wrap_detected": wrap_detected,
        "is_outlier": is_outlier,
    }


def evaluate_dataset_tap_path_consistency(
    data: dict[str, Any],
    matrix_source: str = "A",
    nfft: int = 2048,
    window: str = "hann",
    power_floor: float = 1e-12,
    half_window_bins: int = 2,
    overlap_policy: str = "skip",
    low_snr_rel_threshold: float = 1e-6,
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
                half_window_bins=half_window_bins,
                overlap_policy=overlap_policy,
                low_snr_rel_threshold=low_snr_rel_threshold,
                outlier_tau_factor=outlier_tau_factor,
                outlier_xpd_db=outlier_xpd_db,
            )
            r["scenario_id"] = str(sid)
            r["case_id"] = str(cid)
            r["material"] = str(case.get("params", {}).get("material", "NA"))
            entries.append(r)

    if len(entries) == 0:
        return {"entries": []}

    d_tau = np.asarray([float(e.get("delta_tau_s", np.nan)) for e in entries], dtype=float)
    d_xpd = np.asarray([float(e.get("delta_xpd_db", np.nan)) for e in entries], dtype=float)
    wraps = np.asarray([bool(e.get("wrap_detected", False)) for e in entries], dtype=bool)
    outliers = np.asarray([bool(e.get("is_outlier", False)) for e in entries], dtype=bool)
    overlaps = np.asarray([bool(e.get("overlap", False)) for e in entries], dtype=bool)
    d_xpd_non_overlap = np.asarray(
        [float(e.get("delta_xpd_db", np.nan)) for e in entries if (not bool(e.get("overlap", False))) and np.isfinite(float(e.get("delta_xpd_db", np.nan)))],
        dtype=float,
    )
    reason_counts: dict[str, int] = {}
    for e in entries:
        k = str(e.get("outlier_reason", "NONE"))
        reason_counts[k] = int(reason_counts.get(k, 0) + 1)
    return {
        "entries": entries,
        "matrix_source": matrix_source,
        "n_cases": int(len(entries)),
        "half_window_bins": int(half_window_bins),
        "overlap_policy": str(overlap_policy),
        "delta_tau_median_s": float(np.nanmedian(d_tau)),
        "delta_tau_max_s": float(np.nanmax(d_tau)),
        "delta_xpd_median_db": float(np.nanmedian(d_xpd)),
        "delta_xpd_max_db": float(np.nanmax(d_xpd)),
        "delta_xpd_non_overlap_max_db": float(np.nanmax(d_xpd_non_overlap)) if len(d_xpd_non_overlap) else np.nan,
        "delta_xpd_non_overlap_median_db": float(np.nanmedian(d_xpd_non_overlap)) if len(d_xpd_non_overlap) else np.nan,
        "wrap_detected_cases": int(np.sum(wraps)),
        "overlap_cases": int(np.sum(overlaps)),
        "overlap_labeled_cases": int(sum(1 for e in entries if str(e.get("outlier_reason", "NONE")) == "OVERLAP")),
        "outlier_cases": int(np.sum(outliers)),
        "outlier_reason_counts": reason_counts,
    }


def write_outlier_csv(path: str | Path, entries: list[dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "scenario_id",
        "case_id",
        "strongest_tau_s",
        "strongest_bounce_count",
        "material",
        "strongest_incidence_angle_deg",
        "overlap_count",
        "overlap",
        "delta_xpd_db",
        "delta_tau_s",
        "outlier_reason",
        "status",
        "n_paths",
        "wrap_detected",
    ]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for e in entries:
            if not bool(e.get("is_outlier", False)):
                continue
            row = {k: e.get(k, "") for k in cols}
            w.writerow(row)
    return p

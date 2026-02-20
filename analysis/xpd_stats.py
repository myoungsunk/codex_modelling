"""XPD/XPR estimation and conditional statistics.

This module provides power-based path/tap metrics that are stable for UWB,
including support for both embedded matrices (A_f) and propagation-only
matrices (J_f).
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


EPS = 1e-15
MatrixSource = Literal["A", "J"]


def _matrix_from_path(path: dict[str, Any], matrix_source: MatrixSource = "A") -> NDArray[np.complex128]:
    if matrix_source == "J" and "J_f" in path:
        return np.asarray(path["J_f"], dtype=np.complex128)
    return np.asarray(path["A_f"], dtype=np.complex128)


def _co_cross_power_spectra(M_f: NDArray[np.complex128]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    co_f = np.abs(M_f[:, 0, 0]) ** 2 + np.abs(M_f[:, 1, 1]) ** 2
    cross_f = np.abs(M_f[:, 0, 1]) ** 2 + np.abs(M_f[:, 1, 0]) ** 2
    return co_f.astype(float), cross_f.astype(float)


def make_subbands(nf: int, num_subbands: int) -> list[tuple[int, int]]:
    if num_subbands <= 0:
        raise ValueError("num_subbands must be positive")
    idx = np.linspace(0, nf, num_subbands + 1, dtype=int)
    out = []
    for i in range(num_subbands):
        s, e = int(idx[i]), int(idx[i + 1])
        if e > s:
            out.append((s, e))
    return out


def subband_centers_hz(f_hz: NDArray[np.float64], subbands: list[tuple[int, int]]) -> NDArray[np.float64]:
    f = np.asarray(f_hz, dtype=float)
    return np.asarray([float(np.mean(f[s:e])) for s, e in subbands], dtype=float)


def xpd_xpr_from_power(co_power: float, cross_power: float, power_floor: float = 1e-12) -> dict[str, float]:
    co = float(co_power)
    cross = max(float(cross_power), float(power_floor))
    xpd_linear = (co + EPS) / (cross + EPS)
    xpd_db = float(10.0 * np.log10(xpd_linear))
    return {
        "co_power": co,
        "cross_power": cross,
        "xpd_linear": float(xpd_linear),
        "xpr_linear": float(xpd_linear),
        "xpd_db": xpd_db,
        "xpr_db": xpd_db,
    }


def pathwise_xpd(
    paths: list[dict[str, Any]],
    subbands: list[tuple[int, int]] | None = None,
    exact_bounce: int | None = None,
    bounce_filter: set[int] | None = None,
    power_floor: float = 1e-12,
    matrix_source: MatrixSource = "A",
) -> list[dict[str, Any]]:
    """Compute path-wise XPD/XPR using mean POWER over frequency.

    For each path and frequency span:
      co = mean(|M11|^2 + |M22|^2)
      cross = mean(|M12|^2 + |M21|^2)
      XPD = 10log10(co/cross)

    where M is A_f (embedded) or J_f (propagation-only).
    """

    out: list[dict[str, Any]] = []
    for i, p in enumerate(paths):
        M_f = _matrix_from_path(p, matrix_source=matrix_source)
        meta = p.get("meta", {})
        bcnt = int(meta.get("bounce_count", 0))

        if exact_bounce is not None and bcnt != exact_bounce:
            continue
        if bounce_filter is not None and bcnt not in bounce_filter:
            continue

        co_f, cross_f = _co_cross_power_spectra(M_f)
        base = {
            "path_index": i,
            "bounce_count": bcnt,
            "parity": "even" if bcnt % 2 == 0 else "odd",
            "tau_s": float(p.get("tau_s", 0.0)),
            "matrix_source": matrix_source,
        }

        if subbands is None:
            stats = xpd_xpr_from_power(float(np.mean(co_f)), float(np.mean(cross_f)), power_floor=power_floor)
            out.append({**base, **stats})
            continue

        for bidx, (s, e) in enumerate(subbands):
            stats = xpd_xpr_from_power(float(np.mean(co_f[s:e])), float(np.mean(cross_f[s:e])), power_floor=power_floor)
            out.append({**base, "subband": bidx, **stats})

    return out


def tapwise_xpd(
    h_tau: NDArray[np.complex128],
    tau_s: NDArray[np.float64],
    win_s: tuple[float, float] | None = None,
    power_floor: float = 1e-12,
) -> dict[str, NDArray[np.float64]]:
    p = np.abs(h_tau) ** 2
    co = p[:, 0, 0] + p[:, 1, 1]
    cross = np.maximum(p[:, 0, 1] + p[:, 1, 0], power_floor)
    xpd = 10.0 * np.log10((co + EPS) / (cross + EPS))
    if win_s is None:
        m = np.ones_like(tau_s, dtype=bool)
    else:
        m = (tau_s >= win_s[0]) & (tau_s <= win_s[1])
    return {"tau_s": tau_s[m], "xpd_db": xpd[m], "co": co[m], "cross": cross[m]}


def early_late_split(samples: list[dict[str, Any]], split_tau_s: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    early = [s for s in samples if float(s.get("tau_s", 0.0)) <= split_tau_s]
    late = [s for s in samples if float(s.get("tau_s", 0.0)) > split_tau_s]
    return early, late


def fit_normal_db(values: NDArray[np.float64]) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"mu": np.nan, "sigma": np.nan, "n": 0}
    mu = float(np.mean(v))
    sigma = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return {"mu": mu, "sigma": sigma, "n": int(len(v))}


def conditional_fit(samples: list[dict[str, Any]], keys: list[str], value_key: str = "xpd_db") -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {}
    for s in samples:
        k = "|".join(str(s.get(x, "NA")) for x in keys)
        buckets.setdefault(k, []).append(float(s[value_key]))
    return {k: fit_normal_db(np.asarray(v, dtype=float)) for k, v in buckets.items()}


def fit_linear_mu_frequency(
    subband_samples: list[dict[str, Any]],
    subband_centers: NDArray[np.float64],
    condition_keys: list[str],
    value_key: str = "xpd_db",
) -> dict[str, dict[str, Any]]:
    """Fit mu(f)=mu0 + mu1*(f-fc) using subband-wise means."""

    centers = np.asarray(subband_centers, dtype=float)
    fc = float(np.mean(centers)) if len(centers) else 0.0
    groups: dict[str, dict[int, list[float]]] = {}

    for s in subband_samples:
        if "subband" not in s:
            continue
        g = "|".join(str(s.get(k, "NA")) for k in condition_keys)
        b = int(s["subband"])
        groups.setdefault(g, {}).setdefault(b, []).append(float(s[value_key]))

    out: dict[str, dict[str, Any]] = {}
    for g, bmap in groups.items():
        xs, ys = [], []
        for b, vals in sorted(bmap.items()):
            if 0 <= b < len(centers) and len(vals) > 0:
                xs.append(float(centers[b]))
                ys.append(float(np.mean(vals)))
        if len(xs) >= 2:
            x = np.asarray(xs, dtype=float) - fc
            y = np.asarray(ys, dtype=float)
            mu1, mu0 = np.polyfit(x, y, 1)
        elif len(xs) == 1:
            mu0, mu1 = ys[0], 0.0
        else:
            mu0, mu1 = np.nan, np.nan
        out[g] = {
            "mu0_db": float(mu0),
            "mu1_db_per_hz": float(mu1),
            "fc_hz": fc,
            "num_points": len(xs),
            "subband_centers_hz": xs,
            "subband_mu_db": ys,
        }
    return out


def incidence_angle_bin_label(incidence_angles_rad: list[float], bins_deg: list[float]) -> str:
    if not incidence_angles_rad:
        return "NA"
    ang_deg = float(np.rad2deg(np.nanmean(np.asarray(incidence_angles_rad, dtype=float))))
    b = np.asarray(bins_deg, dtype=float)
    if len(b) < 2:
        return f"{ang_deg:.1f}deg"
    idx = int(np.digitize([ang_deg], b, right=False)[0]) - 1
    idx = int(np.clip(idx, 0, len(b) - 2))
    return f"[{b[idx]:.0f},{b[idx+1]:.0f})"


def save_stats_json(path: str | Path, stats_obj: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats_obj, indent=2), encoding="utf-8")
    return p


def coupling_eps_from_params(cross_pol_leakage_db: float, axial_ratio_db: float, enable_coupling: bool = True) -> float:
    """Return coupling epsilon that matches rt_core.antenna.Antenna._coupling_matrix."""

    if not enable_coupling:
        return 0.0
    leak = 10.0 ** (-float(cross_pol_leakage_db) / 20.0)
    ar = 10.0 ** (float(axial_ratio_db) / 20.0)
    ar_leak = abs((ar - 1.0) / (ar + 1.0))
    return float(np.clip(leak + ar_leak, 0.0, 0.49))


def estimate_leakage_floor_db(
    tx_cross_pol_leakage_db: float,
    rx_cross_pol_leakage_db: float,
    tx_axial_ratio_db: float = 0.0,
    rx_axial_ratio_db: float = 0.0,
    tx_enable_coupling: bool = True,
    rx_enable_coupling: bool = True,
) -> dict[str, float]:
    """Predict XPD floor from Tx/Rx coupling parameters.

    floor_db ~= 20*log10(1/(eps_tx + eps_rx + 1e-15))
    """

    eps_tx = coupling_eps_from_params(
        cross_pol_leakage_db=tx_cross_pol_leakage_db,
        axial_ratio_db=tx_axial_ratio_db,
        enable_coupling=tx_enable_coupling,
    )
    eps_rx = coupling_eps_from_params(
        cross_pol_leakage_db=rx_cross_pol_leakage_db,
        axial_ratio_db=rx_axial_ratio_db,
        enable_coupling=rx_enable_coupling,
    )
    floor_db = float(20.0 * np.log10(1.0 / (eps_tx + eps_rx + 1e-15)))
    return {"eps_tx": eps_tx, "eps_rx": eps_rx, "xpd_floor_db": floor_db}


def estimate_leakage_floor_from_antenna_config(antenna_config: dict[str, Any]) -> dict[str, float]:
    """Convenience wrapper to estimate leakage floor from runner antenna_config dict."""

    en = bool(antenna_config.get("enable_coupling", True))
    return estimate_leakage_floor_db(
        tx_cross_pol_leakage_db=float(antenna_config.get("tx_cross_pol_leakage_db", 35.0)),
        rx_cross_pol_leakage_db=float(antenna_config.get("rx_cross_pol_leakage_db", 35.0)),
        tx_axial_ratio_db=float(antenna_config.get("tx_axial_ratio_db", 0.0)),
        rx_axial_ratio_db=float(antenna_config.get("rx_axial_ratio_db", 0.0)),
        tx_enable_coupling=en,
        rx_enable_coupling=en,
    )


def leakage_limited_summary(
    xpd_samples_db: list[float] | NDArray[np.float64],
    xpd_floor_db: float,
    median_tolerance_db: float = 1.0,
    sigma_threshold_db: float = 1.0,
) -> dict[str, float | bool]:
    """Return leakage-floor proximity and trigger flag for XPD samples."""

    v = np.asarray(xpd_samples_db, dtype=float)
    if len(v) == 0:
        return {
            "median_xpd_db": np.nan,
            "sigma_xpd_db": np.nan,
            "delta_floor_db": np.nan,
            "is_leakage_limited": False,
        }
    median_xpd_db = float(np.median(v))
    sigma_xpd_db = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    delta_floor_db = float(abs(median_xpd_db - float(xpd_floor_db)))
    is_limited = bool(delta_floor_db < float(median_tolerance_db) and sigma_xpd_db < float(sigma_threshold_db))
    return {
        "median_xpd_db": median_xpd_db,
        "sigma_xpd_db": sigma_xpd_db,
        "delta_floor_db": delta_floor_db,
        "is_leakage_limited": is_limited,
    }

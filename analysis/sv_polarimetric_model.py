"""Stochastic SV-style polarimetric channel model generator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from analysis.xpd_stats import pathwise_xpd, conditional_fit


def _matrix_from_path(path: dict[str, Any], matrix_source: str = "A") -> NDArray[np.complex128]:
    use_j = str(matrix_source).upper() == "J" and "J_f" in path
    return np.asarray(path["J_f"] if use_j else path["A_f"], dtype=np.complex128)


def ray_matrix_from_kappa(kappa: float, rng: np.random.Generator) -> NDArray[np.complex128]:
    """Generate a 2x2 ray matrix with XPR/XPD control via kappa."""

    kk = max(float(kappa), 1e-6)
    inv = 1.0 / np.sqrt(kk)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=4)
    h = np.array(
        [
            [np.exp(1j * phi[0]), inv * np.exp(1j * phi[1])],
            [inv * np.exp(1j * phi[2]), np.exp(1j * phi[3])],
        ],
        dtype=np.complex128,
    )
    return h


def _sample_kappa_linear(parity: str, parity_fit: dict[str, dict[str, float]], rng: np.random.Generator) -> float:
    p = parity_fit.get(parity) or parity_fit.get("NA") or {"mu": 10.0, "sigma": 3.0}
    mu = float(p.get("mu", 10.0))
    sigma = max(float(p.get("sigma", 0.0)), 0.0)
    xpd_db = float(rng.normal(mu, sigma)) if sigma > 0 else mu
    return float(10.0 ** (xpd_db / 10.0))


def offdiag_phases(paths: list[dict[str, Any]], matrix_source: str = "A") -> NDArray[np.float64]:
    vals: list[float] = []
    for p in paths:
        m = _matrix_from_path(p, matrix_source=matrix_source)
        if m.ndim != 3 or m.shape[1:] != (2, 2):
            continue
        vals.extend(np.angle(m[:, 0, 1]).tolist())
        vals.extend(np.angle(m[:, 1, 0]).tolist())
    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def kuiper_uniform_test(
    angles_rad: NDArray[np.float64] | list[float],
    bootstrap_B: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Rotation-invariant circular-uniformity test via Kuiper statistic + bootstrap p-value."""

    a = np.asarray(angles_rad, dtype=float)
    a = a[np.isfinite(a)]
    n = int(len(a))
    if n < 2:
        return {"n": n, "V": np.nan, "p_boot": np.nan}

    u = np.sort((a % (2.0 * np.pi)) / (2.0 * np.pi))
    i = np.arange(1, n + 1, dtype=float)
    d_plus = float(np.max(i / n - u))
    d_minus = float(np.max(u - (i - 1.0) / n))
    V = float(d_plus + d_minus)

    B = int(max(0, bootstrap_B))
    if B <= 0:
        return {"n": n, "V": V, "p_boot": np.nan}

    rng = np.random.default_rng(int(seed))
    v_boot = np.zeros(B, dtype=float)
    for b in range(B):
        ub = np.sort(rng.uniform(0.0, 1.0, size=n))
        dbp = float(np.max(i / n - ub))
        dbm = float(np.max(ub - (i - 1.0) / n))
        v_boot[b] = dbp + dbm
    p_boot = float(np.mean(v_boot >= V))
    return {"n": n, "V": V, "p_boot": p_boot}


def generate_synthetic_paths(
    f_hz: NDArray[np.float64],
    num_rays: int,
    delay_samples_s: NDArray[np.float64],
    power_samples: NDArray[np.float64],
    parity_probs: dict[str, float],
    parity_fit: dict[str, dict[str, float]],
    parity_slope_model: dict[str, dict[str, Any]] | None = None,
    matrix_source: str = "A",
    xpd_freq_noise_sigma_db: float = 0.0,
    sample_slope: bool = False,
    slope_sigma_db_per_hz: float = 0.0,
    kappa_min: float = 1e-6,
    kappa_max: float = 1e12,
    return_diagnostics: bool = False,
    seed: int = 0,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate synthetic per-ray paths with 2x2 matrices and delays."""

    rng = np.random.default_rng(seed)
    freq = np.asarray(f_hz, dtype=float)
    n_f = len(freq)

    d = np.asarray(delay_samples_s, dtype=float)
    p = np.asarray(power_samples, dtype=float)
    if len(d) == 0:
        d = np.array([0.0], dtype=float)
    if len(p) == 0:
        p = np.array([1.0], dtype=float)
    p = np.maximum(p, 1e-15)
    p = p / np.sum(p)

    parity_labels = ["odd", "even"]
    probs = np.array([parity_probs.get("odd", 0.5), parity_probs.get("even", 0.5)], dtype=float)
    if probs.sum() <= 0:
        probs[:] = 0.5
    probs /= probs.sum()

    paths: list[dict[str, Any]] = []
    kappa_clamp_count = 0
    kappa_total = 0
    src = str(matrix_source).upper()
    for i in range(num_rays):
        idx_d = int(rng.integers(0, len(d)))
        idx_p = int(rng.choice(np.arange(len(p)), p=p))
        tau = float(d[idx_d])
        ray_power = float(max(power_samples[idx_p], 1e-15)) if len(power_samples) > idx_p else float(max(np.mean(power_samples), 1e-15))

        parity = str(rng.choice(parity_labels, p=probs))
        pfit = parity_fit.get(parity) or parity_fit.get("NA") or {"mu": 10.0, "sigma": 3.0}
        mu = float(pfit.get("mu", 10.0))
        sigma = max(float(pfit.get("sigma", 0.0)), 0.0)
        slope = 0.0
        fc = float(np.mean(freq)) if n_f > 0 else 0.0
        if parity_slope_model is not None and parity in parity_slope_model:
            slope = float(parity_slope_model[parity].get("mu1_db_per_hz", 0.0))
            fc = float(parity_slope_model[parity].get("fc_hz", fc))
        if sample_slope:
            slope_sig = float(max(slope_sigma_db_per_hz, 0.0))
            if slope_sig > 0.0:
                slope = float(rng.normal(slope, slope_sig))

        xpd0_db = float(rng.normal(mu, sigma)) if sigma > 0.0 else mu
        phi = rng.uniform(0.0, 2.0 * np.pi, size=4)
        M_f = np.zeros((n_f, 2, 2), dtype=np.complex128)
        for k in range(n_f):
            xpd_db_k = xpd0_db + slope * (freq[k] - fc)
            if xpd_freq_noise_sigma_db > 0.0:
                xpd_db_k += float(rng.normal(0.0, xpd_freq_noise_sigma_db))
            kappa_raw = float(10.0 ** (xpd_db_k / 10.0))
            kappa_k = float(np.clip(kappa_raw, float(kappa_min), float(kappa_max)))
            kappa_total += 1
            if not np.isclose(kappa_k, kappa_raw):
                kappa_clamp_count += 1
            inv = 1.0 / np.sqrt(max(kappa_k, 1e-15))
            H = np.array(
                [
                    [np.exp(1j * phi[0]), inv * np.exp(1j * phi[1])],
                    [inv * np.exp(1j * phi[2]), np.exp(1j * phi[3])],
                ],
                dtype=np.complex128,
            )
            M_f[k] = H

        mean_power = float(np.mean(np.sum(np.abs(M_f) ** 2, axis=(1, 2))))
        scale = np.sqrt(ray_power / max(mean_power, 1e-15))
        M_f *= scale
        eye = np.eye(2, dtype=np.complex128)
        g_tx = np.repeat(eye[None, :, :], n_f, axis=0)
        g_rx = np.repeat(eye[None, :, :], n_f, axis=0)
        scalar = np.ones((n_f,), dtype=float)
        j_f = M_f.copy()
        a_f = M_f.copy()
        paths.append(
            {
                "tau_s": tau,
                "A_f": a_f,
                "J_f": j_f,
                "G_tx_f": g_tx,
                "G_rx_f": g_rx,
                "scalar_gain_f": scalar,
                "meta": {
                    "bounce_count": 1 if parity == "odd" else 2,
                    "parity": parity,
                    "synthetic_matrix_source": src,
                    "synthetic_xpd0_db": float(xpd0_db),
                    "synthetic_slope_db_per_hz": float(slope),
                    "interactions": ["synthetic"],
                    "surface_ids": [],
                    "incidence_angles": [],
                    "AoD": [0.0, 0.0, 0.0],
                    "AoA": [0.0, 0.0, 0.0],
                    "segment_basis_uv": [],
                },
            }
        )
    diagnostics = {
        "matrix_source": src,
        "xpd_freq_noise_sigma_db": float(xpd_freq_noise_sigma_db),
        "sample_slope": bool(sample_slope),
        "slope_sigma_db_per_hz": float(max(slope_sigma_db_per_hz, 0.0)),
        "kappa_min": float(kappa_min),
        "kappa_max": float(kappa_max),
        "kappa_clamp_count": int(kappa_clamp_count),
        "kappa_total": int(kappa_total),
        "kappa_clamp_rate": float(kappa_clamp_count / max(kappa_total, 1)),
    }
    if return_diagnostics:
        return paths, diagnostics
    return paths


def summarize_rt_vs_synth(
    rt_paths: list[dict[str, Any]],
    synth_paths: list[dict[str, Any]],
    subbands: list[tuple[int, int]],
    rt_matrix_source: str = "A",
    synth_matrix_source: str = "A",
    phase_bootstrap_B: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    rt_samples = pathwise_xpd(rt_paths, matrix_source=rt_matrix_source)
    sy_samples = pathwise_xpd(synth_paths, matrix_source=synth_matrix_source)
    rt_par = conditional_fit(rt_samples, ["parity"])
    sy_par = conditional_fit(sy_samples, ["parity"])
    rt_x = np.asarray([float(s["xpd_db"]) for s in rt_samples], dtype=float)
    sy_x = np.asarray([float(s["xpd_db"]) for s in sy_samples], dtype=float)
    f3_mu_rt = float(np.mean(rt_x)) if len(rt_x) else np.nan
    f3_mu_sy = float(np.mean(sy_x)) if len(sy_x) else np.nan
    f3_delta = float(abs(f3_mu_rt - f3_mu_sy)) if np.isfinite(f3_mu_rt) and np.isfinite(f3_mu_sy) else np.nan

    rt_sub = pathwise_xpd(rt_paths, subbands=subbands, matrix_source=rt_matrix_source)
    sy_sub = pathwise_xpd(synth_paths, subbands=subbands, matrix_source=synth_matrix_source)

    def _mu_sigma(samples: list[dict[str, Any]], nb: int) -> tuple[list[float], list[float]]:
        mu: list[float] = []
        sg: list[float] = []
        for b in range(nb):
            vals = np.asarray([float(s["xpd_db"]) for s in samples if int(s.get("subband", -1)) == b], dtype=float)
            if len(vals) == 0:
                mu.append(np.nan)
                sg.append(np.nan)
            elif len(vals) == 1:
                mu.append(float(vals[0]))
                sg.append(0.0)
            else:
                mu.append(float(np.mean(vals)))
                sg.append(float(np.std(vals, ddof=1)))
        return mu, sg

    nb = len(subbands)
    mu_rt, sg_rt = _mu_sigma(rt_sub, nb)
    mu_sy, sg_sy = _mu_sigma(sy_sub, nb)
    a_rt = np.asarray(mu_rt, dtype=float)
    a_sy = np.asarray(mu_sy, dtype=float)
    valid = np.isfinite(a_rt) & np.isfinite(a_sy)
    mu_rmse = float(np.sqrt(np.mean((a_rt[valid] - a_sy[valid]) ** 2))) if np.any(valid) else np.nan
    span_rt = float(np.nanmax(a_rt) - np.nanmin(a_rt)) if np.any(np.isfinite(a_rt)) else np.nan
    span_sy = float(np.nanmax(a_sy) - np.nanmin(a_sy)) if np.any(np.isfinite(a_sy)) else np.nan

    ph_rt = offdiag_phases(rt_paths, matrix_source=rt_matrix_source)
    ph_sy = offdiag_phases(synth_paths, matrix_source=synth_matrix_source)
    ku_rt = kuiper_uniform_test(ph_rt, bootstrap_B=phase_bootstrap_B, seed=seed)
    ku_sy = kuiper_uniform_test(ph_sy, bootstrap_B=phase_bootstrap_B, seed=seed + 1)

    return {
        "rt_parity_xpd": rt_par,
        "synthetic_parity_xpd": sy_par,
        "f3_xpd_mu_rt_db": f3_mu_rt,
        "f3_xpd_mu_synth_db": f3_mu_sy,
        "f3_xpd_mu_delta_abs_db": f3_delta,
        "rt_num_paths": len(rt_paths),
        "synthetic_num_paths": len(synth_paths),
        "rt_subband_count": len(rt_sub),
        "synthetic_subband_count": len(sy_sub),
        "subband_mu_rt_db": mu_rt,
        "subband_mu_synth_db": mu_sy,
        "subband_sigma_rt_db": sg_rt,
        "subband_sigma_synth_db": sg_sy,
        "subband_mu_span_rt": span_rt,
        "subband_mu_span_synth": span_sy,
        "subband_mu_rmse": mu_rmse,
        "phase_uniformity_V_rt": float(ku_rt["V"]) if np.isfinite(ku_rt["V"]) else np.nan,
        "phase_uniformity_p_rt": float(ku_rt["p_boot"]) if np.isfinite(ku_rt["p_boot"]) else np.nan,
        "phase_uniformity_V_synth": float(ku_sy["V"]) if np.isfinite(ku_sy["V"]) else np.nan,
        "phase_uniformity_p_synth": float(ku_sy["p_boot"]) if np.isfinite(ku_sy["p_boot"]) else np.nan,
        "phase_uniformity_p": float(ku_sy["p_boot"]) if np.isfinite(ku_sy["p_boot"]) else np.nan,
    }

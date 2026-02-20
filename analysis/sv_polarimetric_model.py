"""Stochastic SV-style polarimetric channel model generator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from analysis.xpd_stats import pathwise_xpd, conditional_fit


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


def generate_synthetic_paths(
    f_hz: NDArray[np.float64],
    num_rays: int,
    delay_samples_s: NDArray[np.float64],
    power_samples: NDArray[np.float64],
    parity_probs: dict[str, float],
    parity_fit: dict[str, dict[str, float]],
    parity_slope_model: dict[str, dict[str, Any]] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
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

        phi = rng.uniform(0.0, 2.0 * np.pi, size=4)
        A_f = np.zeros((n_f, 2, 2), dtype=np.complex128)
        for k in range(n_f):
            xpd_db_k = mu + slope * (freq[k] - fc)
            if sigma > 0.0:
                xpd_db_k += float(rng.normal(0.0, sigma))
            kappa_k = float(10.0 ** (xpd_db_k / 10.0))
            inv = 1.0 / np.sqrt(max(kappa_k, 1e-6))
            H = np.array(
                [
                    [np.exp(1j * phi[0]), inv * np.exp(1j * phi[1])],
                    [inv * np.exp(1j * phi[2]), np.exp(1j * phi[3])],
                ],
                dtype=np.complex128,
            )
            A_f[k] = H

        mean_power = float(np.mean(np.sum(np.abs(A_f) ** 2, axis=(1, 2))))
        scale = np.sqrt(ray_power / max(mean_power, 1e-15))
        A_f *= scale
        paths.append(
            {
                "tau_s": tau,
                "A_f": A_f,
                "meta": {
                    "bounce_count": 1 if parity == "odd" else 2,
                    "parity": parity,
                    "interactions": ["synthetic"],
                    "surface_ids": [],
                    "incidence_angles": [],
                    "AoD": [0.0, 0.0, 0.0],
                    "AoA": [0.0, 0.0, 0.0],
                    "segment_basis_uv": [],
                },
            }
        )
    return paths


def summarize_rt_vs_synth(
    rt_paths: list[dict[str, Any]],
    synth_paths: list[dict[str, Any]],
    subbands: list[tuple[int, int]],
    rt_matrix_source: str = "A",
) -> dict[str, Any]:
    rt_samples = pathwise_xpd(rt_paths, matrix_source=rt_matrix_source)
    sy_samples = pathwise_xpd(synth_paths)
    rt_par = conditional_fit(rt_samples, ["parity"])
    sy_par = conditional_fit(sy_samples, ["parity"])

    rt_sub = pathwise_xpd(rt_paths, subbands=subbands, matrix_source=rt_matrix_source)
    sy_sub = pathwise_xpd(synth_paths, subbands=subbands)

    return {
        "rt_parity_xpd": rt_par,
        "synthetic_parity_xpd": sy_par,
        "rt_num_paths": len(rt_paths),
        "synthetic_num_paths": len(synth_paths),
        "rt_subband_count": len(rt_sub),
        "synthetic_subband_count": len(sy_sub),
    }

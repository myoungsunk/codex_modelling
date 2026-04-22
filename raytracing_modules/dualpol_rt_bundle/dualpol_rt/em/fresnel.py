from __future__ import annotations

import numpy as np

from dualpol_rt.em.materials import Material


EPS0 = 8.8541878128e-12


def fresnel_reflection(material: Material, theta_i: float, freqs_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.asarray(freqs_hz, dtype=float)
    if material.kind.lower() == "pec":
        gamma_te = -np.ones(freqs.shape, dtype=np.complex128)
        gamma_tm = np.ones(freqs.shape, dtype=np.complex128)
        return gamma_te, gamma_tm

    omega = 2.0 * np.pi * np.maximum(freqs, 1.0)
    eps_r = material.eps_r(freqs)
    sigma = material.sigma(freqs)
    eps_complex = eps_r - 1j * sigma / (omega * EPS0)
    sin2 = float(np.sin(theta_i) ** 2)
    cos_theta = float(np.cos(theta_i))
    root = np.sqrt(eps_complex - sin2)
    root = np.where(np.real(root) < 0.0, -root, root)
    root = np.where((np.real(root) == 0.0) & (np.imag(root) < 0.0), -root, root)
    gamma_te = (cos_theta - root) / (cos_theta + root)
    gamma_tm = (eps_complex * cos_theta - root) / (eps_complex * cos_theta + root)

    def _passive(gamma: np.ndarray) -> np.ndarray:
        mag = np.minimum(np.abs(gamma), 1.0)
        return (mag * np.exp(1j * np.angle(gamma))).astype(np.complex128)

    return _passive(gamma_te.astype(np.complex128)), _passive(gamma_tm.astype(np.complex128))

"""Polarization utilities for local TE/TM basis and Jones accumulation.

Example:
    >>> import numpy as np
    >>> from rt_core.polarization import transverse_basis, basis_change
    >>> k = np.array([1.0, 0.0, 0.0])
    >>> u, v = transverse_basis(k, np.array([0.0, 0.0, 1.0]))
    >>> M = basis_change(np.column_stack([u, v]), np.column_stack([u, v]))
    >>> np.allclose(M, np.eye(2))
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from rt_core.geometry import Material, normalize

Vec3 = NDArray[np.float64]
CVec3 = NDArray[np.complex128]
Mat2 = NDArray[np.complex128]


@dataclass(frozen=True)
class DepolConfig:
    enabled: bool = False
    apply_mode: str = "event"  # event|path
    side_mode: str = "both"  # both|left|right
    rho_func: Callable[[dict], float] | None = None
    seed: int = 0


def transverse_basis(k: Vec3, up_hint: Vec3) -> tuple[Vec3, Vec3]:
    """Build orthonormal transverse basis (u,v) for direction k."""

    kk = normalize(np.asarray(k, dtype=float))
    up = np.asarray(up_hint, dtype=float)
    u = up - float(np.dot(up, kk)) * kk
    if np.linalg.norm(u) < 1e-9:
        alt = np.array([1.0, 0.0, 0.0]) if abs(kk[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        u = alt - float(np.dot(alt, kk)) * kk
    u = normalize(u)
    v = normalize(np.cross(kk, u))
    return u, v


def basis_change(src_basis: NDArray[np.complex128], dst_basis: NDArray[np.complex128]) -> Mat2:
    """Return matrix mapping coeffs from src basis to dst basis."""

    return (dst_basis.conj().T @ src_basis).astype(np.complex128)


def _safe_s(k_in: Vec3, n_eff: Vec3) -> Vec3:
    s = np.cross(k_in, n_eff)
    if np.linalg.norm(s) < 1e-9:
        alt = np.array([0.0, 0.0, 1.0]) if abs(k_in[2]) < 0.8 else np.array([0.0, 1.0, 0.0])
        s = np.cross(k_in, alt)
    return normalize(s)


def local_sp_bases(k_in: Vec3, k_out: Vec3, normal: Vec3) -> tuple[Vec3, Vec3, Vec3, Vec3, float, Vec3]:
    """Return (s_in, p_in, s_out, p_out, incidence_angle, effective_normal).

    Sign convention:
    - effective normal points against incoming wave so cos(theta_i) >= 0.
    - (s_in, p_in, k_in) and (s_out, p_out, k_out) are right-handed.
    """

    kin = normalize(np.asarray(k_in, dtype=float))
    kout = normalize(np.asarray(k_out, dtype=float))
    n = normalize(np.asarray(normal, dtype=float))
    if float(np.dot(kin, n)) > 0.0:
        n = -n
    cos_i = -float(np.dot(kin, n))
    cos_i = float(np.clip(cos_i, 0.0, 1.0))
    theta_i = float(np.arccos(cos_i))
    s_in = _safe_s(kin, n)
    s_out = _safe_s(kout, n)
    p_in = normalize(np.cross(kin, s_in))
    p_out = normalize(np.cross(kout, s_out))
    return s_in, p_in, s_out, p_out, theta_i, n


def fresnel_reflection(material: Material, theta_i: float, f_hz: NDArray[np.float64]) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Return (Gamma_s, Gamma_p) over frequency."""

    freq = np.asarray(f_hz, dtype=float)
    if material.kind.upper() == "PEC":
        ones = -np.ones(freq.shape, dtype=np.complex128)
        return ones, ones

    sin2 = float(np.sin(theta_i) ** 2)
    cos_i = float(np.cos(theta_i))
    if material.complex_eps_r is not None:
        eps_c = np.full(freq.shape, material.complex_eps_r, dtype=np.complex128)
    else:
        eps_c = np.full(freq.shape, material.eps_r * (1.0 - 1j * material.tan_delta), dtype=np.complex128)

    root = np.sqrt(eps_c - sin2)
    gamma_s = (cos_i - root) / (cos_i + root)
    gamma_p = (eps_c * cos_i - root) / (eps_c * cos_i + root)
    return gamma_s.astype(np.complex128), gamma_p.astype(np.complex128)


def jones_reflection(material: Material, theta_i: float, f_hz: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Return R(f) with shape (Nf,2,2) in local (s,p) basis."""

    gs, gp = fresnel_reflection(material, theta_i, f_hz)
    n = len(f_hz)
    out = np.zeros((n, 2, 2), dtype=np.complex128)
    out[:, 0, 0] = gs
    out[:, 1, 1] = gp
    return out


def depol_matrix(rho: float, rng: np.random.Generator) -> Mat2:
    """Construct random depolarization mixer D(rho)."""

    rr = float(np.clip(rho, 0.0, 1.0))
    phi1 = float(rng.uniform(0.0, 2.0 * np.pi))
    phi2 = float(rng.uniform(0.0, 2.0 * np.pi))
    a = np.sqrt(1.0 - rr)
    b = np.sqrt(rr)
    return np.array(
        [[a, b * np.exp(1j * phi1)], [b * np.exp(1j * phi2), a]],
        dtype=np.complex128,
    )


def apply_depol(A: NDArray[np.complex128], D_in: Mat2, D_out: Mat2, side_mode: str = "both") -> NDArray[np.complex128]:
    """Apply depolarization mixer to A(f)."""

    if side_mode == "left":
        return (D_out[None, :, :] @ A).astype(np.complex128)
    if side_mode == "right":
        return (A @ D_in[None, :, :]).astype(np.complex128)
    return (D_out[None, :, :] @ A @ D_in[None, :, :]).astype(np.complex128)

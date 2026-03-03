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


def _fallback_tangent_from_normal(n_eff: Vec3) -> Vec3:
    """Return deterministic tangent vector orthogonal to effective normal."""
    n = normalize(np.asarray(n_eff, dtype=float))
    alt = np.array([0.0, 0.0, 1.0], dtype=float) if abs(n[2]) < 0.8 else np.array([0.0, 1.0, 0.0], dtype=float)
    t = np.cross(n, alt)
    if np.linalg.norm(t) < 1e-9:
        t = np.cross(n, np.array([1.0, 0.0, 0.0], dtype=float))
    return normalize(t)


def local_sp_bases(k_in: Vec3, k_out: Vec3, normal: Vec3) -> tuple[Vec3, Vec3, Vec3, Vec3, float, Vec3]:
    """Return (s_in, p_in, s_out, p_out, incidence_angle, effective_normal).

    Sign convention:
    - effective normal points against incoming wave so cos(theta_i) >= 0.
    - (s_in, p_in, k_in) and (s_out, p_out, k_out) are right-handed.
    """

    kin = normalize(np.asarray(k_in, dtype=float))
    kout = normalize(np.asarray(k_out, dtype=float))
    n = normalize(np.asarray(normal, dtype=float))
    # Deterministically orient normal against incoming direction.
    # Tie-break near grazing with outgoing direction to keep reversal stable.
    dot_in = float(np.dot(kin, n))
    dot_out = float(np.dot(kout, n))
    if dot_in > 0.0 or (abs(dot_in) <= 1e-12 and dot_out > 0.0):
        n = -n
    cos_i = -float(np.dot(kin, n))
    cos_i = float(np.clip(cos_i, 0.0, 1.0))
    theta_i = float(np.arccos(cos_i))
    s_in_raw = np.cross(kin, n)
    s_out_raw = np.cross(kout, n)
    n_in = float(np.linalg.norm(s_in_raw))
    n_out = float(np.linalg.norm(s_out_raw))
    eps = 1e-9

    # Near normal incidence, k x n can vanish for incident/reflected directions.
    # Use a shared deterministic tangent to avoid arbitrary basis jumps between s_in/s_out.
    if n_in < eps and n_out < eps:
        s_ref = _fallback_tangent_from_normal(n)
        s_in = s_ref.copy()
        s_out = s_ref.copy()
    elif n_in < eps:
        s_out = normalize(s_out_raw)
        s_in = s_out.copy()
    elif n_out < eps:
        s_in = normalize(s_in_raw)
        s_out = s_in.copy()
    else:
        s_in = normalize(s_in_raw)
        s_out = normalize(s_out_raw)
        # Keep in/out transverse axes aligned when both are well-defined.
        if float(np.dot(s_in, s_out)) < 0.0:
            s_out = -s_out

    p_in = normalize(np.cross(kin, s_in))
    p_out = normalize(np.cross(kout, s_out))
    return s_in, p_in, s_out, p_out, theta_i, n


def fresnel_reflection(material: Material, theta_i: float, f_hz: NDArray[np.float64]) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Return (Gamma_s, Gamma_p) over frequency."""

    freq = np.asarray(f_hz, dtype=float)
    if material.kind.upper() == "PEC":
        gamma_s = -np.ones(freq.shape, dtype=np.complex128)
        tm_sign = float(getattr(material, "pec_tm_sign", -1.0))
        gamma_p = (np.ones(freq.shape, dtype=np.complex128) if tm_sign >= 0.0 else -np.ones(freq.shape, dtype=np.complex128))
        return gamma_s, gamma_p

    sin2 = float(np.sin(theta_i) ** 2)
    cos_i = float(np.cos(theta_i))
    if material.complex_eps_r is not None:
        eps_c = np.full(freq.shape, material.complex_eps_r, dtype=np.complex128)
    else:
        mode = str(getattr(material, "dispersion_model", "const")).lower()
        if mode == "table" and material.table_f_hz and material.table_eps_r:
            f_tab = np.asarray(material.table_f_hz, dtype=float)
            e_tab = np.asarray(material.table_eps_r, dtype=float)
            t_tab = (
                np.asarray(material.table_tan_delta, dtype=float)
                if material.table_tan_delta is not None and len(material.table_tan_delta) == len(f_tab)
                else np.full(f_tab.shape, float(material.tan_delta), dtype=float)
            )
            if np.any(f_tab <= 0.0):
                f_tab = np.maximum(f_tab, 1.0)
            ord_idx = np.argsort(f_tab)
            f_tab = f_tab[ord_idx]
            e_tab = e_tab[ord_idx]
            t_tab = t_tab[ord_idx]
            logf = np.log10(np.maximum(freq, 1.0))
            logft = np.log10(np.maximum(f_tab, 1.0))
            eps_r = np.interp(logf, logft, e_tab, left=e_tab[0], right=e_tab[-1])
            tan = np.interp(logf, logft, t_tab, left=t_tab[0], right=t_tab[-1])
            eps_r = np.maximum(eps_r, 1.0)
            tan = np.maximum(tan, 0.0)
            eps_c = eps_r * (1.0 - 1j * tan)
        elif mode == "debye" and material.debye_eps_inf is not None and material.debye_delta_eps and material.debye_tau_s:
            de = np.asarray(material.debye_delta_eps, dtype=float)
            ts = np.asarray(material.debye_tau_s, dtype=float)
            if len(de) != len(ts) or len(de) == 0:
                eps_c = np.full(freq.shape, material.eps_r * (1.0 - 1j * material.tan_delta), dtype=np.complex128)
            else:
                w = 2.0 * np.pi * freq
                eps_c = np.full(freq.shape, float(material.debye_eps_inf), dtype=np.complex128)
                for d_eps, tau in zip(de, ts):
                    eps_c += d_eps / (1.0 + 1j * w * max(float(tau), 1e-15))
                # Debye dispersion already models frequency-dependent loss via Im{eps(omega)}.
                # Do not apply `tan_delta` again here to avoid double-counting dielectric loss.
        else:
            eps_c = np.full(freq.shape, material.eps_r * (1.0 - 1j * material.tan_delta), dtype=np.complex128)

    root = np.sqrt(eps_c - sin2)
    gamma_s = (cos_i - root) / (cos_i + root)
    gamma_p = (eps_c * cos_i - root) / (eps_c * cos_i + root)
    # Passive-media guard: preserve phase while limiting |Gamma|<=1.
    # Small numerical branch-cut errors can otherwise produce |Gamma| slightly above 1.
    def _clamp_passive(g: NDArray[np.complex128]) -> NDArray[np.complex128]:
        mag = np.abs(g)
        phase = np.exp(1j * np.angle(g))
        mag_c = np.minimum(mag, 1.0)
        return (mag_c * phase).astype(np.complex128)

    return _clamp_passive(gamma_s.astype(np.complex128)), _clamp_passive(gamma_p.astype(np.complex128))


def jones_reflection(material: Material, theta_i: float, f_hz: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Return R(f) with shape (Nf,2,2) in local (s,p) basis.

    Default model is diagonal Fresnel reflection (smooth, isotropic interface).
    Optional effective off-diagonal coupling can be enabled via material fields:
    - material.xpol_coupling_db
    - material.xpol_coupling_phase_deg
    This is an empirical extension hook (not full rough-surface BRDF physics).
    """

    gs, gp = fresnel_reflection(material, theta_i, f_hz)
    n = len(f_hz)
    out = np.zeros((n, 2, 2), dtype=np.complex128)
    out[:, 0, 0] = gs
    out[:, 1, 1] = gp
    xdb = getattr(material, "xpol_coupling_db", None)
    if xdb is not None and np.isfinite(float(xdb)):
        # Amplitude coupling relative to diagonal magnitude scale.
        amp = float(10.0 ** (-float(xdb) / 20.0))
        phi = float(np.deg2rad(float(getattr(material, "xpol_coupling_phase_deg", 0.0))))
        phase = np.exp(1j * phi)
        # Bound cross term by geometric mean diagonal magnitude.
        diag_scale = np.sqrt(np.maximum(np.abs(gs * gp), 0.0))
        c = (amp * diag_scale * phase).astype(np.complex128)
        out[:, 0, 1] = c
        out[:, 1, 0] = np.conj(c)
    return out


def depol_matrix(rho: float, rng: np.random.Generator) -> Mat2:
    """Construct power-preserving SU(2) depolarization mixer D(rho).

    rho controls mixing as sin^2(theta)=rho.
    """

    rr = float(np.clip(rho, 0.0, 1.0))
    theta = float(np.arcsin(np.sqrt(rr)))
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, np.exp(1j * phi) * s], [-np.exp(-1j * phi) * s, c]], dtype=np.complex128)


def apply_depol(A: NDArray[np.complex128], D_in: Mat2, D_out: Mat2, side_mode: str = "both") -> NDArray[np.complex128]:
    """Apply depolarization mixer to A(f)."""

    if side_mode == "left":
        return (D_out[None, :, :] @ A).astype(np.complex128)
    if side_mode == "right":
        return (A @ D_in[None, :, :]).astype(np.complex128)
    return (D_out[None, :, :] @ A @ D_in[None, :, :]).astype(np.complex128)

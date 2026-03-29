from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dualpol_rt.geometry.visibility import normalize


Vec3 = NDArray[np.float64]
Mat2 = NDArray[np.complex128]


def transverse_basis(k: Vec3, up_hint: Vec3 | None = None) -> tuple[Vec3, Vec3]:
    direction = normalize(np.asarray(k, dtype=float))
    up = np.asarray([0.0, 0.0, 1.0] if up_hint is None else up_hint, dtype=float)
    u = up - float(np.dot(up, direction)) * direction
    if np.linalg.norm(u) < 1e-9:
        alt = np.array([1.0, 0.0, 0.0], dtype=float) if abs(direction[0]) < 0.8 else np.array([0.0, 1.0, 0.0], dtype=float)
        u = alt - float(np.dot(alt, direction)) * direction
    u = normalize(u)
    v = normalize(np.cross(direction, u))
    return u, v


def basis_change(src_basis: NDArray[np.complex128], dst_basis: NDArray[np.complex128]) -> Mat2:
    return (dst_basis.conj().T @ src_basis).astype(np.complex128)


def _fallback_tangent(normal: Vec3) -> Vec3:
    n = normalize(np.asarray(normal, dtype=float))
    alt = np.array([0.0, 0.0, 1.0], dtype=float) if abs(n[2]) < 0.8 else np.array([0.0, 1.0, 0.0], dtype=float)
    t = np.cross(n, alt)
    if np.linalg.norm(t) < 1e-9:
        t = np.cross(n, np.array([1.0, 0.0, 0.0], dtype=float))
    return normalize(t)


def local_te_tm_bases(
    k_in: Vec3,
    k_out: Vec3,
    normal: Vec3,
) -> tuple[Vec3, Vec3, Vec3, Vec3, float, Vec3]:
    kin = normalize(np.asarray(k_in, dtype=float))
    kout = normalize(np.asarray(k_out, dtype=float))
    n = normalize(np.asarray(normal, dtype=float))
    if float(np.dot(kin, n)) > 0.0:
        n = -n
    cos_theta = float(np.clip(-np.dot(kin, n), 0.0, 1.0))
    theta_i = float(np.arccos(cos_theta))
    te_in_raw = np.cross(kin, n)
    te_out_raw = np.cross(kout, n)
    if np.linalg.norm(te_in_raw) < 1e-9 and np.linalg.norm(te_out_raw) < 1e-9:
        te = _fallback_tangent(n)
        te_in = te.copy()
        te_out = te.copy()
    elif np.linalg.norm(te_in_raw) < 1e-9:
        te_out = normalize(te_out_raw)
        te_in = te_out.copy()
    elif np.linalg.norm(te_out_raw) < 1e-9:
        te_in = normalize(te_in_raw)
        te_out = te_in.copy()
    else:
        te_in = normalize(te_in_raw)
        te_out = normalize(te_out_raw)
        if float(np.dot(te_in, te_out)) < 0.0:
            te_out = -te_out
    tm_in = normalize(np.cross(kin, te_in))
    tm_out = normalize(np.cross(kout, te_out))
    return te_in, tm_in, te_out, tm_out, theta_i, n

from __future__ import annotations

import numpy as np

from dualpol_rt.channel.path_record import PathRecord
from dualpol_rt.em.basis import basis_change, canonical_up_hint, local_te_tm_bases, transverse_basis
from dualpol_rt.em.fresnel import fresnel_reflection


LIGHT_SPEED_M_PER_S = 299792458.0


def _reflection_mixing_matrix(xpol_coupling_db: float) -> np.ndarray:
    amplitude = float(10.0 ** (-max(float(xpol_coupling_db), 0.0) / 20.0))
    amplitude = min(amplitude, 1.0 - 1e-12)
    co_pol = float(np.sqrt(max(1.0 - amplitude**2, 0.0)))
    return np.array(
        [
            [co_pol, 1j * amplitude],
            [1j * amplitude, co_pol],
        ],
        dtype=np.complex128,
    )


def path_jones_response(
    path: PathRecord,
    freqs_hz: np.ndarray,
    up_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.asarray(freqs_hz, dtype=float)
    basis_up_hint = canonical_up_hint(up_hint)
    jones_f = np.repeat(np.eye(2, dtype=np.complex128)[None, :, :], len(freqs), axis=0)
    points = [np.asarray(point, dtype=float) for point in path.points]
    for bounce_idx in range(path.bounce_count):
        k_in = points[bounce_idx + 1] - points[bounce_idx]
        k_out = points[bounce_idx + 2] - points[bounce_idx + 1]
        w_in = np.column_stack(transverse_basis(k_in, up_hint=basis_up_hint)).astype(np.complex128)
        w_out = np.column_stack(transverse_basis(k_out, up_hint=basis_up_hint)).astype(np.complex128)
        te_in, tm_in, te_out, tm_out, theta_i, _n_eff = local_te_tm_bases(k_in, k_out, path.normals[bounce_idx])
        local_in = np.column_stack([te_in, tm_in]).astype(np.complex128)
        local_out = np.column_stack([te_out, tm_out]).astype(np.complex128)
        in_rot = basis_change(w_in, local_in)
        out_rot = basis_change(local_out, w_out)
        gamma_te, gamma_tm = fresnel_reflection(path.materials[bounce_idx], theta_i, freqs)
        refl = np.zeros((len(freqs), 2, 2), dtype=np.complex128)
        mixing = _reflection_mixing_matrix(path.materials[bounce_idx].xpol_coupling_db)
        refl[:, 0, 0] = mixing[0, 0] * gamma_te
        refl[:, 0, 1] = mixing[0, 1] * gamma_tm
        refl[:, 1, 0] = mixing[1, 0] * gamma_te
        refl[:, 1, 1] = mixing[1, 1] * gamma_tm
        event = np.einsum("ab,kbc,cd->kad", out_rot, refl, in_rot)
        jones_f = np.einsum("kab,kbc->kac", event, jones_f)
    scalar_factor = (LIGHT_SPEED_M_PER_S / np.maximum(freqs, 1.0)) / (4.0 * np.pi * max(path.path_length_m, 1e-12))
    return jones_f, scalar_factor.astype(np.complex128)


def attach_em_response(
    path: PathRecord,
    freqs_hz: np.ndarray,
    up_hint: np.ndarray | None = None,
) -> PathRecord:
    basis_up_hint = canonical_up_hint(up_hint)
    jones_f, scalar_factor_f = path_jones_response(path, freqs_hz=freqs_hz, up_hint=basis_up_hint)
    return path.with_em_response(jones_f=jones_f, scalar_factor_f=scalar_factor_f, basis_up_hint=basis_up_hint)

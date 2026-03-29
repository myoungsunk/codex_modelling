from __future__ import annotations

import numpy as np

from dualpol_rt.channel.path_record import PathRecord
from dualpol_rt.em.basis import canonical_up_hint, transverse_basis
from dualpol_rt.em.jones import attach_em_response


def _circular_basis_matrix(convention: str = "IEEE-RHCP", circular_order: str = "RL") -> np.ndarray:
    if convention.upper() != "IEEE-RHCP":
        raise ValueError(f"unsupported circular convention: {convention}")
    left = np.array([1.0, 1j], dtype=np.complex128) / np.sqrt(2.0)
    right = np.array([1.0, -1j], dtype=np.complex128) / np.sqrt(2.0)
    if circular_order.upper() == "LR":
        return np.column_stack([left, right]).astype(np.complex128)
    if circular_order.upper() == "RL":
        return np.column_stack([right, left]).astype(np.complex128)
    raise ValueError(f"unsupported circular order: {circular_order}")


def convert_basis(
    H_f: np.ndarray,
    src: str,
    dst: str,
    convention: str = "IEEE-RHCP",
    circular_order: str = "RL",
) -> np.ndarray:
    src_norm = str(src).lower()
    dst_norm = str(dst).lower()
    if src_norm == dst_norm:
        return np.asarray(H_f, dtype=np.complex128)
    if {src_norm, dst_norm} != {"linear", "circular"}:
        raise ValueError(f"unsupported basis conversion: {src}->{dst}")
    U = _circular_basis_matrix(convention=convention, circular_order=circular_order)
    H = np.asarray(H_f, dtype=np.complex128)
    if src_norm == "linear" and dst_norm == "circular":
        return np.einsum("ab,kbc,cd->kad", U.conj().T, H, U)
    return np.einsum("ab,kbc,cd->kad", U, H, U.conj().T)


def _wave_basis(direction_world: np.ndarray, up_hint: np.ndarray | None = None) -> np.ndarray:
    return np.column_stack(transverse_basis(direction_world, up_hint=up_hint)).astype(np.complex128)


def _emit_matrix(pattern, freqs_hz: np.ndarray, direction_world: np.ndarray, up_hint: np.ndarray | None = None) -> np.ndarray:
    wave_basis = _wave_basis(direction_world, up_hint=up_hint)
    out = np.zeros((len(freqs_hz), 2, pattern.port_count), dtype=np.complex128)
    for port_id in range(pattern.port_count):
        field_world = pattern.port_vector_world(port_id, freqs_hz, direction_world)
        out[:, :, port_id] = np.einsum("ab,kb->ka", wave_basis.conj().T, field_world)
    return out


def _receive_matrix(pattern, freqs_hz: np.ndarray, propagation_direction_world: np.ndarray, up_hint: np.ndarray | None = None) -> np.ndarray:
    look_direction_world = -np.asarray(propagation_direction_world, dtype=float)
    look_basis = _wave_basis(look_direction_world, up_hint=up_hint)
    out = np.zeros((len(freqs_hz), pattern.port_count, 2), dtype=np.complex128)
    for port_id in range(pattern.port_count):
        field_world = pattern.port_vector_world(port_id, freqs_hz, look_direction_world)
        # Reciprocity is evaluated in the antenna look-direction basis. This
        # avoids the artificial phi-axis sign flip between +/-k.
        out[:, port_id, :] = np.einsum("kb,ba->ka", field_world.conj(), look_basis)
    return out


def _matching_up_hint(path: PathRecord, up_hint: np.ndarray) -> bool:
    path_up_hint = canonical_up_hint(path.basis_up_hint)
    return bool(np.allclose(path_up_hint, up_hint, atol=1e-12, rtol=0.0))


def build_channel(
    paths: list[PathRecord],
    tx_pattern,
    rx_pattern,
    freqs_hz: np.ndarray,
    eval_basis: str = "linear",
    convention: str = "IEEE-RHCP",
    circular_order: str = "RL",
    up_hint: np.ndarray | None = None,
) -> np.ndarray:
    freqs = np.asarray(freqs_hz, dtype=float)
    basis_up_hint = canonical_up_hint(up_hint)
    H = np.zeros((len(freqs), rx_pattern.port_count, tx_pattern.port_count), dtype=np.complex128)
    for path in paths:
        if path.jones_f is not None and path.scalar_factor_f is not None and _matching_up_hint(path, basis_up_hint):
            enriched = path
        else:
            enriched = attach_em_response(path, freqs_hz=freqs, up_hint=basis_up_hint)
        g_tx = _emit_matrix(tx_pattern, freqs, path.launch_dir, up_hint=basis_up_hint)
        g_rx = _receive_matrix(rx_pattern, freqs, path.arrival_dir, up_hint=basis_up_hint)
        core = np.einsum(
            "kab,kbc,kcd->kad",
            g_rx,
            enriched.jones_f * enriched.scalar_factor_f[:, None, None],
            g_tx,
        )
        phase = np.exp(-1j * 2.0 * np.pi * freqs * path.delay_s)[:, None, None]
        H += core * phase
    if str(eval_basis).lower() != "linear":
        H = convert_basis(H, src="linear", dst=eval_basis, convention=convention, circular_order=circular_order)
    return H

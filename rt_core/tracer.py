"""Deterministic specular polarimetric ray tracer.

Example:
    >>> import numpy as np
    >>> from rt_core.antenna import Antenna
    >>> from rt_core.geometry import Plane, Material
    >>> from rt_core.tracer import trace_paths
    >>> tx = Antenna(np.array([0,0,0]), np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]))
    >>> rx = Antenna(np.array([2,0,0]), np.array([-1,0,0]), np.array([0,1,0]), np.array([0,0,1]))
    >>> paths = trace_paths([], tx, rx, np.linspace(6e9,7e9,8), max_bounce=0)
    >>> len(paths)
    1
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from rt_core.antenna import Antenna
from rt_core.geometry import Plane, line_plane_intersection, normalize, path_length, reflect_direction, reflect_point
from rt_core.polarization import DepolConfig, apply_depol, basis_change, depol_matrix, jones_reflection, local_sp_bases, transverse_basis


@dataclass
class PathResult:
    tau_s: float
    A_f: NDArray[np.complex128]  # (Nf,2,2)
    bounce_count: int
    interactions: list[str]
    surface_ids: list[int]
    incidence_angles: list[float]
    AoD: NDArray[np.float64]
    AoA: NDArray[np.float64]
    points: list[NDArray[np.float64]]

    def meta_dict(self) -> dict:
        return {
            "bounce_count": self.bounce_count,
            "interactions": self.interactions,
            "surface_ids": self.surface_ids,
            "incidence_angles": self.incidence_angles,
            "AoD": self.AoD.tolist(),
            "AoA": self.AoA.tolist(),
        }


def _enumerate_sequences(num_planes: int, bounce: int) -> list[tuple[int, ...]]:
    if bounce == 0:
        return [tuple()]
    return list(product(range(num_planes), repeat=bounce))


def _construct_reflection_points(tx: NDArray[np.float64], rx: NDArray[np.float64], seq: Sequence[Plane]) -> list[NDArray[np.float64]] | None:
    if not seq:
        return [tx, rx]

    images: list[NDArray[np.float64]] = [tx]
    for pl in seq:
        images.append(reflect_point(images[-1], pl))

    points: list[NDArray[np.float64] | None] = [None] * len(seq)
    target = rx
    for i in range(len(seq) - 1, -1, -1):
        p = line_plane_intersection(images[i + 1], target, seq[i])
        if p is None:
            return None
        points[i] = p
        target = p

    out = [tx]
    out.extend(p for p in points if p is not None)
    out.append(rx)
    return out


def _path_valid(points: list[NDArray[np.float64]]) -> bool:
    for i in range(len(points) - 1):
        if np.linalg.norm(points[i + 1] - points[i]) < 1e-7:
            return False
    return True


def _default_rho(_: dict) -> float:
    return 0.0


def trace_paths(
    planes: Sequence[Plane],
    tx: Antenna,
    rx: Antenna,
    f_hz: NDArray[np.float64],
    max_bounce: int = 2,
    los_enabled: bool = True,
    c0: float = 299_792_458.0,
    depol: DepolConfig | None = None,
) -> list[PathResult]:
    """Trace paths and return per-path delay and 2x2 transfer matrix over frequency."""

    if max_bounce not in (0, 1, 2):
        raise ValueError("max_bounce must be 0, 1, or 2")

    dep = depol or DepolConfig(enabled=False)
    rng = np.random.default_rng(dep.seed)
    rho_fn: Callable[[dict], float] = dep.rho_func or _default_rho

    freq = np.asarray(f_hz, dtype=float)
    n_f = len(freq)
    results: list[PathResult] = []

    for b in range(max_bounce + 1):
        if b == 0 and not los_enabled:
            continue
        for idx_seq in _enumerate_sequences(len(planes), b):
            seq = [planes[i] for i in idx_seq]
            points = _construct_reflection_points(tx.position, rx.position, seq)
            if points is None or not _path_valid(points):
                continue

            dirs = [normalize(points[i + 1] - points[i]) for i in range(len(points) - 1)]
            total_len = path_length(points)
            if total_len <= 0.0:
                continue

            # Baseband path amplitude only (phase applied later in channel synthesis).
            scalar_gain = 1.0 / total_len

            wave_basis = tx.wave_basis(dirs[0])
            A = np.repeat(tx.tx_emit_matrix(dirs[0], wave_basis=wave_basis)[None, :, :], n_f, axis=0).astype(np.complex128)

            incidence_angles: list[float] = []
            interactions: list[str] = []
            surface_ids: list[int] = []

            for i, pl in enumerate(seq):
                k_in = dirs[i]
                k_out = dirs[i + 1]
                s, p_in, p_out, theta_i, _ = local_sp_bases(k_in, k_out, pl.unit_normal())
                incidence_angles.append(theta_i)
                interactions.append("reflection")
                surface_ids.append(pl.id)

                sp_in = np.column_stack([s, p_in]).astype(np.complex128)
                sp_out = np.column_stack([s, p_out]).astype(np.complex128)

                t_in = basis_change(wave_basis, sp_in)
                next_u, next_v = transverse_basis(k_out, pl.unit_normal())
                next_basis = np.column_stack([next_u, next_v]).astype(np.complex128)
                t_out = basis_change(next_basis, sp_out)
                r_f = jones_reflection(pl.material, theta_i, freq)

                updated = np.zeros_like(A)
                for k in range(n_f):
                    updated[k] = t_out.conj().T @ r_f[k] @ t_in @ A[k]
                A = updated
                wave_basis = next_basis

                if dep.enabled and dep.apply_mode == "event":
                    rho = float(np.clip(rho_fn({"surface_id": pl.id, "bounce_index": i, "theta_i": theta_i}), 0.0, 1.0))
                    D_in = depol_matrix(rho, rng)
                    D_out = depol_matrix(rho, rng)
                    A = apply_depol(A, D_in=D_in, D_out=D_out, side_mode=dep.side_mode)

            rx_mat = rx.rx_receive_matrix(dirs[-1], wave_basis=wave_basis)
            A = np.einsum("ab,kbc->kac", rx_mat, A).astype(np.complex128)
            A *= scalar_gain

            if dep.enabled and dep.apply_mode == "path":
                rho = float(np.clip(rho_fn({"bounce_count": b, "surface_ids": surface_ids}), 0.0, 1.0))
                D_in = depol_matrix(rho, rng)
                D_out = depol_matrix(rho, rng)
                A = apply_depol(A, D_in=D_in, D_out=D_out, side_mode=dep.side_mode)

            results.append(
                PathResult(
                    tau_s=total_len / c0,
                    A_f=A,
                    bounce_count=b,
                    interactions=interactions,
                    surface_ids=surface_ids,
                    incidence_angles=incidence_angles,
                    AoD=dirs[0],
                    AoA=dirs[-1],
                    points=[np.asarray(p, dtype=float) for p in points],
                )
            )

    results.sort(key=lambda p: p.tau_s)
    return results

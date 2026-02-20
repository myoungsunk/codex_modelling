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
from rt_core.geometry import Plane, line_plane_intersection, normalize, path_length, ray_plane_intersection, reflect_point
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
    segment_basis_uv: list[dict[str, list[float]]]

    def meta_dict(self) -> dict:
        return {
            "bounce_count": self.bounce_count,
            "interactions": self.interactions,
            "surface_ids": self.surface_ids,
            "incidence_angles": self.incidence_angles,
            "AoD": self.AoD.tolist(),
            "AoA": self.AoA.tolist(),
            "segment_basis_uv": self.segment_basis_uv,
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


def _segment_blocked(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    planes: Sequence[Plane],
    skip_ids: set[int],
    eps: float = 1e-7,
) -> bool:
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    seg_len = float(np.linalg.norm(d))
    if seg_len < eps:
        return True
    direction = d / seg_len
    for pl in planes:
        if pl.id in skip_ids:
            continue
        hit = ray_plane_intersection(np.asarray(a, dtype=float), direction, pl, eps=eps)
        if hit is None:
            continue
        if eps < hit.t < seg_len - eps and pl.contains_point(hit.point, eps=eps):
            return True
    return False


def _path_occluded(points: list[NDArray[np.float64]], planes: Sequence[Plane], seq: Sequence[Plane], eps: float = 1e-7) -> bool:
    n_seg = len(points) - 1
    for i in range(n_seg):
        skip = set()
        if i > 0:
            skip.add(seq[i - 1].id)
        if i < len(seq):
            skip.add(seq[i].id)
        if _segment_blocked(points[i], points[i + 1], planes, skip_ids=skip, eps=eps):
            return True
    return False


def _path_key(points: list[NDArray[np.float64]], seq: Sequence[Plane], tau_s: float) -> tuple:
    sid = tuple(pl.id for pl in seq)
    pflat = tuple(tuple(np.round(p, 6).tolist()) for p in points[1:-1])
    return (sid, pflat, round(float(tau_s), 12))


def trace_paths(
    planes: Sequence[Plane],
    tx: Antenna,
    rx: Antenna,
    f_hz: NDArray[np.float64],
    max_bounce: int = 2,
    los_enabled: bool = True,
    use_fspl: bool = True,
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
    seen: set[tuple] = set()

    for b in range(max_bounce + 1):
        if b == 0 and not los_enabled:
            continue
        for idx_seq in _enumerate_sequences(len(planes), b):
            seq = [planes[i] for i in idx_seq]
            points = _construct_reflection_points(tx.position, rx.position, seq)
            if points is None or not _path_valid(points):
                continue
            if any(not seq[i].contains_point(points[i + 1]) for i in range(len(seq))):
                continue
            if _path_occluded(points, planes, seq):
                continue

            dirs = [normalize(points[i + 1] - points[i]) for i in range(len(points) - 1)]
            total_len = path_length(points)
            if total_len <= 0.0:
                continue
            tau = total_len / c0
            key = _path_key(points, seq, tau)
            if key in seen:
                continue
            seen.add(key)

            if use_fspl:
                scalar_gain_f = (c0 / freq) / (4.0 * np.pi * total_len)
            else:
                scalar_gain_f = np.full(n_f, 1.0 / total_len, dtype=float)

            wave_basis = tx.wave_basis(dirs[0])
            J = np.repeat(np.eye(2, dtype=np.complex128)[None, :, :], n_f, axis=0)
            seg_basis_uv: list[dict[str, list[float]]] = []
            u0 = np.real(wave_basis[:, 0]).astype(float)
            v0 = np.real(wave_basis[:, 1]).astype(float)
            seg_basis_uv.append({"k": dirs[0].tolist(), "u": u0.tolist(), "v": v0.tolist()})

            incidence_angles: list[float] = []
            interactions: list[str] = []
            surface_ids: list[int] = []

            for i, pl in enumerate(seq):
                k_in = dirs[i]
                k_out = dirs[i + 1]
                s_in, p_in, s_out, p_out, theta_i, _ = local_sp_bases(k_in, k_out, pl.unit_normal())
                incidence_angles.append(theta_i)
                interactions.append("reflection")
                surface_ids.append(pl.id)

                sp_in = np.column_stack([s_in, p_in]).astype(np.complex128)
                sp_out = np.column_stack([s_out, p_out]).astype(np.complex128)

                t_in = basis_change(wave_basis, sp_in)
                next_u, next_v = transverse_basis(k_out, pl.unit_normal())
                next_basis = np.column_stack([next_u, next_v]).astype(np.complex128)
                t_out = basis_change(next_basis, sp_out)
                r_f = jones_reflection(pl.material, theta_i, freq)

                updated = np.zeros_like(J)
                for k in range(n_f):
                    updated[k] = t_out.conj().T @ r_f[k] @ t_in @ J[k]
                J = updated
                wave_basis = next_basis
                seg_basis_uv.append({"k": k_out.tolist(), "u": next_u.tolist(), "v": next_v.tolist()})

                if dep.enabled and dep.apply_mode == "event":
                    rho = float(np.clip(rho_fn({"surface_id": pl.id, "bounce_index": i, "theta_i": theta_i}), 0.0, 1.0))
                    D_in = depol_matrix(rho, rng)
                    D_out = depol_matrix(rho, rng)
                    J = apply_depol(J, D_in=D_in, D_out=D_out, side_mode=dep.side_mode)

            g_tx = tx.tx_port_to_wave(dirs[0], freq, wave_basis=tx.wave_basis(dirs[0]))
            g_rx_h = rx.rx_wave_to_port(dirs[-1], freq, wave_basis=wave_basis)
            A = np.einsum("kab,kbc,kcd->kad", g_rx_h, J, g_tx).astype(np.complex128)
            A *= scalar_gain_f[:, None, None]
            if rx.basis == "circular" and (b % 2 == 1):
                # Specular mirror reflection flips helicity; odd bounce swaps R/L at receive side.
                swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
                A = np.einsum("ab,kbc->kac", swap, A)

            if dep.enabled and dep.apply_mode == "path":
                rho = float(np.clip(rho_fn({"bounce_count": b, "surface_ids": surface_ids}), 0.0, 1.0))
                D_in = depol_matrix(rho, rng)
                D_out = depol_matrix(rho, rng)
                A = apply_depol(A, D_in=D_in, D_out=D_out, side_mode=dep.side_mode)

            results.append(
                PathResult(
                    tau_s=tau,
                    A_f=A,
                    bounce_count=b,
                    interactions=interactions,
                    surface_ids=surface_ids,
                    incidence_angles=incidence_angles,
                    AoD=dirs[0],
                    AoA=dirs[-1],
                    points=[np.asarray(p, dtype=float) for p in points],
                    segment_basis_uv=seg_basis_uv,
                )
            )

    results.sort(key=lambda p: p.tau_s)
    return results

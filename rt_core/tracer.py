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
from typing import Callable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from rt_core.antenna import Antenna
from rt_core.geometry import Plane, line_plane_intersection, normalize, path_length, ray_plane_intersection, reflect_point
from rt_core.polarization import DepolConfig, apply_depol, basis_change, depol_matrix, jones_reflection, local_sp_bases, transverse_basis


@dataclass
class PathResult:
    tau_s: float
    path_length_m: float
    segment_count: int
    A_f: NDArray[np.complex128]  # (Nf,2,2)
    J_f: NDArray[np.complex128]  # (Nf,2,2), propagation-only before antenna and FSPL
    scalar_gain_f: NDArray[np.float64]  # (Nf,)
    G_tx_f: NDArray[np.complex128]  # (Nf,2,2), tx port -> wave
    G_rx_f: NDArray[np.complex128]  # (Nf,2,2), wave -> rx port
    bounce_count: int
    interactions: list[str]
    surface_ids: list[int]
    incidence_angles: list[float]
    bounce_normals: list[NDArray[np.float64]]
    AoD: NDArray[np.float64]
    AoA: NDArray[np.float64]
    points: list[NDArray[np.float64]]
    segment_basis_uv: list[dict[str, list[float]]]

    @staticmethod
    def _complex_array_to_dict(x: NDArray[np.complex128]) -> dict:
        return {"real": np.real(x).tolist(), "imag": np.imag(x).tolist()}

    def meta_dict(self) -> dict:
        return {
            "path_length_m": float(self.path_length_m),
            "segment_count": int(self.segment_count),
            "bounce_count": self.bounce_count,
            "interactions": self.interactions,
            "surface_ids": self.surface_ids,
            "incidence_angles": self.incidence_angles,
            "bounce_normals": [np.asarray(n, dtype=float).tolist() for n in self.bounce_normals],
            "AoD": self.AoD.tolist(),
            "AoA": self.AoA.tolist(),
            "segment_basis_uv": self.segment_basis_uv,
            "J_f": self._complex_array_to_dict(self.J_f),
            "scalar_gain_f": self.scalar_gain_f.tolist(),
            "G_tx_f": self._complex_array_to_dict(self.G_tx_f),
            "G_rx_f": self._complex_array_to_dict(self.G_rx_f),
        }


def _enumerate_sequences(
    num_planes: int,
    bounce: int,
    max_candidates: int | None = None,
    forbid_immediate_repeat: bool = False,
) -> Iterator[tuple[int, ...]]:
    if bounce == 0:
        yield tuple()
        return
    if num_planes <= 0 or bounce < 0:
        return
    cap = int(max_candidates) if max_candidates is not None and int(max_candidates) > 0 else None
    emitted = 0
    for seq in product(range(num_planes), repeat=bounce):
        if bool(forbid_immediate_repeat) and any(seq[i] == seq[i - 1] for i in range(1, bounce)):
            continue
        yield tuple(seq)
        emitted += 1
        if cap is not None and emitted >= cap:
            break


def _is_occluder_only_plane(plane: Plane) -> bool:
    """Return True when a plane should block rays but never generate reflections.

    LOS blockers are exported with absorber-like material names and dedicated IDs.
    Keep them in occlusion checks, but remove from bounce-sequence enumeration.
    """

    try:
        mat_name = str(getattr(plane.material, "name", "") or "").strip().lower()
    except Exception:
        mat_name = ""
    pid = int(getattr(plane, "id", -1))
    if "absorber_proxy" in mat_name or "occluder" in mat_name:
        return True
    if pid in {9101, 9201, 9301, 9401}:
        return True
    return False


def _construct_reflection_points_source_images(
    tx: NDArray[np.float64],
    rx: NDArray[np.float64],
    seq: Sequence[Plane],
) -> list[NDArray[np.float64]] | None:
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


def _construct_reflection_points_receiver_images(
    tx: NDArray[np.float64],
    rx: NDArray[np.float64],
    seq: Sequence[Plane],
) -> list[NDArray[np.float64]] | None:
    if not seq:
        return [tx, rx]
    n = len(seq)
    rx_images: list[NDArray[np.float64]] = [np.asarray(rx, dtype=float)]
    for j in range(n):
        rx_images.append(reflect_point(rx_images[-1], seq[n - 1 - j]))

    out: list[NDArray[np.float64]] = [np.asarray(tx, dtype=float)]
    curr = out[0]
    for i, pl in enumerate(seq):
        target = rx_images[n - i]
        p = line_plane_intersection(curr, target, pl)
        if p is None:
            return None
        out.append(np.asarray(p, dtype=float))
        curr = out[-1]
    out.append(np.asarray(rx, dtype=float))
    return out


def _construct_reflection_points(tx: NDArray[np.float64], rx: NDArray[np.float64], seq: Sequence[Plane]) -> list[NDArray[np.float64]] | None:
    """Solve specular reflection points for a plane sequence.

    We try source-image and receiver-image formulations to avoid
    directional degeneracies (important for reciprocity reversal).
    """

    pts = _construct_reflection_points_source_images(tx, rx, seq)
    if pts is not None:
        return pts
    return _construct_reflection_points_receiver_images(tx, rx, seq)


def _path_valid(points: list[NDArray[np.float64]]) -> bool:
    for i in range(len(points) - 1):
        if np.linalg.norm(points[i + 1] - points[i]) < 1e-7:
            return False
    return True


def _fit_reflection_points_to_bounds(
    points: list[NDArray[np.float64]],
    seq: Sequence[Plane],
    eps: float = 1e-6,
    clamp_tol: float = 1e-6,
) -> list[NDArray[np.float64]] | None:
    """Clamp tiny out-of-bound reflection drift to finite plate bounds consistently."""

    if len(seq) == 0:
        return points
    out: list[NDArray[np.float64]] = [np.asarray(points[0], dtype=float)]
    for i, pl in enumerate(seq):
        p = np.asarray(points[i + 1], dtype=float)
        if pl.contains_point(p, eps=eps):
            out.append(p)
            continue
        p_c, d_c = pl.clamp_to_bounds(p)
        if d_c <= clamp_tol and pl.contains_point(p_c, eps=eps):
            out.append(np.asarray(p_c, dtype=float))
            continue
        return None
    out.append(np.asarray(points[-1], dtype=float))
    return out


def _default_rho(_: dict) -> float:
    return 0.0


def _transport_wave_basis(k_out: NDArray[np.float64], prev_basis: NDArray[np.complex128], fallback_up: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Parallel-transport-like wave basis update to reduce gauge jumps."""
    k = normalize(np.asarray(k_out, dtype=float))
    u_prev = np.real(np.asarray(prev_basis[:, 0], dtype=np.complex128)).astype(float)
    v_prev = np.real(np.asarray(prev_basis[:, 1], dtype=np.complex128)).astype(float)
    u = u_prev - float(np.dot(u_prev, k)) * k
    if np.linalg.norm(u) < 1e-9:
        u = v_prev - float(np.dot(v_prev, k)) * k
    if np.linalg.norm(u) < 1e-9:
        u, v = transverse_basis(k, np.asarray(fallback_up, dtype=float))
        return np.column_stack([u, v]).astype(np.complex128)
    u = normalize(u)
    v = normalize(np.cross(k, u))
    return np.column_stack([u, v]).astype(np.complex128)


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


def _path_key(
    points: list[NDArray[np.float64]],
    seq: Sequence[Plane],
    tau_s: float,
    point_tol_m: float = 1e-7,
    tau_tol_s: float = 1e-13,
) -> tuple:
    sid = tuple(pl.id for pl in seq)
    ptol = float(max(abs(float(point_tol_m)), 1e-12))
    ttol = float(max(abs(float(tau_tol_s)), 1e-18))
    pflat = tuple(
        tuple(int(np.rint(float(x) / ptol)) for x in np.asarray(p, dtype=float))
        for p in points[1:-1]
    )
    tau_q = int(np.rint(float(tau_s) / ttol))
    return (sid, pflat, tau_q)


def trace_paths(
    planes: Sequence[Plane],
    tx: Antenna,
    rx: Antenna,
    f_hz: NDArray[np.float64],
    max_bounce: int = 2,
    los_enabled: bool = True,
    use_fspl: bool = True,
    force_cp_swap_on_odd_reflection: bool = False,
    c0: float = 299_792_458.0,
    depol: DepolConfig | None = None,
    diffuse_enabled: bool = False,
    diffuse_model: str = "lambertian",
    diffuse_factor: float = 0.0,
    diffuse_lobe_alpha: float = 8.0,
    diffuse_rays_per_hit: int = 0,
    diffuse_seed: int = 0,
    wave_basis_mode: str = "transport",
    min_path_power_db: float | None = None,
    max_paths_per_case: int | None = None,
    max_sequence_candidates_per_bounce: int | None = None,
    forbid_immediate_surface_repeat: bool = False,
    dedup_point_tol_m: float = 1e-7,
    dedup_tau_tol_s: float = 1e-13,
) -> list[PathResult]:
    """Trace paths and return per-path delay and 2x2 transfer matrix over frequency."""

    if int(max_bounce) < 0:
        raise ValueError("max_bounce must be >= 0")

    dep = depol or DepolConfig(enabled=False)
    wb_mode = str(wave_basis_mode).strip().lower()
    if wb_mode not in {"transport", "global_up"}:
        raise ValueError("wave_basis_mode must be one of: transport, global_up")
    rng = np.random.default_rng(dep.seed)
    drng = np.random.default_rng(int(diffuse_seed))
    rho_fn: Callable[[dict], float] = dep.rho_func or _default_rho

    freq = np.asarray(f_hz, dtype=float)
    n_f = len(freq)
    results: list[PathResult] = []
    seen: set[tuple] = set()
    plate_eps = 1e-6
    # Use all planes for occlusion checks, but only reflective planes for bounce generation.
    reflective_planes = [pl for pl in planes if not _is_occluder_only_plane(pl)]

    for b in range(max_bounce + 1):
        if b == 0 and not los_enabled:
            continue
        if b > 0 and len(reflective_planes) == 0:
            continue
        for idx_seq in _enumerate_sequences(
            len(reflective_planes),
            b,
            max_candidates=max_sequence_candidates_per_bounce,
            forbid_immediate_repeat=forbid_immediate_surface_repeat,
        ):
            seq = [reflective_planes[i] for i in idx_seq]
            points = _construct_reflection_points(tx.position, rx.position, seq)
            if points is None or not _path_valid(points):
                continue
            points = _fit_reflection_points_to_bounds(points, seq, eps=plate_eps, clamp_tol=plate_eps)
            if points is None or not _path_valid(points):
                continue
            if any(not seq[i].contains_point(points[i + 1], eps=plate_eps) for i in range(len(seq))):
                continue
            if _path_occluded(points, planes, seq, eps=plate_eps):
                continue

            dirs = [normalize(points[i + 1] - points[i]) for i in range(len(points) - 1)]
            total_len = path_length(points)
            if total_len <= 0.0:
                continue
            tau = total_len / c0
            key = _path_key(
                points,
                seq,
                tau,
                point_tol_m=float(dedup_point_tol_m),
                tau_tol_s=float(dedup_tau_tol_s),
            )
            if key in seen:
                continue
            seen.add(key)

            if use_fspl:
                # Amplitude-domain Friis factor (lambda/4piR). Power scaling is its squared magnitude.
                # Directional antenna gain is injected below as sqrt(G_tx * G_rx).
                scalar_gain_f = (c0 / freq) / (4.0 * np.pi * total_len)
            else:
                scalar_gain_f = np.full(n_f, 1.0 / total_len, dtype=float)
            # Optional directional pattern gain (power) from antenna boresight model.
            # Defaults are isotropic (G_tx=G_rx=1), preserving existing behavior.
            g_tx_dir = float(max(tx.tx_directional_gain_linear(dirs[0]), 0.0))
            g_rx_dir = float(max(rx.rx_directional_gain_linear(dirs[-1]), 0.0))
            scalar_gain_f = scalar_gain_f * float(np.sqrt(max(g_tx_dir * g_rx_dir, 0.0)))

            wave_basis = tx.wave_basis(dirs[0])
            J = np.repeat(np.eye(2, dtype=np.complex128)[None, :, :], n_f, axis=0)
            seg_basis_uv: list[dict[str, list[float]]] = []
            u0 = np.real(wave_basis[:, 0]).astype(float)
            v0 = np.real(wave_basis[:, 1]).astype(float)
            seg_basis_uv.append({"k": dirs[0].tolist(), "u": u0.tolist(), "v": v0.tolist()})

            incidence_angles: list[float] = []
            bounce_normals: list[NDArray[np.float64]] = []
            interactions: list[str] = []
            surface_ids: list[int] = []

            for i, pl in enumerate(seq):
                k_in = dirs[i]
                k_out = dirs[i + 1]
                s_in, p_in, s_out, p_out, theta_i, n_eff = local_sp_bases(k_in, k_out, pl.unit_normal())
                incidence_angles.append(theta_i)
                bounce_normals.append(np.asarray(n_eff, dtype=float))
                interactions.append("reflection")
                surface_ids.append(pl.id)

                sp_in = np.column_stack([s_in, p_in]).astype(np.complex128)
                sp_out = np.column_stack([s_out, p_out]).astype(np.complex128)

                t_in = basis_change(wave_basis, sp_in)
                if wb_mode == "global_up":
                    next_basis = tx.wave_basis(k_out)
                else:
                    next_basis = _transport_wave_basis(k_out, wave_basis, tx.global_up)
                t_out = basis_change(next_basis, sp_out)
                r_f = jones_reflection(pl.material, theta_i, freq)

                updated = np.zeros_like(J)
                for k in range(n_f):
                    updated[k] = t_out.conj().T @ r_f[k] @ t_in @ J[k]
                J = updated
                wave_basis = next_basis
                seg_basis_uv.append(
                    {
                        "k": k_out.tolist(),
                        "u": np.real(wave_basis[:, 0]).astype(float).tolist(),
                        "v": np.real(wave_basis[:, 1]).astype(float).tolist(),
                    }
                )

                if dep.enabled and dep.apply_mode == "event":
                    rho = float(np.clip(rho_fn({"surface_id": pl.id, "bounce_index": i, "theta_i": theta_i}), 0.0, 1.0))
                    D_in = depol_matrix(rho, rng)
                    D_out = depol_matrix(rho, rng)
                    J = apply_depol(J, D_in=D_in, D_out=D_out, side_mode=dep.side_mode)

            if dep.enabled and dep.apply_mode == "path":
                rho = float(np.clip(rho_fn({"bounce_count": b, "surface_ids": surface_ids}), 0.0, 1.0))
                D_in = depol_matrix(rho, rng)
                D_out = depol_matrix(rho, rng)
                J = apply_depol(J, D_in=D_in, D_out=D_out, side_mode=dep.side_mode)

            g_tx = tx.tx_port_to_wave(dirs[0], freq, wave_basis=tx.wave_basis(dirs[0]))
            g_rx_h = rx.rx_wave_to_port(dirs[-1], freq, wave_basis=wave_basis)
            A = np.einsum("kab,kbc,kcd->kad", g_rx_h, J, g_tx).astype(np.complex128)
            A *= scalar_gain_f[:, None, None]

            # Optional heuristic: odd mirror reflections can swap CP handedness near normal incidence.
            # Keep disabled by default; for oblique incidence Fresnel imbalance can produce elliptical states.
            if force_cp_swap_on_odd_reflection and rx.basis == "circular" and (b % 2 == 1):
                swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
                A = np.einsum("ab,kbc->kac", swap, A)

            results.append(
                PathResult(
                    tau_s=tau,
                    path_length_m=float(total_len),
                    segment_count=int(len(points) - 1),
                    A_f=A,
                    J_f=J.copy(),
                    scalar_gain_f=scalar_gain_f.astype(float),
                    G_tx_f=g_tx.copy(),
                    G_rx_f=g_rx_h.copy(),
                    bounce_count=b,
                    interactions=interactions,
                    surface_ids=surface_ids,
                    incidence_angles=incidence_angles,
                    bounce_normals=bounce_normals,
                    AoD=dirs[0],
                    AoA=dirs[-1],
                    points=[np.asarray(p, dtype=float) for p in points],
                    segment_basis_uv=seg_basis_uv,
                )
            )
            if diffuse_enabled and b > 0 and diffuse_factor > 0.0 and diffuse_rays_per_hit > 0:
                # NOTE:
                # Current diffuse branch is an empirical proxy, not a geometric BRDF scatter solver.
                # It perturbs polarization (SU(2) mixer) and excess delay/power around the parent
                # specular path while preserving parent AoD/AoA/path vertices.
                # Use for stress/sensitivity studies, not absolute scatter-angle validation.
                nrho = float(np.clip(diffuse_factor, 0.0, 1.0))
                if str(diffuse_model).lower() == "directive":
                    jitter_ns = 0.08 / np.sqrt(max(diffuse_lobe_alpha, 1e-6))
                elif str(diffuse_model).lower() == "directive_backscatter":
                    jitter_ns = 0.12 / np.sqrt(max(diffuse_lobe_alpha, 1e-6))
                else:
                    jitter_ns = 0.2
                n_spawn = int(max(1, diffuse_rays_per_hit * b))
                for _ in range(n_spawn):
                    D_in = depol_matrix(nrho, drng)
                    D_out = depol_matrix(nrho, drng)
                    Jd = apply_depol(J, D_in=D_in, D_out=D_out, side_mode="both")
                    # Keep diffuse energy controlled relative to parent path.
                    sg_scale = np.sqrt(max(diffuse_factor, 0.0) / float(n_spawn))
                    scalar_d = scalar_gain_f * float(sg_scale)
                    A_d = np.einsum("kab,kbc,kcd->kad", g_rx_h, Jd, g_tx).astype(np.complex128)
                    A_d *= scalar_d[:, None, None]
                    tau_jitter = abs(float(drng.normal(loc=0.0, scale=jitter_ns))) * 1e-9
                    tau_d = float(tau + tau_jitter)
                    results.append(
                        PathResult(
                            tau_s=tau_d,
                            path_length_m=float(tau_d * c0),
                            segment_count=int(len(points) - 1),
                            A_f=A_d,
                            J_f=Jd.copy(),
                            scalar_gain_f=scalar_d.astype(float),
                            G_tx_f=g_tx.copy(),
                            G_rx_f=g_rx_h.copy(),
                            bounce_count=b,
                            interactions=["diffuse"] * b,
                            surface_ids=surface_ids.copy(),
                            incidence_angles=incidence_angles.copy(),
                            bounce_normals=[np.asarray(nv, dtype=float) for nv in bounce_normals],
                            AoD=dirs[0],
                            AoA=dirs[-1],
                            points=[np.asarray(p, dtype=float) for p in points],
                            segment_basis_uv=seg_basis_uv,
                        )
                    )

    if min_path_power_db is not None:
        pmin = float(min_path_power_db)
        keep: list[PathResult] = []
        for p in results:
            pd = 10.0 * np.log10(float(np.mean(np.abs(p.A_f) ** 2)) + 1e-18)
            if pd >= pmin:
                keep.append(p)
        results = keep

    if max_paths_per_case is not None and int(max_paths_per_case) > 0 and len(results) > int(max_paths_per_case):
        k = int(max_paths_per_case)
        idx = np.argsort(
            -np.asarray([float(np.mean(np.abs(p.A_f) ** 2)) for p in results], dtype=float)
        )[:k]
        results = [results[int(i)] for i in idx]

    results.sort(key=lambda p: p.tau_s)
    return results

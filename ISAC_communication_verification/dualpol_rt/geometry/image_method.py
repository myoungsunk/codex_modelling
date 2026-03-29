from __future__ import annotations

from itertools import product

import numpy as np

from dualpol_rt.channel.path_record import PathRecord
from dualpol_rt.geometry.visibility import line_plane_intersection, normalize, path_length, reflect_point, segment_blocked
from dualpol_rt.scene.surfaces import Scene, Surface


LIGHT_SPEED_M_PER_S = 299792458.0


def _enumerate_sequences(num_surfaces: int, bounce_count: int):
    if bounce_count == 0:
        yield tuple()
        return
    for seq in product(range(num_surfaces), repeat=bounce_count):
        if any(seq[idx] == seq[idx - 1] for idx in range(1, len(seq))):
            continue
        yield seq


def _construct_points_source_images(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    sequence: tuple[Surface, ...],
) -> list[np.ndarray] | None:
    if not sequence:
        return [np.asarray(tx_pos, dtype=float), np.asarray(rx_pos, dtype=float)]
    images = [np.asarray(tx_pos, dtype=float)]
    for surface in sequence:
        images.append(reflect_point(images[-1], surface))
    target = np.asarray(rx_pos, dtype=float)
    reflection_points: list[np.ndarray | None] = [None] * len(sequence)
    for idx in range(len(sequence) - 1, -1, -1):
        hit = line_plane_intersection(images[idx + 1], target, sequence[idx])
        if hit is None:
            return None
        point, _t = hit
        reflection_points[idx] = point
        target = point
    return [np.asarray(tx_pos, dtype=float), *[np.asarray(p, dtype=float) for p in reflection_points if p is not None], np.asarray(rx_pos, dtype=float)]


def _construct_points_receiver_images(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    sequence: tuple[Surface, ...],
) -> list[np.ndarray] | None:
    if not sequence:
        return [np.asarray(tx_pos, dtype=float), np.asarray(rx_pos, dtype=float)]
    rx_images = [np.asarray(rx_pos, dtype=float)]
    for surface in reversed(sequence):
        rx_images.append(reflect_point(rx_images[-1], surface))
    points = [np.asarray(tx_pos, dtype=float)]
    current = points[0]
    for idx, surface in enumerate(sequence):
        target = rx_images[len(sequence) - idx]
        hit = line_plane_intersection(current, target, surface)
        if hit is None:
            return None
        point, _t = hit
        points.append(np.asarray(point, dtype=float))
        current = points[-1]
    points.append(np.asarray(rx_pos, dtype=float))
    return points


def _construct_points(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    sequence: tuple[Surface, ...],
) -> list[np.ndarray] | None:
    points = _construct_points_source_images(tx_pos, rx_pos, sequence)
    if points is not None:
        return points
    return _construct_points_receiver_images(tx_pos, rx_pos, sequence)


def _path_key(points: list[np.ndarray], surface_ids: tuple[int, ...], tol_m: float = 1e-6) -> tuple:
    q_points = tuple(tuple(int(round(float(coord) / tol_m)) for coord in point) for point in points[1:-1])
    return surface_ids, q_points


def _validate_path(points: list[np.ndarray], sequence: tuple[Surface, ...], scene: Scene) -> tuple[bool, bool]:
    if any(np.linalg.norm(points[idx + 1] - points[idx]) <= 1e-9 for idx in range(len(points) - 1)):
        return False, False
    for idx, surface in enumerate(sequence):
        if not surface.contains_point(points[idx + 1], eps=1e-6):
            return False, False
    for idx in range(len(points) - 1):
        if segment_blocked(points[idx], points[idx + 1], scene, eps=1e-7):
            return False, True
    return True, False


def enumerate_paths(
    scene: Scene,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    max_reflections: int,
) -> list[PathRecord]:
    if max_reflections not in {0, 1, 2}:
        raise ValueError("Phase 1 baseline supports max_reflections in {0,1,2}")
    tx = np.asarray(tx_pos, dtype=float)
    rx = np.asarray(rx_pos, dtype=float)
    out: list[PathRecord] = []
    seen: set[tuple] = set()
    for bounce_count in range(max_reflections + 1):
        for seq_idx in _enumerate_sequences(len(scene.surfaces), bounce_count):
            surfaces = tuple(scene.surfaces[idx] for idx in seq_idx)
            points = _construct_points(tx, rx, surfaces)
            if points is None:
                continue
            valid, blocked = _validate_path(points, surfaces, scene)
            if not valid:
                continue
            key = _path_key(points, tuple(surface.surface_id for surface in surfaces))
            if key in seen:
                continue
            seen.add(key)
            total_length = path_length(points)
            launch_dir = normalize(points[1] - points[0])
            arrival_dir = normalize(points[-1] - points[-2])
            incidence_angles = []
            normals = []
            for idx, surface in enumerate(surfaces):
                kin = normalize(points[idx + 1] - points[idx])
                normal = np.asarray(surface.normal, dtype=float)
                if float(np.dot(kin, normal)) > 0.0:
                    normal = -normal
                normals.append(normal.astype(float))
                cos_theta = float(np.clip(-np.dot(kin, normal), 0.0, 1.0))
                incidence_angles.append(float(np.arccos(cos_theta)))
            out.append(
                PathRecord(
                    points=tuple(np.asarray(point, dtype=float) for point in points),
                    surface_ids=tuple(surface.surface_id for surface in surfaces),
                    surface_names=tuple(surface.name for surface in surfaces),
                    materials=tuple(surface.material for surface in surfaces),
                    bounce_count=bounce_count,
                    path_length_m=float(total_length),
                    delay_s=float(total_length / LIGHT_SPEED_M_PER_S),
                    launch_dir=launch_dir,
                    arrival_dir=arrival_dir,
                    incidence_angles_rad=tuple(incidence_angles),
                    normals=tuple(normals),
                    blocked=blocked,
                    valid=True,
                )
            )
    out.sort(key=lambda path: (path.delay_s, path.surface_ids))
    return out

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dualpol_rt.scene.surfaces import Scene, Surface


Vec3 = NDArray[np.float64]
GEOM_EPS = 1e-9


def normalize(vec: Vec3) -> Vec3:
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero-length vector")
    return (v / n).astype(float)


def path_length(points: list[Vec3] | tuple[Vec3, ...]) -> float:
    total = 0.0
    for idx in range(len(points) - 1):
        total += float(np.linalg.norm(np.asarray(points[idx + 1], dtype=float) - np.asarray(points[idx], dtype=float)))
    return total


def line_plane_intersection(a: Vec3, b: Vec3, surface: Surface, eps: float = GEOM_EPS) -> tuple[Vec3, float] | None:
    p0 = np.asarray(a, dtype=float)
    p1 = np.asarray(b, dtype=float)
    d = p1 - p0
    denom = float(np.dot(surface.normal, d))
    if abs(denom) <= eps:
        return None
    t = float(np.dot(surface.normal, surface.point - p0) / denom)
    point = p0 + t * d
    return np.asarray(point, dtype=float), t


def segment_plane_intersection(a: Vec3, b: Vec3, surface: Surface, eps: float = GEOM_EPS) -> tuple[Vec3, float] | None:
    hit = line_plane_intersection(a, b, surface, eps=eps)
    if hit is None:
        return None
    point, t = hit
    if t < -eps or t > 1.0 + eps:
        return None
    if not surface.contains_point(point, eps=max(eps, 1e-7)):
        return None
    return point, t


def reflect_point(point: Vec3, surface: Surface) -> Vec3:
    p = np.asarray(point, dtype=float)
    delta = p - surface.point
    return np.asarray(p - 2.0 * np.dot(delta, surface.normal) * surface.normal, dtype=float)


def segment_blocked(
    a: Vec3,
    b: Vec3,
    scene: Scene,
    ignore_surface_ids: set[int] | None = None,
    eps: float = 1e-7,
) -> bool:
    ignore = set(ignore_surface_ids or set())
    for surface in scene.surfaces:
        if surface.surface_id in ignore:
            continue
        hit = segment_plane_intersection(a, b, surface, eps=eps)
        if hit is None:
            continue
        _point, t = hit
        if eps < t < 1.0 - eps:
            return True
    return False

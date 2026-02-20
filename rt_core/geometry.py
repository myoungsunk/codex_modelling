"""Geometry primitives for polarimetric UWB ray tracing.

Example:
    >>> import numpy as np
    >>> from rt_core.geometry import Plane, Material, ray_plane_intersection
    >>> pl = Plane(id=1, p0=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 1.0]), material=Material.pec())
    >>> hit = ray_plane_intersection(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]), pl)
    >>> np.allclose(hit.point, np.array([0.0, 0.0, 0.0]))
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from numpy.typing import NDArray

Vec3 = NDArray[np.float64]


@dataclass(frozen=True)
class Material:
    """Surface material model for reflection coefficients."""

    kind: str
    eps_r: float = 1.0
    tan_delta: float = 0.0
    complex_eps_r: complex | None = None

    @staticmethod
    def pec() -> "Material":
        return Material(kind="PEC")

    @staticmethod
    def dielectric(eps_r: float, tan_delta: float = 0.0, complex_eps_r: complex | None = None) -> "Material":
        return Material(kind="dielectric", eps_r=eps_r, tan_delta=tan_delta, complex_eps_r=complex_eps_r)


@dataclass(frozen=True)
class Plane:
    """Infinite plane used by the deterministic specular tracer."""

    id: int
    p0: Vec3
    normal: Vec3
    material: Material
    u_axis: Vec3 | None = None
    v_axis: Vec3 | None = None
    half_extent_u: float | None = None
    half_extent_v: float | None = None

    def unit_normal(self) -> Vec3:
        n = np.asarray(self.normal, dtype=float)
        nn = np.linalg.norm(n)
        if nn == 0:
            raise ValueError("Plane normal cannot be zero")
        return n / nn

    def local_axes(self) -> tuple[Vec3, Vec3]:
        n = self.unit_normal()
        if self.u_axis is not None:
            u = np.asarray(self.u_axis, dtype=float)
            u = u - float(np.dot(u, n)) * n
            u = normalize(u)
        else:
            hint = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = normalize(np.cross(hint, n))
        if self.v_axis is not None:
            v = np.asarray(self.v_axis, dtype=float)
            v = v - float(np.dot(v, n)) * n - float(np.dot(v, u)) * u
            v = normalize(v)
        else:
            v = normalize(np.cross(n, u))
        return u, v

    def contains_point(self, point: Vec3, eps: float = 1e-7) -> bool:
        """Return True for infinite plane or if point is inside finite plate bounds."""

        if self.half_extent_u is None or self.half_extent_v is None:
            return True
        u, v = self.local_axes()
        d = np.asarray(point, dtype=float) - np.asarray(self.p0, dtype=float)
        uu = float(np.dot(d, u))
        vv = float(np.dot(d, v))
        return abs(uu) <= float(self.half_extent_u) + eps and abs(vv) <= float(self.half_extent_v) + eps


@dataclass(frozen=True)
class Hit:
    t: float
    point: Vec3


def normalize(v: Vec3, eps: float = 1e-12) -> Vec3:
    vv = np.asarray(v, dtype=float)
    n = np.linalg.norm(vv)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return vv / n


def ray_plane_intersection(origin: Vec3, direction: Vec3, plane: Plane, eps: float = 1e-9) -> Optional[Hit]:
    """Compute ray-plane intersection for t >= 0."""

    o = np.asarray(origin, dtype=float)
    d = normalize(np.asarray(direction, dtype=float))
    n = plane.unit_normal()
    denom = float(np.dot(d, n))
    if abs(denom) < eps:
        return None
    t = float(np.dot(plane.p0 - o, n) / denom)
    if t < eps:
        return None
    p = o + t * d
    return Hit(t=t, point=p)


def line_plane_intersection(a: Vec3, b: Vec3, plane: Plane, eps: float = 1e-9) -> Optional[Vec3]:
    """Intersection between infinite line through points a,b and a plane."""

    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if np.linalg.norm(d) < eps:
        return None
    hit = ray_plane_intersection(np.asarray(a, dtype=float), d, plane, eps=eps)
    if hit is None:
        return None
    return hit.point


def reflect_point(point: Vec3, plane: Plane) -> Vec3:
    """Mirror a point across the plane."""

    p = np.asarray(point, dtype=float)
    n = plane.unit_normal()
    signed_dist = float(np.dot(p - plane.p0, n))
    return p - 2.0 * signed_dist * n


def reflect_direction(direction: Vec3, normal: Vec3) -> Vec3:
    """Specularly reflect a direction around the given unit normal."""

    d = normalize(np.asarray(direction, dtype=float))
    n = normalize(np.asarray(normal, dtype=float))
    return normalize(d - 2.0 * float(np.dot(d, n)) * n)


def path_length(points: Iterable[Vec3]) -> float:
    pts = [np.asarray(p, dtype=float) for p in points]
    if len(pts) < 2:
        return 0.0
    return float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dualpol_rt.em.materials import Material


Vec3 = NDArray[np.float64]


def normalize(vec: Vec3) -> Vec3:
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero-length vector")
    return (v / n).astype(float)


@dataclass(frozen=True)
class Surface:
    surface_id: int
    name: str
    point: Vec3
    normal: Vec3
    u_axis: Vec3
    v_axis: Vec3
    half_u: float
    half_v: float
    material: Material

    def __post_init__(self) -> None:
        n = normalize(self.normal)
        u = np.asarray(self.u_axis, dtype=float)
        u = u - float(np.dot(u, n)) * n
        u = normalize(u)
        v = np.asarray(self.v_axis, dtype=float)
        v = v - float(np.dot(v, n)) * n - float(np.dot(v, u)) * u
        v = normalize(v)
        p = np.asarray(self.point, dtype=float)
        for arr in (p, n, u, v):
            arr.setflags(write=False)
        object.__setattr__(self, "point", p)
        object.__setattr__(self, "normal", n)
        object.__setattr__(self, "u_axis", u)
        object.__setattr__(self, "v_axis", v)
        object.__setattr__(self, "half_u", float(self.half_u))
        object.__setattr__(self, "half_v", float(self.half_v))

    def local_coordinates(self, point: Vec3) -> tuple[float, float]:
        d = np.asarray(point, dtype=float) - self.point
        return float(np.dot(d, self.u_axis)), float(np.dot(d, self.v_axis))

    def contains_point(self, point: Vec3, eps: float = 1e-7) -> bool:
        u, v = self.local_coordinates(point)
        return abs(u) <= self.half_u + eps and abs(v) <= self.half_v + eps


@dataclass(frozen=True)
class Scene:
    surfaces: tuple[Surface, ...]

    def __iter__(self):
        return iter(self.surfaces)

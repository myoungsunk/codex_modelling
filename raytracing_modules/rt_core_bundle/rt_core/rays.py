"""Ray/path data structures and helpers.

Example:
    >>> import numpy as np
    >>> from rt_core.rays import Ray
    >>> r = Ray(origin=np.array([0.0, 0.0, 0.0]), direction=np.array([1.0, 0.0, 0.0]))
    >>> np.allclose(r.direction, np.array([1.0, 0.0, 0.0]))
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from numpy.typing import NDArray

from rt_core.geometry import normalize, reflect_direction

Vec3 = NDArray[np.float64]


@dataclass(frozen=True)
class Ray:
    origin: Vec3
    direction: Vec3

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=float))
        object.__setattr__(self, "direction", normalize(np.asarray(self.direction, dtype=float)))

    def reflect(self, normal: Vec3) -> "Ray":
        return Ray(origin=self.origin, direction=reflect_direction(self.direction, normal))


@dataclass
class PathAccumulator:
    """Accumulates geometric metadata as a path is built."""

    points: List[Vec3] = field(default_factory=list)
    interactions: List[str] = field(default_factory=list)
    surface_ids: List[int] = field(default_factory=list)
    incidence_angles: List[float] = field(default_factory=list)

    def add_point(self, p: Vec3) -> None:
        self.points.append(np.asarray(p, dtype=float))

    def add_reflection(self, surface_id: int, incidence_angle_rad: float) -> None:
        self.interactions.append("reflection")
        self.surface_ids.append(int(surface_id))
        self.incidence_angles.append(float(incidence_angle_rad))

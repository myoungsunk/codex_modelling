from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PathRecord:
    points: tuple[NDArray[np.float64], ...]
    surface_ids: tuple[int, ...]
    surface_names: tuple[str, ...]
    materials: tuple[object, ...]
    bounce_count: int
    path_length_m: float
    delay_s: float
    launch_dir: NDArray[np.float64]
    arrival_dir: NDArray[np.float64]
    incidence_angles_rad: tuple[float, ...]
    normals: tuple[NDArray[np.float64], ...]
    blocked: bool
    valid: bool
    jones_f: NDArray[np.complex128] | None = None
    scalar_factor_f: NDArray[np.complex128] | None = None

    @property
    def bounce_points(self) -> tuple[NDArray[np.float64], ...]:
        return self.points[1:-1]

    @property
    def aod_dir(self) -> NDArray[np.float64]:
        return self.launch_dir

    @property
    def aoa_dir(self) -> NDArray[np.float64]:
        return self.arrival_dir

    def with_em_response(
        self,
        jones_f: NDArray[np.complex128],
        scalar_factor_f: NDArray[np.complex128],
    ) -> "PathRecord":
        return replace(self, jones_f=jones_f, scalar_factor_f=scalar_factor_f)


@dataclass(frozen=True)
class GridResult:
    rx_x: NDArray[np.float64]
    rx_y: NDArray[np.float64]
    linear_rate_maps: dict[float, NDArray[np.float64]]
    circular_rate_maps: dict[float, NDArray[np.float64]]
    linear_capacity_maps: dict[float, NDArray[np.float64]]
    circular_capacity_maps: dict[float, NDArray[np.float64]]
    delta_rate_maps: dict[float, NDArray[np.float64]]
    xpr_db_map: NDArray[np.float64]
    condition_number_map: NDArray[np.float64]

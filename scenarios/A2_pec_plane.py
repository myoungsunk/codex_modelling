"""A2 single PEC plane one-bounce with LOS blocked."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_core.geometry import Material, Plane
from rt_core.tracer import trace_paths
from scenarios.common import default_antennas


def build_scene(y_plane: float = 2.0) -> list[Plane]:
    return [Plane(id=1, p0=np.array([0.0, y_plane, 0.0]), normal=np.array([0.0, -1.0, 0.0]), material=Material.pec())]


def build_sweep_params() -> list[dict[str, Any]]:
    params = []
    for y in [1.5, 2.0, 2.5]:
        for x in [4.0, 6.0, 8.0]:
            params.append({"y_plane": y, "distance_m": x})
    return params


def run_case(params: dict[str, Any], f_hz, basis: str = "linear"):
    tx, rx = default_antennas(basis=basis)
    rx.position[:] = [params["distance_m"], 0.0, 1.5]
    scene = build_scene(y_plane=params["y_plane"])
    return trace_paths(scene, tx, rx, f_hz, max_bounce=1, los_enabled=False)

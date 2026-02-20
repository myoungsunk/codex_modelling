"""A4 dielectric material sweep with one-bounce reflections."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_core.geometry import Material, Plane
from rt_core.tracer import trace_paths
from scenarios.common import default_antennas


MATERIALS = {
    "glass": Material.dielectric(eps_r=6.5, tan_delta=0.005),
    "wood": Material.dielectric(eps_r=2.2, tan_delta=0.03),
    "gypsum": Material.dielectric(eps_r=2.8, tan_delta=0.02),
}


def build_scene(material_name: str, y_plane: float = 2.0) -> list[Plane]:
    return [
        Plane(
            id=1,
            p0=np.array([0.0, y_plane, 0.0]),
            normal=np.array([0.0, -1.0, 0.0]),
            material=MATERIALS[material_name],
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=20.0,
            half_extent_v=20.0,
        )
    ]


def build_sweep_params() -> list[dict[str, Any]]:
    params = []
    for mat in MATERIALS:
        for y in [1.5, 2.0, 2.5]:
            params.append({"material": mat, "y_plane": y, "distance_m": 6.0})
    return params


def run_case(params: dict[str, Any], f_hz, basis: str = "linear"):
    tx, rx = default_antennas(basis=basis)
    rx.position[:] = [params["distance_m"], 0.0, 1.5]
    scene = build_scene(params["material"], y_plane=params["y_plane"])
    return trace_paths(scene, tx, rx, f_hz, max_bounce=1, los_enabled=False)

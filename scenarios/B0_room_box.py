"""B0 indoor room-box multipath scenario for richer path statistics.

This scenario builds a 6-plane room (4 walls + floor + ceiling), keeps LOS,
and traces up to 2-bounce specular paths over a 5x5 Rx grid.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_core.geometry import Material, Plane
from rt_core.tracer import trace_paths
from scenarios.common import default_antennas


def build_scene(
    length_x: float = 10.0,
    width_y: float = 8.0,
    height_z: float = 3.0,
    material: Material | None = None,
) -> list[Plane]:
    """Create 6 finite planes that enclose a rectangular room."""

    mat = material or Material.pec()
    hx = 0.5 * length_x
    hy = 0.5 * width_y
    hz = 0.5 * height_z
    cx = 0.5 * length_x

    # Room coordinates:
    # x in [0, length_x], y in [-hy, +hy], z in [0, height_z]
    return [
        # Left wall x=0, normal +x.
        Plane(
            id=101,
            p0=np.array([0.0, 0.0, hz]),
            normal=np.array([1.0, 0.0, 0.0]),
            material=mat,
            u_axis=np.array([0.0, 1.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=hy,
            half_extent_v=hz,
        ),
        # Right wall x=length_x, normal -x.
        Plane(
            id=102,
            p0=np.array([length_x, 0.0, hz]),
            normal=np.array([-1.0, 0.0, 0.0]),
            material=mat,
            u_axis=np.array([0.0, 1.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=hy,
            half_extent_v=hz,
        ),
        # Back wall y=-hy, normal +y.
        Plane(
            id=103,
            p0=np.array([cx, -hy, hz]),
            normal=np.array([0.0, 1.0, 0.0]),
            material=mat,
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=hx,
            half_extent_v=hz,
        ),
        # Front wall y=+hy, normal -y.
        Plane(
            id=104,
            p0=np.array([cx, hy, hz]),
            normal=np.array([0.0, -1.0, 0.0]),
            material=mat,
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=hx,
            half_extent_v=hz,
        ),
        # Floor z=0, normal +z.
        Plane(
            id=105,
            p0=np.array([cx, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            material=mat,
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 1.0, 0.0]),
            half_extent_u=hx,
            half_extent_v=hy,
        ),
        # Ceiling z=height_z, normal -z.
        Plane(
            id=106,
            p0=np.array([cx, 0.0, height_z]),
            normal=np.array([0.0, 0.0, -1.0]),
            material=mat,
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 1.0, 0.0]),
            half_extent_u=hx,
            half_extent_v=hy,
        ),
    ]


def build_sweep_params() -> list[dict[str, Any]]:
    """5x5 Rx grid inside room for richer per-condition statistics."""

    params: list[dict[str, Any]] = []
    x_vals = np.linspace(3.0, 8.0, 5)
    y_vals = np.linspace(-3.0, 3.0, 5)
    for ix, rx_x in enumerate(x_vals):
        for iy, rx_y in enumerate(y_vals):
            params.append(
                {
                    "rx_x": float(rx_x),
                    "rx_y": float(rx_y),
                    "rx_z": 1.5,
                    "grid_i": ix,
                    "grid_j": iy,
                    "max_bounce": 2,
                }
            )
    return params


def run_case(
    params: dict[str, Any],
    f_hz,
    basis: str = "linear",
    antenna_config: dict[str, Any] | None = None,
    force_cp_swap_on_odd_reflection: bool = False,
):
    tx, rx = default_antennas(basis=basis, **(antenna_config or {}))
    tx.position[:] = [2.0, 0.0, 1.5]
    rx.position[:] = [params["rx_x"], params["rx_y"], params.get("rx_z", 1.5)]
    scene = build_scene()
    return trace_paths(
        scene,
        tx,
        rx,
        f_hz,
        max_bounce=int(params.get("max_bounce", 2)),
        los_enabled=True,
        force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
    )


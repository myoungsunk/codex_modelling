"""A3 two-plate corner two-bounce even-parity scenario."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_core.geometry import Material, Plane
from rt_core.tracer import trace_paths
from scenarios.common import default_antennas


def build_scene(offset: float = 3.0) -> list[Plane]:
    return [
        Plane(
            id=1,
            p0=np.array([0.0, offset, 0.0]),
            normal=np.array([0.0, -1.0, 0.0]),
            material=Material.pec(),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=20.0,
            half_extent_v=20.0,
        ),
        Plane(
            id=2,
            p0=np.array([offset, 0.0, 0.0]),
            normal=np.array([-1.0, 0.0, 0.0]),
            material=Material.pec(),
            u_axis=np.array([0.0, 1.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=20.0,
            half_extent_v=20.0,
        ),
    ]


def build_sweep_params() -> list[dict[str, Any]]:
    return [
        {"offset": 3.0, "rx_x": 3.0, "rx_y": 4.0},
        {"offset": 3.0, "rx_x": 4.0, "rx_y": 3.0},
        {"offset": 3.5, "rx_x": 3.5, "rx_y": 4.5},
        {"offset": 4.0, "rx_x": 4.0, "rx_y": 5.0},
    ]


def run_case(
    params: dict[str, Any],
    f_hz,
    basis: str = "linear",
    antenna_config: dict[str, Any] | None = None,
    force_cp_swap_on_odd_reflection: bool = False,
):
    tx, rx = default_antennas(basis=basis, **(antenna_config or {}))
    tx.position[:] = [0.0, 0.0, 1.5]
    rx.position[:] = [params["rx_x"], params["rx_y"], 1.5]
    scene = build_scene(offset=params["offset"])
    return trace_paths(
        scene,
        tx,
        rx,
        f_hz,
        max_bounce=2,
        los_enabled=False,
        force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
    )

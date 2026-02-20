"""A5 depolarization stress with optional parity collapse."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_core.geometry import Material, Plane
from rt_core.polarization import DepolConfig
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
    return [{"offset": 3.5, "rx_x": 3.5, "rx_y": 4.5, "rho": rho} for rho in [0.05, 0.15, 0.25, 0.35, 0.45]]


def run_case(params: dict[str, Any], f_hz, basis: str = "linear"):
    tx, rx = default_antennas(basis=basis)
    rx.position[:] = [params["rx_x"], params["rx_y"], 1.5]
    scene = build_scene(offset=params["offset"])

    def rho_hook(ctx: dict) -> float:
        _ = ctx
        return float(params["rho"])

    dep = DepolConfig(enabled=True, apply_mode="event", side_mode="both", rho_func=rho_hook, seed=1234)
    return trace_paths(scene, tx, rx, f_hz, max_bounce=2, los_enabled=False, depol=dep)

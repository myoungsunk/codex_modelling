"""A2 single PEC plane one-bounce with LOS blocked."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_core.geometry import Material, Plane
from rt_core.tracer import trace_paths
from scenarios.common import default_antennas, make_los_blocker_plane


def build_scene(y_plane: float = 2.0) -> list[Plane]:
    return [
        Plane(
            id=1,
            p0=np.array([0.0, y_plane, 0.0]),
            normal=np.array([0.0, -1.0, 0.0]),
            material=Material.pec(),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=20.0,
            half_extent_v=20.0,
        )
    ]


def build_sweep_params() -> list[dict[str, Any]]:
    params = []
    for y in [1.5, 2.0, 2.5]:
        for x in [4.0, 6.0, 8.0]:
            params.append({"y_plane": y, "distance_m": x})
    return params


def run_case(
    params: dict[str, Any],
    f_hz,
    basis: str = "linear",
    antenna_config: dict[str, Any] | None = None,
    force_cp_swap_on_odd_reflection: bool = False,
    max_bounce_override: int | None = None,
    diffuse_config: dict[str, Any] | None = None,
    los_blocker: bool = False,
    los_enabled_override: bool | None = None,
):
    tx, rx = default_antennas(basis=basis, **(antenna_config or {}))
    rx.position[:] = [params["distance_m"], 0.0, 1.5]
    scene = build_scene(y_plane=params["y_plane"])
    if bool(los_blocker):
        scene.append(make_los_blocker_plane(tx.position, rx.position, plane_id=9101))
    if los_enabled_override is None:
        los_enabled = True if bool(los_blocker) else False
    else:
        los_enabled = bool(los_enabled_override)
    trace_kwargs = dict(diffuse_config or {})
    return trace_paths(
        scene,
        tx,
        rx,
        f_hz,
        max_bounce=int(max_bounce_override) if max_bounce_override is not None else 1,
        los_enabled=bool(los_enabled),
        force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
        **trace_kwargs,
    )

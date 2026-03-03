"""A3 two-plate corner two-bounce even-parity scenario."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_core.geometry import Material, Plane
from rt_core.tracer import trace_paths
from scenarios.common import default_antennas, make_los_blocker_plane


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
    out: list[dict[str, Any]] = []
    # Keep Rx on the same side of both corner planes (x<offset, y<offset) to avoid
    # physically ambiguous "through-wall" interpretation while preserving strong 2-bounce paths.
    for off in [3.0, 3.5, 4.0]:
        for rx_x, rx_y in [(0.5, 1.5), (1.5, 0.5), (0.5, 2.5), (2.5, 0.5)]:
            out.append({"offset": float(off), "rx_x": float(rx_x), "rx_y": float(rx_y)})
    return out


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
    off = float(params["offset"])
    rx_x = float(params["rx_x"])
    rx_y = float(params["rx_y"])
    if not (rx_x < off and rx_y < off):
        raise ValueError(
            "A3 invalid geometry: Rx must satisfy rx_x<offset and rx_y<offset "
            f"(rx_x={rx_x}, rx_y={rx_y}, offset={off})"
        )
    tx = tx.with_position([0.0, 0.0, 1.5])
    rx = rx.with_position([rx_x, rx_y, 1.5])
    scene = build_scene(offset=off)
    if bool(los_blocker):
        scene.append(make_los_blocker_plane(tx.position, rx.position, plane_id=9201))
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
        max_bounce=int(max_bounce_override) if max_bounce_override is not None else 2,
        los_enabled=bool(los_enabled),
        force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
        **trace_kwargs,
    )

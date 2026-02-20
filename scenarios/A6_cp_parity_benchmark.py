"""A6 controlled CP parity benchmark under near-normal PEC reflections.

This benchmark is designed for near-normal incidence where handedness trends
are expected to be observable. It does not claim universal odd/even behavior
for arbitrary oblique-incidence geometries.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_core.geometry import Material, Plane
from rt_core.tracer import trace_paths
from scenarios.common import default_antennas


def _odd_scene() -> list[Plane]:
    # Single PEC plane (one-bounce benchmark).
    return [
        Plane(
            id=301,
            p0=np.array([0.0, 0.0, 1.5]),
            normal=np.array([0.0, -1.0, 0.0]),
            material=Material.pec(),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=20.0,
            half_extent_v=20.0,
        )
    ]


def _even_scene() -> list[Plane]:
    # Two parallel PEC planes to form a controlled two-bounce channel.
    return [
        Plane(
            id=302,
            p0=np.array([0.0, 0.0, 1.5]),
            normal=np.array([0.0, -1.0, 0.0]),
            material=Material.pec(),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=20.0,
            half_extent_v=20.0,
        ),
        Plane(
            id=303,
            p0=np.array([0.0, 4.0, 1.5]),
            normal=np.array([0.0, 1.0, 0.0]),
            material=Material.pec(),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=20.0,
            half_extent_v=20.0,
        ),
    ]


def build_scene(mode: str) -> list[Plane]:
    if mode == "odd":
        return _odd_scene()
    if mode == "even":
        return _even_scene()
    raise ValueError(f"unknown mode: {mode}")


def build_sweep_params() -> list[dict[str, Any]]:
    """Generate odd/even near-normal cases with small perturbations."""

    params: list[dict[str, Any]] = []
    dx_vals = [-0.2, 0.0, 0.2]
    dz_vals = [-0.1, 0.0, 0.1]

    for dx in dx_vals:
        for dz in dz_vals:
            params.append(
                {
                    "mode": "odd",
                    "target_bounce": 1,
                    "rx_x": 2.4 + dx,
                    "rx_y": -2.0,
                    "rx_z": 1.5 + dz,
                    "incidence_max_deg": 15.0,
                }
            )

    for dx in dx_vals:
        for dz in dz_vals:
            params.append(
                {
                    "mode": "even",
                    "target_bounce": 2,
                    "rx_x": 2.4 + dx,
                    "rx_y": 1.0,
                    "rx_z": 1.5 + dz,
                    "incidence_max_deg": 15.0,
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
    mode = str(params["mode"])
    scene = build_scene(mode)
    target_bounce = int(params["target_bounce"])
    max_bounce = target_bounce

    if mode == "odd":
        tx.position[:] = [2.0, -2.0, 1.5]
    else:
        tx.position[:] = [2.0, 1.0, 1.5]

    rx.position[:] = [params["rx_x"], params["rx_y"], params["rx_z"]]

    paths = trace_paths(
        scene,
        tx,
        rx,
        f_hz,
        max_bounce=max_bounce,
        los_enabled=False,
        force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
    )

    # Keep only target bounce and near-normal incidence paths.
    max_theta = float(np.deg2rad(params.get("incidence_max_deg", 15.0)))
    out = []
    for p in paths:
        if int(p.bounce_count) != target_bounce:
            continue
        if p.incidence_angles and max(float(a) for a in p.incidence_angles) > max_theta:
            continue
        out.append(p)
    return out


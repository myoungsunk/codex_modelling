"""A5 depolarization stress with optional parity collapse."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_core.geometry import Material, Plane
from rt_core.polarization import DepolConfig
from rt_core.tracer import trace_paths
from scenarios.common import default_antennas

RHO_VALUES = [0.05, 0.15, 0.25, 0.35, 0.45]
N_REPS_PER_RHO = 6
BASE_SEED = 1234


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
    params: list[dict[str, Any]] = []
    for rho_idx, rho in enumerate(RHO_VALUES):
        for rep_id in range(N_REPS_PER_RHO):
            params.append(
                {
                    "offset": 3.5,
                    "rx_x": 3.5,
                    "rx_y": 4.5,
                    "rho": rho,
                    "rep_id": rep_id,
                    "seed": BASE_SEED + rho_idx * 100 + rep_id,
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
    rx.position[:] = [params["rx_x"], params["rx_y"], 1.5]
    scene = build_scene(offset=params["offset"])

    def rho_hook(ctx: dict) -> float:
        _ = ctx
        return float(params["rho"])

    dep = DepolConfig(
        enabled=True,
        apply_mode="event",
        side_mode="both",
        rho_func=rho_hook,
        seed=int(params.get("seed", BASE_SEED)),
    )
    return trace_paths(
        scene,
        tx,
        rx,
        f_hz,
        max_bounce=2,
        los_enabled=False,
        depol=dep,
        force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
    )

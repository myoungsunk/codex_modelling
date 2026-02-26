"""A5 depolarization stress with optional parity collapse."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_core.geometry import Material, Plane
from rt_core.polarization import DepolConfig
from rt_core.tracer import trace_paths
from scenarios.common import default_antennas, make_los_blocker_plane

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


def _append_stress_scatterers(scene: list[Plane], seed: int, count: int) -> None:
    if int(count) <= 0:
        return
    rng = np.random.default_rng(int(seed))
    for i in range(int(count)):
        px = 2.8 + 1.8 * float(rng.random())
        py = 1.2 + 2.8 * float(rng.random())
        pz = 0.8 + 1.4 * float(rng.random())
        yaw = (float(rng.random()) - 0.5) * np.pi
        n = np.array([np.cos(yaw), np.sin(yaw), 0.2 * (float(rng.random()) - 0.5)], dtype=float)
        n = n / float(np.linalg.norm(n))
        scene.append(
            Plane(
                id=9500 + int(i),
                p0=np.array([px, py, pz], dtype=float),
                normal=n,
                material=Material.pec(),
                u_axis=np.array([0.0, 0.0, 1.0], dtype=float),
                v_axis=np.array([-n[1], n[0], 0.0], dtype=float),
                half_extent_u=0.2,
                half_extent_v=0.25,
            )
        )


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
    max_bounce_override: int | None = None,
    diffuse_config: dict[str, Any] | None = None,
    stress_mode: str = "synthetic",
    scatterer_count: int = 0,
    los_blocker: bool = False,
    los_enabled_override: bool | None = None,
):
    tx, rx = default_antennas(basis=basis, **(antenna_config or {}))
    rx.position[:] = [params["rx_x"], params["rx_y"], 1.5]
    scene = build_scene(offset=params["offset"])
    mode = str(stress_mode).lower().strip()
    if mode in {"geometry", "hybrid"} and int(scatterer_count) > 0:
        _append_stress_scatterers(scene, seed=int(params.get("seed", BASE_SEED)), count=int(scatterer_count))
    if bool(los_blocker):
        scene.append(make_los_blocker_plane(tx.position, rx.position, plane_id=9401))

    dep = None
    if mode in {"synthetic", "hybrid"}:
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
        depol=dep,
        force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
        **trace_kwargs,
    )

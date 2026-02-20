"""A3_rotated_dihedral: rotated corner with optional floor/ceiling multipath overlap."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_core.geometry import Material, Plane, normalize
from rt_core.tracer import trace_paths
from scenarios.common import default_antennas


def _rot_z(v: np.ndarray, deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return (R @ v).astype(float)


def build_scene(offset: float = 3.0, yaw_deg: float = 20.0, with_floor_ceiling: bool = True) -> list[Plane]:
    n1 = normalize(_rot_z(np.array([0.0, -1.0, 0.0], dtype=float), yaw_deg))
    n2 = normalize(_rot_z(np.array([-1.0, 0.0, 0.0], dtype=float), yaw_deg))
    planes = [
        Plane(
            id=201,
            p0=np.array([0.0, offset, 0.0]),
            normal=n1,
            material=Material.pec(),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=25.0,
            half_extent_v=25.0,
        ),
        Plane(
            id=202,
            p0=np.array([offset, 0.0, 0.0]),
            normal=n2,
            material=Material.pec(),
            u_axis=np.array([0.0, 1.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=25.0,
            half_extent_v=25.0,
        ),
    ]
    if with_floor_ceiling:
        planes.extend(
            [
                Plane(
                    id=203,
                    p0=np.array([0.0, 0.0, 0.0]),
                    normal=np.array([0.0, 0.0, 1.0]),
                    material=Material.pec(),
                    u_axis=np.array([1.0, 0.0, 0.0]),
                    v_axis=np.array([0.0, 1.0, 0.0]),
                    half_extent_u=25.0,
                    half_extent_v=25.0,
                ),
                Plane(
                    id=204,
                    p0=np.array([0.0, 0.0, 3.0]),
                    normal=np.array([0.0, 0.0, -1.0]),
                    material=Material.pec(),
                    u_axis=np.array([1.0, 0.0, 0.0]),
                    v_axis=np.array([0.0, 1.0, 0.0]),
                    half_extent_u=25.0,
                    half_extent_v=25.0,
                ),
            ]
        )
    return planes


def build_sweep_params() -> list[dict[str, Any]]:
    return [
        {"offset": 3.0, "yaw_deg": 20.0, "rx_x": 3.0, "rx_y": 4.0, "with_floor_ceiling": True},
        {"offset": 3.0, "yaw_deg": 25.0, "rx_x": 3.0, "rx_y": 4.0, "with_floor_ceiling": True},
        {"offset": 3.0, "yaw_deg": 30.0, "rx_x": 3.0, "rx_y": 4.0, "with_floor_ceiling": True},
        {"offset": 3.0, "yaw_deg": 25.0, "rx_x": 3.5, "rx_y": 4.2, "with_floor_ceiling": True},
    ]


def run_case(
    params: dict[str, Any],
    f_hz,
    basis: str = "linear",
    antenna_config: dict[str, Any] | None = None,
    force_cp_swap_on_odd_reflection: bool = False,
):
    tx, rx = default_antennas(basis=basis, **(antenna_config or {}))
    tx.position[:] = [0.0, -0.8, 1.4]
    rx.position[:] = [params["rx_x"], params["rx_y"], 1.6]
    scene = build_scene(
        offset=float(params["offset"]),
        yaw_deg=float(params["yaw_deg"]),
        with_floor_ceiling=bool(params.get("with_floor_ceiling", True)),
    )
    return trace_paths(
        scene,
        tx,
        rx,
        f_hz,
        max_bounce=2,
        los_enabled=False,
        force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
    )

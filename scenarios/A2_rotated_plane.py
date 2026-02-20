"""A2_rotated_plane: one-bounce PEC with rotated plane normal and wide incidence sweep."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_core.geometry import Material, Plane, normalize
from rt_core.tracer import trace_paths
from scenarios.common import default_antennas


def _rot_x(v: np.ndarray, deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)
    return (R @ v).astype(float)


def _rot_z(v: np.ndarray, deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return (R @ v).astype(float)


def build_scene(y_plane: float = 2.0, tilt_x_deg: float = 20.0, yaw_z_deg: float = 15.0) -> list[Plane]:
    n0 = np.array([0.0, -1.0, 0.0], dtype=float)
    n = normalize(_rot_z(_rot_x(n0, tilt_x_deg), yaw_z_deg))
    return [
        Plane(
            id=101,
            p0=np.array([0.0, y_plane, 0.0]),
            normal=n,
            material=Material.pec(),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=25.0,
            half_extent_v=25.0,
        )
    ]


def build_sweep_params() -> list[dict[str, Any]]:
    out = []
    for tilt in [10.0, 20.0, 30.0, 40.0]:
        for yaw in [-25.0, -10.0, 10.0, 25.0]:
            out.append({"y_plane": 2.0, "distance_m": 6.0, "tilt_x_deg": tilt, "yaw_z_deg": yaw})
    return out


def run_case(
    params: dict[str, Any],
    f_hz,
    basis: str = "linear",
    antenna_config: dict[str, Any] | None = None,
    force_cp_swap_on_odd_reflection: bool = False,
):
    tx, rx = default_antennas(basis=basis, **(antenna_config or {}))
    tx.position[:] = [0.0, -1.0, 1.5]
    rx.position[:] = [params["distance_m"], 1.0, 1.7]
    scene = build_scene(
        y_plane=float(params["y_plane"]),
        tilt_x_deg=float(params["tilt_x_deg"]),
        yaw_z_deg=float(params["yaw_z_deg"]),
    )
    return trace_paths(
        scene,
        tx,
        rx,
        f_hz,
        max_bounce=1,
        los_enabled=False,
        force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
    )

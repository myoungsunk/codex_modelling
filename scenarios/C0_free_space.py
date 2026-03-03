"""C0 free-space LOS sanity scenario."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_core.tracer import trace_paths
from scenarios.common import default_antennas


def build_scene() -> list:
    return []


def build_sweep_params() -> list[dict[str, Any]]:
    return [{"distance_m": d} for d in [1.0, 2.0, 3.0, 4.0, 5.0]]


def run_case(
    params: dict[str, Any],
    f_hz,
    basis: str = "linear",
    antenna_config: dict[str, Any] | None = None,
    force_cp_swap_on_odd_reflection: bool = False,
    max_bounce_override: int | None = None,
    diffuse_config: dict[str, Any] | None = None,
):
    tx, rx = default_antennas(basis=basis, **(antenna_config or {}))
    yaw_deg = float(params.get("yaw_deg", 0.0))
    pitch_deg = float(params.get("pitch_deg", 0.0))

    def _rot_z(v: np.ndarray, deg: float) -> np.ndarray:
        t = np.deg2rad(float(deg))
        c, s = float(np.cos(t)), float(np.sin(t))
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        return R @ np.asarray(v, dtype=float)

    def _rot_y(v: np.ndarray, deg: float) -> np.ndarray:
        t = np.deg2rad(float(deg))
        c, s = float(np.cos(t)), float(np.sin(t))
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
        return R @ np.asarray(v, dtype=float)

    rx_b = _rot_y(_rot_z(np.asarray(rx.boresight, dtype=float), yaw_deg), pitch_deg)
    rx_h = _rot_y(_rot_z(np.asarray(rx.h_axis, dtype=float), yaw_deg), pitch_deg)
    rx_v = _rot_y(_rot_z(np.asarray(rx.v_axis, dtype=float), yaw_deg), pitch_deg)
    rx = rx.with_orientation(boresight=rx_b, h_axis=rx_h, v_axis=rx_v).with_position([params["distance_m"], 0.0, 1.5])
    trace_kwargs = dict(diffuse_config or {})
    return trace_paths(
        build_scene(),
        tx,
        rx,
        f_hz,
        max_bounce=int(max_bounce_override) if max_bounce_override is not None else 0,
        los_enabled=True,
        force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
        **trace_kwargs,
    )

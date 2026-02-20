"""C0 free-space LOS sanity scenario."""

from __future__ import annotations

from typing import Any

from rt_core.tracer import trace_paths
from scenarios.common import default_antennas


def build_scene() -> list:
    return []


def build_sweep_params() -> list[dict[str, Any]]:
    return [{"distance_m": d} for d in [3.0, 6.0, 9.0]]


def run_case(
    params: dict[str, Any],
    f_hz,
    basis: str = "linear",
    antenna_config: dict[str, Any] | None = None,
    force_cp_swap_on_odd_reflection: bool = False,
):
    tx, rx = default_antennas(basis=basis, **(antenna_config or {}))
    rx.position[:] = [params["distance_m"], 0.0, 1.5]
    return trace_paths(
        build_scene(),
        tx,
        rx,
        f_hz,
        max_bounce=0,
        los_enabled=True,
        force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
    )

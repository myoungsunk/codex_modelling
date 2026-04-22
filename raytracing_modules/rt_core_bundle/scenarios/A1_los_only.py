"""A1 nearly LOS-only scenario (max_bounce=0)."""

from __future__ import annotations

from typing import Any

from rt_core.tracer import trace_paths
from scenarios.common import default_antennas


def build_scene() -> list:
    return []


def build_sweep_params() -> list[dict[str, Any]]:
    return [{"distance_m": d} for d in [4.0, 6.0, 8.0]]


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
    rx = rx.with_position([params["distance_m"], 0.2, 1.5])
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

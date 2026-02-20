"""A3 two-plate corner two-bounce even-parity scenario."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_core.geometry import Material, Plane
from rt_core.tracer import trace_paths
from scenarios.common import default_antennas


def build_scene(offset: float = 3.0) -> list[Plane]:
    return [
        Plane(id=1, p0=np.array([0.0, offset, 0.0]), normal=np.array([0.0, -1.0, 0.0]), material=Material.pec()),
        Plane(id=2, p0=np.array([offset, 0.0, 0.0]), normal=np.array([-1.0, 0.0, 0.0]), material=Material.pec()),
    ]


def build_sweep_params() -> list[dict[str, Any]]:
    return [{"offset": 3.0, "delta_m": d} for d in [0.0, 0.5, 1.0, 1.5]]


def run_case(params: dict[str, Any], f_hz, basis: str = "linear"):
    tx, rx = default_antennas(basis=basis)
    tx.position[:] = [0.0, 0.0, 1.5]
    rx.position[:] = [3.0 + params["delta_m"], 4.0 + params["delta_m"], 1.5]
    scene = build_scene(offset=params["offset"])
    return trace_paths(scene, tx, rx, f_hz, max_bounce=2, los_enabled=False)

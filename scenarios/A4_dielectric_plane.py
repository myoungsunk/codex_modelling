"""A4 dielectric material sweep with one-bounce reflections."""

from __future__ import annotations

from typing import Any

import numpy as np

from rt_core.geometry import Plane
from rt_core.materials import DEFAULT_MATERIAL_SPECS, resolve_material_library
from rt_core.tracer import trace_paths
from scenarios.common import default_antennas

MATERIALS = resolve_material_library(None, material_dispersion="off")


def _material_map(
    materials_db: str | None = None,
    material_dispersion: str = "off",
) -> dict[str, Any]:
    return resolve_material_library(
        materials_db_path=materials_db,
        material_dispersion=material_dispersion,
        default_specs=DEFAULT_MATERIAL_SPECS,
    )


def build_scene(
    material_name: str,
    y_plane: float = 2.0,
    materials_db: str | None = None,
    material_dispersion: str = "off",
) -> list[Plane]:
    mats = _material_map(materials_db=materials_db, material_dispersion=material_dispersion)
    mat = mats.get(material_name, MATERIALS.get(material_name))
    if mat is None:
        raise KeyError(f"unknown material: {material_name}")
    return [
        Plane(
            id=1,
            p0=np.array([0.0, y_plane, 0.0]),
            normal=np.array([0.0, -1.0, 0.0]),
            material=mat,
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=20.0,
            half_extent_v=20.0,
        )
    ]


def build_sweep_params() -> list[dict[str, Any]]:
    params = []
    for mat in DEFAULT_MATERIAL_SPECS:
        for y in [1.5, 2.0, 2.5]:
            params.append({"material": mat, "y_plane": y, "distance_m": 6.0})
    return params


def run_case(
    params: dict[str, Any],
    f_hz,
    basis: str = "linear",
    antenna_config: dict[str, Any] | None = None,
    force_cp_swap_on_odd_reflection: bool = False,
    materials_db: str | None = None,
    material_dispersion: str = "off",
    max_bounce_override: int | None = None,
    diffuse_config: dict[str, Any] | None = None,
):
    tx, rx = default_antennas(basis=basis, **(antenna_config or {}))
    rx.position[:] = [params["distance_m"], 0.0, 1.5]
    scene = build_scene(
        params["material"],
        y_plane=params["y_plane"],
        materials_db=materials_db,
        material_dispersion=material_dispersion,
    )
    trace_kwargs = dict(diffuse_config or {})
    return trace_paths(
        scene,
        tx,
        rx,
        f_hz,
        max_bounce=int(max_bounce_override) if max_bounce_override is not None else 1,
        los_enabled=False,
        force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
        **trace_kwargs,
    )

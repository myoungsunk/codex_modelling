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


def _even_scene(even_layout: str = "localized") -> list[Plane]:
    # Two PEC planes for controlled two-bounce benchmark.
    # - full: wide parallel planes (legacy, may admit symmetric dual paths)
    # - localized: finite reflector patches around intended bounce regions
    layout = str(even_layout).strip().lower()
    if layout not in {"full", "localized"}:
        layout = "localized"
    if layout == "full":
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
    return [
        Plane(
            id=302,
            p0=np.array([2.10, 0.0, 1.5]),
            normal=np.array([0.0, -1.0, 0.0]),
            material=Material.pec(),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            # Expanded along x to keep near-normal even cases with rx_x=2.2 viable.
            half_extent_u=0.20,
            half_extent_v=0.25,
        ),
        Plane(
            id=303,
            p0=np.array([2.29, 4.0, 1.5]),
            normal=np.array([0.0, 1.0, 0.0]),
            material=Material.pec(),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=0.20,
            half_extent_v=0.25,
        ),
    ]


def build_scene(mode: str, even_layout: str = "localized") -> list[Plane]:
    if mode == "odd":
        return _odd_scene()
    if mode == "even":
        return _even_scene(even_layout=even_layout)
    raise ValueError(f"unknown mode: {mode}")


def build_sweep_params(case_set: str = "full") -> list[dict[str, Any]]:
    """Generate odd/even near-normal cases.

    case_set:
      - "full":  odd/even 각각 3x3 perturbation (총 18개)
      - "minimal": odd/even 각각 중심점 1개 (총 2개)
      - "both": full + minimal을 함께 반환 (총 20개)
    """

    set_key = str(case_set).strip().lower()
    if set_key not in {"full", "minimal", "both"}:
        set_key = "full"

    def _params_for(name: str) -> list[dict[str, Any]]:
        if name == "minimal":
            dx_vals = [0.0]
            dz_vals = [0.0]
        else:
            dx_vals = [-0.2, 0.0, 0.2]
            dz_vals = [-0.1, 0.0, 0.1]
        out: list[dict[str, Any]] = []
        for dx in dx_vals:
            for dz in dz_vals:
                out.append(
                    {
                        "mode": "odd",
                        "target_bounce": 1,
                        "rx_x": 2.4 + dx,
                        "rx_y": -2.0,
                        "rx_z": 1.5 + dz,
                        "incidence_max_deg": 15.0,
                        "a6_case_set": str(name),
                    }
                )
        for dx in dx_vals:
            for dz in dz_vals:
                out.append(
                    {
                        "mode": "even",
                        "target_bounce": 2,
                        "rx_x": 2.4 + dx,
                        "rx_y": 1.0,
                        "rx_z": 1.5 + dz,
                        "incidence_max_deg": 15.0,
                        "a6_case_set": str(name),
                    }
                )
        return out

    if set_key == "both":
        return _params_for("full") + _params_for("minimal")
    return _params_for(set_key)


def run_case(
    params: dict[str, Any],
    f_hz,
    basis: str = "linear",
    antenna_config: dict[str, Any] | None = None,
    force_cp_swap_on_odd_reflection: bool = False,
    max_bounce_override: int | None = None,
    diffuse_config: dict[str, Any] | None = None,
    even_path_policy: str = "canonical",
    even_layout: str = "localized",
):
    tx, rx = default_antennas(basis=basis, **(antenna_config or {}))
    mode = str(params["mode"])
    scene = build_scene(mode, even_layout=even_layout)
    target_bounce = int(params["target_bounce"])
    max_bounce = int(max_bounce_override) if max_bounce_override is not None else target_bounce

    if mode == "odd":
        tx = tx.with_position([2.0, -2.0, 1.5])
    else:
        tx = tx.with_position([2.0, 1.0, 1.5])

    rx = rx.with_position([params["rx_x"], params["rx_y"], params["rx_z"]])

    paths = trace_paths(
        scene,
        tx,
        rx,
        f_hz,
        max_bounce=max_bounce,
        los_enabled=False,
        force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
        **dict(diffuse_config or {}),
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

    # For A6 even-bounce benchmark, two symmetric 2-bounce solutions may coexist.
    # Default to a single canonical path to avoid equal-length dual-path ambiguity.
    mode_l = str(mode).strip().lower()
    pol = str(even_path_policy).strip().lower()
    if mode_l == "even" and len(out) > 1 and pol in {"canonical", "dominant"}:
        if pol == "canonical":
            canon = [p for p in out if list(getattr(p, "surface_ids", [])) == [302, 303]]
            if canon:
                out = canon
            else:
                # Fallback to dominant path if canonical ordering not found.
                pol = "dominant"
        if pol == "dominant":
            def _path_power_key(pp) -> float:
                af = np.asarray(getattr(pp, "A_f", np.zeros((0, 2, 2), dtype=np.complex128)))
                if af.size == 0:
                    return -np.inf
                p_lin = float(np.nanmean(np.abs(af) ** 2))
                return p_lin

            j = int(np.argmax([_path_power_key(p) for p in out]))
            out = [out[j]]
    return out

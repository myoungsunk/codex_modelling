"""Build ray-table style latent variables from RT paths."""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis.power_utils import path_power


def _material_summary(meta: dict[str, Any]) -> str:
    mids = meta.get("material_ids", [])
    if isinstance(mids, list) and mids:
        return "|".join(str(x) for x in mids)
    sids = meta.get("surface_ids", [])
    if isinstance(sids, list) and sids:
        return "|".join(str(x) for x in sids)
    return "NA"


def build_ray_table_from_rt(
    paths: list[dict[str, Any]],
    matrix_source: str = "A",
    include_material: bool = True,
    include_angles: bool = True,
) -> list[dict[str, Any]]:
    """Build per-path latent variable table from RT path records."""

    rows: list[dict[str, Any]] = []
    for i, p in enumerate(paths):
        meta = dict(p.get("meta", {}))
        n_bounce = int(meta.get("bounce_count", 0))
        tau_s = float(p.get("tau_s", np.nan))
        L_m = float(p.get("path_length_m", np.nan))
        P_lin = float(path_power(p, matrix_source=matrix_source))
        row: dict[str, Any] = {
            "ray_index": int(i),
            "tau_s": tau_s,
            "L_m": L_m,
            "n_bounce": n_bounce,
            "P_lin": P_lin,
            "parity": "even" if (n_bounce % 2 == 0) else "odd",
            "los_flag_ray": int(n_bounce == 0),
            "surface_seq": "|".join(str(x) for x in meta.get("surface_ids", []))
            if isinstance(meta.get("surface_ids", []), list)
            else "NA",
        }
        if include_material:
            row["material_class"] = _material_summary(meta)
        if include_angles:
            angles = np.asarray(meta.get("incidence_angles", []), dtype=float)
            row["incidence_deg"] = float(np.rad2deg(np.nanmean(angles))) if angles.size else np.nan
        rows.append(row)
    rows.sort(key=lambda r: float(r.get("tau_s", np.inf)))
    return rows

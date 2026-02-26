"""Excess-loss proxy computations for ray/link tables."""

from __future__ import annotations

from typing import Any

import numpy as np

from physics.fspl import fspl_db


EPS = 1e-15


def add_el_db(
    ray_rows: list[dict[str, Any]],
    f_center_hz: float,
    method: str = "fspl",
) -> list[dict[str, Any]]:
    """Attach EL_db to each ray row.

    EL_db = P_fs_db - P_db,
    where P_fs_db is free-space received power (relative) at identical length.
    """

    out = []
    for r in ray_rows:
        row = dict(r)
        L_m = float(row.get("L_m", np.nan))
        P_lin = float(row.get("P_lin", np.nan))
        if not np.isfinite(L_m) or not np.isfinite(P_lin) or P_lin <= 0.0:
            row["EL_db"] = np.nan
        else:
            if str(method).lower() == "fspl":
                P_fs_db = -float(fspl_db(L_m, f_center_hz))
            else:
                P_fs_db = float(-20.0 * np.log10(max(L_m, EPS)))
            P_db = float(10.0 * np.log10(max(P_lin, EPS)))
            row["EL_db"] = float(P_fs_db - P_db)
        out.append(row)
    return out

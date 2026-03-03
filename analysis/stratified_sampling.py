"""Stratified sampling helper for measurement-point selection."""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_rows(points_df_or_rows: Any) -> list[dict[str, Any]]:
    if isinstance(points_df_or_rows, list):
        return [dict(x) for x in points_df_or_rows if isinstance(x, dict)]
    try:
        import pandas as pd  # type: ignore

        if isinstance(points_df_or_rows, pd.DataFrame):
            return points_df_or_rows.to_dict(orient="records")
    except Exception:
        pass
    raise ValueError("Unsupported points input type")


def stratified_sample(
    points_df: Any,
    bins: dict[str, int] | None = None,
    per_bin: int = 5,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rows = _to_rows(points_df)
    if not rows:
        return []
    bin_cfg = dict(bins or {"EL_proxy_db": 4, "LOSflag": 2})
    rng = np.random.default_rng(int(seed))

    key_specs: dict[str, dict[str, Any]] = {}
    for k, nb in bin_cfg.items():
        if int(nb) <= 1:
            key_specs[k] = {"mode": "raw"}
            continue
        vals = np.asarray([float(r.get(k, np.nan)) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            key_specs[k] = {"mode": "raw"}
            continue
        rounded = np.round(vals)
        unique_int = np.unique(rounded.astype(int))
        if np.all(np.isclose(vals, rounded)) and len(unique_int) <= int(nb):
            key_specs[k] = {"mode": "categorical_int"}
            continue
        q = np.linspace(0.0, 1.0, int(nb) + 1)
        edges = np.quantile(vals, q)
        key_specs[k] = {"mode": "quantile", "edges": np.asarray(edges, dtype=float)}

    idx_by_key: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        parts = []
        for k, nb in bin_cfg.items():
            if k not in r:
                parts.append(f"{k}=NA")
                continue
            spec = key_specs.get(k, {"mode": "raw"})
            try:
                x = float(r.get(k, np.nan))
                if np.isfinite(x) and int(nb) > 1:
                    if spec.get("mode") == "categorical_int":
                        parts.append(f"{k}={int(np.round(x))}")
                    elif spec.get("mode") == "quantile":
                        edges = np.asarray(spec.get("edges", []), dtype=float)
                        b = int(np.searchsorted(edges[1:-1], x, side="right"))
                        parts.append(f"{k}={b}")
                    else:
                        parts.append(f"{k}={str(r.get(k))}")
                else:
                    parts.append(f"{k}={str(r.get(k))}")
            except Exception:
                parts.append(f"{k}={str(r.get(k))}")
        key = "|".join(parts)
        idx_by_key.setdefault(key, []).append(i)

    out_idx: list[int] = []
    for _, idxs in sorted(idx_by_key.items(), key=lambda kv: kv[0]):
        if len(idxs) <= int(per_bin):
            out_idx.extend(idxs)
            continue
        pick = rng.choice(np.asarray(idxs, dtype=int), size=int(per_bin), replace=False)
        out_idx.extend([int(x) for x in pick.tolist()])
    out_idx = sorted(set(out_idx))
    return [rows[i] for i in out_idx]

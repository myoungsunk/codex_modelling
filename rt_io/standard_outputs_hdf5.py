"""HDF5/CSV exporters for standard outputs schema v1."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from rt_types.standard_outputs import SCHEMA_VERSION, StandardOutputBundle


def _json(obj: Any) -> str:
    def _default(x: Any) -> Any:
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return str(x)

    return json.dumps(obj, sort_keys=True, default=_default)


def _write_scalar_attrs(group: h5py.Group, payload: dict[str, Any]) -> None:
    for k, v in payload.items():
        if isinstance(v, (str, int, float, bool, np.bool_)):
            group.attrs[str(k)] = v
        else:
            group.attrs[str(k)] = _json(v)


def _write_ray_table(group: h5py.Group, rows: list[dict[str, Any]]) -> None:
    if not rows:
        group.create_dataset("n_rows", data=np.asarray(0, dtype=np.int32))
        return
    cols = sorted({k for r in rows for k in r.keys()})
    n = len(rows)
    group.create_dataset("n_rows", data=np.asarray(n, dtype=np.int32))
    str_dt = h5py.string_dtype(encoding="utf-8")
    for c in cols:
        vals = [r.get(c, "") for r in rows]
        if all(isinstance(v, (int, np.integer, bool, np.bool_)) for v in vals):
            group.create_dataset(c, data=np.asarray(vals, dtype=np.int64))
        elif all(isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)) for v in vals):
            group.create_dataset(c, data=np.asarray(vals, dtype=np.float64))
        else:
            group.create_dataset(c, data=np.asarray([str(v) for v in vals], dtype=object), dtype=str_dt)


def save_run(
    run_meta: dict[str, Any],
    bundles: list[StandardOutputBundle],
    out_h5: str | Path,
    run_id: str | None = None,
) -> Path:
    p = Path(out_h5)
    p.parent.mkdir(parents=True, exist_ok=True)
    rid = str(run_id or run_meta.get("run_id", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))

    with h5py.File(p, "a") as f:
        runs = f.require_group("runs")
        if rid in runs:
            del runs[rid]
        rg = runs.create_group(rid)
        meta_g = rg.create_group("meta")
        meta = dict(run_meta)
        meta.setdefault("schema_version", SCHEMA_VERSION)
        meta.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        _write_scalar_attrs(meta_g, meta)
        meta_g.attrs["meta_json"] = _json(meta)

        links_g = rg.create_group("links")
        for b in bundles:
            lg = links_g.create_group(str(b.link_id))
            lg.attrs["schema_version"] = str(b.schema_version)
            lg.attrs["scenario_id"] = str(b.scenario_id)
            lg.attrs["case_id"] = str(b.case_id)

            rays_g = lg.create_group("rays")
            _write_ray_table(rays_g, b.rays.rows)

            pdp_g = lg.create_group("pdp")
            pdp_g.create_dataset("delay_tau_s", data=np.asarray(b.pdp.delay_tau_s, dtype=float))
            pdp_g.create_dataset("P_co", data=np.asarray(b.pdp.P_co, dtype=float))
            pdp_g.create_dataset("P_cross", data=np.asarray(b.pdp.P_cross, dtype=float))
            if b.pdp.XPD_tau_db is not None:
                pdp_g.create_dataset("XPD_tau_db", data=np.asarray(b.pdp.XPD_tau_db, dtype=float))

            met_g = lg.create_group("metrics")
            _write_scalar_attrs(met_g, b.metrics.to_dict())
            met_g.attrs["metrics_json"] = _json(b.metrics.to_dict())

            u_g = lg.create_group("U")
            _write_scalar_attrs(u_g, b.conditions.to_dict())
            u_g.attrs["u_json"] = _json(b.conditions.to_dict())

            prov_g = lg.create_group("provenance")
            _write_scalar_attrs(prov_g, b.provenance)
            prov_g.attrs["provenance_json"] = _json(b.provenance)
    return p


def _read_ray_table(group: h5py.Group) -> list[dict[str, Any]]:
    keys = [k for k in group.keys() if k != "n_rows"]
    if not keys:
        return []
    n = int(group["n_rows"][()])
    out = []
    for i in range(n):
        row: dict[str, Any] = {}
        for k in keys:
            v = group[k][i]
            if isinstance(v, bytes):
                row[k] = v.decode("utf-8")
            elif isinstance(v, np.generic):
                row[k] = v.item()
            else:
                row[k] = v
        out.append(row)
    return out


def load_run(path: str | Path, run_id: str) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        rg = f[f"runs/{run_id}"]
        meta = json.loads(str(rg["meta"].attrs.get("meta_json", "{}")))
        out = {"run_id": str(run_id), "meta": meta, "bundles": []}
        for link_id, lg in rg["links"].items():
            rays = _read_ray_table(lg["rays"])
            pdp = {
                "delay_tau_s": np.asarray(lg["pdp"]["delay_tau_s"][:], dtype=float),
                "P_co": np.asarray(lg["pdp"]["P_co"][:], dtype=float),
                "P_cross": np.asarray(lg["pdp"]["P_cross"][:], dtype=float),
            }
            if "XPD_tau_db" in lg["pdp"]:
                pdp["XPD_tau_db"] = np.asarray(lg["pdp"]["XPD_tau_db"][:], dtype=float)
            bundle = {
                "schema_version": str(lg.attrs.get("schema_version", SCHEMA_VERSION)),
                "link_id": str(link_id),
                "scenario_id": str(lg.attrs.get("scenario_id", "")),
                "case_id": str(lg.attrs.get("case_id", "")),
                "rays": {"rows": rays},
                "pdp": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in pdp.items()},
                "metrics": json.loads(str(lg["metrics"].attrs.get("metrics_json", "{}"))),
                "conditions": json.loads(str(lg["U"].attrs.get("u_json", "{}"))),
                "provenance": json.loads(str(lg["provenance"].attrs.get("provenance_json", "{}"))),
            }
            out["bundles"].append(bundle)
    return out


def export_csv(
    bundles: list[StandardOutputBundle],
    out_dir: str | Path,
    link_metrics_name: str = "link_metrics.csv",
    rays_name: str = "rays.csv",
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    link_metrics_csv = out / link_metrics_name
    rays_csv = out / rays_name

    link_rows: list[dict[str, Any]] = []
    ray_rows: list[dict[str, Any]] = []
    for b in bundles:
        lr = {
            "link_id": b.link_id,
            "scenario_id": b.scenario_id,
            "case_id": b.case_id,
            **b.metrics.to_dict(),
            **b.conditions.to_dict(),
        }
        lr["provenance_json"] = _json(b.provenance)
        link_rows.append(lr)
        for r in b.rays.rows:
            ray_rows.append(
                {
                    "link_id": b.link_id,
                    "scenario_id": b.scenario_id,
                    "case_id": b.case_id,
                    **dict(r),
                }
            )
        np.savez_compressed(
            out / f"pdp_{b.link_id}.npz",
            delay_tau_s=np.asarray(b.pdp.delay_tau_s, dtype=float),
            P_co=np.asarray(b.pdp.P_co, dtype=float),
            P_cross=np.asarray(b.pdp.P_cross, dtype=float),
            XPD_tau_db=np.asarray(b.pdp.XPD_tau_db, dtype=float)
            if b.pdp.XPD_tau_db is not None
            else np.asarray([], dtype=float),
        )

    def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        import csv

        keys = sorted({k for r in rows for k in r.keys()}) if rows else []
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                out_row = {}
                for k in keys:
                    v = r.get(k, "")
                    if isinstance(v, (dict, list, tuple)):
                        out_row[k] = _json(v)
                    else:
                        out_row[k] = v
                w.writerow(out_row)

    _write_rows(link_metrics_csv, link_rows)
    _write_rows(rays_csv, ray_rows)
    return {
        "link_metrics_csv": str(link_metrics_csv),
        "rays_csv": str(rays_csv),
    }

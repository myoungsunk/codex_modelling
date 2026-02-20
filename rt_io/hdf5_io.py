"""HDF5 schema for polarimetric RT sweeps.

Example:
    >>> import numpy as np
    >>> from rt_io.hdf5_io import save_rt_dataset, load_rt_dataset
    >>> data = {
    ...   "meta": {"basis": "linear", "convention": "IEEE-RHCP"},
    ...   "frequency": np.linspace(6e9, 7e9, 4),
    ...   "scenarios": {"C0": {"cases": {"0": {"params": {"x": 1}, "paths": []}}}},
    ... }
    >>> _ = save_rt_dataset('/tmp/rt_demo.h5', data)
    >>> out = load_rt_dataset('/tmp/rt_demo.h5')
    >>> list(out['scenarios'].keys())
    ['C0']
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis.ctf_cir import synthesize_ctf


def _write_string_array(group: h5py.Group, name: str, values: list[str]) -> None:
    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dt)


def save_rt_dataset(path: str | Path, data: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(p, "w") as f:
        meta = f.create_group("meta")
        now = datetime.now(timezone.utc).isoformat()
        meta.attrs["created_at"] = data.get("meta", {}).get("created_at", now)
        meta.attrs["basis"] = data.get("meta", {}).get("basis", "linear")
        meta.attrs["convention"] = data.get("meta", {}).get("convention", "IEEE-RHCP")

        freq = np.asarray(data["frequency"], dtype=float)
        f.create_dataset("frequency", data=freq)

        sc_root = f.create_group("scenarios")
        for scenario_id, scenario in data.get("scenarios", {}).items():
            sc_g = sc_root.create_group(str(scenario_id))
            cases_g = sc_g.create_group("cases")
            for case_id, case in scenario.get("cases", {}).items():
                c_g = cases_g.create_group(str(case_id))
                c_g.create_dataset("params", data=json.dumps(case.get("params", {})))

                paths = case.get("paths", [])
                p_g = c_g.create_group("paths")
                l = len(paths)
                p_g.create_dataset("tau_s", data=np.asarray([pp["tau_s"] for pp in paths], dtype=float))
                if l == 0:
                    p_g.create_dataset("A_f", data=np.zeros((0, len(freq), 2, 2), dtype=np.complex128))
                    p_g.create_dataset("bounce_count", data=np.zeros((0,), dtype=np.int32))
                    _write_string_array(p_g, "interactions", [])
                    p_g.create_dataset("surface_ids", data=np.zeros((0, 0), dtype=np.int32))
                    p_g.create_dataset("incidence_angles", data=np.zeros((0, 0), dtype=float))
                    p_g.create_dataset("AoD", data=np.zeros((0, 3), dtype=float))
                    p_g.create_dataset("AoA", data=np.zeros((0, 3), dtype=float))
                    continue

                p_g.create_dataset("A_f", data=np.asarray([pp["A_f"] for pp in paths], dtype=np.complex128))
                p_g.create_dataset("bounce_count", data=np.asarray([pp["meta"]["bounce_count"] for pp in paths], dtype=np.int32))
                _write_string_array(p_g, "interactions", ["|".join(pp["meta"]["interactions"]) for pp in paths])

                max_sid = max((len(pp["meta"]["surface_ids"]) for pp in paths), default=0)
                max_ang = max((len(pp["meta"]["incidence_angles"]) for pp in paths), default=0)
                sid = np.full((l, max_sid), -1, dtype=np.int32)
                ang = np.full((l, max_ang), np.nan, dtype=float)
                for i, pp in enumerate(paths):
                    sv = np.asarray(pp["meta"]["surface_ids"], dtype=np.int32)
                    av = np.asarray(pp["meta"]["incidence_angles"], dtype=float)
                    sid[i, : len(sv)] = sv
                    ang[i, : len(av)] = av
                p_g.create_dataset("surface_ids", data=sid)
                p_g.create_dataset("incidence_angles", data=ang)
                p_g.create_dataset("AoD", data=np.asarray([pp["meta"]["AoD"] for pp in paths], dtype=float))
                p_g.create_dataset("AoA", data=np.asarray([pp["meta"]["AoA"] for pp in paths], dtype=float))
                _write_string_array(
                    p_g,
                    "segment_basis_uv",
                    [json.dumps(pp["meta"].get("segment_basis_uv", [])) for pp in paths],
                )
    return p


def load_rt_dataset(path: str | Path) -> dict[str, Any]:
    out: dict[str, Any] = {"meta": {}, "frequency": None, "scenarios": {}}
    with h5py.File(path, "r") as f:
        out["meta"] = {
            "created_at": f["meta"].attrs.get("created_at", ""),
            "basis": f["meta"].attrs.get("basis", "linear"),
            "convention": f["meta"].attrs.get("convention", "IEEE-RHCP"),
        }
        out["frequency"] = np.asarray(f["frequency"][:], dtype=float)

        for scenario_id, sc_g in f["scenarios"].items():
            sc = {"cases": {}}
            for case_id, c_g in sc_g["cases"].items():
                params = json.loads(c_g["params"][()].decode("utf-8") if isinstance(c_g["params"][()], bytes) else c_g["params"][()])
                p_g = c_g["paths"]
                tau = np.asarray(p_g["tau_s"][:], dtype=float)
                af = np.asarray(p_g["A_f"][:], dtype=np.complex128)
                bc = np.asarray(p_g["bounce_count"][:], dtype=int)
                inter_raw = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in p_g["interactions"][:]]
                sid = np.asarray(p_g["surface_ids"][:], dtype=int)
                ang = np.asarray(p_g["incidence_angles"][:], dtype=float)
                aod = np.asarray(p_g["AoD"][:], dtype=float)
                aoa = np.asarray(p_g["AoA"][:], dtype=float)
                segb = (
                    [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in p_g["segment_basis_uv"][:]]
                    if "segment_basis_uv" in p_g
                    else ["[]"] * len(tau)
                )

                paths = []
                for i in range(len(tau)):
                    sids = [int(x) for x in sid[i].tolist() if x >= 0]
                    angs = [float(x) for x in ang[i].tolist() if np.isfinite(x)]
                    paths.append(
                        {
                            "tau_s": float(tau[i]),
                            "A_f": af[i],
                            "meta": {
                                "bounce_count": int(bc[i]),
                                "interactions": [z for z in inter_raw[i].split("|") if z],
                                "surface_ids": sids,
                                "incidence_angles": angs,
                                "AoD": aod[i].tolist(),
                                "AoA": aoa[i].tolist(),
                                "segment_basis_uv": json.loads(segb[i]),
                            },
                        }
                    )
                sc["cases"][case_id] = {"params": params, "paths": paths}
            out["scenarios"][scenario_id] = sc
    return out


def self_test_reproducibility(path: str | Path, scenario_id: str, case_id: str, atol: float = 1e-9) -> bool:
    """Check that H(f) is preserved through save/load."""

    original = load_rt_dataset(path)
    freq = np.asarray(original["frequency"], dtype=float)
    case = original["scenarios"][scenario_id]["cases"][case_id]
    h1 = synthesize_ctf(case["paths"], freq)

    tmp = Path(path).with_suffix(".roundtrip.h5")
    save_rt_dataset(tmp, original)
    loaded = load_rt_dataset(tmp)
    case2 = loaded["scenarios"][scenario_id]["cases"][case_id]
    h2 = synthesize_ctf(case2["paths"], freq)
    return bool(np.allclose(h1, h2, atol=atol))

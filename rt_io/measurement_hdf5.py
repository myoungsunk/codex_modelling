"""HDF5 IO for imported measurement-domain channel matrices."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import h5py
import numpy as np


MEASUREMENT_SCHEMA_VERSION = "measurement_v1"


def _json_dumps(value: Any) -> str:
    def _default(obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_default)


def _json_loads_safe(text: str, fallback: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return fallback


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_provenance(
    source_paths: dict[str, str | Path],
    command: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    abs_paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for k, v in source_paths.items():
        p = Path(v).expanduser().resolve()
        abs_paths[str(k)] = str(p)
        hashes[str(k)] = sha256_file(p)
    out: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "command": str(command),
        "source_paths": abs_paths,
        "file_sha256": hashes,
    }
    if extra:
        out["extra"] = dict(extra)
    return out


def _validate_case_arrays(frequency_hz: np.ndarray, H_f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(frequency_hz, dtype=float)
    h = np.asarray(H_f, dtype=np.complex128)
    if f.ndim != 1 or len(f) == 0:
        raise ValueError("frequency_hz must be non-empty 1D.")
    if not np.all(np.isfinite(f)):
        raise ValueError("frequency_hz contains NaN/Inf.")
    if h.shape != (len(f), 2, 2):
        raise ValueError(f"H_f must have shape {(len(f), 2, 2)}.")
    if not np.all(np.isfinite(h)):
        raise ValueError("H_f contains NaN/Inf.")
    return f, h


def _write_attrs_json(group: h5py.Group, key: str, value: Any) -> None:
    group.attrs[str(key)] = _json_dumps(value)


def append_measurement_case(
    path: str | Path,
    scenario_id: str,
    case_id: str,
    frequency_hz: np.ndarray,
    H_f: np.ndarray,
    meta: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f, h = _validate_case_arrays(frequency_hz, H_f)
    m = dict(meta or {})
    prov = dict(provenance or {})
    now = datetime.now(timezone.utc).isoformat()

    basis = str(m.get("basis", "circular"))
    convention = str(m.get("convention", "IEEE-RHCP"))

    with h5py.File(p, "a") as hf:
        root = hf.require_group("measurement")
        mg = root.require_group("meta")
        if "created_at" not in mg.attrs:
            mg.attrs["created_at"] = now
        mg.attrs["updated_at"] = now
        mg.attrs["schema_version"] = MEASUREMENT_SCHEMA_VERSION
        mg.attrs["basis"] = basis
        mg.attrs["convention"] = convention
        _write_attrs_json(mg, "meta_json", m)

        sc_root = root.require_group("scenarios")
        sc = sc_root.require_group(str(scenario_id))
        cases = sc.require_group("cases")

        cid = str(case_id)
        if cid in cases:
            if overwrite:
                del cases[cid]
            else:
                raise ValueError(f"measurement case exists and overwrite=False: {scenario_id}/{case_id}")

        cg = cases.create_group(cid)
        cg.create_dataset("frequency_hz", data=f)
        cg.create_dataset("H_f", data=h)

        cm = cg.create_group("meta")
        cm.attrs["basis"] = basis
        cm.attrs["convention"] = convention
        _write_attrs_json(cm, "meta_json", m)

        pg = cg.create_group("provenance")
        _write_attrs_json(pg, "json", prov)
        for k, v in prov.items():
            if isinstance(v, (str, int, float, bool, np.bool_)):
                pg.attrs[str(k)] = v
            else:
                pg.attrs[str(k)] = _json_dumps(v)
    return p


def list_measurement_cases(path: str | Path, scenario_id: str | None = None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with h5py.File(path, "r") as hf:
        if "measurement/scenarios" not in hf:
            return out
        sc_root = hf["measurement/scenarios"]
        if scenario_id is not None:
            sid = str(scenario_id)
            if sid not in sc_root:
                return out
            for cid in sc_root[sid]["cases"].keys():
                out.append((sid, str(cid)))
            return out
        for sid in sc_root.keys():
            for cid in sc_root[sid]["cases"].keys():
                out.append((str(sid), str(cid)))
    return out


def load_measurement_case(path: str | Path, scenario_id: str, case_id: str) -> dict[str, Any]:
    with h5py.File(path, "r") as hf:
        if "measurement" not in hf:
            raise ValueError("missing /measurement group")
        mg = hf["measurement/meta"]
        cg = hf[f"measurement/scenarios/{scenario_id}/cases/{case_id}"]
        cm = cg.get("meta")
        pg = cg.get("provenance")
        root_meta = _json_loads_safe(str(mg.attrs.get("meta_json", "{}")), {})
        case_meta = _json_loads_safe(str(cm.attrs.get("meta_json", "{}")) if cm is not None else "{}", {})
        prov = _json_loads_safe(str(pg.attrs.get("json", "{}")) if pg is not None else "{}", {})
        return {
            "scenario_id": str(scenario_id),
            "case_id": str(case_id),
            "frequency_hz": np.asarray(cg["frequency_hz"][:], dtype=float),
            "H_f": np.asarray(cg["H_f"][:], dtype=np.complex128),
            "meta": {
                "basis": str((cm.attrs.get("basis") if cm is not None else mg.attrs.get("basis", "circular"))),
                "convention": str(
                    (cm.attrs.get("convention") if cm is not None else mg.attrs.get("convention", "IEEE-RHCP"))
                ),
                "root_meta": root_meta if isinstance(root_meta, dict) else {},
                "case_meta": case_meta if isinstance(case_meta, dict) else {},
            },
            "provenance": prov if isinstance(prov, dict) else {},
        }


def iter_measurement_cases(path: str | Path, scenario_id: str | None = None) -> list[dict[str, Any]]:
    return [load_measurement_case(path, sid, cid) for sid, cid in list_measurement_cases(path, scenario_id=scenario_id)]

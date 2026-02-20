"""HDF5 schema for polarimetric RT sweeps.

Schema versions:
- v1: basic per-path arrays (tau_s, A_f, metadata)
- v2: adds decomposition arrays (J_f, G_tx_f, G_rx_f, scalar_gain_f)

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
    >>> out['meta']['schema_version']
    'v2'
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis.ctf_cir import synthesize_ctf


SCHEMA_VERSION = "v2"


def _write_string_array(group: h5py.Group, name: str, values: list[str]) -> None:
    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dt)


def _json_dataset_value(ds: h5py.Dataset) -> dict[str, Any]:
    raw = ds[()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _default_identity_stack(l: int, nf: int) -> np.ndarray:
    eye = np.eye(2, dtype=np.complex128)
    out = np.zeros((l, nf, 2, 2), dtype=np.complex128)
    out[:] = eye
    return out


def _read_dataset_or_default(
    group: h5py.Group,
    name: str,
    dtype: Any,
    default: np.ndarray,
) -> np.ndarray:
    if name in group:
        return np.asarray(group[name][:], dtype=dtype)
    return np.asarray(default, dtype=dtype)


def _decomposed_path_to_A(path: dict[str, Any], freq: np.ndarray) -> np.ndarray:
    a_f = np.asarray(path["A_f"], dtype=np.complex128)
    n_f = len(freq)

    j_f = path.get("J_f")
    g_tx_f = path.get("G_tx_f")
    g_rx_f = path.get("G_rx_f")
    scalar_gain_f = path.get("scalar_gain_f")

    if j_f is None or g_tx_f is None or g_rx_f is None or scalar_gain_f is None:
        return a_f

    j_f = np.asarray(j_f, dtype=np.complex128)
    g_tx_f = np.asarray(g_tx_f, dtype=np.complex128)
    g_rx_f = np.asarray(g_rx_f, dtype=np.complex128)
    scalar_gain_f = np.asarray(scalar_gain_f, dtype=float)

    if j_f.shape != (n_f, 2, 2):
        return a_f
    if g_tx_f.shape != (n_f, 2, 2) or g_rx_f.shape != (n_f, 2, 2):
        return a_f
    if scalar_gain_f.shape != (n_f,):
        return a_f

    a_rec = np.einsum("kab,kbc,kcd->kad", g_rx_f, j_f, g_tx_f).astype(np.complex128)
    a_rec *= scalar_gain_f[:, None, None]
    return a_rec


def synthesize_ctf_from_decomposed(paths: list[dict[str, Any]], f_hz: np.ndarray) -> np.ndarray:
    """Synthesize H(f) using decomposed terms if available, with fallback to A_f."""

    freq = np.asarray(f_hz, dtype=float)
    h_f = np.zeros((len(freq), 2, 2), dtype=np.complex128)
    for p in paths:
        tau = float(p["tau_s"])
        a = _decomposed_path_to_A(p, freq)
        phase = np.exp(-1j * 2.0 * np.pi * freq * tau)[:, None, None]
        h_f += a * phase
    return h_f


def save_rt_dataset(path: str | Path, data: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(p, "w") as f:
        meta = f.create_group("meta")
        now = datetime.now(timezone.utc).isoformat()
        src_meta = data.get("meta", {})
        meta.attrs["created_at"] = src_meta.get("created_at", now)
        meta.attrs["basis"] = src_meta.get("basis", "linear")
        meta.attrs["convention"] = src_meta.get("convention", "IEEE-RHCP")
        meta.attrs["schema_version"] = src_meta.get("schema_version", SCHEMA_VERSION)

        freq = np.asarray(data["frequency"], dtype=float)
        nf = len(freq)
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
                p_g.create_dataset("tau_s", data=np.asarray([pp.get("tau_s", 0.0) for pp in paths], dtype=float))

                if l == 0:
                    p_g.create_dataset("A_f", data=np.zeros((0, nf, 2, 2), dtype=np.complex128))
                    p_g.create_dataset("J_f", data=np.zeros((0, nf, 2, 2), dtype=np.complex128))
                    p_g.create_dataset("G_tx_f", data=np.zeros((0, nf, 2, 2), dtype=np.complex128))
                    p_g.create_dataset("G_rx_f", data=np.zeros((0, nf, 2, 2), dtype=np.complex128))
                    p_g.create_dataset("scalar_gain_f", data=np.zeros((0, nf), dtype=np.float64))
                    p_g.create_dataset("bounce_count", data=np.zeros((0,), dtype=np.int32))
                    _write_string_array(p_g, "interactions", [])
                    p_g.create_dataset("surface_ids", data=np.zeros((0, 0), dtype=np.int32))
                    p_g.create_dataset("incidence_angles", data=np.zeros((0, 0), dtype=float))
                    p_g.create_dataset("AoD", data=np.zeros((0, 3), dtype=float))
                    p_g.create_dataset("AoA", data=np.zeros((0, 3), dtype=float))
                    p_g.create_dataset("bounce_normals", data=np.zeros((0, 0, 3), dtype=float))
                    _write_string_array(p_g, "segment_basis_uv", [])
                    continue

                def _a(pp: dict[str, Any]) -> np.ndarray:
                    return np.asarray(pp.get("A_f", np.zeros((nf, 2, 2), dtype=np.complex128)), dtype=np.complex128)

                p_g.create_dataset("A_f", data=np.asarray([_a(pp) for pp in paths], dtype=np.complex128))
                p_g.create_dataset(
                    "J_f",
                    data=np.asarray([pp.get("J_f", np.zeros((nf, 2, 2), dtype=np.complex128)) for pp in paths], dtype=np.complex128),
                )
                p_g.create_dataset(
                    "G_tx_f",
                    data=np.asarray([pp.get("G_tx_f", _default_identity_stack(1, nf)[0]) for pp in paths], dtype=np.complex128),
                )
                p_g.create_dataset(
                    "G_rx_f",
                    data=np.asarray([pp.get("G_rx_f", _default_identity_stack(1, nf)[0]) for pp in paths], dtype=np.complex128),
                )
                p_g.create_dataset(
                    "scalar_gain_f",
                    data=np.asarray([pp.get("scalar_gain_f", np.ones(nf, dtype=float)) for pp in paths], dtype=np.float64),
                )

                p_g.create_dataset(
                    "bounce_count",
                    data=np.asarray([pp.get("meta", {}).get("bounce_count", 0) for pp in paths], dtype=np.int32),
                )
                _write_string_array(
                    p_g,
                    "interactions",
                    ["|".join(pp.get("meta", {}).get("interactions", [])) for pp in paths],
                )

                max_sid = max((len(pp.get("meta", {}).get("surface_ids", [])) for pp in paths), default=0)
                max_ang = max((len(pp.get("meta", {}).get("incidence_angles", [])) for pp in paths), default=0)
                max_nrm = max((len(pp.get("meta", {}).get("bounce_normals", [])) for pp in paths), default=0)
                sid = np.full((l, max_sid), -1, dtype=np.int32)
                ang = np.full((l, max_ang), np.nan, dtype=float)
                nrm = np.full((l, max_nrm, 3), np.nan, dtype=float)
                for i, pp in enumerate(paths):
                    sv = np.asarray(pp.get("meta", {}).get("surface_ids", []), dtype=np.int32)
                    av = np.asarray(pp.get("meta", {}).get("incidence_angles", []), dtype=float)
                    nv = np.asarray(pp.get("meta", {}).get("bounce_normals", []), dtype=float)
                    sid[i, : len(sv)] = sv
                    ang[i, : len(av)] = av
                    if nv.size > 0:
                        nrm[i, : len(nv), :] = nv.reshape(len(nv), 3)
                p_g.create_dataset("surface_ids", data=sid)
                p_g.create_dataset("incidence_angles", data=ang)
                p_g.create_dataset("bounce_normals", data=nrm)
                p_g.create_dataset("AoD", data=np.asarray([pp.get("meta", {}).get("AoD", [0.0, 0.0, 0.0]) for pp in paths], dtype=float))
                p_g.create_dataset("AoA", data=np.asarray([pp.get("meta", {}).get("AoA", [0.0, 0.0, 0.0]) for pp in paths], dtype=float))
                _write_string_array(
                    p_g,
                    "segment_basis_uv",
                    [json.dumps(pp.get("meta", {}).get("segment_basis_uv", [])) for pp in paths],
                )
    return p


def load_rt_dataset(path: str | Path) -> dict[str, Any]:
    out: dict[str, Any] = {"meta": {}, "frequency": None, "scenarios": {}}
    with h5py.File(path, "r") as f:
        if "meta" in f:
            meta_g = f["meta"]
            out["meta"] = {
                "created_at": meta_g.attrs.get("created_at", ""),
                "basis": meta_g.attrs.get("basis", "linear"),
                "convention": meta_g.attrs.get("convention", "IEEE-RHCP"),
                "schema_version": meta_g.attrs.get("schema_version", "v1"),
            }
        else:
            out["meta"] = {
                "created_at": "",
                "basis": "linear",
                "convention": "IEEE-RHCP",
                "schema_version": "v1",
            }

        out["frequency"] = np.asarray(f["frequency"][:], dtype=float)
        nf = len(out["frequency"])

        for scenario_id, sc_g in f["scenarios"].items():
            sc = {"cases": {}}
            for case_id, c_g in sc_g["cases"].items():
                params = _json_dataset_value(c_g["params"])
                p_g = c_g["paths"]

                tau = _read_dataset_or_default(p_g, "tau_s", float, np.zeros((0,), dtype=float))
                l = len(tau)
                af = _read_dataset_or_default(p_g, "A_f", np.complex128, np.zeros((l, nf, 2, 2), dtype=np.complex128))
                jf = _read_dataset_or_default(p_g, "J_f", np.complex128, np.zeros((l, nf, 2, 2), dtype=np.complex128))
                gtx = _read_dataset_or_default(p_g, "G_tx_f", np.complex128, _default_identity_stack(l, nf))
                grx = _read_dataset_or_default(p_g, "G_rx_f", np.complex128, _default_identity_stack(l, nf))
                sg = _read_dataset_or_default(p_g, "scalar_gain_f", float, np.ones((l, nf), dtype=float))
                bc = _read_dataset_or_default(p_g, "bounce_count", int, np.zeros((l,), dtype=np.int32))

                if "interactions" in p_g:
                    inter_raw = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in p_g["interactions"][:]]
                else:
                    inter_raw = [""] * l

                sid = _read_dataset_or_default(p_g, "surface_ids", int, np.full((l, 0), -1, dtype=np.int32))
                ang = _read_dataset_or_default(p_g, "incidence_angles", float, np.full((l, 0), np.nan, dtype=float))
                nrm = _read_dataset_or_default(p_g, "bounce_normals", float, np.full((l, 0, 3), np.nan, dtype=float))
                aod = _read_dataset_or_default(p_g, "AoD", float, np.zeros((l, 3), dtype=float))
                aoa = _read_dataset_or_default(p_g, "AoA", float, np.zeros((l, 3), dtype=float))
                if "segment_basis_uv" in p_g:
                    segb = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in p_g["segment_basis_uv"][:]]
                else:
                    segb = ["[]"] * l

                paths = []
                for i in range(l):
                    sids = [int(x) for x in np.asarray(sid[i]).tolist() if x >= 0]
                    angs = [float(x) for x in np.asarray(ang[i]).tolist() if np.isfinite(x)]
                    nrms = [list(map(float, row)) for row in np.asarray(nrm[i], dtype=float).tolist() if np.all(np.isfinite(row))]
                    paths.append(
                        {
                            "tau_s": float(tau[i]),
                            "A_f": af[i],
                            "J_f": jf[i],
                            "G_tx_f": gtx[i],
                            "G_rx_f": grx[i],
                            "scalar_gain_f": sg[i],
                            "meta": {
                                "bounce_count": int(bc[i]),
                                "interactions": [z for z in inter_raw[i].split("|") if z],
                                "surface_ids": sids,
                                "incidence_angles": angs,
                                "bounce_normals": nrms,
                                "AoD": np.asarray(aod[i], dtype=float).tolist(),
                                "AoA": np.asarray(aoa[i], dtype=float).tolist(),
                                "segment_basis_uv": json.loads(segb[i]) if segb[i] else [],
                            },
                        }
                    )

                sc["cases"][case_id] = {"params": params, "paths": paths}
            out["scenarios"][scenario_id] = sc
    return out


def self_test_reproducibility(path: str | Path, scenario_id: str, case_id: str, atol: float = 1e-9) -> bool:
    """Check roundtrip reproducibility and decomposition consistency.

    Validates all of the following:
    1) H(f) from stored A_f is preserved through save/load roundtrip.
    2) H(f) from decomposed terms (G_rx*J*G_tx*scalar_gain) matches H(f) from A_f.
    3) The decomposition-based H(f) is also preserved through roundtrip.
    """

    original = load_rt_dataset(path)
    freq = np.asarray(original["frequency"], dtype=float)
    case = original["scenarios"][scenario_id]["cases"][case_id]

    h_af_1 = synthesize_ctf(case["paths"], freq)
    h_dec_1 = synthesize_ctf_from_decomposed(case["paths"], freq)

    tmp = Path(path).with_suffix(".roundtrip.h5")
    save_rt_dataset(tmp, original)
    loaded = load_rt_dataset(tmp)
    case2 = loaded["scenarios"][scenario_id]["cases"][case_id]

    h_af_2 = synthesize_ctf(case2["paths"], freq)
    h_dec_2 = synthesize_ctf_from_decomposed(case2["paths"], freq)

    return bool(
        np.allclose(h_af_1, h_af_2, atol=atol)
        and np.allclose(h_dec_1, h_dec_2, atol=atol)
        and np.allclose(h_af_1, h_dec_1, atol=atol)
        and np.allclose(h_af_2, h_dec_2, atol=atol)
    )

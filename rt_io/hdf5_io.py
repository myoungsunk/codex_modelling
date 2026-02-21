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
import subprocess
from typing import Any

import h5py
import numpy as np

from analysis.ctf_cir import synthesize_ctf


SCHEMA_VERSION = "v2"
V2_REQUIRED_META_ATTRS = (
    "schema_version",
    "git_commit",
    "git_dirty",
    "cmdline",
    "seed_json",
    "basis",
    "convention",
    "xpd_matrix_source",
    "exact_bounce_defaults_json",
    "physics_validation_mode",
    "antenna_config_json",
    "release_mode",
)
META_COMPARE_KEYS = (
    "schema_version",
    "git_commit",
    "git_dirty",
    "cmdline",
    "basis",
    "convention",
    "xpd_matrix_source",
    "exact_bounce_defaults",
    "physics_validation_mode",
    "antenna_config",
    "release_mode",
    "seed_json",
)


def _git_meta() -> tuple[str, bool]:
    """Best-effort git commit/dirty metadata."""

    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
            .strip()
        )
    except Exception:
        commit = "unknown"
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True)
        dirty = bool(status.strip())
    except Exception:
        dirty = True
    return commit, dirty


def _git_diff(max_chars: int = 200_000) -> str:
    """Best-effort git diff text (optionally truncated)."""

    try:
        diff = subprocess.check_output(["git", "diff", "--no-color"], stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return ""
    if len(diff) > int(max_chars):
        keep = max(int(max_chars) - 64, 0)
        diff = diff[:keep] + "\n... [truncated]\n"
    return diff


def _write_string_array(group: h5py.Group, name: str, values: list[str]) -> None:
    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dt)


def _json_dumps_canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_loads_safe(text: str, fallback: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return fallback


def _stringify_seed(seed_val: Any) -> str:
    if isinstance(seed_val, str):
        return seed_val
    if seed_val is None:
        return ""
    if isinstance(seed_val, (dict, list, tuple, int, float, bool)):
        return _json_dumps_canonical(seed_val)
    return str(seed_val)


def _normalize_meta_for_compare(meta: dict[str, Any], default_schema_version: str = "v1") -> dict[str, Any]:
    out = dict(meta)
    out["schema_version"] = str(out.get("schema_version", default_schema_version))
    out["git_commit"] = str(out.get("git_commit", "unknown"))
    out["git_dirty"] = bool(out.get("git_dirty", True))
    out["cmdline"] = str(out.get("cmdline", ""))
    out["basis"] = str(out.get("basis", "linear"))
    out["convention"] = str(out.get("convention", "IEEE-RHCP"))
    out["xpd_matrix_source"] = str(out.get("xpd_matrix_source", out.get("matrix_source", "")))
    out["release_mode"] = bool(out.get("release_mode", False))
    out["physics_validation_mode"] = bool(
        out.get(
            "physics_validation_mode",
            not bool((out.get("antenna_config", {}) or {}).get("enable_coupling", True)),
        )
    )
    ac = out.get("antenna_config", {})
    if isinstance(ac, str):
        ac = _json_loads_safe(ac, {})
    out["antenna_config"] = ac if isinstance(ac, dict) else {}
    eb = out.get("exact_bounce_defaults", {})
    if isinstance(eb, str):
        eb = _json_loads_safe(eb, {})
    out["exact_bounce_defaults"] = eb if isinstance(eb, dict) else {}
    out["seed_json"] = str(out.get("seed_json", _stringify_seed(out.get("seed", ""))))
    return out


def _require_v2_meta_attrs(meta_g: h5py.Group) -> None:
    missing = [k for k in V2_REQUIRED_META_ATTRS if k not in meta_g.attrs]
    if missing:
        raise ValueError(
            "Missing required v2 meta attrs: " + ", ".join(missing)
        )


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
        src_meta = _normalize_meta_for_compare(data.get("meta", {}), default_schema_version=SCHEMA_VERSION)
        git_commit, git_dirty = _git_meta()
        meta.attrs["created_at"] = src_meta.get("created_at", now)
        meta.attrs["basis"] = src_meta.get("basis", "linear")
        meta.attrs["convention"] = src_meta.get("convention", "IEEE-RHCP")
        meta.attrs["xpd_matrix_source"] = src_meta.get("xpd_matrix_source", src_meta.get("matrix_source", ""))
        meta.attrs["matrix_source"] = src_meta.get("xpd_matrix_source", src_meta.get("matrix_source", ""))  # compat
        meta.attrs["schema_version"] = src_meta.get("schema_version", SCHEMA_VERSION)
        meta.attrs["git_commit"] = src_meta.get("git_commit", git_commit)
        meta.attrs["git_dirty"] = bool(src_meta.get("git_dirty", git_dirty))
        meta.attrs["cmdline"] = str(src_meta.get("cmdline", ""))
        seed_json = str(src_meta.get("seed_json", _stringify_seed(src_meta.get("seed", ""))))
        meta.attrs["seed_json"] = seed_json
        meta.attrs["seed"] = seed_json  # compat
        exact_bounce_defaults = src_meta.get("exact_bounce_defaults", {})
        antenna_cfg = src_meta.get("antenna_config", {})
        meta.attrs["exact_bounce_defaults_json"] = _json_dumps_canonical(exact_bounce_defaults)
        meta.attrs["antenna_config_json"] = _json_dumps_canonical(antenna_cfg)
        meta.attrs["physics_validation_mode"] = bool(src_meta.get("physics_validation_mode", False))
        meta.attrs["release_mode"] = bool(src_meta.get("release_mode", False))
        git_diff = src_meta.get("git_diff", None)
        if git_diff is None:
            git_diff = _git_diff()
        meta.attrs["git_diff"] = str(git_diff)

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
            schema_version = str(meta_g.attrs.get("schema_version", "v1"))
            if schema_version == SCHEMA_VERSION:
                _require_v2_meta_attrs(meta_g)
            seed_json = str(meta_g.attrs.get("seed_json", meta_g.attrs.get("seed", "")))
            exact_json = str(meta_g.attrs.get("exact_bounce_defaults_json", "{}"))
            antenna_json = str(meta_g.attrs.get("antenna_config_json", "{}"))
            out["meta"] = {
                "created_at": meta_g.attrs.get("created_at", ""),
                "basis": meta_g.attrs.get("basis", "linear"),
                "convention": meta_g.attrs.get("convention", "IEEE-RHCP"),
                "xpd_matrix_source": meta_g.attrs.get("xpd_matrix_source", meta_g.attrs.get("matrix_source", "")),
                "matrix_source": meta_g.attrs.get("xpd_matrix_source", meta_g.attrs.get("matrix_source", "")),
                "schema_version": schema_version,
                "git_commit": meta_g.attrs.get("git_commit", "unknown"),
                "git_dirty": bool(meta_g.attrs.get("git_dirty", True)),
                "cmdline": meta_g.attrs.get("cmdline", ""),
                "seed_json": seed_json,
                "seed": _json_loads_safe(seed_json, seed_json),
                "exact_bounce_defaults_json": exact_json,
                "exact_bounce_defaults": _json_loads_safe(exact_json, {}),
                "physics_validation_mode": bool(meta_g.attrs.get("physics_validation_mode", False)),
                "antenna_config_json": antenna_json,
                "antenna_config": _json_loads_safe(antenna_json, {}),
                "release_mode": bool(meta_g.attrs.get("release_mode", False)),
                "git_diff": meta_g.attrs.get("git_diff", ""),
            }
        else:
            out["meta"] = {
                "created_at": "",
                "basis": "linear",
                "convention": "IEEE-RHCP",
                "xpd_matrix_source": "",
                "matrix_source": "",
                "schema_version": "v1",
                "git_commit": "unknown",
                "git_dirty": True,
                "cmdline": "",
                "seed_json": "",
                "seed": "",
                "exact_bounce_defaults_json": "{}",
                "exact_bounce_defaults": {},
                "physics_validation_mode": False,
                "antenna_config_json": "{}",
                "antenna_config": {},
                "release_mode": False,
                "git_diff": "",
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


def self_test_meta_roundtrip(path: str | Path, expected_meta: dict[str, Any] | None = None) -> bool:
    """Validate HDF5 meta contract + array shapes after save/load.

    Checks:
    - v2 required attrs exist and are loadable
    - optional expected_meta matches loaded meta on stable keys
    - all path arrays have expected shapes
    """

    loaded = load_rt_dataset(path)
    meta_loaded = _normalize_meta_for_compare(loaded.get("meta", {}))
    if str(meta_loaded.get("schema_version", "")) == SCHEMA_VERSION:
        missing = [k for k in V2_REQUIRED_META_ATTRS if k not in meta_loaded and k != "seed_json"]
        if missing:
            return False
    if expected_meta is not None:
        exp = _normalize_meta_for_compare(expected_meta)
        for k in META_COMPARE_KEYS:
            if exp.get(k) != meta_loaded.get(k):
                return False

    freq = np.asarray(loaded.get("frequency", []), dtype=float)
    nf = len(freq)
    for sc in loaded.get("scenarios", {}).values():
        for case in sc.get("cases", {}).values():
            for p in case.get("paths", []):
                if np.asarray(p.get("A_f", np.zeros((0, 2, 2)))).shape != (nf, 2, 2):
                    return False
                if np.asarray(p.get("J_f", np.zeros((0, 2, 2)))).shape != (nf, 2, 2):
                    return False
                if np.asarray(p.get("G_tx_f", np.zeros((0, 2, 2)))).shape != (nf, 2, 2):
                    return False
                if np.asarray(p.get("G_rx_f", np.zeros((0, 2, 2)))).shape != (nf, 2, 2):
                    return False
                if np.asarray(p.get("scalar_gain_f", np.zeros((0,)))).shape != (nf,):
                    return False
    return True


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

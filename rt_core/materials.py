"""Material database loader and helpers for dispersive dielectric models.

Example:
    >>> from rt_core.materials import resolve_material_library
    >>> mats = resolve_material_library(None, material_dispersion="off")
    >>> sorted(mats.keys())[:2]
    ['glass', 'gypsum']
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rt_core.geometry import Material


DEFAULT_MATERIAL_SPECS: dict[str, dict[str, Any]] = {
    "glass": {"model": "const", "eps_r": 6.5, "tan_delta": 0.005},
    "wood": {"model": "const", "eps_r": 2.2, "tan_delta": 0.03},
    "gypsum": {"model": "const", "eps_r": 2.8, "tan_delta": 0.02},
}


def materials_db_hash(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_material_specs(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"materials db not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("YAML materials db requires PyYAML; use JSON or install PyYAML") from exc
        obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"unsupported materials db extension: {suffix}")

    if isinstance(obj, dict) and "materials" in obj and isinstance(obj["materials"], dict):
        obj = obj["materials"]
    if not isinstance(obj, dict):
        raise ValueError("materials db must be a mapping of material_name -> spec")
    out: dict[str, dict[str, Any]] = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            out[str(k)] = dict(v)
    return out


def _table_to_const(spec: dict[str, Any], fallback_tan: float = 0.0) -> tuple[float, float]:
    eps = np.asarray(spec.get("eps_r", []), dtype=float)
    tan = np.asarray(spec.get("tan_delta", []), dtype=float)
    if eps.size == 0:
        return 2.0, float(max(fallback_tan, 0.0))
    eps_c = float(max(np.nanmean(eps), 1.0))
    if tan.size == 0:
        tan_c = float(max(fallback_tan, 0.0))
    else:
        tan_c = float(max(np.nanmean(tan), 0.0))
    return eps_c, tan_c


def _debye_to_const(spec: dict[str, Any], fallback_tan: float = 0.0) -> tuple[float, float]:
    eps_inf = float(spec.get("eps_inf", 1.0))
    de = np.asarray(spec.get("delta_eps", []), dtype=float)
    eps = float(max(eps_inf + float(np.nansum(de)), 1.0))
    tan = float(max(float(spec.get("tan_delta", fallback_tan)), 0.0))
    return eps, tan


def material_from_spec(name: str, spec: dict[str, Any], material_dispersion: str = "off") -> Material:
    mode = str(material_dispersion).lower()
    model = str(spec.get("model", "const")).lower()
    xpol_db_raw = spec.get("xpol_coupling_db", None)
    xpol_db = (None if xpol_db_raw is None else float(xpol_db_raw))
    xpol_ph = float(spec.get("xpol_coupling_phase_deg", 0.0))
    xpol_hv_db_raw = spec.get("xpol_coupling_hv_db", None)
    xpol_hv_db = (None if xpol_hv_db_raw is None else float(xpol_hv_db_raw))
    xpol_hv_ph = float(spec.get("xpol_coupling_hv_phase_deg", 0.0))
    xpol_vh_db_raw = spec.get("xpol_coupling_vh_db", None)
    xpol_vh_db = (None if xpol_vh_db_raw is None else float(xpol_vh_db_raw))
    xpol_vh_ph = float(spec.get("xpol_coupling_vh_phase_deg", 0.0))
    pec_tm_sign = float(spec.get("pec_tm_sign", -1.0))

    if model == "pec":
        return Material.pec(
            xpol_coupling_db=xpol_db,
            xpol_coupling_phase_deg=xpol_ph,
            xpol_coupling_hv_db=xpol_hv_db,
            xpol_coupling_hv_phase_deg=xpol_hv_ph,
            xpol_coupling_vh_db=xpol_vh_db,
            xpol_coupling_vh_phase_deg=xpol_vh_ph,
            pec_tm_sign=pec_tm_sign,
        )

    if mode == "off":
        if model == "table":
            eps_r, tan_delta = _table_to_const(spec, fallback_tan=float(spec.get("tan_delta_const", 0.0)))
            return Material.dielectric(
                eps_r=eps_r,
                tan_delta=tan_delta,
                name=name,
                xpol_coupling_db=xpol_db,
                xpol_coupling_phase_deg=xpol_ph,
                xpol_coupling_hv_db=xpol_hv_db,
                xpol_coupling_hv_phase_deg=xpol_hv_ph,
                xpol_coupling_vh_db=xpol_vh_db,
                xpol_coupling_vh_phase_deg=xpol_vh_ph,
            )
        if model == "debye":
            eps_r, tan_delta = _debye_to_const(spec)
            return Material.dielectric(
                eps_r=eps_r,
                tan_delta=tan_delta,
                name=name,
                xpol_coupling_db=xpol_db,
                xpol_coupling_phase_deg=xpol_ph,
                xpol_coupling_hv_db=xpol_hv_db,
                xpol_coupling_hv_phase_deg=xpol_hv_ph,
                xpol_coupling_vh_db=xpol_vh_db,
                xpol_coupling_vh_phase_deg=xpol_vh_ph,
            )
        return Material.dielectric(
            eps_r=float(max(float(spec.get("eps_r", 2.0)), 1.0)),
            tan_delta=float(max(float(spec.get("tan_delta", 0.0)), 0.0)),
            name=name,
            xpol_coupling_db=xpol_db,
            xpol_coupling_phase_deg=xpol_ph,
            xpol_coupling_hv_db=xpol_hv_db,
            xpol_coupling_hv_phase_deg=xpol_hv_ph,
            xpol_coupling_vh_db=xpol_vh_db,
            xpol_coupling_vh_phase_deg=xpol_vh_ph,
        )

    # dispersion ON/DEBYE mode: keep spec model if available.
    if model == "table":
        return Material.dielectric_table(
            f_hz=[float(x) for x in spec.get("f_hz", [])],
            eps_r=[float(x) for x in spec.get("eps_r", [])],
            tan_delta=[float(x) for x in spec.get("tan_delta", [])] if "tan_delta" in spec else None,
            name=name,
            xpol_coupling_db=xpol_db,
            xpol_coupling_phase_deg=xpol_ph,
            xpol_coupling_hv_db=xpol_hv_db,
            xpol_coupling_hv_phase_deg=xpol_hv_ph,
            xpol_coupling_vh_db=xpol_vh_db,
            xpol_coupling_vh_phase_deg=xpol_vh_ph,
        )
    if model == "debye":
        return Material.dielectric_debye(
            eps_inf=float(spec.get("eps_inf", 1.0)),
            delta_eps=[float(x) for x in spec.get("delta_eps", [])],
            tau_s=[float(x) for x in spec.get("tau_s", [])],
            tan_delta=float(spec.get("tan_delta", 0.0)),
            name=name,
            xpol_coupling_db=xpol_db,
            xpol_coupling_phase_deg=xpol_ph,
            xpol_coupling_hv_db=xpol_hv_db,
            xpol_coupling_hv_phase_deg=xpol_hv_ph,
            xpol_coupling_vh_db=xpol_vh_db,
            xpol_coupling_vh_phase_deg=xpol_vh_ph,
        )
    return Material.dielectric(
        eps_r=float(max(float(spec.get("eps_r", 2.0)), 1.0)),
        tan_delta=float(max(float(spec.get("tan_delta", 0.0)), 0.0)),
        name=name,
        xpol_coupling_db=xpol_db,
        xpol_coupling_phase_deg=xpol_ph,
        xpol_coupling_hv_db=xpol_hv_db,
        xpol_coupling_hv_phase_deg=xpol_hv_ph,
        xpol_coupling_vh_db=xpol_vh_db,
        xpol_coupling_vh_phase_deg=xpol_vh_ph,
    )


def resolve_material_library(
    materials_db_path: str | Path | None,
    material_dispersion: str = "off",
    default_specs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Material]:
    specs = dict(default_specs or DEFAULT_MATERIAL_SPECS)
    specs.update(load_material_specs(materials_db_path))
    out: dict[str, Material] = {}
    for name, spec in specs.items():
        out[str(name)] = material_from_spec(str(name), spec, material_dispersion=material_dispersion)
    return out

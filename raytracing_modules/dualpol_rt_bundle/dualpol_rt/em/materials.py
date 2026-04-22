from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


EPS0 = 8.8541878128e-12
DEFAULT_XPOL_COUPLING_DB = 35.0
DEFAULT_XPOL_COUPLING_SWEEP_DB = (20.0, 25.0, 30.0, 35.0, 40.0)


@dataclass(frozen=True)
class Material:
    name: str
    kind: str = "dielectric"
    eps_r_const: float = 1.0
    sigma_const: float = 0.0
    tan_delta_const: float | None = None
    eps_r_power_a: float | None = None
    eps_r_power_b: float | None = None
    sigma_power_c: float | None = None
    sigma_power_d: float | None = None
    thickness_m: float | None = None
    xpol_coupling_db: float = DEFAULT_XPOL_COUPLING_DB
    pec_tm_sign: float = -1.0

    def __post_init__(self) -> None:
        validate_material(self)

    def eps_r(self, freqs_hz: np.ndarray) -> np.ndarray:
        if self.eps_r_power_a is not None and self.eps_r_power_b is not None:
            freqs_ghz = np.maximum(np.asarray(freqs_hz, dtype=float) / 1.0e9, 1.0e-12)
            return (float(self.eps_r_power_a) * (freqs_ghz ** float(self.eps_r_power_b))).astype(float)
        return np.full(np.asarray(freqs_hz, dtype=float).shape, float(self.eps_r_const), dtype=float)

    def sigma(self, freqs_hz: np.ndarray) -> np.ndarray:
        if self.sigma_power_c is not None and self.sigma_power_d is not None:
            freqs_ghz = np.maximum(np.asarray(freqs_hz, dtype=float) / 1.0e9, 1.0e-12)
            return (float(self.sigma_power_c) * (freqs_ghz ** float(self.sigma_power_d))).astype(float)
        if self.tan_delta_const is not None:
            freqs = np.asarray(freqs_hz, dtype=float)
            omega = 2.0 * np.pi * np.maximum(freqs, 1.0)
            return (omega * EPS0 * self.eps_r(freqs) * float(self.tan_delta_const)).astype(float)
        return np.full(np.asarray(freqs_hz, dtype=float).shape, float(self.sigma_const), dtype=float)

    @staticmethod
    def pec(
        name: str = "pec_debug",
        *,
        xpol_coupling_db: float = 45.0,
        pec_tm_sign: float = -1.0,
    ) -> "Material":
        return Material(
            name=name,
            kind="PEC",
            eps_r_const=1.0,
            sigma_const=0.0,
            xpol_coupling_db=xpol_coupling_db,
            pec_tm_sign=pec_tm_sign,
        )

    @staticmethod
    def itu_p2040(
        name: str,
        a: float,
        b: float,
        c: float,
        d: float,
        *,
        kind: str = "dielectric",
        thickness_m: float | None = None,
        xpol_coupling_db: float = DEFAULT_XPOL_COUPLING_DB,
        pec_tm_sign: float = -1.0,
    ) -> "Material":
        return Material(
            name=name,
            kind=kind,
            eps_r_const=float(a),
            sigma_const=float(c),
            eps_r_power_a=float(a),
            eps_r_power_b=float(b),
            sigma_power_c=float(c),
            sigma_power_d=float(d),
            thickness_m=thickness_m,
            xpol_coupling_db=float(xpol_coupling_db),
            pec_tm_sign=float(pec_tm_sign),
        )

    def with_xpol_coupling_db(self, xpol_coupling_db: float) -> "Material":
        return replace(self, xpol_coupling_db=float(xpol_coupling_db))


def validate_material(mat: Material) -> None:
    if not np.isfinite(float(mat.xpol_coupling_db)):
        raise ValueError(f"{mat.name}: xpol_coupling_db must be finite")
    if float(mat.xpol_coupling_db) < 0.0:
        raise ValueError(f"{mat.name}: xpol_coupling_db must be non-negative")
    if str(mat.kind).upper() == "PEC":
        assert float(mat.pec_tm_sign) == -1.0, "PEC TM sign must be -1.0 (IEEE convention)"


from dualpol_rt.em.materials_db import MATERIAL_PRESETS, MATERIALS


def material_named(name: str, xpol_coupling_db: float | None = None) -> Material:
    try:
        material = MATERIALS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown material name: {name}") from exc
    return material if xpol_coupling_db is None else material.with_xpol_coupling_db(xpol_coupling_db)

def material_bundle(name: str, xpol_coupling_db: float | None = None) -> dict[str, Material]:
    try:
        bundle = MATERIAL_PRESETS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown material preset: {name}") from exc
    if xpol_coupling_db is None:
        return dict(bundle)
    return {key: material.with_xpol_coupling_db(xpol_coupling_db) for key, material in bundle.items()}


def material_preset(name: str, xpol_coupling_db: float | None = None) -> Material:
    bundle = material_bundle(name, xpol_coupling_db=xpol_coupling_db)
    return bundle["walls"]

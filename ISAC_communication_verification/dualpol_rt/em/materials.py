from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    kind: str = "dielectric"
    eps_r_const: float = 1.0
    sigma_const: float = 0.0
    eps_r_power_a: float | None = None
    eps_r_power_b: float | None = None
    sigma_power_c: float | None = None
    sigma_power_d: float | None = None
    thickness_m: float | None = None

    def eps_r(self, freqs_hz: np.ndarray) -> np.ndarray:
        if self.eps_r_power_a is not None and self.eps_r_power_b is not None:
            freqs_ghz = np.maximum(np.asarray(freqs_hz, dtype=float) / 1.0e9, 1.0e-12)
            return (float(self.eps_r_power_a) * (freqs_ghz ** float(self.eps_r_power_b))).astype(float)
        return np.full(np.asarray(freqs_hz, dtype=float).shape, float(self.eps_r_const), dtype=float)

    def sigma(self, freqs_hz: np.ndarray) -> np.ndarray:
        if self.sigma_power_c is not None and self.sigma_power_d is not None:
            freqs_ghz = np.maximum(np.asarray(freqs_hz, dtype=float) / 1.0e9, 1.0e-12)
            return (float(self.sigma_power_c) * (freqs_ghz ** float(self.sigma_power_d))).astype(float)
        return np.full(np.asarray(freqs_hz, dtype=float).shape, float(self.sigma_const), dtype=float)

    @staticmethod
    def pec(name: str = "pec_debug") -> "Material":
        return Material(name=name, kind="pec", eps_r_const=1.0, sigma_const=0.0)

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
        )


_MATERIAL_LIBRARY = {
    "pec_debug": Material.pec("pec_debug"),
    "concrete": Material.itu_p2040(name="concrete", a=5.24, b=0.0, c=0.0462, d=0.7822),
    "plasterboard": Material.itu_p2040(name="plasterboard", a=2.73, b=0.0, c=0.0085, d=0.9395),
    "wood": Material.itu_p2040(name="wood", a=1.99, b=0.0, c=0.0047, d=1.0718),
    "metal": Material.pec("metal"),
    "human_blocker_proxy": Material(name="human_blocker_proxy", eps_r_const=38.0, sigma_const=1.5),
}


def material_named(name: str) -> Material:
    try:
        return _MATERIAL_LIBRARY[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown material name: {name}") from exc


_MATERIAL_DB = {
    "pec_debug": {
        "walls": material_named("pec_debug"),
        "floor": material_named("pec_debug"),
        "ceiling": material_named("pec_debug"),
    },
    "uniform_concrete": {
        "walls": material_named("concrete"),
        "floor": material_named("concrete"),
        "ceiling": material_named("concrete"),
    },
    "indoor_mixed": {
        "walls": material_named("plasterboard"),
        "floor": material_named("concrete"),
        "ceiling": material_named("plasterboard"),
    },
    "indoor_open_office": {
        "walls": material_named("plasterboard"),
        "floor": material_named("concrete"),
        "ceiling": material_named("plasterboard"),
    },
    "corridor_concrete_metal": {
        "walls": material_named("concrete"),
        "floor": material_named("concrete"),
        "ceiling": material_named("concrete"),
    },
    "factory_shell": {
        "walls": material_named("concrete"),
        "floor": material_named("concrete"),
        "ceiling": material_named("metal"),
    },
}


def material_bundle(name: str) -> dict[str, Material]:
    try:
        bundle = _MATERIAL_DB[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown material preset: {name}") from exc
    return dict(bundle)


def material_preset(name: str) -> Material:
    bundle = material_bundle(name)
    return bundle["walls"]

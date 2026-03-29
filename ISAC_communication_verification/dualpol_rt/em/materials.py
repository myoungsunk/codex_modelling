from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    kind: str = "dielectric"
    eps_r_const: float = 1.0
    sigma_const: float = 0.0
    thickness_m: float | None = None

    def eps_r(self, freqs_hz: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(freqs_hz, dtype=float).shape, float(self.eps_r_const), dtype=float)

    def sigma(self, freqs_hz: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(freqs_hz, dtype=float).shape, float(self.sigma_const), dtype=float)

    @staticmethod
    def pec(name: str = "pec_debug") -> "Material":
        return Material(name=name, kind="pec", eps_r_const=1.0, sigma_const=0.0)


_MATERIAL_LIBRARY = {
    "pec_debug": Material.pec("pec_debug"),
    "concrete": Material(name="concrete", eps_r_const=5.31, sigma_const=0.0326),
    "plasterboard": Material(name="plasterboard", eps_r_const=2.73, sigma_const=0.0085),
    "wood": Material(name="wood", eps_r_const=1.99, sigma_const=0.0047),
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

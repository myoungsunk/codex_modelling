"""Dataclasses for standard output schema v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


SCHEMA_VERSION = "standard_outputs_v1"


def _to_list(x: Any, dtype: Any = float) -> list[Any]:
    arr = np.asarray(x, dtype=dtype)
    return arr.tolist()


@dataclass
class RayTable:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"rows": [dict(r) for r in self.rows]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RayTable":
        return cls(rows=[dict(x) for x in data.get("rows", [])])


@dataclass
class DualCPPDP:
    delay_tau_s: np.ndarray
    P_co: np.ndarray
    P_cross: np.ndarray
    XPD_tau_db: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "delay_tau_s": _to_list(self.delay_tau_s, float),
            "P_co": _to_list(self.P_co, float),
            "P_cross": _to_list(self.P_cross, float),
        }
        if self.XPD_tau_db is not None:
            out["XPD_tau_db"] = _to_list(self.XPD_tau_db, float)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DualCPPDP":
        xpd = data.get("XPD_tau_db")
        return cls(
            delay_tau_s=np.asarray(data.get("delay_tau_s", []), dtype=float),
            P_co=np.asarray(data.get("P_co", []), dtype=float),
            P_cross=np.asarray(data.get("P_cross", []), dtype=float),
            XPD_tau_db=np.asarray(xpd, dtype=float) if xpd is not None else None,
        )


@dataclass
class LinkMetricsZ:
    XPD_early_db: float
    XPD_late_db: float
    rho_early_lin: float
    rho_early_db: float
    L_pol_db: float
    delay_spread_rms_s: float
    early_energy_fraction: float
    window: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "XPD_early_db": float(self.XPD_early_db),
            "XPD_late_db": float(self.XPD_late_db),
            "rho_early_lin": float(self.rho_early_lin),
            "rho_early_db": float(self.rho_early_db),
            "L_pol_db": float(self.L_pol_db),
            "delay_spread_rms_s": float(self.delay_spread_rms_s),
            "early_energy_fraction": float(self.early_energy_fraction),
            "window": dict(self.window),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinkMetricsZ":
        return cls(
            XPD_early_db=float(data.get("XPD_early_db", np.nan)),
            XPD_late_db=float(data.get("XPD_late_db", np.nan)),
            rho_early_lin=float(data.get("rho_early_lin", np.nan)),
            rho_early_db=float(data.get("rho_early_db", np.nan)),
            L_pol_db=float(data.get("L_pol_db", np.nan)),
            delay_spread_rms_s=float(data.get("delay_spread_rms_s", np.nan)),
            early_energy_fraction=float(data.get("early_energy_fraction", np.nan)),
            window=dict(data.get("window", {})),
        )


@dataclass
class LinkConditionsU:
    d_m: float
    LOSflag: int
    EL_proxy_db: float
    material_class: str = "NA"
    roughness_flag: int = 0
    human_flag: int = 0
    obstacle_flag: int = 0
    dominant_parity_early: str = "NA"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "d_m": float(self.d_m),
            "LOSflag": int(self.LOSflag),
            "EL_proxy_db": float(self.EL_proxy_db),
            "material_class": str(self.material_class),
            "roughness_flag": int(self.roughness_flag),
            "human_flag": int(self.human_flag),
            "obstacle_flag": int(self.obstacle_flag),
            "dominant_parity_early": str(self.dominant_parity_early),
        }
        out.update(dict(self.extras))
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinkConditionsU":
        known = {
            "d_m",
            "LOSflag",
            "EL_proxy_db",
            "material_class",
            "roughness_flag",
            "human_flag",
            "obstacle_flag",
            "dominant_parity_early",
        }
        extras = {k: v for k, v in data.items() if k not in known}
        return cls(
            d_m=float(data.get("d_m", np.nan)),
            LOSflag=int(data.get("LOSflag", 0)),
            EL_proxy_db=float(data.get("EL_proxy_db", np.nan)),
            material_class=str(data.get("material_class", "NA")),
            roughness_flag=int(data.get("roughness_flag", 0)),
            human_flag=int(data.get("human_flag", 0)),
            obstacle_flag=int(data.get("obstacle_flag", 0)),
            dominant_parity_early=str(data.get("dominant_parity_early", "NA")),
            extras=extras,
        )


@dataclass
class StandardOutputBundle:
    link_id: str
    scenario_id: str
    case_id: str
    rays: RayTable
    pdp: DualCPPDP
    metrics: LinkMetricsZ
    conditions: LinkConditionsU
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "link_id": str(self.link_id),
            "scenario_id": str(self.scenario_id),
            "case_id": str(self.case_id),
            "rays": self.rays.to_dict(),
            "pdp": self.pdp.to_dict(),
            "metrics": self.metrics.to_dict(),
            "conditions": self.conditions.to_dict(),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StandardOutputBundle":
        return cls(
            link_id=str(data.get("link_id", "")),
            scenario_id=str(data.get("scenario_id", "")),
            case_id=str(data.get("case_id", "")),
            rays=RayTable.from_dict(dict(data.get("rays", {}))),
            pdp=DualCPPDP.from_dict(dict(data.get("pdp", {}))),
            metrics=LinkMetricsZ.from_dict(dict(data.get("metrics", {}))),
            conditions=LinkConditionsU.from_dict(dict(data.get("conditions", {}))),
            provenance=dict(data.get("provenance", {})),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        )

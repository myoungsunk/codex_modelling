"""XPR proxy models used for dual-CP PDP synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class BaseXPRModel:
    def sample_xpr_db(self, ray_row: dict[str, Any], link_U: dict[str, Any], rng: np.random.Generator) -> float:
        raise NotImplementedError


@dataclass
class ConstantXPR(BaseXPRModel):
    xpr_db: float = 10.0

    def sample_xpr_db(self, ray_row: dict[str, Any], link_U: dict[str, Any], rng: np.random.Generator) -> float:
        _ = ray_row, link_U, rng
        return float(self.xpr_db)


@dataclass
class ConditionalLinearXPR(BaseXPRModel):
    a0: float = 10.0
    a_el: float = 0.0
    a_late: float = 0.0
    a_incidence: float = 0.0
    a_rough: float = -2.0
    a_human: float = -3.0
    sigma_db: float = 0.0
    material_bias: dict[str, float] = field(default_factory=dict)

    def sample_xpr_db(self, ray_row: dict[str, Any], link_U: dict[str, Any], rng: np.random.Generator) -> float:
        tau = float(ray_row.get("tau_s", np.nan))
        tau0 = float(link_U.get("tau0_s", tau))
        Te_s = float(link_U.get("Te_s", 0.0))
        late = 1.0 if np.isfinite(tau) and np.isfinite(tau0) and tau > (tau0 + Te_s) else 0.0
        el = float(ray_row.get("EL_db", link_U.get("EL_proxy_db", 0.0)))
        inc = float(ray_row.get("incidence_deg", 0.0))
        mat = str(ray_row.get("material_class", link_U.get("material_class", "NA")))
        m_bias = float(self.material_bias.get(mat, 0.0))
        rough = float(link_U.get("roughness_flag", 0))
        human = float(link_U.get("human_flag", 0))

        mu = (
            float(self.a0)
            + float(self.a_el) * el
            + float(self.a_late) * late
            + float(self.a_incidence) * inc
            + float(self.a_rough) * rough
            + float(self.a_human) * human
            + m_bias
        )
        if float(self.sigma_db) > 0.0:
            return float(rng.normal(mu, float(self.sigma_db)))
        return float(mu)


@dataclass
class BinnedXPR(BaseXPRModel):
    """Lookup/binned XPR model.

    `bins` entries:
      {
        "el_min": -inf, "el_max": inf,
        "material": "NA" or "*",
        "late": 0/1/"*",
        "xpr_db": 10.0, "sigma_db": 0.0
      }
    """

    default_xpr_db: float = 10.0
    bins: list[dict[str, Any]] = field(default_factory=list)

    def sample_xpr_db(self, ray_row: dict[str, Any], link_U: dict[str, Any], rng: np.random.Generator) -> float:
        el = float(ray_row.get("EL_db", link_U.get("EL_proxy_db", 0.0)))
        mat = str(ray_row.get("material_class", link_U.get("material_class", "NA")))
        tau = float(ray_row.get("tau_s", np.nan))
        tau0 = float(link_U.get("tau0_s", tau))
        Te_s = float(link_U.get("Te_s", 0.0))
        late = 1 if np.isfinite(tau) and np.isfinite(tau0) and tau > (tau0 + Te_s) else 0

        for b in self.bins:
            lo = float(b.get("el_min", -np.inf))
            hi = float(b.get("el_max", np.inf))
            bmat = str(b.get("material", "*"))
            blate = b.get("late", "*")
            if not (lo <= el < hi):
                continue
            if bmat not in {"*", mat}:
                continue
            if str(blate) not in {"*", str(late)}:
                continue
            mu = float(b.get("xpr_db", self.default_xpr_db))
            sig = float(b.get("sigma_db", 0.0))
            return float(rng.normal(mu, sig)) if sig > 0.0 else mu
        return float(self.default_xpr_db)

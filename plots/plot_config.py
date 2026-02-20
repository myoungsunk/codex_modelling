"""Plot configuration for interpretation-safe validation figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PlotConfig:
    scenario_id: str | None = None
    case_id: str | None = None
    top_k_paths: int = 5
    matrix_source: Literal["A", "J"] = "A"
    apply_exact_bounce: bool = True
    cp_eval_basis: Literal["circular"] = "circular"
    convention: str = "IEEE-RHCP"
    tau_plot_max_ns: float | None = None
    power_floor_db: float = -120.0

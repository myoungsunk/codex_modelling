"""Markdown report builder helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_SCENARIO_MEANING: dict[str, str] = {
    "C0": "Calibration-only baseline: floor reference and alignment sensitivity (W_floor).",
    "A2": "Odd parity isolation anchor (A2_off core evidence); A2_on is observability bridge set.",
    "A2_ON": "LOS-on bridge observability counterpart of A2_off; direct LOS + odd-path coexistence check.",
    "A3": "A3_supp supplementary mechanism scene (target-window evidence only, not system early baseline); A3_on is LOS-on observability bridge set.",
    "A3_ON": "LOS-on bridge observability counterpart of A3_supp; direct LOS + even-path coexistence check.",
    "A4": "Material branch: A4_iso primary (late_panel=off, dispersion=off), A4_bridge secondary (late_panel=on, dispersion=on); A4_on is LOS-on observability bridge set.",
    "A4_ON": "LOS-on bridge observability counterpart of A4 material branch; direct LOS under material contrast.",
    "A5": "A5_pair contamination-response pair (base/on): use for paired contamination/stress-response only.",
    "A6": "Near-normal PEC parity benchmark; primary G2 sign evidence.",
    "A6_ON": "LOS-on bridge observability counterpart of A6 parity benchmark; direct LOS + odd/even coexistence check.",
    "B1": "Room grid LOS anchor for coverage-aware leverage mapping (viable strata only; not universal).",
    "B2": "Room grid with partition obstacle for coverage-aware NLOS leverage mapping (structural-hole aware).",
    "B3": "Room grid corner-obstacle stress region for coverage-aware leverage mapping (structural-hole aware).",
}


def scenario_meaning(scenario_id: Any) -> str:
    sid = str(scenario_id).upper().strip()
    return _SCENARIO_MEANING.get(sid, "Scenario-specific control/sweep for proxy Z/U validation.")


def final_structure_rows() -> list[dict[str, str]]:
    """Agreed final scenario structure for reporting and review."""
    return [
        {"unit": "C0", "role": "calibration only", "notes": "floor_reference 강화"},
        {"unit": "A2_off", "role": "G1 primary evidence", "notes": "odd isolation, keep fixed"},
        {"unit": "A6", "role": "G2 primary evidence", "notes": "near-normal PEC, incidence <= 15 deg"},
        {"unit": "A3_supp", "role": "supplementary mechanism", "notes": "mechanism-only scope; WARN is role lock, no sign-off"},
        {"unit": "A4_iso", "role": "L2-M primary", "notes": "late_panel=false, dispersion=off"},
        {"unit": "A4_bridge", "role": "L2-M secondary support", "notes": "bridge/support scope; WARN is role lock, not weakness"},
        {"unit": "A5_pair", "role": "L2-S contamination-response", "notes": "paired base/on contamination-response only"},
        {"unit": "A2_on/A3_on/A4_on/A6_on", "role": "bridge observability set", "notes": "LOS-on contrast bridge"},
        {"unit": "B1/B2/B3", "role": "R1/R2 coverage-aware leverage map", "notes": "viable strata/support count required; no universal claim"},
    ]


def relpath(target: str | Path, base_dir: str | Path) -> str:
    t = Path(target).resolve()
    b = Path(base_dir).resolve()
    try:
        return str(t.relative_to(b))
    except Exception:
        return str(t)


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [head, sep]
    for r in rows:
        vals = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                if v != v:
                    vals.append("nan")
                else:
                    vals.append(f"{v:.4g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_json(path: str | Path, obj: Any) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return str(p)


def write_text(path: str | Path, text: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return str(p)

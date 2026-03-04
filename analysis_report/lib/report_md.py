"""Markdown report builder helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_SCENARIO_MEANING: dict[str, str] = {
    "C0": "Free-space LOS calibration baseline for floor/alignment uncertainty.",
    "A2": "LOS-blocked single-bounce(odd) control to probe early cross-leakage increase.",
    "A3": "LOS-blocked double-bounce(even) control to test co-dominant recovery trend.",
    "A4": "LOS-blocked material/angle sweep to isolate material-conditional leakage statistics.",
    "A5": "LOS-blocked stress scenario: geometric/hybrid => delay-path response, synthetic => polarization-only stress.",
    "B1": "Room grid baseline (mostly LOS) for spatial Z/U trend mapping.",
    "B2": "Room grid with partition obstacle to induce partial NLOS/blocked regions.",
    "B3": "Room grid with corner obstacles for stronger NLOS and multipath complexity.",
}


def scenario_meaning(scenario_id: Any) -> str:
    sid = str(scenario_id).upper().strip()
    return _SCENARIO_MEANING.get(sid, "Scenario-specific control/sweep for proxy Z/U validation.")


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

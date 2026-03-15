"""Index builder for report artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


_INDEX_COLUMNS = [
    "scenario_id",
    "case_id",
    "case_label",
    "input_run_dir",
    "link_metrics_csv",
    "rays_csv",
    "scene_debug_json",
    "scene_png_path",
    "key_plots",
    "report_refs",
]


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("scenario_id", "NA")), str(row.get("case_id", ""))


def _row_norm(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if isinstance(out.get("key_plots"), (list, tuple)):
        out["key_plots"] = json.dumps(list(out["key_plots"]))
    if isinstance(out.get("report_refs"), (list, tuple, dict)):
        out["report_refs"] = json.dumps(out["report_refs"])
    for c in _INDEX_COLUMNS:
        out.setdefault(c, "")
    return out


def load_index(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def save_index(path: str | Path, rows: list[dict[str, Any]]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_INDEX_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow(_row_norm(r))
    return str(p)


def update_index(path: str | Path, new_rows: list[dict[str, Any]]) -> str:
    cur = load_index(path)
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for r in cur:
        merged[_key(r)] = dict(r)
    for r in new_rows:
        k = _key(r)
        merged[k] = {**merged.get(k, {}), **dict(r)}
    rows = [merged[k] for k in sorted(merged.keys())]
    return save_index(path, rows)


def write_index_md(path: str | Path, rows: list[dict[str, Any]]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Report Index", "", "| scenario | case | label | scene |", "|---|---|---|---|"]
    for r in rows:
        scen = str(r.get("scenario_id", "NA"))
        cid = str(r.get("case_id", ""))
        label = str(r.get("case_label", ""))
        scene_png = str(r.get("scene_png_path", ""))
        if scene_png:
            lines.append(f"| {scen} | {cid} | {label} | [{Path(scene_png).name}]({scene_png}) |")
        else:
            lines.append(f"| {scen} | {cid} | {label} |  |")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)

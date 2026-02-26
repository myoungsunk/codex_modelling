"""Markdown report builder helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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

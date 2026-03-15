"""Helpers to compare circular-CP and linear-basis baseline outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_KEYS = ("scenario_id", "case_id")
DEFAULT_METRICS = (
    "XPD_early_db",
    "XPD_late_db",
    "rho_early_db",
    "L_pol_db",
    "delay_spread_rms_s",
    "early_energy_fraction",
    "XPD_early_excess_db",
    "XPD_late_excess_db",
)


def _maybe_num(v: str) -> Any:
    s = str(v).strip()
    if s == "":
        return ""
    try:
        x = float(s)
        return int(x) if x.is_integer() else x
    except Exception:
        return s


def _load_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        return [{k: _maybe_num(v) for k, v in r.items()} for r in rd]


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()}) if rows else []
    with p.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            out = {}
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, (dict, list, tuple)):
                    out[k] = json.dumps(v, ensure_ascii=True, sort_keys=True)
                else:
                    out[k] = v
            wr.writerow(out)


def _num(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def compare_cp_lp_metrics(
    cp_metrics_csv: str | Path,
    lp_metrics_csv: str | Path,
    out_pairs_csv: str | Path | None = None,
    out_report_md: str | Path | None = None,
    key_cols: tuple[str, str] = DEFAULT_KEYS,
    metric_cols: tuple[str, ...] = DEFAULT_METRICS,
) -> dict[str, Any]:
    cp_rows = _load_csv(cp_metrics_csv)
    lp_rows = _load_csv(lp_metrics_csv)
    idx_lp: dict[tuple[str, str], dict[str, Any]] = {}
    for r in lp_rows:
        idx_lp[(str(r.get(key_cols[0], "")), str(r.get(key_cols[1], "")))] = r

    pairs: list[dict[str, Any]] = []
    for rc in cp_rows:
        key = (str(rc.get(key_cols[0], "")), str(rc.get(key_cols[1], "")))
        rl = idx_lp.get(key)
        if rl is None:
            continue
        row: dict[str, Any] = {
            key_cols[0]: key[0],
            key_cols[1]: key[1],
            "link_id_cp": str(rc.get("link_id", "")),
            "link_id_lp": str(rl.get("link_id", "")),
        }
        for m in metric_cols:
            vc = _num(rc.get(m, np.nan))
            vl = _num(rl.get(m, np.nan))
            row[f"cp_{m}"] = vc
            row[f"lp_{m}"] = vl
            row[f"delta_{m}_cp_minus_lp"] = float(vc - vl) if np.isfinite(vc) and np.isfinite(vl) else float("nan")
        pairs.append(row)

    summary: dict[str, Any] = {
        "n_cp": int(len(cp_rows)),
        "n_lp": int(len(lp_rows)),
        "n_pairs": int(len(pairs)),
        "metrics": {},
    }
    for m in metric_cols:
        d = np.asarray([_num(r.get(f"delta_{m}_cp_minus_lp", np.nan)) for r in pairs], dtype=float)
        d = d[np.isfinite(d)]
        summary["metrics"][m] = {
            "n": int(len(d)),
            "mean_delta_cp_minus_lp": float(np.mean(d)) if len(d) else float("nan"),
            "median_delta_cp_minus_lp": float(np.median(d)) if len(d) else float("nan"),
            "p10_delta_cp_minus_lp": float(np.percentile(d, 10.0)) if len(d) else float("nan"),
            "p90_delta_cp_minus_lp": float(np.percentile(d, 90.0)) if len(d) else float("nan"),
        }

    if out_pairs_csv is not None:
        _write_csv(out_pairs_csv, pairs)

    if out_report_md is not None:
        p = Path(out_report_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# CP vs LP Baseline Comparison", ""]
        lines.append(f"- cp_metrics_csv: {cp_metrics_csv}")
        lines.append(f"- lp_metrics_csv: {lp_metrics_csv}")
        lines.append(f"- n_cp: {summary['n_cp']}")
        lines.append(f"- n_lp: {summary['n_lp']}")
        lines.append(f"- n_pairs: {summary['n_pairs']}")
        lines.append("")
        for m in metric_cols:
            s = summary["metrics"].get(m, {})
            lines.append(f"## {m}")
            lines.append("")
            lines.append(f"- mean_delta_cp_minus_lp: {s.get('mean_delta_cp_minus_lp')}")
            lines.append(f"- median_delta_cp_minus_lp: {s.get('median_delta_cp_minus_lp')}")
            lines.append(f"- p10_delta_cp_minus_lp: {s.get('p10_delta_cp_minus_lp')}")
            lines.append(f"- p90_delta_cp_minus_lp: {s.get('p90_delta_cp_minus_lp')}")
            lines.append("")
        p.write_text("\n".join(lines), encoding="utf-8")

    return {"summary": summary, "pairs": pairs}


"""Select stratified measurement points from link-level outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.stratified_sampling import stratified_sample


def _maybe_num(v: str) -> Any:
    s = str(v).strip()
    if s == "":
        return ""
    try:
        x = float(s)
        return int(x) if x.is_integer() else x
    except Exception:
        return s


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
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
            wr.writerow({k: r.get(k, "") for k in keys})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--link-metrics-csv", required=True, type=str)
    parser.add_argument("--out-csv", default="outputs/selected_points.csv", type=str)
    parser.add_argument("--bins", type=str, default="EL_proxy_db:4,LOSflag:2")
    parser.add_argument("--per-bin", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = _read_csv(args.link_metrics_csv)
    bins: dict[str, int] = {}
    for tok in str(args.bins).split(","):
        t = tok.strip()
        if ":" not in t:
            continue
        k, v = t.split(":", 1)
        try:
            bins[k.strip()] = int(v.strip())
        except ValueError:
            continue
    sel = stratified_sample(rows, bins=bins, per_bin=int(args.per_bin), seed=int(args.seed))
    _write_csv(args.out_csv, sel)
    print(str(args.out_csv))


if __name__ == "__main__":
    main()

"""Export standard output bundles (v1) to HDF5/CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from rt_io.standard_outputs_hdf5 import export_csv, save_run
from rt_types.standard_outputs import StandardOutputBundle


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _load_bundles(path: str | Path) -> list[StandardOutputBundle]:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        arr = obj.get("bundles", [])
    elif isinstance(obj, list):
        arr = obj
    else:
        arr = []
    bundles = []
    for x in arr:
        if not isinstance(x, dict):
            continue
        bundles.append(StandardOutputBundle.from_dict(x))
    return bundles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True, type=str, help="JSON file with bundle dict(s)")
    parser.add_argument("--out-h5", required=True, type=str)
    parser.add_argument("--out-csv-dir", required=True, type=str)
    parser.add_argument("--run-id", type=str, default="run_standard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario-id", type=str, default="NA")
    args = parser.parse_args()

    bundles = _load_bundles(args.input_json)
    meta: dict[str, Any] = {
        "run_id": str(args.run_id),
        "seed": int(args.seed),
        "scenario_id": str(args.scenario_id),
        "cmdline": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "git_branch": _git_branch(),
        "input_json": str(args.input_json),
    }
    save_run(meta, bundles, out_h5=args.out_h5, run_id=args.run_id)
    out = export_csv(bundles, out_dir=args.out_csv_dir)
    print(str(args.out_h5))
    print(out["link_metrics_csv"])
    print(out["rays_csv"])


if __name__ == "__main__":
    main()

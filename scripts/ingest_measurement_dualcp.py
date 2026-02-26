"""Ingest dual-CP sequential CSV measurements into measurement HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any

from analysis.measurement_compare import load_measurement_dualcp_two_csv
from rt_io.measurement_hdf5 import append_measurement_case, build_provenance


def _load_meta_json(arg: str | None) -> dict[str, Any]:
    if arg is None or str(arg).strip() == "":
        return {}
    p = Path(arg)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(str(arg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-id", required=True, type=str)
    parser.add_argument("--case-id", required=True, type=str)
    parser.add_argument("--co-csv", required=True, type=str)
    parser.add_argument("--cross-csv", required=True, type=str)
    parser.add_argument("--out-h5", required=True, type=str)
    parser.add_argument("--meta-json", type=str, default=None)
    args = parser.parse_args()

    meta = _load_meta_json(args.meta_json)
    basis = str(meta.get("basis", "circular"))
    convention = str(meta.get("convention", "IEEE-RHCP"))
    meas = load_measurement_dualcp_two_csv(
        co_csv=args.co_csv,
        cross_csv=args.cross_csv,
        basis=basis,
        convention=convention,
    )

    merged_meta = dict(meta)
    merged_meta.setdefault("basis", str(meas.meta.get("basis", basis)))
    merged_meta.setdefault("convention", str(meas.meta.get("convention", convention)))
    merged_meta.setdefault("format", "dualcp_two_csv")

    provenance = build_provenance(
        source_paths={"co_csv": args.co_csv, "cross_csv": args.cross_csv},
        command=" ".join(shlex.quote(x) for x in sys.argv),
        extra={
            "script": "scripts/ingest_measurement_dualcp.py",
            "scenario_id": str(args.scenario_id),
            "case_id": str(args.case_id),
        },
    )

    out = append_measurement_case(
        path=args.out_h5,
        scenario_id=args.scenario_id,
        case_id=args.case_id,
        frequency_hz=meas.frequency_hz,
        H_f=meas.H_f,
        meta=merged_meta,
        provenance=provenance,
        overwrite=True,
    )
    print(str(out))


if __name__ == "__main__":
    main()

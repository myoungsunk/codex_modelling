"""Run CP and LP standard simulations, then build baseline comparison report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.lp_baseline_compare import compare_cp_lp_metrics


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        required=True,
        choices=["C0", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "A2_on", "A3_on", "A4_on", "A6_on"],
    )
    parser.add_argument("--out-root", type=str, default="outputs/cp_lp_baseline")
    parser.add_argument("--run-id", type=str, default=None)
    args, passthrough = parser.parse_known_args()

    run_id = str(args.run_id or f"{args.scenario}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    out_root = Path(args.out_root)
    cp_dir = out_root / f"{run_id}_cp"
    lp_dir = out_root / f"{run_id}_lp"
    cp_h5 = out_root / f"{run_id}_cp.h5"
    lp_h5 = out_root / f"{run_id}_lp.h5"
    out_root.mkdir(parents=True, exist_ok=True)

    common = ["python3", "scripts/run_standard_sim.py", "--scenario", str(args.scenario), *passthrough]
    cmd_cp = [*common, "--basis", "circular", "--out-h5", str(cp_h5), "--out-dir", str(cp_dir), "--run-id", f"{run_id}_cp"]
    cmd_lp = [*common, "--basis", "linear", "--out-h5", str(lp_h5), "--out-dir", str(lp_dir), "--run-id", f"{run_id}_lp"]
    _run(cmd_cp)
    _run(cmd_lp)

    cp_metrics = cp_dir / "link_metrics.csv"
    lp_metrics = lp_dir / "link_metrics.csv"
    pairs_csv = out_root / f"{run_id}_cp_lp_pairs.csv"
    report_md = out_root / f"{run_id}_cp_lp_report.md"
    comp = compare_cp_lp_metrics(
        cp_metrics_csv=cp_metrics,
        lp_metrics_csv=lp_metrics,
        out_pairs_csv=pairs_csv,
        out_report_md=report_md,
    )

    summary = {
        "run_id": run_id,
        "scenario": str(args.scenario),
        "cp_h5": str(cp_h5),
        "lp_h5": str(lp_h5),
        "cp_metrics_csv": str(cp_metrics),
        "lp_metrics_csv": str(lp_metrics),
        "cp_lp_pairs_csv": str(pairs_csv),
        "cp_lp_report_md": str(report_md),
        "n_pairs": int(comp.get("summary", {}).get("n_pairs", 0)),
        "cmd_cp": " ".join(shlex.quote(x) for x in cmd_cp),
        "cmd_lp": " ".join(shlex.quote(x) for x in cmd_lp),
    }
    out_json = out_root / f"{run_id}_cp_lp_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(out_json))
    print(str(report_md))
    print(str(pairs_csv))


if __name__ == "__main__":
    main()

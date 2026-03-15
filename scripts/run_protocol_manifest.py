"""Run standardized simulations from a reproducible protocol manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


def _load_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"manifest root must be object: {p}")
    return obj


def _parse_only(raw: str) -> set[str]:
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    return {x for x in items}


def _run(cmd: list[str], dry_run: bool = False) -> None:
    print(" ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0:
        raise SystemExit(
            f"command failed ({p.returncode})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        default="analysis_report/manifests/protocol_repro_manifest.v1.json",
    )
    ap.add_argument("--output-root", type=str, default="")
    ap.add_argument("--run-group", type=str, default="")
    ap.add_argument("--analysis-config-out", type=str, default="")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = _load_manifest(args.manifest)
    runs = manifest.get("runs", [])
    if not isinstance(runs, list) or len(runs) == 0:
        raise SystemExit("manifest.runs is empty")

    output_root = Path(str(args.output_root or manifest.get("output_root", "outputs/protocol_repro_v1"))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    only = _parse_only(args.only)
    tag = str(args.tag).strip()

    common_args = manifest.get("common_args", [])
    if not isinstance(common_args, list):
        raise SystemExit("manifest.common_args must be list")

    input_runs: list[str] = []
    for entry in runs:
        if not isinstance(entry, dict):
            continue
        rid = str(entry.get("id", "")).strip()
        scenario = str(entry.get("scenario", "")).strip()
        if not rid or not scenario:
            continue
        if only and rid not in only and scenario not in only:
            continue

        extra_args = entry.get("args", [])
        if not isinstance(extra_args, list):
            raise SystemExit(f"run args must be list: id={rid}")

        out_dir = output_root / rid
        out_h5 = output_root / f"{rid}.h5"
        if tag:
            out_dir = output_root / f"{rid}__{tag}"
            out_h5 = output_root / f"{rid}__{tag}.h5"
        run_id = f"{rid}__{tag}" if tag else rid

        if args.skip_existing and out_dir.exists() and (out_dir / "run_summary.json").exists():
            print(f"[skip-existing] {rid}: {out_dir}")
            input_runs.append(str(out_dir))
            continue

        cmd = [
            sys.executable,
            "scripts/run_standard_sim.py",
            "--scenario",
            scenario,
            "--out-h5",
            str(out_h5),
            "--out-dir",
            str(out_dir),
            "--run-id",
            run_id,
            *common_args,
            *extra_args,
        ]
        _run(cmd, dry_run=bool(args.dry_run))
        input_runs.append(str(out_dir))

    if args.dry_run:
        return

    tpl = manifest.get("analysis_report_template", {})
    if not isinstance(tpl, dict):
        tpl = {}
    run_group = str(args.run_group or manifest.get("run_group", "diag_protocol_repro_v1"))
    cfg = dict(tpl)
    cfg["run_group"] = run_group
    cfg["input_runs"] = input_runs
    cfg_out = Path(
        str(
            args.analysis_config_out
            or (Path("analysis_report") / f"config.{run_group}.json")
        )
    )
    cfg_out.parent.mkdir(parents=True, exist_ok=True)
    cfg_out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(str(cfg_out))


if __name__ == "__main__":
    main()

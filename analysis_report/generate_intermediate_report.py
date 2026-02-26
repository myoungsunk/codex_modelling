"""Generate intermediate report from standard outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis_report.lib.io import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_group = str(cfg.get("run_group", "run_group"))
    out_dir = Path("analysis_report") / "out" / run_group
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "intermediate_report.md").write_text(
        "# Intermediate Report\n\nReport generator wiring complete.\n",
        encoding="utf-8",
    )
    print(str(out_dir / "intermediate_report.md"))


if __name__ == "__main__":
    main()

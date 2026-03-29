from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dualpol_rt import build_phase2_patterns_from_standard_files, canonical_scenarios, run_indoor_campaign


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the indoor richness and robustness campaign.")
    parser.add_argument("--ffd-dir", required=True, help="Directory containing LP_+45, LP_-45, RHCP, and LHCP FFD files.")
    parser.add_argument("--output-dir", required=True, help="Directory for maps.npz, summary.json, and CSV artifacts.")
    parser.add_argument("--scenarios", nargs="*", help="Optional subset of canonical scenario names.")
    parser.add_argument("--screening-n-freq", type=int, default=21)
    parser.add_argument("--screening-grid-step", type=float, default=0.5)
    parser.add_argument("--full-n-freq", type=int, default=101)
    parser.add_argument("--full-grid-step", type=float, default=0.2)
    parser.add_argument("--pose-seed-count", type=int, default=64)
    parser.add_argument("--pose-grid-x", type=int, default=10)
    parser.add_argument("--pose-grid-y", type=int, default=10)
    parser.add_argument("--fc-ghz", type=float, default=6.5)
    parser.add_argument("--bw-mhz", type=float, default=500.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    families = build_phase2_patterns_from_standard_files(args.ffd_dir)
    scenario_names = tuple(args.scenarios) if args.scenarios else tuple(spec.name for spec in canonical_scenarios(include_proxy=True))
    result = run_indoor_campaign(
        families["tx_lp"],
        families["rx_lp"],
        families["tx_cp"],
        families["rx_cp"],
        scenario_names=scenario_names,
        output_dir=args.output_dir,
        fc=args.fc_ghz * 1e9,
        bw=args.bw_mhz * 1e6,
        screening_n_freq=args.screening_n_freq,
        screening_grid_step=args.screening_grid_step,
        full_n_freq=args.full_n_freq,
        full_grid_step=args.full_grid_step,
        pose_seed_count=args.pose_seed_count,
        pose_grid_shape=(args.pose_grid_x, args.pose_grid_y),
    )
    print("Selected scenes:", sorted(result.full_compare_results.keys()))
    print("Artifacts:", result.artifact_paths)


if __name__ == "__main__":
    main()

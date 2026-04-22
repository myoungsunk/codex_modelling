from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dualpol_rt import ExperimentConfig
from dualpol_rt.experiments.realistic_compare import (
    build_phase2_patterns,
    build_phase2_patterns_from_standard_files,
    evaluate_realistic_debug_point,
    evaluate_realistic_rx_grid,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 realistic dual-linear-slant vs dual-CP comparison.")
    parser.add_argument("--ffd-dir", help="Directory containing the standard four-file bundle: LP_+45, LP_-45, RHCP, LHCP")
    parser.add_argument("--tx-lp-plus45", help="Tx +45 deg linear FFD file")
    parser.add_argument("--tx-lp-minus45", help="Tx -45 deg linear FFD file")
    parser.add_argument("--rx-lp-plus45", help="Rx +45 deg linear FFD file")
    parser.add_argument("--rx-lp-minus45", help="Rx -45 deg linear FFD file")
    parser.add_argument("--tx-rhcp", help="Tx RHCP FFD file")
    parser.add_argument("--tx-lhcp", help="Tx LHCP FFD file")
    parser.add_argument("--rx-rhcp", help="Rx RHCP FFD file")
    parser.add_argument("--rx-lhcp", help="Rx LHCP FFD file")
    parser.add_argument("--normalization", default="raw", choices=("raw", "family_radiated_power"))
    parser.add_argument("--full-grid", action="store_true", help="Run the full Rx grid in addition to the debug point.")
    parser.add_argument("--n-freq", type=int, default=101, help="Number of channel frequency samples across the 500 MHz band.")
    parser.add_argument("--fc-ghz", type=float, default=6.5, help="Center frequency in GHz.")
    parser.add_argument("--bw-mhz", type=float, default=500.0, help="Bandwidth in MHz.")
    return parser.parse_args()


def _build_families(args: argparse.Namespace):
    if args.ffd_dir:
        return build_phase2_patterns_from_standard_files(args.ffd_dir, normalization_mode=args.normalization)

    required = {
        "--tx-lp-plus45": args.tx_lp_plus45,
        "--tx-lp-minus45": args.tx_lp_minus45,
        "--rx-lp-plus45": args.rx_lp_plus45,
        "--rx-lp-minus45": args.rx_lp_minus45,
        "--tx-rhcp": args.tx_rhcp,
        "--tx-lhcp": args.tx_lhcp,
        "--rx-rhcp": args.rx_rhcp,
        "--rx-lhcp": args.rx_lhcp,
    }
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Missing FFD inputs. Provide --ffd-dir or all explicit file flags: {missing}")
    return build_phase2_patterns(
        tx_lp_files=(args.tx_lp_plus45, args.tx_lp_minus45),
        rx_lp_files=(args.rx_lp_plus45, args.rx_lp_minus45),
        tx_cp_files=(args.tx_rhcp, args.tx_lhcp),
        rx_cp_files=(args.rx_rhcp, args.rx_lhcp),
        normalization_mode=args.normalization,
    )


def main() -> None:
    args = _parse_args()
    cfg = ExperimentConfig(fc=args.fc_ghz * 1e9, bw=args.bw_mhz * 1e6, n_freq=args.n_freq)
    families = _build_families(args)
    debug = evaluate_realistic_debug_point(
        cfg,
        cfg.debug_rx_pos,
        families["tx_lp"],
        families["rx_lp"],
        families["tx_cp"],
        families["rx_cp"],
    )
    print("Debug point:", cfg.debug_rx_pos.tolist())
    print("Paths:", debug["path_count"])
    print("Raw delta R_EP @ 10 dB (CP-LP):", debug["raw_delta"].delta_equal_power_rate.get(10.0))
    print("Normalized delta R_EP @ 10 dB (CP-LP):", debug["normalized_delta"].delta_equal_power_rate.get(10.0))

    if args.full_grid:
        result = evaluate_realistic_rx_grid(
            cfg,
            families["tx_lp"],
            families["rx_lp"],
            families["tx_cp"],
            families["rx_cp"],
        )
        print("Full-grid raw delta R_EP map shape:", result.raw.delta_rate_maps[10.0].shape)
        print("Full-grid normalized delta R_EP map shape:", result.normalized.delta_rate_maps[10.0].shape)


if __name__ == "__main__":
    main()

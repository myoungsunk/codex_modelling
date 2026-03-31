from __future__ import annotations

import tempfile
import unittest
import csv
import json
from pathlib import Path

import numpy as np

from dualpol_rt import build_phase2_patterns, run_indoor_campaign


def _write_multifrequency_ffd(
    path: Path,
    freqs_hz: list[float],
    blocks: list[list[tuple[complex, complex]]],
) -> None:
    lines = [
        "0 90 2",
        "-90 90 2",
        f"Frequencies {len(freqs_hz)}",
    ]
    for freq_hz, block in zip(freqs_hz, blocks):
        lines.append(f"Frequency {freq_hz:.15e}")
        for e_theta, e_phi in block:
            lines.append(f"{e_theta.real:.15e} {e_theta.imag:.15e} {e_phi.real:.15e} {e_phi.imag:.15e}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _base_block(scale: float, skew: float) -> list[tuple[complex, complex]]:
    return [
        (scale * ((1.0 + skew) + (0.5 - 0.1 * skew) * 1j), scale * ((8.0 - 2.0 * skew) + (-1.0 - 0.2 * skew) * 1j)),
        (scale * ((2.0 - 0.5 * skew) + (1.0 + 0.2 * skew) * 1j), scale * ((16.0 + skew) + (-2.0 + 0.1 * skew) * 1j)),
        (scale * ((3.0 + 0.3 * skew) + (1.5 + 0.2 * skew) * 1j), scale * ((24.0 - skew) + (-3.0 - 0.1 * skew) * 1j)),
        (scale * ((4.0 - 0.2 * skew) + (2.0 - 0.3 * skew) * 1j), scale * ((32.0 + 2.0 * skew) + (-4.0 + 0.2 * skew) * 1j)),
    ]


def _write_family(tmp_path: Path, prefix: str, scale0: float, scale1: float) -> tuple[str, str]:
    port0 = tmp_path / f"{prefix}_p0.ffd"
    port1 = tmp_path / f"{prefix}_p1.ffd"
    _write_multifrequency_ffd(port0, [6.0e9, 7.0e9], [_base_block(scale0, 0.1), _base_block(2.0 * scale0, 0.1)])
    _write_multifrequency_ffd(port1, [6.0e9, 7.0e9], [_base_block(scale1, 0.7), _base_block(2.0 * scale1, 0.7)])
    return str(port0), str(port1)


class IndoorCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        lp_files = _write_family(self.root, "lp", 1.0, 0.8)
        cp_files = _write_family(self.root, "cp", 2.0, 1.6)
        self.families = build_phase2_patterns(
            tx_lp_files=lp_files,
            rx_lp_files=lp_files,
            tx_cp_files=cp_files,
            rx_cp_files=cp_files,
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_campaign_writes_artifacts_and_excludes_proxy_from_headline(self) -> None:
        output_dir = self.root / "campaign_out"
        result = run_indoor_campaign(
            self.families["tx_lp"],
            self.families["rx_lp"],
            self.families["tx_cp"],
            self.families["rx_cp"],
            scenario_names=("open_office", "mixed_office", "straight_corridor", "l_corridor_proxy", "lobby_blocker_proxy"),
            output_dir=output_dir,
            screening_n_freq=3,
            screening_grid_step=3.0,
            full_n_freq=3,
            full_grid_step=3.0,
            pose_seed_count=1,
            pose_grid_shape=(1, 1),
            snr_db_list=(10.0,),
            base_seed=1234,
        )
        self.assertTrue((output_dir / "maps.npz").exists())
        self.assertTrue((output_dir / "summary.json").exists())
        self.assertTrue((output_dir / "scene_metrics.csv").exists())
        self.assertTrue((output_dir / "pose_metrics.csv").exists())
        self.assertTrue((output_dir / "checkpoint_status.json").exists())
        self.assertGreaterEqual(len(result.scene_summaries), 5)
        self.assertTrue(all(name not in result.headline_summary["selected_scenes"] for name in ("l_corridor_proxy", "lobby_blocker_proxy")))
        self.assertIn("l_corridor_proxy", result.headline_summary["proxy_scenes"])
        self.assertIn("lobby_blocker_proxy", result.headline_summary["proxy_scenes"])
        self.assertIn("straight_corridor", result.full_compare_results)
        self.assertEqual(result.headline_summary["primary_kpis"], ["raw_total_snr", "best_port_outage", "best_port_rate", "fixed_egc_rank1_rate"])
        self.assertEqual(result.headline_summary["raw_total_snr_threshold_db"], -30.0)
        self.assertEqual(result.headline_summary["best_port_threshold_db"], -30.0)
        checkpoint = json.loads((output_dir / "checkpoint_status.json").read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["stage"], "completed")
        self.assertIn("selected_scenes", checkpoint)
        self.assertIn("completed_selected_scenes", checkpoint)
        self.assertIn("completed_proxy_scenes", checkpoint)
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["notes"]["practical_metrics_are_primary"])
        self.assertTrue(summary["notes"]["normalized_ideal_2x2_results_are_secondary"])
        self.assertTrue(summary["notes"]["normalized_2x2_rate_is_unitary_invariant"])
        self.assertTrue(summary["notes"]["port_constrained_metrics_are_not_unitary_invariant"])
        self.assertTrue(summary["notes"]["raw_power_invariance_is_not_assumed"])

    def test_campaign_reports_delta_rate_p05_as_percentile_difference(self) -> None:
        result = run_indoor_campaign(
            self.families["tx_lp"],
            self.families["rx_lp"],
            self.families["tx_cp"],
            self.families["rx_cp"],
            scenario_names=("open_office", "mixed_office", "straight_corridor"),
            screening_n_freq=3,
            screening_grid_step=3.0,
            full_n_freq=3,
            full_grid_step=3.0,
            pose_seed_count=1,
            pose_grid_shape=(1, 1),
            snr_db_list=(10.0,),
            base_seed=99,
        )
        row = next(r for r in result.pose_rows if r["scenario_name"] == "open_office" and r["protocol"] == "P0")
        self.assertAlmostEqual(row["delta_rate_p05"], row["cp_rate_p05"] - row["lp_rate_p05"], places=12)
        self.assertIn("lp_raw_total_snr_mean_db", row)
        self.assertIn("cp_best_port_snr_mean_db", row)
        self.assertIn("lp_fixed_egc_rate_raw_mean", row)
        self.assertIn("lp_best_port_rate_norm_mean", row)
        self.assertIn("lp_fixed_egc_rate_norm_mean", row)
        self.assertEqual(row["raw_total_snr_threshold_db"], -30.0)
        self.assertEqual(row["best_port_threshold_db"], -30.0)
        self.assertGreater(row["cp_raw_total_snr_mean_db"], row["lp_raw_total_snr_mean_db"])
        self.assertGreater(row["cp_best_port_snr_mean_db"], row["lp_best_port_snr_mean_db"])
        self.assertGreater(row["cp_best_port_rate_raw_mean"], row["lp_best_port_rate_raw_mean"])
        self.assertGreater(row["cp_fixed_egc_rate_raw_mean"], row["lp_fixed_egc_rate_raw_mean"])
        self.assertGreaterEqual(row["cp_best_port_rate_norm_mean"], 0.0)

    def test_campaign_exports_capacity_columns_and_maps(self) -> None:
        output_dir = self.root / "campaign_capacity_out"
        run_indoor_campaign(
            self.families["tx_lp"],
            self.families["rx_lp"],
            self.families["tx_cp"],
            self.families["rx_cp"],
            scenario_names=("open_office", "mixed_office", "straight_corridor"),
            output_dir=output_dir,
            screening_n_freq=3,
            screening_grid_step=3.0,
            full_n_freq=3,
            full_grid_step=3.0,
            pose_seed_count=1,
            pose_grid_shape=(1, 1),
            snr_db_list=(10.0,),
            base_seed=101,
        )
        with (output_dir / "scene_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertIn("delta_capacity_raw_snr_10", reader.fieldnames)
            self.assertIn("delta_capacity_norm_snr_10", reader.fieldnames)
            first_row = next(reader)
            self.assertIn("lp_capacity_raw_snr_10", first_row)
            self.assertIn("lp_capacity_norm_snr_10", first_row)
        arrays = np.load(output_dir / "maps.npz", allow_pickle=True)
        self.assertIn("full_open_office_norm_delta_capacity_snr_10", arrays.files)
        self.assertIn("full_open_office_raw_delta_capacity_snr_10", arrays.files)

    def test_campaign_pose_rows_are_deterministic_for_fixed_seed(self) -> None:
        common_kwargs = dict(
            scenario_names=("open_office", "mixed_office", "straight_corridor", "l_corridor_proxy", "lobby_blocker_proxy"),
            screening_n_freq=3,
            screening_grid_step=3.0,
            full_n_freq=3,
            full_grid_step=3.0,
            pose_seed_count=1,
            pose_grid_shape=(1, 1),
            snr_db_list=(10.0,),
            base_seed=99,
        )
        result_a = run_indoor_campaign(
            self.families["tx_lp"],
            self.families["rx_lp"],
            self.families["tx_cp"],
            self.families["rx_cp"],
            **common_kwargs,
        )
        result_b = run_indoor_campaign(
            self.families["tx_lp"],
            self.families["rx_lp"],
            self.families["tx_cp"],
            self.families["rx_cp"],
            **common_kwargs,
        )
        self.assertEqual(result_a.pose_rows, result_b.pose_rows)


if __name__ == "__main__":
    unittest.main()

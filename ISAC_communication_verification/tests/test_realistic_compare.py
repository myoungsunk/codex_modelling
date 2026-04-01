from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dualpol_rt import ExperimentConfig
from dualpol_rt.experiments.realistic_compare import (
    build_phase2_patterns,
    build_phase2_patterns_from_standard_files,
    evaluate_realistic_debug_point,
    evaluate_realistic_rx_grid,
)


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


class RealisticCompareTests(unittest.TestCase):
    def _build_fixture(self) -> tuple[ExperimentConfig, dict[str, object]]:
        self.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmpdir.name)
        lp_files = _write_family(tmp_path, "lp", 1.0, 0.8)
        cp_files = _write_family(tmp_path, "cp", 2.0, 1.6)
        cfg = ExperimentConfig(
            fc=6.5e9,
            bw=200e6,
            n_freq=5,
            room_size=(2.0, 2.0, 2.0),
            tx_pos=np.array([0.6, 1.0, 1.5], dtype=float),
            rx_plane_z=1.0,
            grid_step=1.0,
            max_reflections=1,
            snr_db_list=(10.0,),
            debug_rx_pos=np.array([1.4, 1.0, 1.0], dtype=float),
        )
        families = build_phase2_patterns(
            tx_lp_files=lp_files,
            rx_lp_files=lp_files,
            tx_cp_files=cp_files,
            rx_cp_files=cp_files,
        )
        return cfg, families

    def tearDown(self) -> None:
        tmpdir = getattr(self, "tmpdir", None)
        if tmpdir is not None:
            tmpdir.cleanup()
            self.tmpdir = None

    def test_same_family_representation_invariance(self) -> None:
        cfg, families = self._build_fixture()
        debug = evaluate_realistic_debug_point(
            cfg,
            cfg.debug_rx_pos,
            families["tx_lp"],
            families["rx_lp"],
            families["tx_cp"],
            families["rx_cp"],
        )
        self.assertLess(debug["same_family_basis_invariance_lp"]["raw"], 1e-8)
        self.assertLess(debug["same_family_basis_invariance_lp"]["normalized"], 1e-8)
        self.assertIsNone(debug["same_family_basis_invariance_cp"])
        self.assertEqual(
            debug["same_family_basis_invariance_note"],
            "Representation invariance is evaluated on the LP-family channel only; same_family_basis_invariance_cp is intentionally omitted because CP-family channels already live in circular port space.",
        )

    def test_realistic_debug_point_smoke(self) -> None:
        cfg, families = self._build_fixture()
        debug = evaluate_realistic_debug_point(
            cfg,
            cfg.debug_rx_pos,
            families["tx_lp"],
            families["rx_lp"],
            families["tx_cp"],
            families["rx_cp"],
        )
        self.assertGreater(debug["path_count"], 0)
        self.assertEqual(debug["H_lp_raw"].shape, (cfg.n_freq, 2, 2))
        self.assertEqual(debug["H_cp_raw"].shape, (cfg.n_freq, 2, 2))
        self.assertEqual(set(debug["lp_raw_summary"].equal_power_rate.keys()), {10.0})
        self.assertEqual(set(debug["cp_normalized_summary"].waterfilled_capacity.keys()), {10.0})
        self.assertGreater(debug["lp_raw_summary"].total_rx_gain, 0.0)
        self.assertEqual(debug["lp_raw_summary"].per_port_rx_gain.shape, (2,))
        self.assertEqual(set(debug["lp_raw_summary"].best_port_rate.keys()), {10.0})
        self.assertEqual(set(debug["lp_raw_summary"].fixed_rank1_rates.keys()), {"single_port", "equal_gain"})
        self.assertEqual(set(debug["lp_raw_summary"].polarization_diversity_gain.keys()), {10.0})
        self.assertEqual(set(debug["lp_raw_summary"].mimo_multiplexing_efficiency.keys()), {10.0})

    def test_delta_sign_is_cp_minus_lp(self) -> None:
        cfg, families = self._build_fixture()
        debug = evaluate_realistic_debug_point(
            cfg,
            cfg.debug_rx_pos,
            families["tx_lp"],
            families["rx_lp"],
            families["tx_cp"],
            families["rx_cp"],
        )
        raw_lp = debug["lp_raw_summary"].equal_power_rate[10.0]
        raw_cp = debug["cp_raw_summary"].equal_power_rate[10.0]
        norm_lp = debug["lp_normalized_summary"].equal_power_rate[10.0]
        norm_cp = debug["cp_normalized_summary"].equal_power_rate[10.0]
        self.assertAlmostEqual(debug["raw_delta"].delta_equal_power_rate[10.0], raw_cp - raw_lp, places=12)
        self.assertAlmostEqual(debug["normalized_delta"].delta_equal_power_rate[10.0], norm_cp - norm_lp, places=12)

    def test_standard_file_helper_uses_fixed_filenames(self) -> None:
        self._build_fixture()
        tmp_path = Path(self.tmpdir.name)
        (tmp_path / "LP_+45_new_6G7G_11pts.ffd").write_text((tmp_path / "lp_p0.ffd").read_text(encoding="utf-8"), encoding="utf-8")
        (tmp_path / "LP_-45_new_6G7G_11pts.ffd").write_text((tmp_path / "lp_p1.ffd").read_text(encoding="utf-8"), encoding="utf-8")
        (tmp_path / "RHCP_new_6G7G_11pts.ffd").write_text((tmp_path / "cp_p0.ffd").read_text(encoding="utf-8"), encoding="utf-8")
        (tmp_path / "LHCP_new_6G7G_11pts.ffd").write_text((tmp_path / "cp_p1.ffd").read_text(encoding="utf-8"), encoding="utf-8")
        families = build_phase2_patterns_from_standard_files(tmp_path)
        self.assertEqual(families["tx_lp"].port_labels, ("+45", "-45"))
        self.assertEqual(families["tx_cp"].port_labels, ("RHCP", "LHCP"))

    def test_realistic_grid_shapes_and_keys(self) -> None:
        cfg, families = self._build_fixture()
        result = evaluate_realistic_rx_grid(
            cfg,
            families["tx_lp"],
            families["rx_lp"],
            families["tx_cp"],
            families["rx_cp"],
        )
        self.assertEqual(result.raw.delta_rate_maps[10.0].shape, (2, 2))
        self.assertEqual(result.normalized.delta_capacity_maps[10.0].shape, (2, 2))
        self.assertEqual(result.raw.lp.xpr_db_map.shape, (2, 2))
        self.assertEqual(result.normalized.cp.condition_number_map.shape, (2, 2))

    def test_raw_vs_normalized_comparison_runs(self) -> None:
        cfg, families = self._build_fixture()
        debug = evaluate_realistic_debug_point(
            cfg,
            cfg.debug_rx_pos,
            families["tx_lp"],
            families["rx_lp"],
            families["tx_cp"],
            families["rx_cp"],
        )
        raw_delta = abs(debug["raw_delta"].delta_equal_power_rate[10.0])
        normalized_delta = abs(debug["normalized_delta"].delta_equal_power_rate[10.0])
        self.assertGreater(raw_delta, 1e-6)
        self.assertLess(normalized_delta, raw_delta)
        self.assertGreater(abs(debug["raw_delta"].delta_best_port_rate[10.0]), 1e-6)
        self.assertGreater(abs(debug["raw_delta"].delta_fixed_rank1_rates["equal_gain"][10.0]), 1e-6)
        self.assertGreater(abs(debug["raw_delta"].delta_polarization_diversity_gain[10.0]), 1e-6)
        self.assertGreater(abs(debug["raw_delta"].delta_mimo_multiplexing_efficiency[10.0]), 1e-6)
        self.assertAlmostEqual(
            debug["lp_raw_summary"].gain_normalized_equal_power_rate[10.0],
            debug["lp_raw_summary"].equal_power_rate[10.0],
            places=12,
        )
        self.assertAlmostEqual(
            debug["cp_raw_summary"].gain_normalized_equal_power_rate[10.0],
            debug["cp_raw_summary"].equal_power_rate[10.0],
            places=12,
        )
        self.assertAlmostEqual(
            debug["lp_normalized_summary"].gain_normalized_equal_power_rate[10.0],
            debug["lp_normalized_summary"].equal_power_rate[10.0],
            places=12,
        )
        self.assertAlmostEqual(
            debug["cp_normalized_summary"].gain_normalized_equal_power_rate[10.0],
            debug["cp_normalized_summary"].equal_power_rate[10.0],
            places=12,
        )


if __name__ == "__main__":
    unittest.main()

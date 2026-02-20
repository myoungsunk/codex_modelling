"""Step 1 acceptance tests for physics-oriented polarimetric RT outputs."""

from __future__ import annotations

import unittest

import numpy as np

from scenarios import A2_pec_plane, A3_corner_2bounce, C0_free_space


def _same_vs_opp_power(A_f: np.ndarray) -> tuple[float, float]:
    same = float(np.mean(np.abs(A_f[:, 0, 0]) ** 2 + np.abs(A_f[:, 1, 1]) ** 2))
    opp = float(np.mean(np.abs(A_f[:, 0, 1]) ** 2 + np.abs(A_f[:, 1, 0]) ** 2))
    return same, opp


class Step1AcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.f = np.linspace(6e9, 10e9, 256)

    def test_linear_los_xpd_not_pinned_to_leakage_floor(self) -> None:
        params = C0_free_space.build_sweep_params()[1]
        paths = C0_free_space.run_case(
            params,
            self.f,
            basis="linear",
            antenna_config={
                "tx_cross_pol_leakage_db": 120.0,
                "rx_cross_pol_leakage_db": 120.0,
                "tx_axial_ratio_db": 0.0,
                "rx_axial_ratio_db": 0.0,
                "enable_coupling": True,
            },
        )
        self.assertEqual(len(paths), 1)
        same, opp = _same_vs_opp_power(paths[0].A_f)
        xpd_db = 10.0 * np.log10((same + 1e-15) / (opp + 1e-15))
        self.assertGreater(xpd_db, 60.0)

    def test_circular_odd_even_ratio_directionality(self) -> None:
        a2_params = A2_pec_plane.build_sweep_params()[4]
        a3_params = A3_corner_2bounce.build_sweep_params()[0]
        antenna_config = {
            "convention": "IEEE-RHCP",
            "tx_cross_pol_leakage_db": 120.0,
            "rx_cross_pol_leakage_db": 120.0,
            "tx_axial_ratio_db": 0.0,
            "rx_axial_ratio_db": 0.0,
            "enable_coupling": True,
        }

        a2_paths = A2_pec_plane.run_case(
            a2_params,
            self.f,
            basis="circular",
            antenna_config=antenna_config,
            force_cp_swap_on_odd_reflection=True,
        )
        a3_paths = A3_corner_2bounce.run_case(
            a3_params,
            self.f,
            basis="circular",
            antenna_config=antenna_config,
            force_cp_swap_on_odd_reflection=True,
        )

        odd = [p for p in a2_paths if p.bounce_count == 1]
        even = [p for p in a3_paths if p.bounce_count == 2]
        self.assertTrue(len(odd) > 0)
        self.assertTrue(len(even) > 0)

        odd_same, odd_opp = _same_vs_opp_power(odd[0].A_f)
        even_same, even_opp = _same_vs_opp_power(even[0].A_f)

        self.assertGreater(odd_opp, odd_same)
        self.assertGreater(even_same, even_opp)


if __name__ == "__main__":
    unittest.main()

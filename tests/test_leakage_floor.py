"""Leakage-floor prediction and trigger checks."""

from __future__ import annotations

import unittest

from analysis.xpd_stats import estimate_leakage_floor_db, leakage_limited_summary


class LeakageFloorTests(unittest.TestCase):
    def test_default_35db_floor_is_about_29db(self) -> None:
        info = estimate_leakage_floor_db(
            tx_cross_pol_leakage_db=35.0,
            rx_cross_pol_leakage_db=35.0,
            tx_axial_ratio_db=0.0,
            rx_axial_ratio_db=0.0,
            tx_enable_coupling=True,
            rx_enable_coupling=True,
        )
        self.assertAlmostEqual(float(info["xpd_floor_db"]), 28.982, places=2)

    def test_physics_validation_mode_does_not_trigger_leakage_warning(self) -> None:
        # Coupling disabled in physics-validation mode.
        info = estimate_leakage_floor_db(
            tx_cross_pol_leakage_db=120.0,
            rx_cross_pol_leakage_db=120.0,
            tx_axial_ratio_db=0.0,
            rx_axial_ratio_db=0.0,
            tx_enable_coupling=False,
            rx_enable_coupling=False,
        )
        chk = leakage_limited_summary([120.0, 121.0, 119.0], xpd_floor_db=float(info["xpd_floor_db"]))
        self.assertFalse(bool(chk["is_leakage_limited"]))


if __name__ == "__main__":
    unittest.main()

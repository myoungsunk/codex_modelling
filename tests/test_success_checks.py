"""Tests for success-check metric selection and stability."""

from __future__ import annotations

import unittest

from analysis.success_checks import check_C0_floor


class SuccessChecksTests(unittest.TestCase):
    def test_c0_floor_prefers_raw_xpd_not_excess(self) -> None:
        rows = [
            {
                "scenario_id": "C0",
                "XPD_early_db": 10.0,
                "XPD_early_excess_db": 0.0,
                "d_m": 3.0,
                "yaw_deg": 0.0,
            },
            {
                "scenario_id": "C0",
                "XPD_early_db": 20.0,
                "XPD_early_excess_db": 0.0,
                "d_m": 6.0,
                "yaw_deg": 5.0,
            },
        ]
        out = check_C0_floor(rows)
        self.assertEqual(out.get("metric_key"), "XPD_early_db")
        self.assertAlmostEqual(float(out.get("xpd_floor_mean_db")), 15.0, places=9)

    def test_c0_floor_falls_back_to_excess_if_raw_missing(self) -> None:
        rows = [
            {"scenario_id": "C0", "XPD_early_excess_db": -1.0, "d_m": 3.0, "yaw_deg": 0.0},
            {"scenario_id": "C0", "XPD_early_excess_db": 1.0, "d_m": 6.0, "yaw_deg": 5.0},
        ]
        out = check_C0_floor(rows)
        self.assertEqual(out.get("metric_key"), "XPD_early_excess_db")
        self.assertAlmostEqual(float(out.get("xpd_floor_mean_db")), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()


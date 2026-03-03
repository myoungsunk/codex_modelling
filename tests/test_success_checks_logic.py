from __future__ import annotations

import unittest

from analysis.success_checks import check_A2_A3_parity_sign, check_A4_A5_breaking


class SuccessChecksLogicTests(unittest.TestCase):
    def test_parity_check_prefers_raw_and_expected_subset(self) -> None:
        rows = []
        # A2: expected odd-dominant and negative early XPD.
        for i in range(6):
            rows.append(
                {
                    "scenario_id": "A2",
                    "XPD_early_db": -6.0 - 0.1 * i,
                    "XPD_early_excess_db": -30.0 - 0.1 * i,
                    "dominant_parity_early": "odd",
                }
            )
        # A3: mix of NA and even; even-dominant subset should be used.
        for _ in range(5):
            rows.append(
                {
                    "scenario_id": "A3",
                    "XPD_early_db": 0.0,
                    "XPD_early_excess_db": -23.0,
                    "dominant_parity_early": "NA",
                }
            )
        for i in range(6):
            rows.append(
                {
                    "scenario_id": "A3",
                    "XPD_early_db": 6.0 + 0.1 * i,
                    "XPD_early_excess_db": -17.0 + 0.1 * i,
                    "dominant_parity_early": "even",
                }
            )

        out = check_A2_A3_parity_sign(rows)
        self.assertEqual(out["metric_key"], "XPD_early_db")
        self.assertTrue(bool(out["used_expected_parity_subset_A2"]))
        self.assertTrue(bool(out["used_expected_parity_subset_A3"]))
        self.assertTrue(bool(out["pass_A2_negative"]))
        self.assertTrue(bool(out["pass_A3_positive"]))

    def test_breaking_check_prefers_raw_xpd(self) -> None:
        rows = []
        # A5 base
        for v in [6.0, 6.2, 5.8, 6.1]:
            rows.append(
                {
                    "scenario_id": "A5",
                    "XPD_early_db": v,
                    "XPD_early_excess_db": v - 23.0,
                    "XPD_late_db": 0.0,
                    "XPD_late_excess_db": -23.0,
                    "roughness_flag": 0,
                    "human_flag": 0,
                }
            )
        # A5 stress
        for v in [-2.0, -1.8, -2.2, -1.9]:
            rows.append(
                {
                    "scenario_id": "A5",
                    "XPD_early_db": v,
                    "XPD_early_excess_db": v - 23.0,
                    "XPD_late_db": 0.0,
                    "XPD_late_excess_db": -23.0,
                    "roughness_flag": 1,
                    "human_flag": 0,
                }
            )

        out = check_A4_A5_breaking(rows)
        self.assertEqual(out["metric_key_early"], "XPD_early_db")
        self.assertEqual(out["metric_key_late"], "XPD_late_db")
        self.assertTrue(bool(out["pass_breaking_trend"]))


if __name__ == "__main__":
    unittest.main()


"""Reciprocity coverage regression tests for canonical scenarios."""

from __future__ import annotations

import unittest

from scenarios.runner import build_dataset, compute_reciprocity_checks


class ReciprocityCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = build_dataset(
            basis="circular",
            nf=16,
            antenna_config={
                "convention": "IEEE-RHCP",
                "tx_cross_pol_leakage_db": 120.0,
                "rx_cross_pol_leakage_db": 120.0,
                "tx_axial_ratio_db": 0.0,
                "rx_axial_ratio_db": 0.0,
                "enable_coupling": False,
            },
            force_cp_swap_on_odd_reflection=False,
        )
        cls.metrics = compute_reciprocity_checks(
            cls.data,
            matrix_source="J",
            scenario_ids=None,
            tau_tol_s=1e-12,
            sigma_tol_db=1e-6,
            require_bidirectional_paths=True,
        )

    def test_checked_cases_cover_all_cases(self) -> None:
        self.assertEqual(int(self.metrics["covered_cases"]), int(self.metrics["checked_cases"]))
        self.assertEqual(int(self.metrics["reverse_empty_cases"]), 0)
        self.assertTrue(bool(self.metrics["coverage_pass"]))

    def test_per_case_bidirectional_and_invariants(self) -> None:
        for entry in self.metrics.get("entries", []):
            sid = str(entry.get("scenario_id", "NA"))
            cid = str(entry.get("case_id", "NA"))
            ctx = f"{sid}/{cid}"
            self.assertGreater(int(entry.get("n_forward", 0)), 0, ctx)
            self.assertGreater(int(entry.get("n_reverse", 0)), 0, ctx)
            self.assertEqual(int(entry.get("n_forward", 0)), int(entry.get("n_reverse", 0)), ctx)
            self.assertLessEqual(float(entry.get("delta_tau_max_s", 0.0)), 1e-12 + 1e-18, ctx)
            self.assertLessEqual(float(entry.get("delta_sigma_max_db", 0.0)), 1e-6 + 1e-12, ctx)
            self.assertLessEqual(float(entry.get("delta_fro_max_db", 0.0)), 1e-6 + 1e-12, ctx)


if __name__ == "__main__":
    unittest.main()

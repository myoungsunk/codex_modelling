"""Tests for XPD GOF model-selection flow."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.xpd_stats import gof_model_selection_db


class GOFModelSelectionTests(unittest.TestCase):
    def test_selects_alternative_for_mixture_data(self) -> None:
        rng = np.random.default_rng(123)
        x1 = rng.normal(20.0, 1.0, size=320)
        x2 = rng.normal(38.0, 1.3, size=320)
        vals = np.concatenate([x1, x2]).astype(float)

        res = gof_model_selection_db(vals, min_n=200, bootstrap_B=40, seed=7)

        self.assertIn(str(res["status"]), {"PASS", "PASS_ALTERNATIVE"})
        self.assertGreaterEqual(int(res["n"]), 640)
        self.assertTrue(bool(res.get("single_normal_fail", False)))
        self.assertNotEqual(str(res.get("best_model", "normal_db")), "normal_db")
        self.assertTrue(bool(res.get("alternative_improved", False)))

    def test_floor_pinned_exclusion_is_reported(self) -> None:
        rng = np.random.default_rng(222)
        floor_cluster = rng.normal(28.98, 0.08, size=280)
        slab = rng.normal(46.0, 2.0, size=320)
        vals = np.concatenate([floor_cluster, slab]).astype(float)

        res = gof_model_selection_db(
            vals,
            min_n=200,
            bootstrap_B=20,
            seed=9,
            floor_db=28.98,
            pinned_tol_db=0.4,
        )

        self.assertIn(str(res["status"]), {"PASS", "PASS_ALTERNATIVE"})
        self.assertGreater(int(res.get("excluded_floor_count", 0)), 100)
        self.assertGreater(int(res.get("n_excluded", 0)), 100)
        self.assertGreaterEqual(int(res.get("n_fit", 0)), 200)
        self.assertGreater(float(res.get("floor_ratio", 0.0)), 0.1)
        self.assertGreater(float(res.get("point_mass_ratio", 0.0)), 0.1)

    def test_insufficient_continuous_part_reports_censoring(self) -> None:
        rng = np.random.default_rng(333)
        floor_cluster = rng.normal(30.0, 0.05, size=60)
        slab = rng.normal(45.0, 0.5, size=8)
        vals = np.concatenate([floor_cluster, slab]).astype(float)

        res = gof_model_selection_db(
            vals,
            min_n=20,
            bootstrap_B=10,
            seed=11,
            floor_db=30.0,
            pinned_tol_db=0.3,
        )

        self.assertEqual(str(res["status"]), "INSUFFICIENT")
        self.assertGreater(float(res.get("floor_ratio", 0.0)), 0.5)
        self.assertGreater(float(res.get("point_mass_ratio", 0.0)), 0.5)
        self.assertIn(
            str(res.get("diagnostic_class", "")),
            {"INSUFFICIENT_N", "POINT_MASS_DOMINANT_INSUFFICIENT"},
        )


if __name__ == "__main__":
    unittest.main()

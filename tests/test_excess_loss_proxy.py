"""Tests for FSPL-based excess-loss proxy computations."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.el_proxy import compute_el_proxy
from analysis.excess_loss import add_el_db
from physics.fspl import fspl_db


class ExcessLossProxyTests(unittest.TestCase):
    def test_add_el_db_consistency_with_fspl_reference(self) -> None:
        f0 = 8e9
        L = 10.0
        fspl = float(fspl_db(L, f0))
        P_ideal = 10.0 ** (-fspl / 10.0)
        rows = [{"tau_s": 1e-9, "L_m": L, "n_bounce": 0, "P_lin": P_ideal}]
        out = add_el_db(rows, f_center_hz=f0, method="fspl")
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out[0]["EL_db"]), 0.0, places=9)

    def test_compute_el_proxy_early_sum(self) -> None:
        f0 = 8e9
        rows = [
            {"tau_s": 1e-9, "L_m": 5.0, "n_bounce": 0, "P_lin": 1e-5, "los_flag_ray": 1, "EL_db": 2.0},
            {"tau_s": 6e-9, "L_m": 7.0, "n_bounce": 1, "P_lin": 2e-6, "los_flag_ray": 0, "EL_db": 8.0},
        ]
        pdp = {
            "delay_tau_s": np.asarray([0.0, 2e-9, 4e-9, 6e-9], dtype=float),
            "P_co": np.asarray([2e-6, 1e-6, 2e-7, 1e-7], dtype=float),
            "P_cross": np.asarray([2e-7, 1e-7, 2e-8, 1e-8], dtype=float),
        }
        early = np.asarray([True, True, False, False], dtype=bool)
        el = compute_el_proxy(rows, pdp=pdp, mode="early_sum", early_mask=early, f_center_hz=f0)
        self.assertTrue(np.isfinite(el))
        # should be in a physically plausible positive-loss range.
        self.assertGreater(el, -200.0)
        self.assertLess(el, 200.0)

    def test_compute_el_proxy_dominant_early_ray_uses_existing_el(self) -> None:
        rows = [
            {"tau_s": 1e-9, "L_m": 5.0, "P_lin": 1e-5, "EL_db": 3.5},
            {"tau_s": 2e-9, "L_m": 6.0, "P_lin": 1e-6, "EL_db": 9.0},
        ]
        el = compute_el_proxy(rows, mode="dominant_early_ray")
        self.assertAlmostEqual(float(el), 3.5, places=9)


if __name__ == "__main__":
    unittest.main()

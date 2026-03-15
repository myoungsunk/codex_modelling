"""Tests for ray->dual-CP PDP synthesis."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.pdp_synthesis import synthesize_dualcp_pdp
from polarization.xpr_models import ConstantXPR


class PDPSynthesisTests(unittest.TestCase):
    def test_single_ray_binning_and_xpd_tau(self) -> None:
        rays = [{"tau_s": 1e-9, "P_lin": 1.0, "n_bounce": 0, "parity": "even"}]
        delay = np.asarray([0.0, 1e-9, 2e-9], dtype=float)
        out = synthesize_dualcp_pdp(rays, delay, xpr_model=ConstantXPR(10.0), link_U={"scenario_id": "A2"})
        self.assertEqual(len(out.P_co), 3)
        self.assertEqual(len(out.P_cross), 3)
        expected_co = 10.0 / 11.0
        expected_cross = 1.0 / 11.0
        self.assertAlmostEqual(float(out.P_co[1]), expected_co, places=9)
        self.assertAlmostEqual(float(out.P_cross[1]), expected_cross, places=9)
        self.assertAlmostEqual(float(out.P_co[0] + out.P_cross[0]), 0.0, places=12)
        self.assertAlmostEqual(float(out.P_co[2] + out.P_cross[2]), 0.0, places=12)
        self.assertAlmostEqual(float(out.XPD_tau_db[1]), 10.0, places=6)


if __name__ == "__main__":
    unittest.main()

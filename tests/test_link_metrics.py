"""Tests for link summary metrics Z definitions."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.link_metrics import compute_link_metrics


class LinkMetricsTests(unittest.TestCase):
    def test_metrics_match_formula(self) -> None:
        tau = np.asarray([0.0, 1.0e-9, 2.0e-9, 3.0e-9], dtype=float)
        pco = np.asarray([10.0, 5.0, 2.0, 1.0], dtype=float)
        pcx = np.asarray([1.0, 1.0, 2.0, 2.0], dtype=float)
        early = np.asarray([True, True, False, False], dtype=bool)
        late = np.asarray([False, False, True, True], dtype=bool)
        out = compute_link_metrics(
            pdp={"P_co": pco, "P_cross": pcx},
            delay_tau_s=tau,
            masks=(early, late),
            ds_reference="total",
            window_params={"Te_s": 2e-9},
        )
        xpd_e = 10.0 * np.log10((np.sum(pco[early]) + 1e-15) / (np.sum(pcx[early]) + 1e-15))
        xpd_l = 10.0 * np.log10((np.sum(pco[late]) + 1e-15) / (np.sum(pcx[late]) + 1e-15))
        rho_lin = (np.sum(pcx[early]) + 1e-15) / (np.sum(pco[early]) + 1e-15)
        w = pco + pcx
        mu = np.sum(w * tau) / np.sum(w)
        ds = np.sqrt(np.sum(w * (tau ** 2)) / np.sum(w) - mu**2)
        frac = np.sum(w[early]) / np.sum(w)

        self.assertAlmostEqual(out.XPD_early_db, float(xpd_e), places=9)
        self.assertAlmostEqual(out.XPD_late_db, float(xpd_l), places=9)
        self.assertAlmostEqual(out.rho_early_lin, float(rho_lin), places=9)
        self.assertAlmostEqual(out.rho_early_db, float(10.0 * np.log10(rho_lin + 1e-15)), places=9)
        self.assertAlmostEqual(out.L_pol_db, float(xpd_e - xpd_l), places=9)
        self.assertAlmostEqual(out.delay_spread_rms_s, float(ds), places=12)
        self.assertAlmostEqual(out.early_energy_fraction, float(frac), places=12)
        self.assertEqual(float(out.window.get("Te_s")), 2e-9)


if __name__ == "__main__":
    unittest.main()

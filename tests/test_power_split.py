"""Tests for parity sign rule and co/cross split."""

from __future__ import annotations

import unittest

from polarization.power_split import split_power, xpd_from_xpr


class PowerSplitTests(unittest.TestCase):
    def test_parity_sign_flip(self) -> None:
        self.assertAlmostEqual(xpd_from_xpr("even", 7.0), 7.0, places=9)
        self.assertAlmostEqual(xpd_from_xpr("odd", 7.0), -7.0, places=9)

    def test_power_conservation(self) -> None:
        P = 1.25
        pco, pcx = split_power(P, xpd_db=10.0)
        self.assertAlmostEqual(pco + pcx, P, places=12)
        self.assertGreater(pco, pcx)

        pco2, pcx2 = split_power(P, xpd_db=-10.0)
        self.assertAlmostEqual(pco2 + pcx2, P, places=12)
        self.assertLess(pco2, pcx2)


if __name__ == "__main__":
    unittest.main()

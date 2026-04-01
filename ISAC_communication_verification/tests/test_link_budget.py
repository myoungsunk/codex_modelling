from __future__ import annotations

import unittest

import numpy as np

from dualpol_rt import convert_basis
from dualpol_rt.metrics.link_budget import (
    best_port_rate,
    mimo_multiplexing_efficiency,
    per_port_rx_gain,
    polarization_diversity_gain,
    port_imbalance_db,
    standard_fixed_rates,
    total_rx_gain,
)


class LinkBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.H_lp = rng.normal(size=(21, 2, 2)) + 1j * rng.normal(size=(21, 2, 2))
        self.H_cp = convert_basis(self.H_lp, src="linear", dst="circular")

    def test_basis_rotation_preserves_total_gain_but_changes_port_split(self) -> None:
        self.assertAlmostEqual(total_rx_gain(self.H_lp), total_rx_gain(self.H_cp), places=12)
        self.assertFalse(np.allclose(per_port_rx_gain(self.H_lp), per_port_rx_gain(self.H_cp), atol=1e-10, rtol=0.0))
        self.assertNotAlmostEqual(port_imbalance_db(self.H_lp), port_imbalance_db(self.H_cp), places=8)

    def test_best_port_and_fixed_rates_are_basis_dependent(self) -> None:
        self.assertNotAlmostEqual(best_port_rate(self.H_lp, 10.0), best_port_rate(self.H_cp, 10.0), places=8)
        rates_lp = standard_fixed_rates(self.H_lp, (10.0,))
        rates_cp = standard_fixed_rates(self.H_cp, (10.0,))
        self.assertNotAlmostEqual(rates_lp["single_port"][10.0], rates_cp["single_port"][10.0], places=8)
        self.assertNotAlmostEqual(rates_lp["equal_gain"][10.0], rates_cp["equal_gain"][10.0], places=8)
        pdg_lp = polarization_diversity_gain(rates_lp["single_port"][10.0], rates_lp["equal_gain"][10.0])
        pdg_cp = polarization_diversity_gain(rates_cp["single_port"][10.0], rates_cp["equal_gain"][10.0])
        self.assertNotAlmostEqual(pdg_lp, pdg_cp, places=8)

    def test_multiplexing_efficiency_helper_is_positive(self) -> None:
        eff = mimo_multiplexing_efficiency(capacity=2.5, best_port_rate_value=0.7, n_streams=2)
        self.assertGreater(eff, 0.0)


if __name__ == "__main__":
    unittest.main()

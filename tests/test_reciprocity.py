"""Reciprocity sanity checks for simple scenarios."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.reciprocity import reciprocity_sanity
from scenarios.C0_free_space import build_scene
from scenarios.common import default_antennas


class ReciprocityTests(unittest.TestCase):
    def test_c0_reciprocity_invariants(self) -> None:
        f_hz = np.linspace(6e9, 7e9, 32)
        tx, rx = default_antennas(
            basis="linear",
            convention="IEEE-RHCP",
            tx_cross_pol_leakage_db=120.0,
            rx_cross_pol_leakage_db=120.0,
            tx_axial_ratio_db=0.0,
            rx_axial_ratio_db=0.0,
            enable_coupling=False,
        )
        rx.position[:] = [6.0, 0.0, 1.5]
        out = reciprocity_sanity(
            scene=build_scene(),
            tx=tx,
            rx=rx,
            f_hz=f_hz,
            max_bounce=0,
            los_enabled=True,
            matrix_source="J",
        )
        self.assertGreaterEqual(float(out["matched_ratio"]), 0.99)
        self.assertLessEqual(float(out["delta_tau_max_s"]), 1e-12)
        self.assertLessEqual(float(out["delta_sigma_max_db"]), 1e-9)


if __name__ == "__main__":
    unittest.main()

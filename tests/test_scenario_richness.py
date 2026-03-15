"""Scenario richness checks for robust statistics."""

from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from scenarios import A5_depol_stress, B0_room_box


class ScenarioRichnessTests(unittest.TestCase):
    def test_a5_has_multiple_repetitions_per_rho(self) -> None:
        params = A5_depol_stress.build_sweep_params()
        self.assertGreater(len(params), 0)

        counts = Counter(float(p["rho"]) for p in params)
        self.assertTrue(all(c >= 2 for c in counts.values()))

        # Seeds should vary by repetition so depol realizations are different.
        self.assertEqual(len({int(p["seed"]) for p in params}), len(params))

    def test_b0_room_box_yields_multipath(self) -> None:
        f_hz = np.linspace(6e9, 7e9, 16)
        params = B0_room_box.build_sweep_params()[0]
        paths = B0_room_box.run_case(params, f_hz, basis="linear", antenna_config={})
        self.assertGreaterEqual(len(paths), 4)


if __name__ == "__main__":
    unittest.main()

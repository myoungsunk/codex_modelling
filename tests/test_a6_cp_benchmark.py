"""Controlled A6 CP parity benchmark checks."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.xpd_stats import pathwise_xpd
from scenarios import A6_cp_parity_benchmark as A6


class A6CPBenchmarkTests(unittest.TestCase):
    def test_a6_near_normal_odd_even_separation_with_propagation_only_j(self) -> None:
        f_hz = np.linspace(6e9, 10e9, 64)
        ant_cfg = {
            "convention": "IEEE-RHCP",
            "tx_cross_pol_leakage_db": 120.0,
            "rx_cross_pol_leakage_db": 120.0,
            "tx_axial_ratio_db": 0.0,
            "rx_axial_ratio_db": 0.0,
            "enable_coupling": False,
        }

        odd_xpd: list[float] = []
        even_xpd: list[float] = []
        odd_angles: list[float] = []
        even_angles: list[float] = []

        for params in A6.build_sweep_params():
            paths = A6.run_case(params, f_hz, basis="circular", antenna_config=ant_cfg)
            records = [
                {
                    "tau_s": p.tau_s,
                    "A_f": p.A_f,
                    "J_f": p.J_f,
                    "meta": {"bounce_count": p.bounce_count, "incidence_angles": p.incidence_angles},
                }
                for p in paths
            ]
            samples = pathwise_xpd(records, matrix_source="J")
            for s, p in zip(samples, paths):
                ang = float(np.rad2deg(max(p.incidence_angles))) if p.incidence_angles else 0.0
                if int(p.bounce_count) == 1:
                    odd_xpd.append(float(s["xpd_db"]))
                    odd_angles.append(ang)
                elif int(p.bounce_count) == 2:
                    even_xpd.append(float(s["xpd_db"]))
                    even_angles.append(ang)

        self.assertGreaterEqual(len(odd_xpd), 6)
        self.assertGreaterEqual(len(even_xpd), 6)
        self.assertLess(float(np.median(odd_xpd)), float(np.median(even_xpd)))
        self.assertLessEqual(max(odd_angles), 15.0 + 1e-6)
        self.assertLessEqual(max(even_angles), 15.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()

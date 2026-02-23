"""Focused unit tests for XPD statistics definitions."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.xpd_stats import pathwise_xpd


class XPDStatsCoreTests(unittest.TestCase):
    def test_pathwise_xpd_uses_power_average(self) -> None:
        nf = 4
        a = np.zeros((nf, 2, 2), dtype=np.complex128)
        # Complex sign flips should not change power-based average.
        a[:, 0, 0] = np.array([1.0, -1.0, 1j, -1j])
        a[:, 1, 1] = 0.0
        a[:, 0, 1] = 0.5 * np.array([1.0, -1.0, 1j, -1j])
        a[:, 1, 0] = 0.0
        path = {"tau_s": 1e-8, "A_f": a, "J_f": a.copy(), "meta": {"bounce_count": 1}}
        out = pathwise_xpd([path], matrix_source="A")
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out[0]["xpd_db"]), 6.0205999, places=5)

    def test_exact_bounce_filter_applies(self) -> None:
        nf = 8
        base = np.zeros((nf, 2, 2), dtype=np.complex128)
        base[:, 0, 0] = 1.0
        base[:, 1, 1] = 1.0
        p0 = {"tau_s": 10e-9, "A_f": base, "J_f": base.copy(), "meta": {"bounce_count": 0}}
        p1 = {"tau_s": 20e-9, "A_f": base, "J_f": base.copy(), "meta": {"bounce_count": 1}}
        p2 = {"tau_s": 30e-9, "A_f": base, "J_f": base.copy(), "meta": {"bounce_count": 2}}
        out = pathwise_xpd([p0, p1, p2], exact_bounce=2)
        self.assertEqual(len(out), 1)
        self.assertEqual(int(out[0]["bounce_count"]), 2)

    def test_matrix_source_switch_prefers_j_when_present(self) -> None:
        nf = 8
        a = np.zeros((nf, 2, 2), dtype=np.complex128)
        j = np.zeros((nf, 2, 2), dtype=np.complex128)
        a[:, 0, 0] = 1.0
        a[:, 1, 1] = 1.0
        a[:, 0, 1] = 1.0  # 0 dB
        j[:, 0, 0] = 1.0
        j[:, 1, 1] = 1.0
        j[:, 0, 1] = 0.1  # higher XPD
        path = {"tau_s": 1e-8, "A_f": a, "J_f": j, "meta": {"bounce_count": 1}}
        xa = pathwise_xpd([path], matrix_source="A")[0]["xpd_db"]
        xj = pathwise_xpd([path], matrix_source="J")[0]["xpd_db"]
        self.assertGreater(float(xj), float(xa))


if __name__ == "__main__":
    unittest.main()

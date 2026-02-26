"""Unit tests for dual-CP floor calibration."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.dualcp_calibration import apply_floor_excess, estimate_xpd_floor_from_cases


class DualCpCalibrationTests(unittest.TestCase):
    def _toy_case(self, freq: np.ndarray, co_amp: float, cross_amp: float, case_id: str) -> dict:
        H = np.zeros((len(freq), 2, 2), dtype=np.complex128)
        H[:, 0, 0] = complex(co_amp, 0.0)
        H[:, 1, 0] = complex(cross_amp, 0.0)
        return {
            "scenario_id": "C0",
            "case_id": case_id,
            "frequency_hz": np.asarray(freq, dtype=float),
            "H_f": H,
        }

    def test_estimate_floor_shape_and_percentiles(self) -> None:
        f = np.linspace(6.0e9, 6.3e9, 4)
        c1 = self._toy_case(f, co_amp=1.0, cross_amp=0.1, case_id="0")  # ~20 dB
        c2 = self._toy_case(f, co_amp=1.0, cross_amp=0.2, case_id="1")  # ~13.98 dB
        out = estimate_xpd_floor_from_cases(
            cases=[c1, c2],
            method="median",
            subbands=[(0, 2), (2, 4)],
            percentiles=(5.0, 95.0),
        )
        self.assertEqual(len(out["frequency_hz"]), 4)
        self.assertEqual(len(out["xpd_floor_db"]), 4)
        self.assertEqual(len(out["xpd_floor_uncert_db"]), 4)
        self.assertEqual(len(out["subbands"]), 2)
        floor = np.asarray(out["xpd_floor_db"], dtype=float)
        self.assertTrue(np.all(np.isfinite(floor)))
        # median([20,13.9794]) = 16.9897...
        self.assertAlmostEqual(float(np.mean(floor)), 16.9897, places=3)
        uncert = np.asarray(out["xpd_floor_uncert_db"], dtype=float)
        self.assertTrue(np.all(uncert > 0.0))

    def test_apply_floor_excess(self) -> None:
        xpd = np.asarray([10.0, 12.0, 14.0], dtype=float)
        floor = np.asarray([7.0, 8.0, 9.0], dtype=float)
        ex = apply_floor_excess(xpd, floor)
        np.testing.assert_allclose(ex, np.asarray([3.0, 4.0, 5.0], dtype=float))
        ex2 = apply_floor_excess(xpd, 8.0)
        np.testing.assert_allclose(ex2, np.asarray([2.0, 4.0, 6.0], dtype=float))


if __name__ == "__main__":
    unittest.main()

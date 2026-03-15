"""Tests for C0 floor XPD models."""

from __future__ import annotations

import unittest

import numpy as np

from calibration.floor_model import AngleSensitiveFloorXPD, ConstantFloorXPD, FreqDependentFloorXPD


class C0FloorModelTests(unittest.TestCase):
    def test_constant_floor(self) -> None:
        m = ConstantFloorXPD(xpd_floor_db=24.0)
        self.assertAlmostEqual(m.sample_floor_xpd_db(f_hz=8e9), 24.0, places=9)

    def test_freq_dependent_floor_changes_with_frequency(self) -> None:
        m = FreqDependentFloorXPD(
            freq_hz=np.asarray([6e9, 8e9, 10e9], dtype=float),
            xpd_floor_db=np.asarray([20.0, 25.0, 30.0], dtype=float),
        )
        a = m.sample_floor_xpd_db(f_hz=6e9)
        b = m.sample_floor_xpd_db(f_hz=10e9)
        self.assertLess(a, b)

    def test_angle_sensitive_floor_changes_with_yaw(self) -> None:
        m = AngleSensitiveFloorXPD(base_db=25.0, yaw_slope_db_per_deg=-0.1, pitch_slope_db_per_deg=0.0)
        a = m.sample_floor_xpd_db(f_hz=8e9, yaw_deg=0.0)
        b = m.sample_floor_xpd_db(f_hz=8e9, yaw_deg=20.0)
        self.assertGreater(a, b)


if __name__ == "__main__":
    unittest.main()

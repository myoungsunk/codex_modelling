"""Tests for frequency-dependent complex antenna coupling model."""

from __future__ import annotations

import unittest

import numpy as np

from rt_core.antenna import Antenna


class AntennaCouplingTests(unittest.TestCase):
    def _ant(self, **kwargs) -> Antenna:
        return Antenna(
            position=np.array([0.0, 0.0, 0.0]),
            boresight=np.array([1.0, 0.0, 0.0]),
            h_axis=np.array([0.0, 1.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            basis="linear",
            **kwargs,
        )

    def test_default_coupling_is_frequency_flat_real(self) -> None:
        ant = self._ant(cross_pol_leakage_db=35.0, axial_ratio_db=0.0, enable_coupling=True)
        f = np.linspace(6e9, 10e9, 5)
        c = ant._coupling_matrix(f)
        off = c[:, 0, 1]
        self.assertTrue(np.allclose(np.imag(off), 0.0, atol=1e-12))
        self.assertAlmostEqual(float(np.max(np.abs(np.diff(np.abs(off))))), 0.0, places=12)

    def test_coupling_leakage_slope_changes_with_frequency(self) -> None:
        ant = self._ant(
            cross_pol_leakage_db=35.0,
            cross_pol_leakage_db_slope_per_ghz=-3.0,
            coupling_ref_freq_hz=8e9,
            enable_coupling=True,
        )
        f = np.array([6e9, 8e9, 10e9], dtype=float)
        c = ant._coupling_matrix(f)
        off_mag = np.abs(c[:, 0, 1])
        self.assertGreater(float(off_mag[2]), float(off_mag[1]))
        self.assertGreater(float(off_mag[1]), float(off_mag[0]))

    def test_coupling_phase_terms_are_complex_and_conjugate_symmetric(self) -> None:
        ant = self._ant(
            cross_pol_leakage_db=35.0,
            cross_coupling_phase_deg=20.0,
            cross_coupling_phase_slope_deg_per_ghz=5.0,
            coupling_ref_freq_hz=8e9,
            enable_coupling=True,
        )
        f = np.array([6e9, 8e9, 10e9], dtype=float)
        c = ant._coupling_matrix(f)
        off12 = c[:, 0, 1]
        off21 = c[:, 1, 0]
        self.assertTrue(np.any(np.abs(np.imag(off12)) > 1e-12))
        self.assertTrue(np.allclose(off21, np.conj(off12), atol=1e-12))
        ph = np.unwrap(np.angle(off12))
        self.assertNotAlmostEqual(float(ph[-1]), float(ph[0]), places=6)

    def test_vectors_are_read_only(self) -> None:
        ant = self._ant()
        self.assertFalse(bool(ant.position.flags.writeable))
        self.assertFalse(bool(ant.boresight.flags.writeable))
        with self.assertRaises(ValueError):
            ant.position[0] = 1.0

    def test_with_orientation_re_normalizes_axes(self) -> None:
        ant = self._ant()
        new_ant = ant.with_orientation(
            boresight=np.array([2.0, 0.0, 0.0]),
            h_axis=np.array([0.5, 1.0, 0.0]),
            v_axis=np.array([0.1, 0.2, 3.0]),
        ).with_position(np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(new_ant.position, np.array([1.0, 2.0, 3.0]), atol=1e-12))
        self.assertAlmostEqual(float(np.linalg.norm(new_ant.boresight)), 1.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(new_ant.h_axis)), 1.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(new_ant.v_axis)), 1.0, places=12)
        self.assertAlmostEqual(float(np.dot(new_ant.boresight, new_ant.h_axis)), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(new_ant.boresight, new_ant.v_axis)), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()

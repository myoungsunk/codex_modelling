"""Core polarization unit tests."""

from __future__ import annotations

import unittest

import numpy as np

from rt_core.geometry import Material
from rt_core.polarization import depol_matrix, fresnel_reflection, local_sp_bases


class PolarizationCoreTests(unittest.TestCase):
    def test_fresnel_pec_is_minus_one(self) -> None:
        f = np.linspace(6e9, 8e9, 9)
        gs, gp = fresnel_reflection(Material.pec(), theta_i=np.deg2rad(33.0), f_hz=f)
        self.assertTrue(np.allclose(gs, -1.0 + 0.0j))
        self.assertTrue(np.allclose(gp, -1.0 + 0.0j))

    def test_dielectric_fresnel_is_passive(self) -> None:
        mat = Material.dielectric(eps_r=4.2, tan_delta=0.02, name="test")
        f = np.linspace(6e9, 10e9, 17)
        gs, gp = fresnel_reflection(mat, theta_i=np.deg2rad(45.0), f_hz=f)
        self.assertTrue(np.all(np.isfinite(gs)))
        self.assertTrue(np.all(np.isfinite(gp)))
        self.assertLessEqual(float(np.max(np.abs(gs))), 1.0000001)
        self.assertLessEqual(float(np.max(np.abs(gp))), 1.0000001)

    def test_depol_matrix_is_unitary(self) -> None:
        rng = np.random.default_rng(42)
        d = depol_matrix(rho=0.37, rng=rng)
        ident = d.conj().T @ d
        self.assertTrue(np.allclose(ident, np.eye(2), atol=1e-12))

    def test_local_sp_bases_are_orthonormal(self) -> None:
        k_in = np.array([0.4, -0.3, -0.8660254], dtype=float)
        k_out = np.array([0.4, -0.3, 0.8660254], dtype=float)
        n = np.array([0.0, 0.0, 1.0], dtype=float)
        s_in, p_in, s_out, p_out, theta_i, n_eff = local_sp_bases(k_in, k_out, n)
        self.assertGreaterEqual(theta_i, 0.0)
        self.assertLessEqual(theta_i, np.pi / 2.0 + 1e-12)
        self.assertAlmostEqual(float(np.dot(s_in, p_in)), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(s_out, p_out)), 0.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(s_in)), 1.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(p_out)), 1.0, places=12)
        # Effective normal must oppose incident direction.
        self.assertLessEqual(float(np.dot(k_in / np.linalg.norm(k_in), n_eff)), 0.0)


if __name__ == "__main__":
    unittest.main()

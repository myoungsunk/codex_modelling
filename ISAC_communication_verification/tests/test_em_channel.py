from __future__ import annotations

import unittest

import numpy as np

from dualpol_rt.channel.builder import build_channel, convert_basis
from dualpol_rt.config import ExperimentConfig
from dualpol_rt.em.basis import local_te_tm_bases
from dualpol_rt.em.fresnel import fresnel_reflection
from dualpol_rt.em.materials import Material
from dualpol_rt.experiments.basis_invariance import evaluate_rx_grid
from dualpol_rt.geometry.image_method import enumerate_paths
from dualpol_rt.metrics.achievable_rate import equal_power_rate, waterfilled_capacity
from dualpol_rt.metrics.xpr import condition_numbers, singular_values
from dualpol_rt.patterns.ideal_pattern import IdealPattern
from dualpol_rt.scene.shoebox import build_shoebox
from dualpol_rt.scene.surfaces import Scene


class EMChannelTests(unittest.TestCase):
    def test_free_space_crosspol_is_negligible(self) -> None:
        scene = Scene(surfaces=tuple())
        tx = np.array([0.0, 0.0, 1.0], dtype=float)
        rx = np.array([2.0, 0.0, 1.0], dtype=float)
        freqs = np.linspace(6.0e9, 7.0e9, 17)
        paths = enumerate_paths(scene, tx, rx, max_reflections=0)
        pattern = IdealPattern(basis="linear")
        H = build_channel(paths, tx_pattern=pattern, rx_pattern=pattern, freqs_hz=freqs)
        cross = np.mean(np.abs(H[:, 0, 1]) ** 2 + np.abs(H[:, 1, 0]) ** 2)
        self.assertLess(cross, 1e-12)

    def test_basis_conversion_consistency_and_rate_invariance(self) -> None:
        scene = build_shoebox(material_preset="uniform_concrete")
        tx = np.array([1.0, 1.0, 1.5], dtype=float)
        rx = np.array([1.6, 1.4, 1.2], dtype=float)
        freqs = np.linspace(6.25e9, 6.75e9, 21)
        paths = enumerate_paths(scene, tx, rx, max_reflections=2)
        pattern = IdealPattern(basis="linear")
        H_lin = build_channel(paths, tx_pattern=pattern, rx_pattern=pattern, freqs_hz=freqs, eval_basis="linear")
        H_circ = convert_basis(H_lin, src="linear", dst="circular")
        H_back = convert_basis(H_circ, src="circular", dst="linear")
        self.assertTrue(np.allclose(H_lin, H_back, atol=1e-10))
        self.assertAlmostEqual(equal_power_rate(H_lin, 10.0), equal_power_rate(H_circ, 10.0), places=10)
        self.assertAlmostEqual(waterfilled_capacity(H_lin, 10.0), waterfilled_capacity(H_circ, 10.0), places=10)
        self.assertTrue(np.allclose(singular_values(H_lin), singular_values(H_circ), atol=1e-10))
        self.assertTrue(np.allclose(condition_numbers(H_lin), condition_numbers(H_circ), atol=1e-10))

    def test_local_te_tm_bases_are_orthonormal(self) -> None:
        kin = np.array([0.4, -0.3, -0.8660254], dtype=float)
        kout = np.array([0.4, -0.3, 0.8660254], dtype=float)
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        te_in, tm_in, te_out, tm_out, theta_i, n_eff = local_te_tm_bases(kin, kout, normal)
        self.assertAlmostEqual(float(np.dot(te_in, tm_in)), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(te_out, tm_out)), 0.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(te_in)), 1.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(tm_out)), 1.0, places=12)
        self.assertGreaterEqual(theta_i, 0.0)
        self.assertLessEqual(float(np.dot(kin / np.linalg.norm(kin), n_eff)), 0.0)

    def test_material_passivity(self) -> None:
        material = Material(name="concrete", eps_r_const=5.31, sigma_const=0.0326)
        freqs = np.linspace(6.0e9, 7.0e9, 21)
        gamma_te, gamma_tm = fresnel_reflection(material, theta_i=np.deg2rad(35.0), freqs_hz=freqs)
        self.assertTrue(np.all(np.abs(gamma_te) <= 1.0 + 1e-12))
        self.assertTrue(np.all(np.abs(gamma_tm) <= 1.0 + 1e-12))

    def test_pec_reflection_signs(self) -> None:
        freqs = np.linspace(6.0e9, 7.0e9, 3)
        gamma_te, gamma_tm = fresnel_reflection(Material.pec(), theta_i=np.deg2rad(20.0), freqs_hz=freqs)
        self.assertTrue(np.allclose(gamma_te, -1.0 + 0.0j))
        self.assertTrue(np.allclose(gamma_tm, 1.0 + 0.0j))

    def test_evaluate_rx_grid_smoke(self) -> None:
        cfg = ExperimentConfig(
            room_size=(2.0, 2.0, 2.0),
            tx_pos=np.array([0.6, 1.0, 1.5], dtype=float),
            rx_plane_z=1.0,
            grid_step=1.0,
            n_freq=9,
            max_reflections=1,
            snr_db_list=(10.0,),
        )
        pattern = IdealPattern(basis="linear")
        result = evaluate_rx_grid(cfg, tx_pattern=pattern, rx_pattern=pattern)
        self.assertEqual(result.delta_rate_maps[10.0].shape, (2, 2))
        self.assertLess(float(np.max(np.abs(result.delta_rate_maps[10.0]))), 1e-10)


if __name__ == "__main__":
    unittest.main()

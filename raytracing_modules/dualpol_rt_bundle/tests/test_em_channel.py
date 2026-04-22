from __future__ import annotations

import unittest

import numpy as np

from dualpol_rt.channel.builder import _receive_matrix, build_channel, convert_basis
from dualpol_rt.config import DepolConfig, ExperimentConfig
from dualpol_rt.em.basis import local_te_tm_bases, transverse_basis
from dualpol_rt.em.fresnel import fresnel_reflection
from dualpol_rt.em.jones import attach_em_response
from dualpol_rt.em.materials import DEFAULT_XPOL_COUPLING_SWEEP_DB, Material, material_bundle
from dualpol_rt.experiments.basis_invariance import evaluate_rx_grid
from dualpol_rt.geometry.image_method import enumerate_paths
from dualpol_rt.metrics.achievable_rate import equal_power_rate, waterfilled_capacity
from dualpol_rt.metrics.xpr import condition_numbers, singular_values
from dualpol_rt.patterns.ideal_pattern import IdealPattern
from dualpol_rt.scene.shoebox import build_shoebox
from dualpol_rt.scene.surfaces import Scene


class _WaveBasisPattern:
    port_count = 2

    def port_vector_world(self, port_id: int, f_hz: float | np.ndarray, k_world: np.ndarray) -> np.ndarray:
        freqs = np.atleast_1d(np.asarray(f_hz, dtype=float))
        u, v = transverse_basis(np.asarray(k_world, dtype=float))
        vec = np.asarray(u if int(port_id) == 0 else v, dtype=np.complex128)
        return np.repeat(vec[None, :], len(freqs), axis=0)


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

    def test_circular_ideal_pattern_defaults_to_rhcp_then_lhcp(self) -> None:
        pattern = IdealPattern(basis="circular")
        vector_rhcp = pattern.port_vector_world(0, 6.5e9, np.array([0.0, 0.0, 1.0], dtype=float))[0]
        vector_lhcp = pattern.port_vector_world(1, 6.5e9, np.array([0.0, 0.0, 1.0], dtype=float))[0]
        expected_rhcp = np.array([1.0, -1j, 0.0], dtype=np.complex128) / np.sqrt(2.0)
        expected_lhcp = np.array([1.0, 1j, 0.0], dtype=np.complex128) / np.sqrt(2.0)
        self.assertTrue(np.allclose(vector_rhcp, expected_rhcp, atol=1e-12))
        self.assertTrue(np.allclose(vector_lhcp, expected_lhcp, atol=1e-12))

    def test_itu_p2040_material_power_law_is_applied(self) -> None:
        material = Material.itu_p2040(name="concrete", a=5.24, b=0.0, c=0.0462, d=0.7822)
        freqs = np.array([6.0e9, 60.0e9], dtype=float)
        self.assertTrue(np.allclose(material.eps_r(freqs), np.array([5.24, 5.24], dtype=float), atol=1e-12))
        expected_sigma = 0.0462 * np.array([6.0, 60.0], dtype=float) ** 0.7822
        self.assertTrue(np.allclose(material.sigma(freqs), expected_sigma, rtol=1e-12, atol=1e-12))

    def test_material_bundle_xpol_override_is_global(self) -> None:
        bundle = material_bundle("indoor_open_office", xpol_coupling_db=20.0)
        self.assertEqual({mat.xpol_coupling_db for mat in bundle.values()}, {20.0})
        self.assertEqual(DEFAULT_XPOL_COUPLING_SWEEP_DB, (20.0, 25.0, 30.0, 35.0, 40.0))

    def test_invalid_pec_tm_sign_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            Material(name="bad_pec", kind="PEC", pec_tm_sign=1.0)

    def test_experiment_config_rejects_enabled_depolarization(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentConfig(depol=DepolConfig(enabled=True))

    def test_receive_matrix_compensates_look_direction_basis_flip(self) -> None:
        pattern = _WaveBasisPattern()
        recv = _receive_matrix(pattern, np.array([6.5e9], dtype=float), np.array([1.0, 0.0, 0.0], dtype=float))
        self.assertTrue(np.allclose(recv[0], np.eye(2, dtype=np.complex128), atol=1e-12))

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
        material = Material.itu_p2040(name="concrete", a=5.24, b=0.0, c=0.0462, d=0.7822)
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

    def test_build_channel_recomputes_stale_jones_for_changed_up_hint(self) -> None:
        scene = build_shoebox(material_preset="uniform_concrete")
        tx = np.array([1.0, 1.0, 1.5], dtype=float)
        rx = np.array([1.6, 1.4, 1.2], dtype=float)
        freqs = np.linspace(6.25e9, 6.75e9, 9)
        path = next(path for path in enumerate_paths(scene, tx, rx, max_reflections=1) if path.bounce_count == 1)
        pattern = IdealPattern(basis="linear")
        custom_up_hint = np.array([1.0, 1.0, 0.3], dtype=float)
        stale = attach_em_response(path, freqs_hz=freqs, up_hint=np.array([0.0, 1.0, 0.2], dtype=float))
        direct = build_channel([path], tx_pattern=pattern, rx_pattern=pattern, freqs_hz=freqs, up_hint=custom_up_hint)
        reused = build_channel([stale], tx_pattern=pattern, rx_pattern=pattern, freqs_hz=freqs, up_hint=custom_up_hint)
        self.assertTrue(np.allclose(direct, reused, atol=1e-10))


if __name__ == "__main__":
    unittest.main()

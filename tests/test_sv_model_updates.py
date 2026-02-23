"""Regression tests for updated SV polarimetric synthesis flow."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.sv_polarimetric_model import (
    generate_synthetic_paths,
    kuiper_uniform_test,
    offdiag_phases,
    summarize_rt_vs_synth,
)


def _simple_path(nf: int, tau_s: float, xpd_db: float, parity: str) -> dict:
    inv = 10.0 ** (-xpd_db / 20.0)
    A = np.zeros((nf, 2, 2), dtype=np.complex128)
    A[:, 0, 0] = 1.0
    A[:, 1, 1] = 1.0
    A[:, 0, 1] = inv
    A[:, 1, 0] = inv
    return {
        "tau_s": float(tau_s),
        "A_f": A,
        "J_f": A.copy(),
        "meta": {
            "bounce_count": 1 if parity == "odd" else 2,
            "parity": parity,
        },
    }


class SVModelUpdatesTests(unittest.TestCase):
    def test_num_paths_mode_match_rt_total_and_no_clamp(self) -> None:
        nf = 16
        f = np.linspace(6e9, 7e9, nf)
        d = np.linspace(10e-9, 40e-9, 12)
        p = np.linspace(1.0, 0.1, 12)
        synth, diag = generate_synthetic_paths(
            f_hz=f,
            num_rays=5,
            num_paths_mode="match_rt_total",
            rt_case_path_counts=np.asarray([3, 4, 5], dtype=np.int64),
            delay_samples_s=d,
            power_samples=p,
            parity_probs={"odd": 0.5, "even": 0.5},
            parity_fit={"odd": {"mu": 15.0, "sigma": 1.0}, "even": {"mu": 20.0, "sigma": 1.0}},
            matrix_source="J",
            kappa_min=1e-3,
            kappa_max=1e3,
            return_diagnostics=True,
            seed=1,
        )
        self.assertEqual(len(synth), len(d))
        self.assertAlmostEqual(float(diag["kappa_clamp_rate"]), 0.0, places=9)
        self.assertLessEqual(float(diag["kappa_truncation_rate"]), 1.0)
        self.assertGreaterEqual(int(diag.get("dropped_invalid_delay_samples", 0)), 0)

    def test_offdiag_phase_per_ray_sampling(self) -> None:
        nf = 32
        paths = [_simple_path(nf, 10e-9, 15.0, "odd"), _simple_path(nf, 20e-9, 18.0, "even")]
        a_all = offdiag_phases(paths, matrix_source="A", per_ray_sampling=False, common_phase_removed=False)
        a_ray = offdiag_phases(paths, matrix_source="A", per_ray_sampling=True, common_phase_removed=True)
        self.assertEqual(len(a_ray), 2 * len(paths))
        self.assertEqual(len(a_all), 2 * len(paths) * nf)

    def test_offdiag_phase_amp_gate_and_debug(self) -> None:
        nf = 8
        p = _simple_path(nf, 10e-9, 60.0, "odd")
        # Make one off-diagonal almost zero to exercise amplitude gating.
        p["A_f"][:, 0, 1] = 1e-12 + 0j
        p["J_f"] = np.asarray(p["A_f"], dtype=np.complex128).copy()
        phases, dbg = offdiag_phases(
            [p],
            matrix_source="A",
            per_ray_sampling=True,
            common_phase_removed=True,
            amp_ratio_min=1e-3,
            return_debug=True,
        )
        self.assertLess(len(phases), 2)
        self.assertEqual(int(dbg.get("n_total_candidates", 0)), 2)
        self.assertGreaterEqual(int(dbg.get("n_rejected_amp_gate", 0)), 1)
        self.assertIn("phase_top_bins", dbg)

    def test_stratified_phase_sampling_stable_uniformity(self) -> None:
        nf = 8
        f = np.linspace(6e9, 7e9, nf)
        d = np.linspace(10e-9, 50e-9, 64)
        p = np.linspace(1.0, 0.1, 64)
        synth = generate_synthetic_paths(
            f_hz=f,
            num_rays=64,
            delay_samples_s=d,
            power_samples=p,
            parity_probs={"odd": 0.5, "even": 0.5},
            parity_fit={"odd": {"mu": 12.0, "sigma": 2.0}, "even": {"mu": 14.0, "sigma": 2.0}},
            matrix_source="J",
            phase_sampling_method="stratified_uniform",
            seed=7,
        )
        ph = offdiag_phases(
            synth,
            matrix_source="J",
            per_ray_sampling=True,
            common_phase_removed=True,
            amp_ratio_min=1e-3,
        )
        ku = kuiper_uniform_test(ph, bootstrap_B=200, seed=3)
        self.assertGreaterEqual(int(ku.get("n", 0)), 50)
        self.assertGreaterEqual(float(ku.get("p_boot", 0.0)), 0.05)

    def test_summary_contains_phase_and_ks_fields(self) -> None:
        nf = 16
        f = np.linspace(6e9, 7e9, nf)
        rt = [_simple_path(nf, 10e-9, 20.0, "odd"), _simple_path(nf, 20e-9, 25.0, "even")]
        sy = [_simple_path(nf, 11e-9, 19.0, "odd"), _simple_path(nf, 22e-9, 24.0, "even")]
        out = summarize_rt_vs_synth(
            rt_paths=rt,
            synth_paths=sy,
            subbands=[(0, 8), (8, 16)],
            rt_matrix_source="J",
            synth_matrix_source="J",
            phase_test_basis="circular",
            common_phase_removed=True,
            per_ray_phase_sampling=True,
            offdiag_amp_ratio_min=1e-3,
            phase_bootstrap_B=20,
            seed=0,
        )
        self.assertIn("f3_xpd_ks2_p", out)
        self.assertIn("phase_test_basis", out)
        self.assertEqual(str(out["phase_uniformity_rt_status"]), "INFO_DETERMINISTIC")
        self.assertIn("phase_offdiag_synth_debug", out)

    def test_num_paths_mode_per_case_and_per_scenario(self) -> None:
        nf = 8
        f = np.linspace(6e9, 7e9, nf)
        d = np.linspace(5e-9, 20e-9, 10)
        p = np.linspace(1.0, 0.2, 10)
        sids = ["S0"] * 6 + ["S1"] * 4
        cids = ["c0"] * 3 + ["c1"] * 3 + ["c2"] * 2 + ["c3"] * 2

        sy_case, dg_case = generate_synthetic_paths(
            f_hz=f,
            num_rays=3,
            num_paths_mode="match_rt_per_case",
            delay_samples_s=d,
            power_samples=p,
            rt_path_scenarios=sids,
            rt_path_cases=cids,
            parity_probs={"odd": 0.5, "even": 0.5},
            parity_fit={"odd": {"mu": 8.0, "sigma": 1.0}, "even": {"mu": 9.0, "sigma": 1.0}},
            matrix_source="J",
            return_diagnostics=True,
            seed=4,
        )
        self.assertEqual(len(sy_case), len(d))
        self.assertEqual(str(dg_case["num_paths_mode"]), "match_rt_per_case")

        sy_scn, dg_scn = generate_synthetic_paths(
            f_hz=f,
            num_rays=3,
            num_paths_mode="match_rt_per_scenario",
            delay_samples_s=d,
            power_samples=p,
            rt_path_scenarios=sids,
            rt_path_cases=cids,
            parity_probs={"odd": 0.5, "even": 0.5},
            parity_fit={"odd": {"mu": 8.0, "sigma": 1.0}, "even": {"mu": 9.0, "sigma": 1.0}},
            matrix_source="J",
            return_diagnostics=True,
            seed=4,
        )
        self.assertEqual(len(sy_scn), len(d))
        self.assertEqual(str(dg_scn["num_paths_mode"]), "match_rt_per_scenario")

    def test_nan_delay_samples_do_not_propagate_to_tau(self) -> None:
        nf = 8
        f = np.linspace(6e9, 7e9, nf)
        d = np.array([10e-9, np.nan, 20e-9, np.inf], dtype=float)
        p = np.array([1.0, 0.5, np.nan, 0.2], dtype=float)
        synth, diag = generate_synthetic_paths(
            f_hz=f,
            num_rays=4,
            delay_samples_s=d,
            power_samples=p,
            parity_probs={"odd": 0.5, "even": 0.5},
            parity_fit={"odd": {"mu": 10.0, "sigma": 1.0}, "even": {"mu": 11.0, "sigma": 1.0}},
            matrix_source="J",
            return_diagnostics=True,
            seed=11,
        )
        self.assertGreaterEqual(int(diag.get("dropped_invalid_delay_samples", 0)), 1)
        self.assertGreaterEqual(int(diag.get("dropped_invalid_power_samples", 0)), 1)
        taus = np.asarray([float(pth["tau_s"]) for pth in synth], dtype=float)
        self.assertTrue(np.all(np.isfinite(taus)))
        self.assertTrue(np.all(taus >= 0.0))


if __name__ == "__main__":
    unittest.main()

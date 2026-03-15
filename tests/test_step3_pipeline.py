"""Tests for Step 3 analysis/modeling pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.run_model_fit import fit_and_generate
from analysis.xpd_stats import pathwise_xpd
from rt_io.hdf5_io import save_rt_dataset


def _make_path(nf: int, tau_s: float, bounce_count: int, cross_lo: float, cross_hi: float, material: str = "NA") -> dict:
    c = np.linspace(cross_lo, cross_hi, nf)
    A = np.zeros((nf, 2, 2), dtype=np.complex128)
    A[:, 0, 0] = 1.0
    A[:, 1, 1] = 1.0
    A[:, 0, 1] = c
    A[:, 1, 0] = c
    return {
        "tau_s": tau_s,
        "A_f": A,
        "J_f": A.copy(),
        "G_tx_f": np.repeat(np.eye(2, dtype=np.complex128)[None, :, :], nf, axis=0),
        "G_rx_f": np.repeat(np.eye(2, dtype=np.complex128)[None, :, :], nf, axis=0),
        "scalar_gain_f": np.ones(nf, dtype=float),
        "meta": {
            "bounce_count": bounce_count,
            "interactions": ["reflection"] * max(bounce_count, 1),
            "surface_ids": [1] * max(bounce_count, 1),
            "incidence_angles": [float(np.deg2rad(20.0 + 10.0 * bounce_count))],
            "AoD": [1.0, 0.0, 0.0],
            "AoA": [-1.0, 0.0, 0.0],
            "segment_basis_uv": [],
            "bounce_normals": [[0.0, 1.0, 0.0]] * max(bounce_count, 1),
            "material": material,
        },
    }


class Step3PipelineTests(unittest.TestCase):
    def test_pathwise_xpd_power_mean_synthetic(self) -> None:
        nf = 4
        A = np.zeros((nf, 2, 2), dtype=np.complex128)
        A[:, 0, 0] = np.array([1.0, -1.0, 1j, -1j])
        A[:, 1, 1] = 0.0
        A[:, 0, 1] = 0.5 * np.array([1.0, -1.0, 1j, -1j])
        A[:, 1, 0] = 0.0
        J = np.zeros_like(A)
        J[:, 0, 0] = 2.0
        J[:, 1, 1] = 0.0
        J[:, 0, 1] = 1.0
        J[:, 1, 0] = 0.0

        path = {
            "tau_s": 1e-8,
            "A_f": A,
            "J_f": J,
            "meta": {"bounce_count": 1, "incidence_angles": []},
        }
        sA = pathwise_xpd([path], matrix_source="A")[0]
        sJ = pathwise_xpd([path], matrix_source="J")[0]

        self.assertAlmostEqual(sA["xpd_db"], 6.0205999, places=5)
        self.assertAlmostEqual(sJ["xpd_db"], 6.0205999, places=5)

    def test_fit_and_generate_outputs_conditioned_model(self) -> None:
        nf = 16
        freq = np.linspace(6e9, 7e9, nf)
        odd_paths = [
            _make_path(nf, 20e-9, 1, 0.4, 0.8, material="glass"),
            _make_path(nf, 24e-9, 1, 0.35, 0.7, material="glass"),
        ]
        even_paths = [
            _make_path(nf, 22e-9, 2, 0.15, 0.05, material="wood"),
            _make_path(nf, 26e-9, 2, 0.20, 0.08, material="wood"),
        ]
        data = {
            "meta": {"basis": "linear", "convention": "IEEE-RHCP"},
            "frequency": freq,
            "scenarios": {
                "S": {
                    "cases": {
                        "0": {"params": {"material": "glass"}, "paths": odd_paths},
                        "1": {"params": {"material": "wood"}, "paths": even_paths},
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            h5 = tdp / "rt.h5"
            out_model = tdp / "model.json"
            out_cmp = tdp / "cmp.json"
            save_rt_dataset(h5, data)

            res = fit_and_generate(
                input_h5=h5,
                output_json=out_model,
                synthetic_compare_json=out_cmp,
                matrix_source="A",
                num_subbands=4,
                num_synth_rays=200,
                seed=42,
            )

            self.assertTrue(out_model.exists())
            self.assertTrue(out_cmp.exists())

            parity_fit = res["model"]["condition_fits"]["parity"]
            self.assertIn("odd", parity_fit)
            self.assertIn("even", parity_fit)

            # Qualitative parity separation should remain in synthetic generation.
            syn_par = res["comparison"]["comparison"]["synthetic_parity_xpd"]
            self.assertIn("odd", syn_par)
            self.assertIn("even", syn_par)
            self.assertGreater(float(syn_par["even"]["mu"]), float(syn_par["odd"]["mu"]))

            # Subband slope model exists and captures odd/even trend difference.
            slope_par = res["model"]["frequency_slope_models"]["parity"]
            self.assertIn("odd", slope_par)
            self.assertIn("even", slope_par)
            self.assertLess(float(slope_par["odd"]["mu1_db_per_hz"]), float(slope_par["even"]["mu1_db_per_hz"]))


if __name__ == "__main__":
    unittest.main()

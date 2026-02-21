"""Numerical roundtrip tests for HDF5 schema v2."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from rt_io.hdf5_io import SCHEMA_VERSION, load_rt_dataset, save_rt_dataset, self_test_meta_roundtrip


class Hdf5RoundtripTests(unittest.TestCase):
    def test_v2_roundtrip_numerical_equality(self) -> None:
        f_hz = np.linspace(6e9, 7e9, 8, dtype=float)
        n_f = len(f_hz)
        A = np.zeros((n_f, 2, 2), dtype=np.complex128)
        J = np.zeros((n_f, 2, 2), dtype=np.complex128)
        A[:, 0, 0] = 1.0 + 0.1j
        A[:, 1, 1] = 0.8 - 0.2j
        J[:, 0, 0] = 0.5 + 0.0j
        J[:, 1, 1] = 0.3 + 0.0j
        eye = np.repeat(np.eye(2, dtype=np.complex128)[None, :, :], n_f, axis=0)
        gain = np.linspace(1.0, 0.9, n_f, dtype=float)

        data = {
            "meta": {
                "schema_version": SCHEMA_VERSION,
                "git_commit": "abc123",
                "git_dirty": False,
                "release_mode": True,
                "cmdline": "python -m scenarios.runner --release-mode",
                "seed_json": "{\"model_seed\":0}",
                "basis": "circular",
                "convention": "IEEE-RHCP",
                "xpd_matrix_source": "J",
                "exact_bounce_defaults": {"A2": 1, "A3": 2},
                "exact_bounce_applied": True,
                "physics_validation_mode": True,
                "antenna_config": {"enable_coupling": False},
            },
            "frequency": f_hz,
            "scenarios": {
                "C0": {
                    "cases": {
                        "0": {
                            "params": {"distance_m": 1.0},
                            "paths": [
                                {
                                    "tau_s": 5e-9,
                                    "path_length_m": 1.49896229,
                                    "segment_count": 1,
                                    "reflection_points": [],
                                    "A_f": A,
                                    "J_f": J,
                                    "G_tx_f": eye,
                                    "G_rx_f": eye,
                                    "scalar_gain_f": gain,
                                    "points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                                    "meta": {
                                        "bounce_count": 0,
                                        "interactions": [],
                                        "surface_ids": [],
                                        "material_ids": [],
                                        "incidence_angles": [],
                                        "AoD": [1.0, 0.0, 0.0],
                                        "AoA": [1.0, 0.0, 0.0],
                                        "segment_basis_uv": [],
                                    },
                                }
                            ],
                        }
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rt_v2.h5"
            save_rt_dataset(p, data)
            self.assertTrue(self_test_meta_roundtrip(p, expected_meta=data["meta"]))

            with h5py.File(p, "r") as f:
                self.assertIn("freq", f)
                self.assertIn("f", f["freq"])
                self.assertTrue(np.allclose(f["freq"]["f"][:], f_hz))

            loaded = load_rt_dataset(p)
            self.assertEqual(str(loaded["meta"]["schema_version"]), SCHEMA_VERSION)
            self.assertTrue(np.allclose(loaded["frequency"], f_hz))
            p0 = loaded["scenarios"]["C0"]["cases"]["0"]["paths"][0]
            self.assertTrue(np.allclose(np.asarray(p0["A_f"], dtype=np.complex128), A))
            self.assertTrue(np.allclose(np.asarray(p0["J_f"], dtype=np.complex128), J))
            self.assertTrue(np.allclose(np.asarray(p0["scalar_gain_f"], dtype=float), gain))
            self.assertAlmostEqual(float(p0["tau_s"]), 5e-9, places=15)
            self.assertAlmostEqual(float(p0["path_length_m"]), 1.49896229, places=9)


if __name__ == "__main__":
    unittest.main()


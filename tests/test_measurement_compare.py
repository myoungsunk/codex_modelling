"""Tests for Step-5 measurement bridge import and compare."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.measurement_compare import MeasurementData, compare_measured_to_dataset, load_measurement_matrix_csv


class MeasurementBridgeTests(unittest.TestCase):
    def test_load_measurement_matrix_csv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "meas.csv"
            with p.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "f_hz",
                        "hh_re",
                        "hh_im",
                        "hv_re",
                        "hv_im",
                        "vh_re",
                        "vh_im",
                        "vv_re",
                        "vv_im",
                    ]
                )
                w.writerow([6.0e9, 1.0, 0.0, 0.1, 0.0, 0.2, 0.0, 1.0, 0.0])
                w.writerow([6.1e9, 0.9, 0.0, 0.1, 0.0, 0.2, 0.0, 0.9, 0.0])
            m = load_measurement_matrix_csv(p)
            self.assertEqual(m.H_f.shape, (2, 2, 2))
            self.assertAlmostEqual(float(m.frequency_hz[0]), 6.0e9, places=3)
            self.assertAlmostEqual(float(np.real(m.H_f[0, 0, 0])), 1.0, places=9)

    def test_compare_measured_to_dataset_embedded(self) -> None:
        nf = 16
        f = np.linspace(6.0e9, 7.0e9, nf)
        A = np.zeros((nf, 2, 2), dtype=np.complex128)
        A[:, 0, 0] = 1.0
        A[:, 1, 1] = 1.0
        A[:, 0, 1] = 0.1
        A[:, 1, 0] = 0.1
        ds = {
            "meta": {"basis": "linear", "convention": "IEEE-RHCP"},
            "frequency": f,
            "scenarios": {
                "S0": {
                    "cases": {
                        "0": {
                            "params": {},
                            "paths": [
                                {
                                    "tau_s": 0.0,
                                    "A_f": A.copy(),
                                    "J_f": A.copy(),
                                    "meta": {"bounce_count": 0},
                                }
                            ],
                        }
                    }
                }
            },
        }
        meas = MeasurementData(frequency_hz=f, H_f=A.copy(), source="unit-test")
        out = compare_measured_to_dataset(
            ds,
            measurement=meas,
            channel_definition="embedded",
            scenario_id="S0",
            case_id="0",
            create_plots=False,
        )
        self.assertEqual(str(out["matrix_source"]), "A")
        self.assertLess(float(out["rmse_mag_db_meas_vs_rt"]), 1e-9)
        self.assertGreaterEqual(float(out["xpd_ks2_p_meas_vs_rt"]), 0.99)


if __name__ == "__main__":
    unittest.main()


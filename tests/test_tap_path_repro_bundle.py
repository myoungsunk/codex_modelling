from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.tap_path_consistency import build_min_repro_bundle, write_repro_bundle_json


class TapPathReproBundleTests(unittest.TestCase):
    def test_build_min_repro_bundle_filters_overlap_and_threshold(self) -> None:
        data = {
            "scenarios": {
                "S": {
                    "cases": {
                        "0": {
                            "paths": [
                                {
                                    "tau_s": 1.0e-8,
                                    "meta": {"bounce_count": 1, "surface_ids": [0], "incidence_angles": [0.2]},
                                }
                            ]
                        }
                    }
                }
            }
        }
        entries = [
            {
                "scenario_id": "S",
                "case_id": "0",
                "delta_xpd_db": 12.0,
                "delta_tau_s": 1.0e-10,
                "overlap": False,
                "outlier_reason": "MISMATCH",
                "window_start_idx": 10,
                "window_end_idx": 14,
                "window_path_indices": [0],
                "strongest_path_index": 0,
                "strongest_tau_s": 1.0e-8,
                "tap_window_tau_s": 1.01e-8,
                "path_tau_s": [1.0e-8],
                "path_alias_tau_s": [1.0e-8],
            },
            {
                "scenario_id": "S",
                "case_id": "0",
                "delta_xpd_db": 20.0,
                "delta_tau_s": 1.0e-10,
                "overlap": True,
                "outlier_reason": "OVERLAP",
            },
        ]
        out = build_min_repro_bundle(data, entries, delta_xpd_threshold_db=10.0, non_overlap_only=True)
        self.assertEqual(out["n_cases"], 1)
        case0 = out["cases"][0]
        self.assertEqual(case0["scenario_id"], "S")
        self.assertEqual(case0["case_id"], "0")
        self.assertEqual(case0["strongest_path_meta"]["bounce_count"], 1)

    def test_write_repro_bundle_json(self) -> None:
        bundle = {"n_cases": 1, "cases": [{"scenario_id": "S", "case_id": "0"}]}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bundle.json"
            out = write_repro_bundle_json(p, bundle)
            self.assertTrue(out.exists())
            loaded = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(loaded["n_cases"], 1)


if __name__ == "__main__":
    unittest.main()

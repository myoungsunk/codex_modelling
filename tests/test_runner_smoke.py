"""Smoke test for standardized simulation runner."""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class RunnerSmokeTests(unittest.TestCase):
    def test_run_standard_sim_c0_outputs_exist(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            out_h5 = Path(td) / "std_c0.h5"
            out_dir = Path(td) / "std_out"
            cmd = [
                "python3",
                "scripts/run_standard_sim.py",
                "--scenario",
                "C0",
                "--out-h5",
                str(out_h5),
                "--out-dir",
                str(out_dir),
                "--max-links",
                "1",
                "--dist-list",
                "3",
                "--yaw-list",
                "0",
                "--pitch-list",
                "0",
                "--seed",
                "1",
            ]
            p = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            if p.returncode != 0:
                self.fail(f"run_standard_sim failed: {p.stderr}")
            self.assertTrue(out_h5.exists())
            self.assertTrue((out_dir / "link_metrics.csv").exists())
            self.assertTrue((out_dir / "rays.csv").exists())
            self.assertTrue((out_dir / "run_summary.json").exists())

    def test_run_standard_sim_c0_default_yaw_and_floor_reference_frequency(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            out_h5 = Path(td) / "std_c0_freq.h5"
            out_dir = Path(td) / "std_out_freq"
            cmd = [
                "python3",
                "scripts/run_standard_sim.py",
                "--scenario",
                "C0",
                "--out-h5",
                str(out_h5),
                "--out-dir",
                str(out_dir),
                "--max-links",
                "3",
                "--dist-list",
                "1",
                "--n-rep",
                "1",
                "--seed",
                "2",
            ]
            p = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            if p.returncode != 0:
                self.fail(f"run_standard_sim failed: {p.stderr}")

            metrics_csv = out_dir / "link_metrics.csv"
            yaw_vals = []
            with metrics_csv.open("r", encoding="utf-8", newline="") as f:
                rd = csv.DictReader(f)
                for r in rd:
                    try:
                        yaw_vals.append(float(r.get("yaw_deg", "nan")))
                    except Exception:
                        pass
            self.assertGreaterEqual(len(set(v for v in yaw_vals if v == v)), 2)

            floor_json = out_dir / "floor_reference.json"
            self.assertTrue(floor_json.exists())
            ref = json.loads(floor_json.read_text(encoding="utf-8"))
            freq = ref.get("frequency_hz", [])
            floor = ref.get("xpd_floor_db", [])
            uncert = ref.get("xpd_floor_uncert_db", [])
            self.assertTrue(isinstance(freq, list) and len(freq) > 0)
            self.assertEqual(len(freq), len(floor))
            self.assertEqual(len(freq), len(uncert))
            self.assertTrue(isinstance(ref.get("subbands", []), list))


if __name__ == "__main__":
    unittest.main()

"""Smoke test for standardized simulation runner."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()

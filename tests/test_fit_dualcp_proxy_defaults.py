"""Smoke tests for fit_dualcp_proxy default key mapping."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class FitDualcpProxyDefaultsTests(unittest.TestCase):
    def test_defaults_match_standard_output_columns(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            metrics_csv = td_path / "link_metrics.csv"
            model_json = td_path / "proxy_model.json"
            report_md = td_path / "proxy_report.md"

            rows = []
            for i in range(12):
                rows.append(
                    {
                        "scenario_id": "A2" if i < 6 else "A3",
                        "case_id": str(i),
                        "XPD_early_excess_db": -6.0 + 1.2 * i,
                        "XPD_late_excess_db": -2.0 + 0.6 * i,
                        "L_pol_db": -4.0 + 0.6 * i,
                        "rho_early_db": -float(i),
                        "LOSflag": 0 if i < 8 else 1,
                        "material_class": "PEC" if i % 2 == 0 else "glass",
                        "roughness_flag": 1 if (i % 3 == 0) else 0,
                        "human_flag": 1 if (i % 4 == 0) else 0,
                        "d_m": 2.0 + 0.25 * i,
                        "EL_proxy_db": 10.0 + 0.5 * i,
                        "dominant_parity_early": "odd" if i % 2 == 0 else "even",
                    }
                )

            with metrics_csv.open("w", encoding="utf-8", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                for r in rows:
                    wr.writerow(r)

            cmd = [
                "python3",
                "scripts/fit_dualcp_proxy.py",
                "--metrics-csv",
                str(metrics_csv),
                "--out-model-json",
                str(model_json),
                "--out-report",
                str(report_md),
            ]
            env = dict(os.environ)
            env["MPLCONFIGDIR"] = str(td_path / "mplconfig")
            p = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, env=env)
            if p.returncode != 0:
                self.fail(f"fit_dualcp_proxy failed: {p.stderr}")

            self.assertTrue(model_json.exists())
            self.assertTrue(report_md.exists())
            out = json.loads(model_json.read_text(encoding="utf-8"))
            models = dict(out.get("models", {}))
            self.assertIn("XPD_early_excess_db", models)
            self.assertIn("XPD_late_excess_db", models)
            self.assertIn("L_pol_db", models)
            self.assertIn("rho_early_db", models)


if __name__ == "__main__":
    unittest.main()

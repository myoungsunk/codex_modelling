"""Tests for standard outputs v1 schema and HDF5/CSV exporters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rt_io.standard_outputs_hdf5 import export_csv, load_run, save_run
from rt_types.standard_outputs import (
    SCHEMA_VERSION,
    DualCPPDP,
    LinkConditionsU,
    LinkMetricsZ,
    RayTable,
    StandardOutputBundle,
)


class StandardOutputsV1Tests(unittest.TestCase):
    def _toy_bundle(self) -> StandardOutputBundle:
        rays = RayTable(
            rows=[
                {"tau_s": 1e-9, "L_m": 0.3, "n_bounce": 0, "P_lin": 1.0, "EL_db": 3.0, "parity": "even"},
                {"tau_s": 4e-9, "L_m": 1.2, "n_bounce": 1, "P_lin": 0.1, "EL_db": 12.0, "parity": "odd"},
            ]
        )
        delay = np.linspace(0.0, 10e-9, 8)
        pco = np.linspace(1.0, 0.2, 8)
        pcx = np.linspace(0.1, 0.05, 8)
        pdp = DualCPPDP(delay_tau_s=delay, P_co=pco, P_cross=pcx, XPD_tau_db=10.0 * np.log10((pco + 1e-15) / (pcx + 1e-15)))
        metrics = LinkMetricsZ(
            XPD_early_db=10.0,
            XPD_late_db=4.0,
            rho_early_lin=0.1,
            rho_early_db=-10.0,
            L_pol_db=6.0,
            delay_spread_rms_s=2e-9,
            early_energy_fraction=0.7,
            window={"Te_s": 3e-9, "Tmax_s": 20e-9, "tau0_method": "threshold"},
        )
        cond = LinkConditionsU(
            d_m=3.0,
            LOSflag=1,
            EL_proxy_db=5.0,
            material_class="PEC",
            roughness_flag=0,
            human_flag=0,
            dominant_parity_early="even",
        )
        return StandardOutputBundle(
            link_id="L0",
            scenario_id="C0",
            case_id="0",
            rays=rays,
            pdp=pdp,
            metrics=metrics,
            conditions=cond,
            provenance={"seed": 7, "cmdline": "unit-test"},
        )

    def test_save_load_schema(self) -> None:
        b = self._toy_bundle()
        with tempfile.TemporaryDirectory() as td:
            out_h5 = Path(td) / "std.h5"
            save_run({"run_id": "r0", "schema_version": SCHEMA_VERSION}, [b], out_h5=out_h5, run_id="r0")
            loaded = load_run(out_h5, run_id="r0")
            self.assertEqual(str(loaded["meta"].get("schema_version")), SCHEMA_VERSION)
            self.assertEqual(len(loaded["bundles"]), 1)
            lb = loaded["bundles"][0]
            self.assertEqual(str(lb["link_id"]), "L0")
            self.assertEqual(str(lb["scenario_id"]), "C0")
            self.assertIn("XPD_early_db", lb["metrics"])
            self.assertEqual(len(lb["pdp"]["delay_tau_s"]), 8)

    def test_csv_export_files_exist(self) -> None:
        b = self._toy_bundle()
        with tempfile.TemporaryDirectory() as td:
            out = export_csv([b], out_dir=td)
            self.assertTrue(Path(out["link_metrics_csv"]).exists())
            self.assertTrue(Path(out["rays_csv"]).exists())
            self.assertTrue((Path(td) / "pdp_L0.npz").exists())


if __name__ == "__main__":
    unittest.main()

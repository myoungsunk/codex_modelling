"""Tests for dispersive material support."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rt_core.geometry import Material
from rt_core.materials import resolve_material_library
from rt_core.polarization import fresnel_reflection


class MaterialDispersionTest(unittest.TestCase):
    def _temp_db(self) -> Path:
        payload = {
            "materials": {
                "table_mat": {
                    "model": "table",
                    "f_hz": [6.0e9, 8.0e9, 10.0e9],
                    "eps_r": [2.0, 3.0, 4.0],
                    "tan_delta": [0.01, 0.02, 0.03],
                },
                "debye_mat": {
                    "model": "debye",
                    "eps_inf": 2.0,
                    "delta_eps": [0.8, 0.2],
                    "tau_s": [1.0e-11, 4.0e-11],
                    "tan_delta": 0.01,
                },
            }
        }
        d = Path(tempfile.mkdtemp())
        p = d / "materials.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    def test_table_material_variation_and_passive_gamma(self) -> None:
        db = self._temp_db()
        mats = resolve_material_library(db, material_dispersion="on")
        f = np.linspace(6.0e9, 10.0e9, 33)
        gs, gp = fresnel_reflection(mats["table_mat"], theta_i=np.deg2rad(45.0), f_hz=f)

        self.assertTrue(np.all(np.isfinite(gs)))
        self.assertTrue(np.all(np.isfinite(gp)))
        self.assertLessEqual(float(np.max(np.abs(gs))), 1.0000001)
        self.assertLessEqual(float(np.max(np.abs(gp))), 1.0000001)
        # Dispersive table should introduce frequency variation.
        self.assertGreater(float(np.std(np.abs(gs))), 1e-4)

    def test_debye_material_variation_and_passive_gamma(self) -> None:
        db = self._temp_db()
        mats = resolve_material_library(db, material_dispersion="debye")
        f = np.linspace(6.0e9, 10.0e9, 33)
        gs, gp = fresnel_reflection(mats["debye_mat"], theta_i=np.deg2rad(35.0), f_hz=f)

        self.assertTrue(np.all(np.isfinite(gs)))
        self.assertTrue(np.all(np.isfinite(gp)))
        self.assertLessEqual(float(np.max(np.abs(gs))), 1.0000001)
        self.assertLessEqual(float(np.max(np.abs(gp))), 1.0000001)
        self.assertGreater(float(np.std(np.abs(gp))), 1e-5)

    def test_debye_tan_delta_is_not_applied_twice(self) -> None:
        f = np.linspace(6.0e9, 10.0e9, 65)
        a = Material.dielectric_debye(
            eps_inf=2.0,
            delta_eps=[0.8, 0.2],
            tau_s=[1.0e-11, 4.0e-11],
            tan_delta=0.0,
            name="debye_a",
        )
        b = Material.dielectric_debye(
            eps_inf=2.0,
            delta_eps=[0.8, 0.2],
            tau_s=[1.0e-11, 4.0e-11],
            tan_delta=0.05,
            name="debye_b",
        )
        gs_a, gp_a = fresnel_reflection(a, theta_i=np.deg2rad(35.0), f_hz=f)
        gs_b, gp_b = fresnel_reflection(b, theta_i=np.deg2rad(35.0), f_hz=f)
        self.assertTrue(np.allclose(gs_a, gs_b, atol=1e-12, rtol=1e-12))
        self.assertTrue(np.allclose(gp_a, gp_b, atol=1e-12, rtol=1e-12))

    def test_dispersion_off_collapses_to_constant(self) -> None:
        db = self._temp_db()
        mats = resolve_material_library(db, material_dispersion="off")
        f = np.linspace(6.0e9, 10.0e9, 33)
        gs, _ = fresnel_reflection(mats["table_mat"], theta_i=np.deg2rad(45.0), f_hz=f)
        self.assertLess(float(np.std(np.abs(gs))), 1e-10)


if __name__ == "__main__":
    unittest.main()

"""Tests for XPR model material-bias resolution."""

from __future__ import annotations

import unittest

import numpy as np

from polarization.xpr_models import ConditionalLinearXPR


class XprModelTests(unittest.TestCase):
    def test_material_bias_falls_back_to_link_material_when_ray_material_is_surface_id(self) -> None:
        m = ConditionalLinearXPR(
            a0=0.0,
            sigma_db=0.0,
            material_bias={"glass": 2.0, "wood": -1.0},
        )
        rng = np.random.default_rng(0)
        ray_row = {"tau_s": 1e-9, "material_class": "1"}
        link_u = {"material_class": "glass", "EL_proxy_db": 0.0}
        x = m.sample_xpr_db(ray_row, link_u, rng)
        self.assertAlmostEqual(float(x), 2.0, places=9)


if __name__ == "__main__":
    unittest.main()


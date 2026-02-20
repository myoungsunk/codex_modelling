"""Unit tests for deterministic specular polarimetric tracer."""

from __future__ import annotations

import unittest

import numpy as np

from rt_core.antenna import Antenna
from rt_core.geometry import Material, Plane
from rt_core.tracer import trace_paths


class TracerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.f = np.linspace(6e9, 7e9, 64)
        self.tx = Antenna(
            position=np.array([0.0, 0.0, 1.0]),
            boresight=np.array([1.0, 0.0, 0.0]),
            h_axis=np.array([0.0, 1.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            basis="linear",
        )
        self.rx = Antenna(
            position=np.array([5.0, 0.0, 1.0]),
            boresight=np.array([-1.0, 0.0, 0.0]),
            h_axis=np.array([0.0, 1.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            basis="linear",
        )

    def test_los_only_cross_is_small(self) -> None:
        paths = trace_paths([], self.tx, self.rx, self.f, max_bounce=0, los_enabled=True)
        self.assertEqual(len(paths), 1)
        a = np.mean(paths[0].A_f, axis=0)
        cross = np.abs(a[0, 1]) ** 2 + np.abs(a[1, 0]) ** 2
        self.assertLess(cross, 1e-8)

    def test_one_bounce_plane_when_los_blocked(self) -> None:
        plane = Plane(id=1, p0=np.array([0.0, 2.0, 0.0]), normal=np.array([0.0, -1.0, 0.0]), material=Material.pec())
        paths = trace_paths([plane], self.tx, self.rx, self.f, max_bounce=1, los_enabled=False)
        self.assertTrue(any(p.bounce_count == 1 for p in paths))
        self.assertFalse(any(p.bounce_count == 0 for p in paths))

    def test_two_bounce_corner_exists(self) -> None:
        p1 = Plane(id=1, p0=np.array([0.0, 3.0, 0.0]), normal=np.array([0.0, -1.0, 0.0]), material=Material.pec())
        p2 = Plane(id=2, p0=np.array([3.0, 0.0, 0.0]), normal=np.array([-1.0, 0.0, 0.0]), material=Material.pec())
        self.rx.position[:] = [3.0, 4.0, 1.0]
        paths = trace_paths([p1, p2], self.tx, self.rx, self.f, max_bounce=2, los_enabled=False)
        self.assertTrue(any(p.bounce_count == 2 for p in paths))


if __name__ == "__main__":
    unittest.main()

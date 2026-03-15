"""Core geometry unit tests."""

from __future__ import annotations

import unittest

import numpy as np

from rt_core.geometry import Material, Plane, line_plane_intersection, ray_plane_intersection, reflect_direction


class GeometryCoreTests(unittest.TestCase):
    def test_line_plane_intersection_is_direction_symmetric(self) -> None:
        plane = Plane(
            id=1,
            p0=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            material=Material.pec(),
        )
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-2.0, 4.0, -1.0])
        p_ab = line_plane_intersection(a, b, plane)
        p_ba = line_plane_intersection(b, a, plane)
        self.assertIsNotNone(p_ab)
        self.assertIsNotNone(p_ba)
        self.assertTrue(np.allclose(p_ab, p_ba, atol=1e-12))

    def test_finite_plate_contains_and_clamp(self) -> None:
        plane = Plane(
            id=2,
            p0=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            material=Material.pec(),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 1.0, 0.0]),
            half_extent_u=2.0,
            half_extent_v=1.0,
        )
        outside = np.array([3.5, 1.5, 0.0])
        clamped, dist = plane.clamp_to_bounds(outside)
        self.assertGreater(dist, 0.0)
        self.assertTrue(plane.contains_point(clamped, eps=1e-12))
        self.assertFalse(plane.contains_point(outside, eps=1e-12))

    def test_reflect_direction_flips_normal_component(self) -> None:
        n = np.array([0.0, 1.0, 0.0], dtype=float)
        d_in = np.array([1.0, -1.0, 0.0], dtype=float)
        d_out = reflect_direction(d_in, n)
        self.assertAlmostEqual(float(np.dot(d_in / np.linalg.norm(d_in), n)), -float(np.dot(d_out, n)), places=12)

    def test_ray_plane_intersection_forward_only(self) -> None:
        plane = Plane(
            id=3,
            p0=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            material=Material.pec(),
        )
        hit = ray_plane_intersection(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]), plane)
        self.assertIsNotNone(hit)
        miss = ray_plane_intersection(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), plane)
        self.assertIsNone(miss)


if __name__ == "__main__":
    unittest.main()

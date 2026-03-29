from __future__ import annotations

import unittest

import numpy as np

from dualpol_rt.em.materials import Material
from dualpol_rt.geometry.image_method import enumerate_paths
from dualpol_rt.scene.surfaces import Scene, Surface


class GeometryPathTests(unittest.TestCase):
    def test_los_free_space_single_path(self) -> None:
        scene = Scene(surfaces=tuple())
        tx = np.array([0.0, 0.0, 1.0], dtype=float)
        rx = np.array([2.0, 0.0, 1.0], dtype=float)
        paths = enumerate_paths(scene, tx, rx, max_reflections=0)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].bounce_count, 0)
        self.assertAlmostEqual(paths[0].path_length_m, 2.0, places=12)

    def test_single_pec_reflection_bounce_point(self) -> None:
        surface = Surface(
            surface_id=1,
            name="wall_y1",
            point=[1.0, 1.0, 1.0],
            normal=[0.0, -1.0, 0.0],
            u_axis=[1.0, 0.0, 0.0],
            v_axis=[0.0, 0.0, 1.0],
            half_u=5.0,
            half_v=5.0,
            material=Material.pec(),
        )
        scene = Scene(surfaces=(surface,))
        tx = np.array([0.0, 0.0, 1.0], dtype=float)
        rx = np.array([2.0, 0.0, 1.0], dtype=float)
        paths = enumerate_paths(scene, tx, rx, max_reflections=1)
        one_bounce = [path for path in paths if path.bounce_count == 1]
        self.assertEqual(len(one_bounce), 1)
        self.assertTrue(np.allclose(one_bounce[0].bounce_points[0], np.array([1.0, 1.0, 1.0]), atol=1e-9))
        self.assertAlmostEqual(one_bounce[0].path_length_m, 2.0 * np.sqrt(2.0), places=9)

    def test_two_bounce_corner_exists(self) -> None:
        p1 = Surface(
            surface_id=1,
            name="wall_y3",
            point=[0.0, 3.0, 1.0],
            normal=[0.0, -1.0, 0.0],
            u_axis=[1.0, 0.0, 0.0],
            v_axis=[0.0, 0.0, 1.0],
            half_u=5.0,
            half_v=5.0,
            material=Material.pec(),
        )
        p2 = Surface(
            surface_id=2,
            name="wall_x3",
            point=[3.0, 0.0, 1.0],
            normal=[-1.0, 0.0, 0.0],
            u_axis=[0.0, 1.0, 0.0],
            v_axis=[0.0, 0.0, 1.0],
            half_u=5.0,
            half_v=5.0,
            material=Material.pec(),
        )
        scene = Scene(surfaces=(p1, p2))
        tx = np.array([0.0, 0.0, 1.0], dtype=float)
        rx = np.array([3.0, 4.0, 1.0], dtype=float)
        paths = enumerate_paths(scene, tx, rx, max_reflections=2)
        self.assertTrue(any(path.bounce_count == 2 for path in paths))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from dualpol_rt.geometry.image_method import enumerate_paths
from dualpol_rt.scene import build_scenario, canonical_scenarios, get_scenario_spec, scenario_grid


class ScenarioBuilderTests(unittest.TestCase):
    def test_canonical_scenarios_build_non_degenerate_scenes(self) -> None:
        specs = canonical_scenarios(include_proxy=True)
        self.assertGreaterEqual(len(specs), 7)
        for spec in specs:
            scene = build_scenario(spec)
            self.assertGreaterEqual(len(scene.surfaces), 6, msg=spec.name)

    def test_obstacle_scenes_expose_invalid_grid_points(self) -> None:
        for name in ("factory_lite", "l_corridor_proxy"):
            rx_x, rx_y, valid_mask = scenario_grid(name, grid_step=1.0)
            self.assertEqual(valid_mask.shape, (len(rx_y), len(rx_x)))
            self.assertTrue(np.any(~valid_mask), msg=name)

    def test_floor_standing_boxes_do_not_add_hidden_bottom_faces(self) -> None:
        scene = build_scenario("open_office")
        names = {surface.name for surface in scene.surfaces}
        self.assertNotIn("desk_a_z0", names)
        self.assertNotIn("desk_b_z0", names)
        self.assertNotIn("cabinet_z0", names)

    def test_mixed_office_adds_more_reflecting_structure_than_open_office(self) -> None:
        open_scene = build_scenario("open_office")
        mixed_scene = build_scenario("mixed_office")
        self.assertGreater(len(mixed_scene.surfaces), len(open_scene.surfaces))

    def test_lobby_proxy_contains_column_blocked_points(self) -> None:
        spec = get_scenario_spec("lobby_blocker_proxy")
        scene = build_scenario(spec)
        rx_pos = np.array([7.5, 0.5, spec.rx_plane_z], dtype=float)
        paths = enumerate_paths(scene, spec.tx_pos, rx_pos, max_reflections=2)
        self.assertFalse(any(path.bounce_count == 0 for path in paths))

    def test_open_office_has_los_reference_point(self) -> None:
        spec = get_scenario_spec("open_office")
        scene = build_scenario(spec)
        rx_pos = np.array([10.5, 4.0, spec.rx_plane_z], dtype=float)
        paths = enumerate_paths(scene, spec.tx_pos, rx_pos, max_reflections=2)
        self.assertTrue(any(path.bounce_count == 0 for path in paths))

    def test_l_corridor_proxy_blocks_direct_path(self) -> None:
        spec = get_scenario_spec("l_corridor_proxy")
        scene = build_scenario(spec)
        rx_pos = np.array([9.5, 6.0, spec.rx_plane_z], dtype=float)
        paths = enumerate_paths(scene, spec.tx_pos, rx_pos, max_reflections=2)
        self.assertFalse(any(path.bounce_count == 0 for path in paths))
        self.assertGreater(len(paths), 0)


if __name__ == "__main__":
    unittest.main()

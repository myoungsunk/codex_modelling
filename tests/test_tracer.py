"""Unit tests for deterministic specular polarimetric tracer."""

from __future__ import annotations

import unittest

import numpy as np

from rt_core.antenna import Antenna
from rt_core.geometry import Material, Plane
from rt_core.tracer import trace_paths
from scenarios.common import make_los_blocker_plane


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

    def test_absorber_proxy_is_occluder_not_reflector(self) -> None:
        blocker = make_los_blocker_plane(self.tx.position, self.rx.position, plane_id=9101, half_extent_u=0.2, half_extent_v=0.3)
        paths = trace_paths([blocker], self.tx, self.rx, self.f, max_bounce=1, los_enabled=False)
        # With LOS disabled and no reflective planes, there should be no generated path.
        self.assertEqual(len(paths), 0)

    def test_los_blocker_blocks_los_but_not_reflector_candidate(self) -> None:
        refl = Plane(id=1, p0=np.array([0.0, 2.0, 0.0]), normal=np.array([0.0, -1.0, 0.0]), material=Material.pec())
        blocker = make_los_blocker_plane(self.tx.position, self.rx.position, plane_id=9101, half_extent_u=0.2, half_extent_v=0.3)
        paths = trace_paths([refl, blocker], self.tx, self.rx, self.f, max_bounce=1, los_enabled=True)
        self.assertFalse(any(p.bounce_count == 0 for p in paths))
        self.assertTrue(any(p.bounce_count == 1 for p in paths))
        # Blocker should never appear as a reflective surface in path metadata.
        self.assertFalse(any(9101 in list(p.surface_ids) for p in paths))

    def test_directional_gain_reduces_off_boresight_los_power(self) -> None:
        tx = Antenna(
            position=np.array([0.0, 0.0, 1.0]),
            boresight=np.array([1.0, 0.0, 0.0]),
            h_axis=np.array([0.0, 1.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            basis="linear",
            tx_pattern_cos_exp=4.0,
        )
        rx_on = Antenna(
            position=np.array([5.0, 0.0, 1.0]),
            boresight=np.array([-1.0, 0.0, 0.0]),
            h_axis=np.array([0.0, 1.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            basis="linear",
            rx_pattern_cos_exp=4.0,
        )
        rx_off = Antenna(
            position=np.array([2.5, 4.3301270189, 1.0]),  # same range ~5 m, ~60 deg off boresight
            boresight=np.array([-1.0, 0.0, 0.0]),
            h_axis=np.array([0.0, 1.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            basis="linear",
            rx_pattern_cos_exp=4.0,
        )
        p_on = trace_paths([], tx, rx_on, self.f, max_bounce=0, los_enabled=True)
        p_off = trace_paths([], tx, rx_off, self.f, max_bounce=0, los_enabled=True)
        self.assertEqual(len(p_on), 1)
        self.assertEqual(len(p_off), 1)
        pow_on = float(np.mean(np.abs(p_on[0].A_f) ** 2))
        pow_off = float(np.mean(np.abs(p_off[0].A_f) ** 2))
        self.assertLess(pow_off, pow_on)

    def test_wave_basis_mode_switch_keeps_power_consistent(self) -> None:
        p1 = Plane(id=1, p0=np.array([0.0, 3.0, 0.0]), normal=np.array([0.0, -1.0, 0.0]), material=Material.pec())
        p2 = Plane(id=2, p0=np.array([3.0, 0.0, 0.0]), normal=np.array([-1.0, 0.0, 0.0]), material=Material.pec())
        rx = Antenna(
            position=np.array([3.0, 4.0, 1.0]),
            boresight=np.array([-1.0, 0.0, 0.0]),
            h_axis=np.array([0.0, 1.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            basis="linear",
        )
        a = trace_paths([p1, p2], self.tx, rx, self.f, max_bounce=2, los_enabled=False, wave_basis_mode="transport")
        b = trace_paths([p1, p2], self.tx, rx, self.f, max_bounce=2, los_enabled=False, wave_basis_mode="global_up")
        self.assertTrue(len(a) > 0 and len(b) > 0)
        pa = max(a, key=lambda p: float(np.mean(np.abs(p.A_f) ** 2)))
        pb = max(b, key=lambda p: float(np.mean(np.abs(p.A_f) ** 2)))
        pwr_a = float(np.mean(np.abs(pa.A_f) ** 2))
        pwr_b = float(np.mean(np.abs(pb.A_f) ** 2))
        rel = abs(pwr_a - pwr_b) / max(pwr_a, pwr_b, 1e-18)
        self.assertLess(rel, 1e-6)

    def test_invalid_wave_basis_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            trace_paths([], self.tx, self.rx, self.f, max_bounce=0, wave_basis_mode="bad_mode")


if __name__ == "__main__":
    unittest.main()

"""Golden tests for geometry/polarization parity behavior."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.ctf_cir import synthesize_ctf
from rt_core.antenna import Antenna
from rt_core.geometry import Material, Plane, reflect_point
from rt_core.polarization import fresnel_reflection
from rt_core.tracer import trace_paths


class GoldenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.f = np.linspace(6e9, 10e9, 256)
        self.c0 = 299_792_458.0
        self.tx_lin = Antenna(
            position=np.array([0.0, 0.0, 1.5]),
            boresight=np.array([1.0, 0.0, 0.0]),
            h_axis=np.array([0.0, 1.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            basis="linear",
            cross_pol_leakage_db=60.0,
        )
        self.rx_lin = Antenna(
            position=np.array([6.0, 0.0, 1.5]),
            boresight=np.array([-1.0, 0.0, 0.0]),
            h_axis=np.array([0.0, 1.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            basis="linear",
            cross_pol_leakage_db=60.0,
        )

    def _cp_pair(self) -> tuple[Antenna, Antenna]:
        tx = Antenna(
            position=self.tx_lin.position.copy(),
            boresight=self.tx_lin.boresight.copy(),
            h_axis=self.tx_lin.h_axis.copy(),
            v_axis=self.tx_lin.v_axis.copy(),
            basis="circular",
            convention="IEEE-RHCP",
            cross_pol_leakage_db=80.0,
        )
        rx = Antenna(
            position=self.rx_lin.position.copy(),
            boresight=self.rx_lin.boresight.copy(),
            h_axis=self.rx_lin.h_axis.copy(),
            v_axis=self.rx_lin.v_axis.copy(),
            basis="circular",
            convention="IEEE-RHCP",
            cross_pol_leakage_db=80.0,
        )
        return tx, rx

    def test_g1_los_delay_and_phase_slope(self) -> None:
        paths = trace_paths([], self.tx_lin, self.rx_lin, self.f, max_bounce=0, los_enabled=True)
        self.assertEqual(len(paths), 1)
        r = np.linalg.norm(self.rx_lin.position - self.tx_lin.position)
        tau_exp = r / self.c0
        self.assertAlmostEqual(paths[0].tau_s, tau_exp, delta=1e-12)

        h = synthesize_ctf([{"tau_s": paths[0].tau_s, "A_f": paths[0].A_f, "meta": {}}], self.f)
        phase = np.unwrap(np.angle(h[:, 0, 0]))
        slope, _ = np.polyfit(self.f, phase, 1)
        self.assertAlmostEqual(slope, -2.0 * np.pi * tau_exp, delta=0.05 * abs(-2.0 * np.pi * tau_exp))

    def test_p1_a2_odd_bounce_opposite_hand_dominant(self) -> None:
        tx, rx = self._cp_pair()
        plane = Plane(
            id=1,
            p0=np.array([0.0, 2.0, 0.0]),
            normal=np.array([0.0, -1.0, 0.0]),
            material=Material.pec(),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=20.0,
            half_extent_v=20.0,
        )
        paths = trace_paths([plane], tx, rx, self.f, max_bounce=1, los_enabled=False)
        odd = [p for p in paths if p.bounce_count == 1]
        self.assertTrue(len(odd) > 0)
        A = np.mean(odd[0].A_f, axis=0)
        same = np.abs(A[0, 0]) ** 2 + np.abs(A[1, 1]) ** 2
        opp = np.abs(A[0, 1]) ** 2 + np.abs(A[1, 0]) ** 2
        self.assertGreater(opp, same)

    def test_p2_a3_even_bounce_same_hand_recovery(self) -> None:
        tx, rx = self._cp_pair()
        rx.position[:] = [3.0, 4.0, 1.5]
        p1 = Plane(
            id=1,
            p0=np.array([0.0, 3.0, 0.0]),
            normal=np.array([0.0, -1.0, 0.0]),
            material=Material.pec(),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=20.0,
            half_extent_v=20.0,
        )
        p2 = Plane(
            id=2,
            p0=np.array([3.0, 0.0, 0.0]),
            normal=np.array([-1.0, 0.0, 0.0]),
            material=Material.pec(),
            u_axis=np.array([0.0, 1.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=20.0,
            half_extent_v=20.0,
        )
        paths = trace_paths([p1, p2], tx, rx, self.f, max_bounce=2, los_enabled=False)
        even = [p for p in paths if p.bounce_count == 2]
        self.assertTrue(len(even) > 0)
        A = np.mean(even[0].A_f, axis=0)
        same = np.abs(A[0, 0]) ** 2 + np.abs(A[1, 1]) ** 2
        opp = np.abs(A[0, 1]) ** 2 + np.abs(A[1, 0]) ** 2
        self.assertGreaterEqual(same, opp)

    def test_g2_image_method_tau_match(self) -> None:
        plane = Plane(
            id=1,
            p0=np.array([0.0, 2.0, 0.0]),
            normal=np.array([0.0, -1.0, 0.0]),
            material=Material.pec(),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            half_extent_u=20.0,
            half_extent_v=20.0,
        )
        paths = trace_paths([plane], self.tx_lin, self.rx_lin, self.f, max_bounce=1, los_enabled=False)
        self.assertTrue(len(paths) > 0)
        tx_img = reflect_point(self.tx_lin.position, plane)
        r_img = np.linalg.norm(self.rx_lin.position - tx_img)
        tau_img = r_img / self.c0
        tau_rt = min(p.tau_s for p in paths if p.bounce_count == 1)
        self.assertAlmostEqual(tau_rt, tau_img, delta=3e-11)

    def test_m1_dielectric_gs_gp_frequency_variation(self) -> None:
        f = np.linspace(6e9, 10e9, 64)
        mat = Material.dielectric(eps_r=6.5, tan_delta=0.01)
        gs, gp = fresnel_reflection(mat, theta_i=np.deg2rad(35.0), f_hz=f)
        self.assertGreater(np.max(np.abs(gs - gp)), 1e-3)


if __name__ == "__main__":
    unittest.main()

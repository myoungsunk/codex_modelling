from __future__ import annotations

import unittest

import numpy as np

from dualpol_rt.channel.path_record import PathRecord
from dualpol_rt.em.materials import material_named
from dualpol_rt.metrics.richness import compute_mrs_scores, compute_path_richness, label_mrs_scores


def _make_path(
    delay_s: float,
    launch_dir: tuple[float, float, float],
    arrival_dir: tuple[float, float, float],
    jones: np.ndarray,
    scalar: np.ndarray,
) -> PathRecord:
    direction_tx = np.asarray(launch_dir, dtype=float)
    direction_rx = np.asarray(arrival_dir, dtype=float)
    return PathRecord(
        points=(np.zeros(3, dtype=float), np.asarray(direction_tx, dtype=float), np.asarray(direction_tx, dtype=float) + np.asarray(direction_rx, dtype=float)),
        surface_ids=tuple(),
        surface_names=tuple(),
        materials=tuple(),
        bounce_count=0,
        path_length_m=delay_s * 299792458.0,
        delay_s=delay_s,
        launch_dir=direction_tx / np.linalg.norm(direction_tx),
        arrival_dir=direction_rx / np.linalg.norm(direction_rx),
        incidence_angles_rad=tuple(),
        normals=tuple(),
        blocked=False,
        valid=True,
        jones_f=np.asarray(jones, dtype=np.complex128),
        scalar_factor_f=np.asarray(scalar, dtype=np.complex128),
    )


class RichnessMetricTests(unittest.TestCase):
    def test_single_path_metrics_are_degenerate(self) -> None:
        freqs = np.array([6.5e9], dtype=float)
        path = _make_path(
            delay_s=1e-9,
            launch_dir=(1.0, 0.0, 0.0),
            arrival_dir=(1.0, 0.0, 0.0),
            jones=np.array([np.eye(2)], dtype=np.complex128),
            scalar=np.array([1.0 + 0.0j], dtype=np.complex128),
        )
        metrics = compute_path_richness([path], freqs_hz=freqs)
        self.assertAlmostEqual(metrics.n_eff, 1.0, places=12)
        self.assertAlmostEqual(metrics.tau_rms_s, 0.0, places=12)
        self.assertAlmostEqual(metrics.dpr_db, 40.0, places=12)
        self.assertAlmostEqual(metrics.pmi_env, 0.0, places=12)
        self.assertAlmostEqual(metrics.mean_aoa_spread_deg, 0.0, places=12)

    def test_two_equal_paths_raise_effective_path_count_and_spread(self) -> None:
        freqs = np.array([6.4e9, 6.6e9], dtype=float)
        identity = np.repeat(np.eye(2, dtype=np.complex128)[None, :, :], len(freqs), axis=0)
        cross = np.repeat(np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)[None, :, :], len(freqs), axis=0)
        path_a = _make_path(
            delay_s=0.0,
            launch_dir=(1.0, 0.0, 0.0),
            arrival_dir=(1.0, 0.0, 0.0),
            jones=identity,
            scalar=np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128),
        )
        path_b = _make_path(
            delay_s=1e-9,
            launch_dir=(0.0, 1.0, 0.0),
            arrival_dir=(0.0, 1.0, 0.0),
            jones=cross,
            scalar=np.array([1.0 / np.sqrt(2.0) + 0.0j, 1.0 / np.sqrt(2.0) + 0.0j], dtype=np.complex128),
        )
        metrics = compute_path_richness([path_a, path_b], freqs_hz=freqs)
        self.assertAlmostEqual(metrics.n_eff, 2.0, places=12)
        self.assertAlmostEqual(metrics.tau_rms_s, 5e-10, places=12)
        self.assertGreater(metrics.aoa_azimuth_spread_deg, 0.0)
        self.assertLess(metrics.dpr_db, 1e-9)
        self.assertGreater(metrics.pmi_env, 0.0)

    def test_mrs_reference_scaling_uses_non_proxy_corpus(self) -> None:
        freqs = np.array([6.5e9], dtype=float)
        weak = compute_path_richness([_make_path(0.0, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), np.array([np.eye(2)], dtype=np.complex128), np.array([1.0 + 0.0j], dtype=np.complex128))], freqs_hz=freqs)
        rich = compute_path_richness([
            _make_path(0.0, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), np.array([np.eye(2)], dtype=np.complex128), np.array([1.0 + 0.0j], dtype=np.complex128)),
            _make_path(8e-10, (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), np.array([np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)]), np.array([1.0 + 0.0j], dtype=np.complex128)),
        ], freqs_hz=freqs)
        proxy_extreme = compute_path_richness([
            _make_path(0.0, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), np.array([np.eye(2)], dtype=np.complex128), np.array([10.0 + 0.0j], dtype=np.complex128)),
            _make_path(5e-9, (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), np.array([np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)]), np.array([10.0 + 0.0j], dtype=np.complex128)),
            _make_path(9e-9, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), np.array([np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)]), np.array([10.0 + 0.0j], dtype=np.complex128)),
        ], freqs_hz=freqs)
        _, scaled_ref = compute_mrs_scores([weak, rich], reference_metrics_list=[weak, rich])
        _, scaled_all = compute_mrs_scores([weak, rich, proxy_extreme], reference_metrics_list=[weak, rich])
        self.assertTrue(np.allclose(scaled_ref, scaled_all[:2]))

    def test_mrs_scoring_and_labeling_return_ordered_buckets(self) -> None:
        freqs = np.array([6.5e9], dtype=float)
        weak = compute_path_richness(
            [
                _make_path(
                    delay_s=0.0,
                    launch_dir=(1.0, 0.0, 0.0),
                    arrival_dir=(1.0, 0.0, 0.0),
                    jones=np.array([np.eye(2)], dtype=np.complex128),
                    scalar=np.array([1.0 + 0.0j], dtype=np.complex128),
                )
            ],
            freqs_hz=freqs,
        )
        mid = compute_path_richness(
            [
                _make_path(0.0, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), np.array([np.eye(2)], dtype=np.complex128), np.array([1.0 + 0.0j], dtype=np.complex128)),
                _make_path(6e-10, (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), np.array([np.eye(2)], dtype=np.complex128), np.array([0.4 + 0.0j], dtype=np.complex128)),
            ],
            freqs_hz=freqs,
        )
        rich = compute_path_richness(
            [
                _make_path(0.0, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), np.array([np.eye(2)], dtype=np.complex128), np.array([1.0 + 0.0j], dtype=np.complex128)),
                _make_path(8e-10, (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), np.array([np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)]), np.array([1.0 + 0.0j], dtype=np.complex128)),
                _make_path(1.5e-9, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), np.array([np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)]), np.array([1.0 + 0.0j], dtype=np.complex128)),
            ],
            freqs_hz=freqs,
        )
        raw_scores, scaled_scores = compute_mrs_scores([weak, mid, rich])
        labels, _low, _high = label_mrs_scores(scaled_scores)
        self.assertEqual(len(raw_scores), 3)
        self.assertEqual(labels[0], "mild")
        self.assertEqual(labels[1], "moderate")
        self.assertEqual(labels[2], "rich")


if __name__ == "__main__":
    unittest.main()

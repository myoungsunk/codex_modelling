from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dualpol_rt.patterns import AntennaPose, FFDPattern


def _write_multifrequency_ffd(
    path: Path,
    freqs_hz: list[float],
    blocks: list[list[tuple[complex, complex]]],
) -> None:
    lines = [
        "0 90 2",
        "-90 90 2",
        f"Frequencies {len(freqs_hz)}",
    ]
    for freq_hz, block in zip(freqs_hz, blocks):
        if len(block) != 4:
            raise ValueError("Each synthetic block must contain 4 theta/phi samples")
        lines.append(f"Frequency {freq_hz:.15e}")
        for e_theta, e_phi in block:
            lines.append(f"{e_theta.real:.15e} {e_theta.imag:.15e} {e_phi.real:.15e} {e_phi.imag:.15e}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_single_frequency_ffd(path: Path, block: list[tuple[complex, complex]]) -> None:
    lines = [
        "0 90 2",
        "-90 90 2",
    ]
    for e_theta, e_phi in block:
        lines.append(f"{e_theta.real:.15e} {e_theta.imag:.15e} {e_phi.real:.15e} {e_phi.imag:.15e}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _base_block(scale: float) -> list[tuple[complex, complex]]:
    return [
        (scale * (1.0 + 1.0j), scale * (10.0 - 1.0j)),
        (scale * (3.0 + 3.0j), scale * (30.0 - 3.0j)),
        (scale * (5.0 + 5.0j), scale * (50.0 - 5.0j)),
        (scale * (7.0 + 7.0j), scale * (70.0 - 7.0j)),
    ]


class FFDPatternTests(unittest.TestCase):
    def test_multifreq_parse_and_freq_interp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_multifrequency_ffd(tmp_path / "port0.ffd", [6.0e9, 7.0e9], [_base_block(1.0), _base_block(2.0)])
            _write_multifrequency_ffd(tmp_path / "port1.ffd", [6.0e9, 7.0e9], [_base_block(0.5), _base_block(1.0)])
            pattern = FFDPattern.from_port_files(
                {"port0": tmp_path / "port0.ffd", "port1": tmp_path / "port1.ffd"},
                port_order=("port0", "port1"),
            )
            direction = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5)], dtype=float)
            exact = pattern.port_field(0, np.array([6.0e9]), direction)[0]
            interp = pattern.port_field(0, np.array([6.5e9]), direction)[0]
            self.assertTrue(np.allclose(exact, np.array([4.0 + 4.0j, 40.0 - 4.0j]), atol=1e-12))
            self.assertTrue(np.allclose(interp, np.array([6.0 + 6.0j, 60.0 - 6.0j]), atol=1e-12))

    def test_hv_loading_rule_is_ephi_etheta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_multifrequency_ffd(tmp_path / "port0.ffd", [6.0e9, 7.0e9], [_base_block(1.0), _base_block(2.0)])
            _write_multifrequency_ffd(tmp_path / "port1.ffd", [6.0e9, 7.0e9], [_base_block(0.5), _base_block(1.0)])
            pattern = FFDPattern.from_port_files(
                {"port0": tmp_path / "port0.ffd", "port1": tmp_path / "port1.ffd"},
                port_order=("port0", "port1"),
            )
            direction = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5)], dtype=float)
            theta_phi = pattern.port_field(0, np.array([6.0e9]), direction)[0]
            hv = pattern.port_field_hv(0, np.array([6.0e9]), direction)[0]
            self.assertTrue(np.allclose(hv, np.array([theta_phi[1], theta_phi[0]]), atol=1e-12))

    def test_out_of_range_frequency_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_multifrequency_ffd(tmp_path / "port0.ffd", [6.0e9, 7.0e9], [_base_block(1.0), _base_block(2.0)])
            _write_multifrequency_ffd(tmp_path / "port1.ffd", [6.0e9, 7.0e9], [_base_block(0.5), _base_block(1.0)])
            pattern = FFDPattern.from_port_files(
                {"port0": tmp_path / "port0.ffd", "port1": tmp_path / "port1.ffd"},
                port_order=("port0", "port1"),
            )
            direction = np.array([0.0, 0.0, 1.0], dtype=float)
            with self.assertRaises(ValueError):
                pattern.port_field(0, np.array([7.1e9]), direction)

    def test_family_normalization_scales_fields_consistently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_multifrequency_ffd(tmp_path / "port0.ffd", [6.0e9, 7.0e9], [_base_block(2.0), _base_block(4.0)])
            _write_multifrequency_ffd(tmp_path / "port1.ffd", [6.0e9, 7.0e9], [_base_block(1.0), _base_block(2.0)])
            pattern = FFDPattern.from_port_files(
                {"port0": tmp_path / "port0.ffd", "port1": tmp_path / "port1.ffd"},
                port_order=("port0", "port1"),
            )
            normalized = pattern.normalized_copy("family_radiated_power")
            direction = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5)], dtype=float)
            raw_sample = pattern.port_field(0, np.array([6.0e9]), direction)[0]
            norm_sample = normalized.port_field(0, np.array([6.0e9]), direction)[0]
            self.assertGreater(float(pattern.radiation_energy_by_freq()[0]), 1.0)
            self.assertTrue(np.allclose(normalized.radiation_energy_by_freq(), np.ones(2, dtype=float), atol=1e-12))
            self.assertTrue(np.allclose(norm_sample, raw_sample * normalized.family_scale_f[0], atol=1e-12))

    def test_facing_pose_rotation_keeps_boresight_on_local_plus_z(self) -> None:
        pose = AntennaPose.facing_from_points(np.array([0.0, 0.0, 0.0], dtype=float), np.array([1.0, 0.0, 0.0], dtype=float))
        world_boresight = np.array([1.0, 0.0, 0.0], dtype=float)
        local = pose.world_to_local(world_boresight)
        world = pose.local_to_world(np.array([0.0, 0.0, 1.0], dtype=float))
        self.assertTrue(np.allclose(local / np.linalg.norm(local), np.array([0.0, 0.0, 1.0]), atol=1e-12))
        self.assertTrue(np.allclose(world / np.linalg.norm(world), np.array([1.0, 0.0, 0.0]), atol=1e-12))

    def test_single_frequency_array_file_backcompat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_single_frequency_ffd(tmp_path / "port0.ffd", _base_block(1.0))
            _write_single_frequency_ffd(tmp_path / "port1.ffd", _base_block(0.5))
            (tmp_path / "array.txt").write_text("port0 port0.ffd\nport1 port1.ffd\n", encoding="utf-8")
            pattern = FFDPattern.from_array_file(tmp_path / "array.txt")
            direction = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5)], dtype=float)
            sample = pattern.port_field(0, np.array([6.5e9]), direction)[0]
            self.assertTrue(np.allclose(sample, np.array([4.0 + 4.0j, 40.0 - 4.0j]), atol=1e-12))


if __name__ == "__main__":
    unittest.main()

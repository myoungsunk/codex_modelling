"""Tests for CTF->CIR diagnostics and grid interpretation helpers."""

from __future__ import annotations

import unittest
import warnings
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ISAC_communication_verification"))

from analysis.ctf_cir import cir_bandlimit_info, convert_basis, ctf_to_cir, linear_to_circular_matrix, tau_resolution_s
from dualpol_rt.channel.builder import convert_basis as isac_convert_basis


class CtfCirTests(unittest.TestCase):
    def test_tau_resolution_uses_nfft_and_df(self) -> None:
        f = np.linspace(6e9, 10e9, 256)
        nfft = 2048
        dt = tau_resolution_s(f, nfft=nfft)
        df = float(np.median(np.diff(f)))
        self.assertAlmostEqual(dt, 1.0 / (nfft * df), places=18)

    def test_cir_bandlimit_info_reports_passband_and_bw_limit(self) -> None:
        f = np.linspace(6e9, 10e9, 256)
        info = cir_bandlimit_info(f, nfft=2048)
        self.assertTrue(bool(info["passband_only"]))
        self.assertAlmostEqual(float(info["bw_hz"]), 4e9, places=3)
        self.assertAlmostEqual(float(info["delay_resolution_bw_limit_s"]), 1.0 / 4e9, places=18)
        self.assertGreater(float(info["tau_unambiguous_s"]), 0.0)

    def test_ctf_to_cir_runs_with_quasi_uniform_grid(self) -> None:
        # Slightly nonuniform grid should still run; diagnostics should expose nonuniformity.
        f = np.linspace(6e9, 10e9, 128)
        f[10] += 1e3
        H = np.ones((len(f), 2, 2), dtype=np.complex128)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h, tau = ctf_to_cir(H, f, nfft=256, window="hann", nonuniform_warn_rel_max=1e-8)
        self.assertEqual(h.shape[0], 256)
        self.assertEqual(len(tau), 256)
        self.assertTrue(any("nonuniform frequency grid" in str(ww.message).lower() for ww in w))
        info = cir_bandlimit_info(f, nfft=256)
        self.assertGreaterEqual(float(info["grid_uniformity_rel_max"]), 0.0)

    def test_linear_to_circular_matrix_defaults_to_rhcp_lhcp_order(self) -> None:
        U = linear_to_circular_matrix()
        expected = np.column_stack(
            [
                np.array([1.0, -1j], dtype=np.complex128) / np.sqrt(2.0),
                np.array([1.0, 1j], dtype=np.complex128) / np.sqrt(2.0),
            ]
        )
        self.assertTrue(np.allclose(U, expected, atol=1e-12))

    def test_analysis_basis_conversion_matches_isac_builder(self) -> None:
        rng = np.random.default_rng(7)
        H = rng.normal(size=(5, 2, 2)) + 1j * rng.normal(size=(5, 2, 2))
        converted_analysis = convert_basis(H, src="linear", dst="circular", circular_order="RL")
        converted_isac = isac_convert_basis(H, src="linear", dst="circular", circular_order="RL")
        self.assertTrue(np.allclose(converted_analysis, converted_isac, atol=1e-12))


if __name__ == "__main__":
    unittest.main()

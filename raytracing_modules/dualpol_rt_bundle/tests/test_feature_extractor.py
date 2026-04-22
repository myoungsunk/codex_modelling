from __future__ import annotations

import unittest

import numpy as np

from dualpol_rt.feature_extractor import (
    compute_gamma_cp_variants,
    extract_all_features,
    extract_first_path,
    ifft_to_cir,
)


class FeatureExtractorTests(unittest.TestCase):
    def test_ifft_to_cir_and_first_path_recover_discrete_delay(self) -> None:
        n_freq = 16
        df_hz = 10.0e6
        freqs = 6.0e9 + np.arange(n_freq, dtype=float) * df_hz
        delay_s = 3.0 / (n_freq * df_hz)
        H = np.exp(-1j * 2.0 * np.pi * freqs * delay_s)[:, None, None]
        cir = ifft_to_cir(H, freqs, window="rect")
        t_fp, idx_fp = extract_first_path(cir, threshold="leading_edge")
        self.assertEqual(idx_fp, 3)
        self.assertAlmostEqual(t_fp, delay_s, places=15)

    def test_compute_gamma_cp_variants_reports_power_imbalance_and_cross_ratio(self) -> None:
        h_rhcp = np.array([2.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
        h_lhcp = np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
        h_cross = np.array([0.2 + 0.0j, 0.2 + 0.0j], dtype=np.complex128)
        features = compute_gamma_cp_variants(h_rhcp, h_lhcp, H_f_cross_rl=h_cross, H_f_cross_lr=h_cross)
        self.assertAlmostEqual(features["gamma_cp_rhcp_to_lhcp_db"], 6.020599913279624, places=9)
        self.assertGreater(features["gamma_cp_copol_to_cross_db"], 10.0)

    def test_extract_all_features_supports_circular_channel_dict(self) -> None:
        n_freq = 16
        df_hz = 10.0e6
        freqs = 6.0e9 + np.arange(n_freq, dtype=float) * df_hz
        delay_s = 4.0 / (n_freq * df_hz)
        phase = np.exp(-1j * 2.0 * np.pi * freqs * delay_s)
        H = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        H[:, 0, 0] = 2.0 * phase
        H[:, 1, 1] = 1.0 * phase
        H[:, 0, 1] = 0.2 * phase
        H[:, 1, 0] = 0.1 * phase
        features = extract_all_features({"H_circular": H, "freqs_hz": freqs, "cir_window": "rect"})
        self.assertEqual(features["channel_basis"], "circular")
        self.assertEqual(features["first_path_index"], 4)
        self.assertAlmostEqual(features["first_path_delay_s"], delay_s, places=15)
        self.assertGreater(features["gamma_cp_rhcp_to_lhcp_db"], 5.0)
        self.assertGreater(features["a_fp_xpr_db"], 9.0)


if __name__ == "__main__":
    unittest.main()

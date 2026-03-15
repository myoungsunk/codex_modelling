"""Unit tests for dual-CP power-domain metrics."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.dualcp_metrics import compute_dualcp_metrics_from_Hf


class DualCpMetricsTests(unittest.TestCase):
    def test_single_early_and_late_tap_xpd_matches_expected(self) -> None:
        n = 256
        df = 1.0e6
        f = np.arange(n, dtype=float) * df

        # Build taps in CIR domain, then FFT to H(f), so metric extraction is deterministic.
        h_tau = np.zeros((n, 2, 2), dtype=np.complex128)
        h_tau[5, 0, 0] = 1.0
        h_tau[5, 1, 0] = 0.1   # early XPD = 20 dB
        h_tau[20, 0, 0] = 0.5
        h_tau[20, 1, 0] = 0.5  # late XPD = 0 dB
        H_f = np.fft.fft(h_tau, axis=0)

        out = compute_dualcp_metrics_from_Hf(
            H_f=H_f,
            f_hz=f,
            params={
                "nfft": n,
                "window": "none",
                "early_window_ns": 20.0,
                "tmax_ns": 120.0,
                "noise_tail_ns": 20.0,
                "threshold_db": 3.0,
            },
        )
        self.assertAlmostEqual(float(out["xpd_early_db"]), 20.0, places=6)
        self.assertAlmostEqual(float(out["xpd_late_db"]), 0.0, places=6)
        self.assertAlmostEqual(float(out["l_pol_db"]), 20.0, places=6)
        self.assertAlmostEqual(float(out["rho_early_linear"]), 0.01, places=9)
        self.assertGreater(float(out["early_energy_concentration"]), 0.0)
        self.assertGreaterEqual(float(out["tau_rms_ns"]), 0.0)


if __name__ == "__main__":
    unittest.main()

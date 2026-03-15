"""Tests for dual-CP sequential CSV import mapping."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.measurement_compare import load_measurement_dualcp_three_csv, load_measurement_dualcp_two_csv


class DualCpImportTests(unittest.TestCase):
    def _write_trace_csv_re_im(self, path: Path, rows: list[tuple[float, complex]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["f_hz", "s21_re", "s21_im"])
            for ff, z in rows:
                w.writerow([ff, float(np.real(z)), float(np.imag(z))])

    def _write_trace_csv_mag_phase(self, path: Path, rows: list[tuple[float, complex]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frequency_hz", "s21_mag_db", "s21_phase_deg"])
            for ff, z in rows:
                mag_db = 20.0 * np.log10(max(np.abs(z), 1e-18))
                phase_deg = np.rad2deg(np.angle(z))
                w.writerow([ff, float(mag_db), float(phase_deg)])

    def test_dualcp_two_csv_maps_to_hf_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            co_csv = Path(td) / "co.csv"
            cross_csv = Path(td) / "cross.csv"
            co_rows = [(6.0e9, 1.0 + 0.0j), (6.1e9, 0.5 + 0.5j)]
            cross_rows = [(6.0e9, 0.1 + 0.2j), (6.1e9, 0.2 + 0.3j)]
            self._write_trace_csv_re_im(co_csv, co_rows)
            self._write_trace_csv_mag_phase(cross_csv, cross_rows)

            m = load_measurement_dualcp_two_csv(co_csv=co_csv, cross_csv=cross_csv)
            self.assertEqual(m.H_f.shape, (2, 2, 2))
            np.testing.assert_allclose(m.H_f[:, 0, 0], np.asarray([x[1] for x in co_rows], dtype=np.complex128), atol=1e-12)
            np.testing.assert_allclose(
                m.H_f[:, 1, 0],
                np.asarray([x[1] for x in cross_rows], dtype=np.complex128),
                atol=1e-12,
            )
            np.testing.assert_allclose(m.H_f[:, 0, 1], 0.0, atol=1e-15)
            np.testing.assert_allclose(m.H_f[:, 1, 1], 0.0, atol=1e-15)
            self.assertEqual(str(m.meta.get("basis")), "circular")
            self.assertEqual(str(m.meta.get("convention")), "IEEE-RHCP")

    def test_dualcp_three_csv_reports_drift(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            co_pre_csv = Path(td) / "co_pre.csv"
            cross_csv = Path(td) / "cross.csv"
            co_post_csv = Path(td) / "co_post.csv"
            f = [6.0e9, 6.1e9]
            co_pre_rows = [(f[0], 1.0 + 0.0j), (f[1], 1.0 + 0.0j)]
            cross_rows = [(f[0], 0.1 + 0.0j), (f[1], 0.1 + 0.0j)]
            co_post_rows = [(f[0], 10 ** (1.0 / 20.0) + 0.0j), (f[1], 10 ** (1.0 / 20.0) + 0.0j)]  # +1 dB
            self._write_trace_csv_re_im(co_pre_csv, co_pre_rows)
            self._write_trace_csv_re_im(cross_csv, cross_rows)
            self._write_trace_csv_re_im(co_post_csv, co_post_rows)

            m = load_measurement_dualcp_three_csv(
                co_pre_csv=co_pre_csv,
                cross_csv=cross_csv,
                co_post_csv=co_post_csv,
            )
            self.assertEqual(m.H_f.shape, (2, 2, 2))
            np.testing.assert_allclose(m.H_f[:, 0, 0], np.asarray([x[1] for x in co_pre_rows], dtype=np.complex128), atol=1e-12)
            self.assertTrue(float(m.meta.get("drift_co_db", 0.0)) > 0.9)
            self.assertEqual(str(m.meta.get("format")), "dualcp_three_csv")


if __name__ == "__main__":
    unittest.main()

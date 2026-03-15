"""Tests for provenance metadata persistence and reporting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from rt_io.hdf5_io import V2_REQUIRED_META_ATTRS, load_rt_dataset, save_rt_dataset
from scenarios.runner import build_quality_report


class ProvenanceMetadataTests(unittest.TestCase):
    def test_hdf5_attrs_include_provenance(self) -> None:
        data = {
            "meta": {
                "basis": "linear",
                "convention": "IEEE-RHCP",
                "git_commit": "abc123",
                "git_dirty": False,
                "cmdline": "python -m scenarios.runner --nf 64",
                "seed": {"model_seed": 42},
                "release_mode": True,
                "git_diff": "diff --git a/x b/x\n+test\n",
            },
            "frequency": np.linspace(6e9, 7e9, 8),
            "scenarios": {"C0": {"cases": {"0": {"params": {}, "paths": []}}}},
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rt.h5"
            save_rt_dataset(p, data)
            with h5py.File(p, "r") as f:
                m = f["meta"].attrs
                for k in V2_REQUIRED_META_ATTRS:
                    self.assertIn(k, m)
                    if k in {"git_dirty", "release_mode", "exact_bounce_applied", "physics_validation_mode"}:
                        continue
                    self.assertTrue(str(m[k]) != "", k)
                self.assertIn("git_diff", m)
                self.assertIn("seed", m)
            loaded = load_rt_dataset(p)
            self.assertEqual(str(loaded["meta"]["git_commit"]), "abc123")
            self.assertFalse(bool(loaded["meta"]["git_dirty"]))
            self.assertIn("scenarios.runner", str(loaded["meta"]["cmdline"]))
            self.assertIn("model_seed", str(loaded["meta"]["seed"]))
            self.assertTrue(bool(loaded["meta"]["release_mode"]))

    def test_report_marks_dirty_release_failure(self) -> None:
        data = {
            "meta": {
                "basis": "circular",
                "convention": "IEEE-RHCP",
                "git_commit": "deadbeef",
                "git_dirty": True,
                "release_mode": True,
                "cmdline": "python -m scenarios.runner --release-mode",
                "seed": {"model_seed": 0},
                "antenna_config": {"enable_coupling": True},
            },
            "frequency": np.linspace(6e9, 7e9, 8),
            "scenarios": {"C0": {"cases": {"0": {"params": {}, "paths": []}}}},
        }
        with tempfile.TemporaryDirectory() as td:
            rep = Path(td) / "report.md"
            build_quality_report(data, rep, matrix_source="A")
            txt = rep.read_text(encoding="utf-8")
            self.assertIn("WARNING: git_dirty=True", txt)
            self.assertIn("FAIL: release_mode requires git_dirty=False", txt)


if __name__ == "__main__":
    unittest.main()

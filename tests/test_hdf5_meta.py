"""HDF5 v2 metadata contract tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from rt_io.hdf5_io import (
    SCHEMA_VERSION,
    V2_REQUIRED_META_ATTRS,
    load_rt_dataset,
    save_rt_dataset,
    self_test_meta_roundtrip,
)


class Hdf5MetaContractTests(unittest.TestCase):
    def _tiny_data(self) -> dict:
        return {
            "meta": {
                "schema_version": SCHEMA_VERSION,
                "git_commit": "abc123",
                "git_dirty": False,
                "cmdline": "python -m scenarios.runner --release-mode",
                "seed": {"model_seed": 7},
                "seed_json": "{\"model_seed\":7}",
                "basis": "circular",
                "convention": "IEEE-RHCP",
                "xpd_matrix_source": "J",
                "exact_bounce_defaults": {"A2": 1, "A3": 2},
                "physics_validation_mode": True,
                "antenna_config": {"enable_coupling": False, "tx_cross_pol_leakage_db": 120.0},
                "release_mode": True,
                "created_at": "2026-02-21T00:00:00+00:00",
            },
            "frequency": np.linspace(6e9, 7e9, 4),
            "scenarios": {"C0": {"cases": {"0": {"params": {}, "paths": []}}}},
        }

    def test_required_meta_attrs_are_written_and_decodable(self) -> None:
        data = self._tiny_data()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rt.h5"
            save_rt_dataset(p, data)
            with h5py.File(p, "r") as f:
                attrs = f["meta"].attrs
                for k in V2_REQUIRED_META_ATTRS:
                    self.assertIn(k, attrs, k)
            loaded = load_rt_dataset(p)
            meta = loaded["meta"]
            self.assertEqual(meta["schema_version"], SCHEMA_VERSION)
            self.assertEqual(meta["xpd_matrix_source"], "J")
            self.assertIsInstance(meta["exact_bounce_defaults"], dict)
            self.assertIsInstance(meta["antenna_config"], dict)
            self.assertTrue(bool(meta["physics_validation_mode"]))
            self.assertTrue(bool(meta["release_mode"]))
            self.assertFalse(bool(meta["git_dirty"]))
            self.assertTrue(isinstance(str(meta["git_commit"]), str))
            self.assertTrue(self_test_meta_roundtrip(p, expected_meta=data["meta"]))

    def test_missing_required_v2_meta_attrs_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "broken.h5"
            with h5py.File(p, "w") as f:
                meta = f.create_group("meta")
                meta.attrs["schema_version"] = SCHEMA_VERSION
                meta.attrs["basis"] = "linear"
                f.create_dataset("frequency", data=np.linspace(6e9, 7e9, 4))
                sc = f.create_group("scenarios")
                sc.create_group("C0").create_group("cases")
            with self.assertRaises(ValueError) as ctx:
                _ = load_rt_dataset(p)
            msg = str(ctx.exception)
            self.assertIn("Missing required v2 meta attrs", msg)
            self.assertIn("git_commit", msg)
            self.assertIn("xpd_matrix_source", msg)


if __name__ == "__main__":
    unittest.main()

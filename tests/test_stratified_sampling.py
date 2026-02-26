"""Tests for stratified sampling helper."""

from __future__ import annotations

import unittest

from analysis.stratified_sampling import stratified_sample


class StratifiedSamplingTests(unittest.TestCase):
    def test_binary_flag_binning_keeps_los_and_nlos(self) -> None:
        rows = [
            {"link_id": "l0", "LOSflag": 0, "EL_proxy_db": 1.0},
            {"link_id": "l1", "LOSflag": 0, "EL_proxy_db": 2.0},
            {"link_id": "l2", "LOSflag": 0, "EL_proxy_db": 3.0},
            {"link_id": "l3", "LOSflag": 1, "EL_proxy_db": 4.0},
            {"link_id": "l4", "LOSflag": 1, "EL_proxy_db": 5.0},
            {"link_id": "l5", "LOSflag": 1, "EL_proxy_db": 6.0},
        ]
        out = stratified_sample(rows, bins={"LOSflag": 2}, per_bin=1, seed=0)
        self.assertEqual(len(out), 2)
        self.assertEqual({int(r["LOSflag"]) for r in out}, {0, 1})


if __name__ == "__main__":
    unittest.main()


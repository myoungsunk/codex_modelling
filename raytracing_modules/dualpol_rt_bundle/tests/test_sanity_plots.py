from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dualpol_rt.experiments.sanity_plots import generate_sanity_plots


class SanityPlotTests(unittest.TestCase):
    def test_generate_sanity_plots_writes_five_pngs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as tmp_dir:
            outdir = Path(tmp_dir)
            paths = generate_sanity_plots(outdir)
            self.assertEqual(
                set(paths),
                {
                    "plot_los_pathloss",
                    "plot_handedness_reversal",
                    "plot_fresnel_brewster",
                    "plot_cir_los",
                    "plot_antenna_compare",
                },
            )
            for path in paths.values():
                file_path = Path(path)
                self.assertTrue(file_path.exists())
                self.assertEqual(file_path.suffix.lower(), ".png")
                self.assertGreater(file_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path

import numpy as np

from dualpol_rt.patterns.ffd_pattern import FFDPattern


def run_ffd_smoke(array_file: str | Path, sample_direction_local: np.ndarray | None = None) -> dict[str, object]:
    pattern = FFDPattern.from_array_file(array_file)
    direction = np.asarray([0.0, 0.0, 1.0] if sample_direction_local is None else sample_direction_local, dtype=float)
    sample0 = pattern.port_field(0, np.array([6.5e9]), direction)[0]
    sample1 = pattern.port_field(1, np.array([6.5e9]), direction)[0]
    return {
        "array_file": str(array_file),
        "port_names": pattern.port_names,
        "sample_direction_local": direction,
        "port0_field_theta_phi": sample0,
        "port1_field_theta_phi": sample1,
    }

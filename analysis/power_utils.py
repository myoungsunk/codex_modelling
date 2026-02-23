"""Shared helpers for per-path power calculations."""

from __future__ import annotations

from typing import Any

import numpy as np


def path_power(path: dict[str, Any], matrix_source: str = "A") -> float:
    """Return average matrix-element power for one path."""
    use_j = str(matrix_source).upper() == "J" and "J_f" in path
    m = np.asarray(path["J_f"] if use_j else path["A_f"], dtype=np.complex128)
    return float(np.mean(np.abs(m) ** 2))


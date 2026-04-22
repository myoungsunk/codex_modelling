from __future__ import annotations

import numpy as np


def xpr_db(H_f: np.ndarray, eps: float = 1e-15) -> float:
    H = np.asarray(H_f, dtype=np.complex128)
    co = np.mean(np.abs(H[:, 0, 0]) ** 2 + np.abs(H[:, 1, 1]) ** 2)
    cross = np.mean(np.abs(H[:, 0, 1]) ** 2 + np.abs(H[:, 1, 0]) ** 2)
    return float(10.0 * np.log10((co + eps) / (cross + eps)))


def singular_values(H_f: np.ndarray) -> np.ndarray:
    H = np.asarray(H_f, dtype=np.complex128)
    return np.asarray([np.linalg.svd(Hk, compute_uv=False) for Hk in H], dtype=float)


def condition_numbers(H_f: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    sv = singular_values(H_f)
    sigma_max = sv[:, 0]
    sigma_min = sv[:, -1]
    return sigma_max / np.maximum(sigma_min, eps)

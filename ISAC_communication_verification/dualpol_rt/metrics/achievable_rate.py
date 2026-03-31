from __future__ import annotations

import numpy as np


def _snr_linear(snr_db: float) -> float:
    return float(10.0 ** (float(snr_db) / 10.0))


def equal_power_rate(H_f: np.ndarray, snr_db: float) -> float:
    H = np.asarray(H_f, dtype=np.complex128)
    nt = H.shape[-1]
    rho = _snr_linear(snr_db)
    vals = []
    for Hk in H:
        gram = Hk @ Hk.conj().T
        vals.append(float(np.log2(np.linalg.det(np.eye(gram.shape[0]) + (rho / nt) * gram)).real))
    return float(np.mean(vals)) if vals else 0.0


def _waterfill_power_allocation(eigs: np.ndarray, total_power: float) -> np.ndarray:
    eigs = np.asarray(np.real(eigs), dtype=float)
    positive = eigs > 1e-15
    if not np.any(positive):
        return np.zeros_like(eigs)
    inv = np.zeros_like(eigs)
    inv[positive] = 1.0 / eigs[positive]
    active = np.where(positive)[0]
    for _ in range(len(active)):
        mu = (total_power + np.sum(inv[active])) / len(active)
        power = mu - inv
        new_active = np.where(power > 0.0)[0]
        if len(new_active) == len(active):
            out = np.maximum(power, 0.0)
            if np.sum(out) > 0.0:
                out *= total_power / np.sum(out)
            return out
        active = new_active
        if len(active) == 0:
            break
    out = np.zeros_like(eigs)
    if len(active):
        mu = (total_power + np.sum(inv[active])) / len(active)
        out = np.maximum(mu - inv, 0.0)
        if np.sum(out) > 0.0:
            out *= total_power / np.sum(out)
    return out


def waterfilled_capacity(H_f: np.ndarray, snr_db: float) -> float:
    H = np.asarray(H_f, dtype=np.complex128)
    rho = _snr_linear(snr_db)
    vals = []
    for Hk in H:
        eigs = np.linalg.eigvalsh(Hk @ Hk.conj().T)
        eigs = np.sort(np.maximum(np.real(eigs), 0.0))[::-1]
        power = _waterfill_power_allocation(eigs, total_power=rho)
        vals.append(float(np.sum(np.log2(1.0 + power * eigs))))
    return float(np.mean(vals)) if vals else 0.0


def gain_normalized_equal_power_rate(H_f: np.ndarray, snr_db: float) -> float:
    # Family-normalized channels should be compared directly under equal-power
    # loading; per-frequency Frobenius renormalization changes the channel.
    return equal_power_rate(np.asarray(H_f, dtype=np.complex128), snr_db)

from __future__ import annotations

import numpy as np

from dualpol_rt.metrics.achievable_rate import equal_power_rate


def _snr_linear(snr_db: float) -> float:
    return float(10.0 ** (float(snr_db) / 10.0))


def _db10(value: float) -> float:
    return float(10.0 * np.log10(max(float(value), 1.0e-30)))


def total_rx_gain(H_f: np.ndarray) -> float:
    """Mean receive gain under equal-power loading across transmit ports."""
    H = np.asarray(H_f, dtype=np.complex128)
    nt = max(int(H.shape[-1]), 1)
    return float(np.mean(np.sum(np.abs(H) ** 2, axis=(1, 2))) / nt)


def per_port_rx_gain(H_f: np.ndarray) -> np.ndarray:
    """Mean receive gain per Rx port under equal-power loading."""
    H = np.asarray(H_f, dtype=np.complex128)
    nt = max(int(H.shape[-1]), 1)
    return np.mean(np.sum(np.abs(H) ** 2, axis=2), axis=0).astype(float) / nt


def port_imbalance_db(H_f: np.ndarray) -> float:
    """Signed 10*log10(P0 / P1) for the first two Rx ports."""
    gains = per_port_rx_gain(H_f)
    if gains.size < 2:
        return 0.0
    return _db10(gains[0] / max(gains[1], 1.0e-30))


def gain_to_snr_db(gain: float, snr_db: float) -> float:
    return _db10(_snr_linear(snr_db) * float(gain))


def total_snr_db(H_f: np.ndarray, snr_db: float) -> float:
    return gain_to_snr_db(total_rx_gain(H_f), snr_db)


def best_port_snr_db(H_f: np.ndarray, snr_db: float) -> float:
    gains = per_port_rx_gain(H_f)
    best_gain = float(np.max(gains)) if gains.size else 0.0
    return gain_to_snr_db(best_gain, snr_db)


def best_port_rate(H_f: np.ndarray, snr_db: float) -> float:
    gains = per_port_rx_gain(H_f)
    if not gains.size:
        return 0.0
    port_index = int(np.argmax(gains))
    H = np.asarray(H_f, dtype=np.complex128)
    return float(equal_power_rate(H[:, port_index : port_index + 1, :], snr_db))


def fixed_combiner_rate(
    H_f: np.ndarray,
    snr_db: float,
    w_r: np.ndarray,
    w_t: np.ndarray,
) -> float:
    """Rank-1 rate with fixed receive/transmit combiner vectors."""
    H = np.asarray(H_f, dtype=np.complex128)
    wr = np.asarray(w_r, dtype=np.complex128)
    wt = np.asarray(w_t, dtype=np.complex128)
    wr /= max(float(np.linalg.norm(wr)), 1.0e-30)
    wt /= max(float(np.linalg.norm(wt)), 1.0e-30)
    heff = np.einsum("r,krt,t->k", wr.conj(), H, wt)
    rho = _snr_linear(snr_db)
    return float(np.mean(np.log2(1.0 + rho * np.abs(heff) ** 2))) if heff.size else 0.0


def standard_fixed_rates(H_f: np.ndarray, snr_db_list: tuple[float, ...]) -> dict[str, dict[float, float]]:
    """Practical fixed-combiner rates aligned with the current campaign KPIs."""
    rt2 = float(np.sqrt(2.0))
    combiners = {
        "single_port": (
            np.array([1.0, 0.0], dtype=np.complex128),
            np.array([1.0, 0.0], dtype=np.complex128),
        ),
        "equal_gain": (
            np.array([1.0, 1.0], dtype=np.complex128) / rt2,
            np.array([1.0, 0.0], dtype=np.complex128),
        ),
    }
    result: dict[str, dict[float, float]] = {}
    for name, (w_r, w_t) in combiners.items():
        result[name] = {
            float(snr): fixed_combiner_rate(H_f, snr, w_r=w_r, w_t=w_t)
            for snr in snr_db_list
        }
    return result


__all__ = [
    "best_port_rate",
    "best_port_snr_db",
    "fixed_combiner_rate",
    "gain_to_snr_db",
    "per_port_rx_gain",
    "port_imbalance_db",
    "standard_fixed_rates",
    "total_rx_gain",
    "total_snr_db",
]

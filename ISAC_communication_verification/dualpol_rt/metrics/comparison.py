from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dualpol_rt.metrics.achievable_rate import equal_power_rate, gain_normalized_equal_power_rate, waterfilled_capacity
from dualpol_rt.metrics.xpr import condition_numbers, singular_values, xpr_db


@dataclass(frozen=True)
class ChannelSummary:
    equal_power_rate: dict[float, float]
    waterfilled_capacity: dict[float, float]
    gain_normalized_equal_power_rate: dict[float, float]
    xpr_db: float
    mean_condition_number: float
    mean_singular_values: np.ndarray


@dataclass(frozen=True)
class ChannelDeltaSummary:
    delta_equal_power_rate: dict[float, float]
    delta_waterfilled_capacity: dict[float, float]
    delta_gain_normalized_equal_power_rate: dict[float, float]
    delta_xpr_db: float
    delta_condition_number: float


def _resolve_gain_normalized_rate(
    H: np.ndarray,
    snr_db: float,
    *,
    gain_normalized_mode: str,
) -> float:
    mode = str(gain_normalized_mode).lower()
    if mode == "fro":
        return float(gain_normalized_equal_power_rate(H, snr_db))
    if mode == "channel_equal_power":
        return float(equal_power_rate(H, snr_db))
    raise ValueError(f"unsupported gain_normalized_mode: {gain_normalized_mode}")


def summarize_channel(
    H_f: np.ndarray,
    snr_db_list: tuple[float, ...],
    *,
    gain_normalized_mode: str = "fro",
) -> ChannelSummary:
    H = np.asarray(H_f, dtype=np.complex128)
    sv = singular_values(H)
    cond = condition_numbers(H)
    return ChannelSummary(
        equal_power_rate={float(snr): equal_power_rate(H, snr) for snr in snr_db_list},
        waterfilled_capacity={float(snr): waterfilled_capacity(H, snr) for snr in snr_db_list},
        gain_normalized_equal_power_rate={
            float(snr): _resolve_gain_normalized_rate(H, snr, gain_normalized_mode=gain_normalized_mode) for snr in snr_db_list
        },
        xpr_db=float(xpr_db(H)),
        mean_condition_number=float(np.mean(cond)),
        mean_singular_values=np.mean(sv, axis=0).astype(float),
    )


def compare_channels(
    H_lp: np.ndarray,
    H_cp: np.ndarray,
    snr_db_list: tuple[float, ...],
    *,
    gain_normalized_mode: str = "fro",
) -> ChannelDeltaSummary:
    """Return CP-minus-LP deltas for matched channel pairs."""
    lp = summarize_channel(H_lp, snr_db_list, gain_normalized_mode=gain_normalized_mode)
    cp = summarize_channel(H_cp, snr_db_list, gain_normalized_mode=gain_normalized_mode)
    return ChannelDeltaSummary(
        delta_equal_power_rate={snr: float(cp.equal_power_rate[snr] - lp.equal_power_rate[snr]) for snr in lp.equal_power_rate},
        delta_waterfilled_capacity={snr: float(cp.waterfilled_capacity[snr] - lp.waterfilled_capacity[snr]) for snr in lp.waterfilled_capacity},
        delta_gain_normalized_equal_power_rate={
            snr: float(cp.gain_normalized_equal_power_rate[snr] - lp.gain_normalized_equal_power_rate[snr]) for snr in lp.gain_normalized_equal_power_rate
        },
        delta_xpr_db=float(cp.xpr_db - lp.xpr_db),
        delta_condition_number=float(cp.mean_condition_number - lp.mean_condition_number),
    )

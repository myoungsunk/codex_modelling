from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dualpol_rt.metrics.achievable_rate import equal_power_rate, gain_normalized_equal_power_rate, waterfilled_capacity
from dualpol_rt.metrics.link_budget import (
    best_port_rate,
    mimo_multiplexing_efficiency,
    per_port_rx_gain,
    polarization_diversity_gain,
    port_imbalance_db,
    standard_fixed_rates,
    total_rx_gain,
)
from dualpol_rt.metrics.xpr import condition_numbers, singular_values, xpr_db


@dataclass(frozen=True)
class ChannelSummary:
    equal_power_rate: dict[float, float]
    waterfilled_capacity: dict[float, float]
    gain_normalized_equal_power_rate: dict[float, float]
    xpr_db: float
    mean_condition_number: float
    mean_singular_values: np.ndarray
    total_rx_gain: float
    per_port_rx_gain: np.ndarray
    port_imbalance_db: float
    best_port_rate: dict[float, float]
    fixed_rank1_rates: dict[str, dict[float, float]]
    polarization_diversity_gain: dict[float, float]
    mimo_multiplexing_efficiency: dict[float, float]


@dataclass(frozen=True)
class ChannelDeltaSummary:
    delta_equal_power_rate: dict[float, float]
    delta_waterfilled_capacity: dict[float, float]
    delta_gain_normalized_equal_power_rate: dict[float, float]
    delta_xpr_db: float
    delta_condition_number: float
    delta_total_rx_gain: float
    delta_per_port_rx_gain: np.ndarray
    delta_port_imbalance_db: float
    delta_best_port_rate: dict[float, float]
    delta_fixed_rank1_rates: dict[str, dict[float, float]]
    delta_polarization_diversity_gain: dict[float, float]
    delta_mimo_multiplexing_efficiency: dict[float, float]


def _resolve_gain_normalized_rate(
    H: np.ndarray,
    snr_db: float,
    *,
    gain_normalized_mode: str,
) -> float:
    mode = str(gain_normalized_mode).lower()
    if mode in {"fro", "channel_equal_power"}:
        return float(gain_normalized_equal_power_rate(H, snr_db))
    raise ValueError(f"unsupported gain_normalized_mode: {gain_normalized_mode}")


def summarize_channel(
    H_f: np.ndarray,
    snr_db_list: tuple[float, ...],
    *,
    gain_normalized_mode: str = "channel_equal_power",
) -> ChannelSummary:
    H = np.asarray(H_f, dtype=np.complex128)
    sv = singular_values(H)
    cond = condition_numbers(H)
    per_port = per_port_rx_gain(H)
    fixed_rates = standard_fixed_rates(H, snr_db_list)
    n_streams = max(min(int(H.shape[-2]), int(H.shape[-1])), 1)
    best_rates = {float(snr): float(best_port_rate(H, snr)) for snr in snr_db_list}
    capacities = {float(snr): waterfilled_capacity(H, snr) for snr in snr_db_list}
    return ChannelSummary(
        equal_power_rate={float(snr): equal_power_rate(H, snr) for snr in snr_db_list},
        waterfilled_capacity=capacities,
        gain_normalized_equal_power_rate={
            float(snr): _resolve_gain_normalized_rate(H, snr, gain_normalized_mode=gain_normalized_mode) for snr in snr_db_list
        },
        xpr_db=float(xpr_db(H)),
        mean_condition_number=float(np.mean(cond)),
        mean_singular_values=np.mean(sv, axis=0).astype(float),
        total_rx_gain=float(total_rx_gain(H)),
        per_port_rx_gain=per_port,
        port_imbalance_db=float(port_imbalance_db(H)),
        best_port_rate=best_rates,
        fixed_rank1_rates=fixed_rates,
        polarization_diversity_gain={
            float(snr): polarization_diversity_gain(fixed_rates["single_port"][float(snr)], fixed_rates["equal_gain"][float(snr)])
            for snr in snr_db_list
        },
        mimo_multiplexing_efficiency={
            float(snr): mimo_multiplexing_efficiency(capacities[float(snr)], best_rates[float(snr)], n_streams)
            for snr in snr_db_list
        },
    )


def compare_channels(
    H_lp: np.ndarray,
    H_cp: np.ndarray,
    snr_db_list: tuple[float, ...],
    *,
    gain_normalized_mode: str = "channel_equal_power",
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
        delta_total_rx_gain=float(cp.total_rx_gain - lp.total_rx_gain),
        delta_per_port_rx_gain=(cp.per_port_rx_gain - lp.per_port_rx_gain).astype(float),
        delta_port_imbalance_db=float(cp.port_imbalance_db - lp.port_imbalance_db),
        delta_best_port_rate={snr: float(cp.best_port_rate[snr] - lp.best_port_rate[snr]) for snr in lp.best_port_rate},
        delta_fixed_rank1_rates={
            name: {
                snr: float(cp.fixed_rank1_rates[name][snr] - lp.fixed_rank1_rates[name][snr])
                for snr in lp.fixed_rank1_rates[name]
            }
            for name in lp.fixed_rank1_rates
        },
        delta_polarization_diversity_gain={
            snr: float(cp.polarization_diversity_gain[snr] - lp.polarization_diversity_gain[snr])
            for snr in lp.polarization_diversity_gain
        },
        delta_mimo_multiplexing_efficiency={
            snr: float(cp.mimo_multiplexing_efficiency[snr] - lp.mimo_multiplexing_efficiency[snr])
            for snr in lp.mimo_multiplexing_efficiency
        },
    )

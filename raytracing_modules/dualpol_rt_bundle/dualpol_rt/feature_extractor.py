from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from dualpol_rt.channel.builder import convert_basis


@dataclass(frozen=True)
class CIRTrace:
    h_t: np.ndarray
    delays_s: np.ndarray
    freq_step_hz: float
    sample_period_s: float
    bandwidth_hz: float
    window: str


def _window_vector(name: str, n_freq: int) -> np.ndarray:
    window = str(name).lower()
    if window in {"rect", "rectangular", "boxcar", "none"}:
        return np.ones(n_freq, dtype=float)
    if window in {"hann", "hanning"}:
        return np.hanning(n_freq).astype(float)
    if window == "hamming":
        return np.hamming(n_freq).astype(float)
    raise ValueError(f"unsupported CIR window: {name}")


def _uniform_freq_step(freqs: np.ndarray) -> float:
    freq_arr = np.asarray(freqs, dtype=float)
    if freq_arr.ndim != 1 or freq_arr.size == 0:
        raise ValueError("freqs must be a non-empty 1-D array")
    if freq_arr.size == 1:
        return 0.0
    diffs = np.diff(freq_arr)
    if np.any(diffs <= 0.0):
        raise ValueError("freqs must be strictly increasing")
    if not np.allclose(diffs, diffs[0], atol=1e-9, rtol=1e-6):
        raise ValueError("freqs must be uniformly spaced for IFFT-based CIR extraction")
    return float(diffs[0])


def _coerce_cir(h_t: CIRTrace | Mapping[str, Any] | np.ndarray) -> CIRTrace:
    if isinstance(h_t, CIRTrace):
        return h_t
    if isinstance(h_t, Mapping):
        samples = np.asarray(h_t.get("h_t", h_t.get("samples")), dtype=np.complex128)
        delays = np.asarray(h_t.get("delays_s", h_t.get("time_s", np.arange(samples.shape[0], dtype=float))), dtype=float)
        if samples.shape[0] != delays.shape[0]:
            raise ValueError("CIR mapping must provide matching samples and delays")
        sample_period = float(h_t.get("sample_period_s", delays[1] - delays[0] if delays.size > 1 else 0.0))
        return CIRTrace(
            h_t=samples,
            delays_s=delays,
            freq_step_hz=float(h_t.get("freq_step_hz", 0.0)),
            sample_period_s=sample_period,
            bandwidth_hz=float(h_t.get("bandwidth_hz", 0.0)),
            window=str(h_t.get("window", "unknown")),
        )
    samples = np.asarray(h_t, dtype=np.complex128)
    delays = np.arange(samples.shape[0], dtype=float)
    return CIRTrace(
        h_t=samples,
        delays_s=delays,
        freq_step_hz=0.0,
        sample_period_s=1.0,
        bandwidth_hz=0.0,
        window="unknown",
    )


def _tap_power(h_t: CIRTrace | Mapping[str, Any] | np.ndarray) -> np.ndarray:
    trace = _coerce_cir(h_t)
    if trace.h_t.ndim == 1:
        return (np.abs(trace.h_t) ** 2).astype(float)
    axes = tuple(range(1, trace.h_t.ndim))
    return np.sum(np.abs(trace.h_t) ** 2, axis=axes).astype(float)


def _safe_db10(num: float, den: float, eps: float = 1e-15) -> float:
    return float(10.0 * np.log10((float(num) + eps) / (float(den) + eps)))


def _resolve_freqs(channel_dict: Mapping[str, Any]) -> np.ndarray:
    for key in ("freqs_hz", "freqs", "frequency_hz"):
        if key in channel_dict:
            return np.asarray(channel_dict[key], dtype=float)
    config = channel_dict.get("config")
    if config is None:
        raise ValueError("channel_dict must provide freqs_hz/freqs or a config with fc/bw/n_freq")
    if hasattr(config, "freqs_hz"):
        return np.asarray(config.freqs_hz, dtype=float)
    if isinstance(config, Mapping):
        if "freqs_hz" in config:
            return np.asarray(config["freqs_hz"], dtype=float)
        if {"fc", "bw", "n_freq"} <= set(config):
            fc = float(config["fc"])
            bw = float(config["bw"])
            n_freq = int(config["n_freq"])
            if n_freq == 1:
                return np.array([fc], dtype=float)
            return np.linspace(fc - bw / 2.0, fc + bw / 2.0, n_freq, dtype=float)
    raise ValueError("unable to resolve frequency grid from channel_dict")


def _resolve_primary_channel(channel_dict: Mapping[str, Any]) -> tuple[str, str, np.ndarray]:
    explicit_basis = str(channel_dict.get("basis", channel_dict.get("channel_basis", ""))).lower()
    candidates = (
        ("H_f", explicit_basis or "linear"),
        ("H_linear", "linear"),
        ("H_lp_raw", "linear"),
        ("H_lp_normalized", "linear"),
        ("H_circular", "circular"),
        ("H_cp_raw", "circular"),
        ("H_cp_normalized", "circular"),
        ("H", explicit_basis or "linear"),
    )
    for key, basis in candidates:
        if key in channel_dict:
            return key, basis, np.asarray(channel_dict[key], dtype=np.complex128)
    raise ValueError("channel_dict must provide one of H_f/H_linear/H_circular/H_lp_raw/H_cp_raw/H")


def ifft_to_cir(H_f: np.ndarray, freqs: np.ndarray, window: str = "hann") -> CIRTrace:
    H = np.asarray(H_f, dtype=np.complex128)
    freq_arr = np.asarray(freqs, dtype=float)
    if H.ndim == 0:
        raise ValueError("H_f must have at least one dimension")
    if H.shape[0] != freq_arr.shape[0]:
        raise ValueError("H_f and freqs must agree along the frequency axis")
    df_hz = _uniform_freq_step(freq_arr)
    taper = _window_vector(window, len(freq_arr))
    reshape = (len(freq_arr),) + (1,) * (H.ndim - 1)
    h_t = np.fft.ifft(H * taper.reshape(reshape), axis=0)
    sample_period_s = 0.0 if len(freq_arr) <= 1 or df_hz == 0.0 else 1.0 / (len(freq_arr) * df_hz)
    delays_s = np.arange(len(freq_arr), dtype=float) * sample_period_s
    bandwidth_hz = 0.0 if len(freq_arr) <= 1 else float(freq_arr[-1] - freq_arr[0])
    return CIRTrace(
        h_t=h_t.astype(np.complex128),
        delays_s=delays_s,
        freq_step_hz=float(df_hz),
        sample_period_s=float(sample_period_s),
        bandwidth_hz=bandwidth_hz,
        window=str(window),
    )


def extract_first_path(h_t: CIRTrace | Mapping[str, Any] | np.ndarray, threshold: str | float = "leading_edge") -> tuple[float, int]:
    trace = _coerce_cir(h_t)
    power = _tap_power(trace)
    if power.size == 0 or float(np.max(power)) <= 0.0:
        return float("nan"), -1
    peak_idx = int(np.argmax(power))
    peak_power = float(power[peak_idx])
    if isinstance(threshold, str):
        mode = threshold.lower()
        if mode == "peak":
            idx = peak_idx
        elif mode == "leading_edge":
            warmup = power[: max(1, min(len(power), max(len(power) // 8, 1)))]
            noise_floor = float(np.median(warmup))
            threshold_power = max(0.1 * peak_power, 5.0 * noise_floor)
            hits = np.flatnonzero(power >= threshold_power)
            idx = int(hits[0]) if hits.size else peak_idx
        else:
            raise ValueError(f"unsupported first-path threshold mode: {threshold}")
    else:
        threshold_power = peak_power * float(threshold) if 0.0 < float(threshold) <= 1.0 else float(threshold)
        hits = np.flatnonzero(power >= threshold_power)
        idx = int(hits[0]) if hits.size else peak_idx
    return float(trace.delays_s[idx]), idx


def compute_gamma_cp_variants(
    H_f_RHCP: np.ndarray,
    H_f_LHCP: np.ndarray,
    H_f_cross_rl: np.ndarray | None = None,
    H_f_cross_lr: np.ndarray | None = None,
    eps: float = 1e-15,
) -> dict[str, float]:
    h_rhcp = np.asarray(H_f_RHCP, dtype=np.complex128).reshape(-1)
    h_lhcp = np.asarray(H_f_LHCP, dtype=np.complex128).reshape(-1)
    if h_rhcp.shape != h_lhcp.shape:
        raise ValueError("RHCP and LHCP responses must have matching shapes")
    power_rhcp = float(np.mean(np.abs(h_rhcp) ** 2))
    power_lhcp = float(np.mean(np.abs(h_lhcp) ** 2))
    corr = np.mean(h_rhcp * np.conj(h_lhcp))
    out = {
        "gamma_cp_rhcp_power": power_rhcp,
        "gamma_cp_lhcp_power": power_lhcp,
        "gamma_cp_rhcp_to_lhcp_db": _safe_db10(power_rhcp, power_lhcp, eps=eps),
        "gamma_cp_lhcp_to_rhcp_db": _safe_db10(power_lhcp, power_rhcp, eps=eps),
        "gamma_cp_imbalance_abs_db": abs(_safe_db10(power_rhcp, power_lhcp, eps=eps)),
        "gamma_cp_complex_correlation_mag": float(abs(corr) / np.sqrt((power_rhcp + eps) * (power_lhcp + eps))),
        "gamma_cp_complex_correlation_phase_rad": float(np.angle(corr)),
    }
    cross_terms = [term for term in (H_f_cross_rl, H_f_cross_lr) if term is not None]
    if cross_terms:
        cross_power = float(sum(np.mean(np.abs(np.asarray(term, dtype=np.complex128).reshape(-1)) ** 2) for term in cross_terms))
        out["gamma_cp_cross_power"] = cross_power
        out["gamma_cp_copol_power"] = power_rhcp + power_lhcp
        out["gamma_cp_copol_to_cross_db"] = _safe_db10(power_rhcp + power_lhcp, cross_power, eps=eps)
    return out


def compute_a_fp_variants(
    h_t: CIRTrace | Mapping[str, Any] | np.ndarray,
    idx_FP: int,
    eps: float = 1e-15,
) -> dict[str, float]:
    trace = _coerce_cir(h_t)
    power = _tap_power(trace)
    if idx_FP < 0 or idx_FP >= power.shape[0]:
        return {
            "a_fp_total_power": float("nan"),
            "a_fp_total_power_db": float("nan"),
            "a_fp_total_magnitude": float("nan"),
            "a_fp_energy_fraction": float("nan"),
            "a_fp_to_peak_ratio": float("nan"),
        }
    tap = np.asarray(trace.h_t[idx_FP], dtype=np.complex128)
    tap_power = np.abs(tap) ** 2
    total_power = float(np.sum(tap_power))
    total_energy = float(np.sum(power))
    peak_power = float(np.max(power)) if power.size else total_power
    out = {
        "a_fp_total_power": total_power,
        "a_fp_total_power_db": float(10.0 * np.log10(total_power + eps)),
        "a_fp_total_magnitude": float(np.sqrt(total_power)),
        "a_fp_energy_fraction": float(total_power / max(total_energy, eps)),
        "a_fp_to_peak_ratio": float(total_power / max(peak_power, eps)),
    }
    if tap.ndim == 2 and tap.shape == (2, 2):
        co_power = float(tap_power[0, 0] + tap_power[1, 1])
        cross_power = float(tap_power[0, 1] + tap_power[1, 0])
        out["a_fp_co_power"] = co_power
        out["a_fp_cross_power"] = cross_power
        out["a_fp_xpr_db"] = _safe_db10(co_power, cross_power, eps=eps)
    return out


def compute_cir_baseline_features(
    h_t: CIRTrace | Mapping[str, Any] | np.ndarray,
    eps: float = 1e-15,
) -> dict[str, float]:
    trace = _coerce_cir(h_t)
    power = _tap_power(trace)
    total_energy = float(np.sum(power))
    peak_idx = int(np.argmax(power)) if power.size else -1
    peak_power = float(np.max(power)) if power.size else 0.0
    if total_energy <= 0.0 or power.size == 0:
        return {
            "cir_total_energy": 0.0,
            "cir_peak_power": 0.0,
            "cir_peak_delay_s": float("nan"),
            "cir_peak_to_average_db": float("nan"),
            "cir_mean_delay_s": float("nan"),
            "cir_rms_delay_spread_s": float("nan"),
            "cir_delay_p50_s": float("nan"),
            "cir_delay_p95_s": float("nan"),
            "cir_effective_tap_count": 0.0,
            "cir_energy_entropy": 0.0,
        }
    weights = power / total_energy
    delays = np.asarray(trace.delays_s, dtype=float)
    mean_delay = float(np.sum(weights * delays))
    rms_delay = float(np.sqrt(max(np.sum(weights * (delays - mean_delay) ** 2), 0.0)))
    cdf = np.cumsum(weights)
    p50_idx = int(np.searchsorted(cdf, 0.5, side="left"))
    p95_idx = int(np.searchsorted(cdf, 0.95, side="left"))
    entropy = -float(np.sum(weights * np.log2(np.maximum(weights, eps))))
    return {
        "cir_total_energy": total_energy,
        "cir_peak_power": peak_power,
        "cir_peak_delay_s": float(delays[peak_idx]),
        "cir_peak_to_average_db": _safe_db10(peak_power, float(np.mean(power)), eps=eps),
        "cir_mean_delay_s": mean_delay,
        "cir_rms_delay_spread_s": rms_delay,
        "cir_delay_p50_s": float(delays[min(p50_idx, len(delays) - 1)]),
        "cir_delay_p95_s": float(delays[min(p95_idx, len(delays) - 1)]),
        "cir_effective_tap_count": float(1.0 / np.sum(weights**2)),
        "cir_energy_entropy": entropy,
    }


def extract_all_features(channel_dict: Mapping[str, Any]) -> dict[str, float | int | str]:
    if not isinstance(channel_dict, Mapping):
        raise TypeError("channel_dict must be a mapping")
    response_key, basis, H_main = _resolve_primary_channel(channel_dict)
    freqs = _resolve_freqs(channel_dict)
    window = str(channel_dict.get("cir_window", "hann"))
    cir = ifft_to_cir(H_main, freqs, window=window)
    t_fp, idx_fp = extract_first_path(cir, threshold=channel_dict.get("first_path_threshold", "leading_edge"))
    features: dict[str, float | int | str] = {
        "feature_source_key": response_key,
        "channel_basis": basis,
        "cir_window": cir.window,
        "cir_freq_step_hz": cir.freq_step_hz,
        "cir_sample_period_s": cir.sample_period_s,
        "cir_bandwidth_hz": cir.bandwidth_hz,
        "first_path_delay_s": t_fp,
        "first_path_index": idx_fp,
    }
    features.update(compute_a_fp_variants(cir, idx_fp))
    features.update(compute_cir_baseline_features(cir))

    H_circular = None
    if "H_circular" in channel_dict:
        H_circular = np.asarray(channel_dict["H_circular"], dtype=np.complex128)
    elif "H_cp_raw" in channel_dict:
        H_circular = np.asarray(channel_dict["H_cp_raw"], dtype=np.complex128)
    elif "H_cp_normalized" in channel_dict:
        H_circular = np.asarray(channel_dict["H_cp_normalized"], dtype=np.complex128)
    elif basis == "circular" and H_main.ndim == 3 and H_main.shape[1:] == (2, 2):
        H_circular = H_main
    elif basis == "linear" and H_main.ndim == 3 and H_main.shape[1:] == (2, 2):
        H_circular = convert_basis(H_main, src="linear", dst="circular")

    if H_circular is not None and H_circular.ndim == 3 and H_circular.shape[1:] == (2, 2):
        features.update(
            compute_gamma_cp_variants(
                H_circular[:, 0, 0],
                H_circular[:, 1, 1],
                H_f_cross_rl=H_circular[:, 0, 1],
                H_f_cross_lr=H_circular[:, 1, 0],
            )
        )
    return features


__all__ = [
    "CIRTrace",
    "ifft_to_cir",
    "extract_first_path",
    "compute_gamma_cp_variants",
    "compute_a_fp_variants",
    "compute_cir_baseline_features",
    "extract_all_features",
]

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dualpol_rt.channel.path_record import PathRecord
from dualpol_rt.em.jones import attach_em_response


@dataclass(frozen=True)
class PathRichnessMetrics:
    path_count: int
    n_eff: float
    mean_delay_s: float
    tau_rms_s: float
    aoa_azimuth_spread_deg: float
    aoa_elevation_spread_deg: float
    aod_azimuth_spread_deg: float
    aod_elevation_spread_deg: float
    mean_aoa_spread_deg: float
    dpr_db: float
    pmi_env: float


def _ensure_enriched_paths(paths: list[PathRecord], freqs_hz: np.ndarray) -> list[PathRecord]:
    freqs = np.asarray(freqs_hz, dtype=float)
    out: list[PathRecord] = []
    for path in paths:
        out.append(path if path.jones_f is not None and path.scalar_factor_f is not None else attach_em_response(path, freqs_hz=freqs))
    return out


def _path_powers(paths: list[PathRecord]) -> np.ndarray:
    powers = np.zeros(len(paths), dtype=float)
    for idx, path in enumerate(paths):
        assert path.jones_f is not None
        assert path.scalar_factor_f is not None
        core = path.jones_f * path.scalar_factor_f[:, None, None]
        powers[idx] = float(np.mean(np.sum(np.abs(core) ** 2, axis=(1, 2))))
    return powers


def _weights_from_powers(powers: np.ndarray) -> np.ndarray:
    total = float(np.sum(powers))
    if total <= 0.0:
        return np.full(powers.shape, 1.0 / max(len(powers), 1), dtype=float)
    return (powers / total).astype(float)


def _direction_azimuth_elevation(direction: np.ndarray) -> tuple[float, float]:
    vec = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return 0.0, 0.0
    azimuth = float(np.arctan2(vec[1], vec[0]))
    elevation = float(np.arctan2(vec[2], np.linalg.norm(vec[:2])))
    return azimuth, elevation


def _weighted_circular_spread_deg(angles_rad: np.ndarray, weights: np.ndarray) -> float:
    phases = np.exp(1j * np.asarray(angles_rad, dtype=float))
    resultant = np.abs(np.sum(np.asarray(weights, dtype=float) * phases))
    resultant = float(np.clip(resultant, 1e-12, 1.0))
    spread_rad = float(np.sqrt(max(-2.0 * np.log(resultant), 0.0)))
    return float(np.rad2deg(spread_rad))


def compute_path_richness(paths: list[PathRecord], freqs_hz: np.ndarray) -> PathRichnessMetrics:
    if not paths:
        return PathRichnessMetrics(
            path_count=0,
            n_eff=0.0,
            mean_delay_s=0.0,
            tau_rms_s=0.0,
            aoa_azimuth_spread_deg=0.0,
            aoa_elevation_spread_deg=0.0,
            aod_azimuth_spread_deg=0.0,
            aod_elevation_spread_deg=0.0,
            mean_aoa_spread_deg=0.0,
            dpr_db=40.0,
            pmi_env=0.0,
        )

    enriched = _ensure_enriched_paths(paths, freqs_hz=freqs_hz)
    powers = _path_powers(enriched)
    weights = _weights_from_powers(powers)
    delays = np.asarray([path.delay_s for path in enriched], dtype=float)
    mean_delay = float(np.sum(weights * delays))
    tau_rms = float(np.sqrt(max(np.sum(weights * (delays - mean_delay) ** 2), 0.0)))
    n_eff = float(1.0 / np.sum(np.square(weights)))

    aoa_angles = np.asarray([_direction_azimuth_elevation(path.aoa_dir) for path in enriched], dtype=float)
    aod_angles = np.asarray([_direction_azimuth_elevation(path.aod_dir) for path in enriched], dtype=float)

    p_max = float(np.max(powers))
    rest = float(np.sum(powers) - p_max)
    if rest <= 1e-15:
        dpr_db = 40.0
    else:
        dpr_db = float(min(40.0, 10.0 * np.log10(max(p_max / rest, 1e-15))))

    numer = 0.0
    denom = 0.0
    for weight, path in zip(weights, enriched):
        assert path.jones_f is not None
        jbar = np.mean(path.jones_f, axis=0)
        offdiag = jbar.copy()
        offdiag[0, 0] = 0.0
        offdiag[1, 1] = 0.0
        numer += float(weight * np.sum(np.abs(offdiag) ** 2))
        denom += float(weight * np.sum(np.abs(jbar) ** 2))
    pmi_env = 0.0 if denom <= 0.0 else float(numer / denom)

    aoa_az_spread = _weighted_circular_spread_deg(aoa_angles[:, 0], weights)
    aoa_el_spread = _weighted_circular_spread_deg(aoa_angles[:, 1], weights)
    aod_az_spread = _weighted_circular_spread_deg(aod_angles[:, 0], weights)
    aod_el_spread = _weighted_circular_spread_deg(aod_angles[:, 1], weights)

    return PathRichnessMetrics(
        path_count=len(enriched),
        n_eff=n_eff,
        mean_delay_s=mean_delay,
        tau_rms_s=tau_rms,
        aoa_azimuth_spread_deg=aoa_az_spread,
        aoa_elevation_spread_deg=aoa_el_spread,
        aod_azimuth_spread_deg=aod_az_spread,
        aod_elevation_spread_deg=aod_el_spread,
        mean_aoa_spread_deg=0.5 * (aoa_az_spread + aoa_el_spread),
        dpr_db=dpr_db,
        pmi_env=pmi_env,
    )


def compute_mrs_scores(
    metrics_list: list[PathRichnessMetrics],
    reference_metrics_list: list[PathRichnessMetrics] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not metrics_list:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    reference = metrics_list if reference_metrics_list is None or len(reference_metrics_list) == 0 else reference_metrics_list

    def _extract(metrics: list[PathRichnessMetrics]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        log_tau = np.log10(np.maximum([metric.tau_rms_s for metric in metrics], 1e-12))
        log_neff = np.log10(np.maximum([metric.n_eff for metric in metrics], 1e-12))
        aoa = np.asarray([metric.mean_aoa_spread_deg for metric in metrics], dtype=float)
        dpr = np.asarray([metric.dpr_db for metric in metrics], dtype=float)
        pmi = np.asarray([metric.pmi_env for metric in metrics], dtype=float)
        return log_tau, log_neff, aoa, dpr, pmi

    ref_log_tau, ref_log_neff, ref_aoa, ref_dpr, ref_pmi = _extract(reference)
    log_tau, log_neff, aoa, dpr, pmi = _extract(metrics_list)

    def _zscore(values: np.ndarray, ref_values: np.ndarray) -> np.ndarray:
        mean = float(np.mean(ref_values))
        std = float(np.std(ref_values))
        if std <= 1e-12:
            return np.zeros(values.shape, dtype=float)
        return ((values - mean) / std).astype(float)

    raw_ref = (_zscore(ref_log_tau, ref_log_tau) + _zscore(ref_log_neff, ref_log_neff) + _zscore(ref_aoa, ref_aoa) - _zscore(ref_dpr, ref_dpr) + _zscore(ref_pmi, ref_pmi)) / 5.0
    raw = (_zscore(log_tau, ref_log_tau) + _zscore(log_neff, ref_log_neff) + _zscore(aoa, ref_aoa) - _zscore(dpr, ref_dpr) + _zscore(pmi, ref_pmi)) / 5.0
    raw = np.asarray(raw, dtype=float)
    low = float(np.min(raw_ref))
    high = float(np.max(raw_ref))
    if high - low <= 1e-12:
        scaled = np.full(raw.shape, 50.0, dtype=float)
    else:
        scaled = 100.0 * (raw - low) / (high - low)
    return raw, scaled.astype(float)


def label_mrs_scores(scores: np.ndarray, low_threshold: float | None = None, high_threshold: float | None = None) -> tuple[list[str], float, float]:
    score_arr = np.asarray(scores, dtype=float)
    if score_arr.size == 0:
        return [], 25.0 if low_threshold is None else float(low_threshold), 75.0 if high_threshold is None else float(high_threshold)
    low = float(np.percentile(score_arr, 25.0) if low_threshold is None else low_threshold)
    high = float(np.percentile(score_arr, 75.0) if high_threshold is None else high_threshold)
    labels: list[str] = []
    for value in score_arr:
        if value <= low:
            labels.append("mild")
        elif value >= high:
            labels.append("rich")
        else:
            labels.append("moderate")
    return labels, low, high


__all__ = [
    "PathRichnessMetrics",
    "compute_path_richness",
    "compute_mrs_scores",
    "label_mrs_scores",
]

"""Power-domain dual-CP metrics for measurement and RT bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from analysis.ctf_cir import convert_basis, ctf_to_cir, synthesize_ctf_with_source
from analysis.dualcp_calibration import apply_floor_excess


EPS = 1e-15


@dataclass(frozen=True)
class DualCPMetricParams:
    nfft: int = 2048
    window: str = "hann"
    early_window_ns: float = 3.0
    early_window_sensitivity_ns: tuple[float, ...] = ()
    tmax_ns: float = 30.0
    noise_tail_ns: float = 8.0
    threshold_db: float = 6.0
    detect_power: str = "total"  # "total" or "co"
    delay_spread_source: str = "total"  # "total" or "co"
    power_floor: float = 1e-18
    eval_basis: str = "circular"
    convention: str = "IEEE-RHCP"


def _to_params(params: dict[str, Any] | DualCPMetricParams | None) -> DualCPMetricParams:
    if params is None:
        return DualCPMetricParams()
    if isinstance(params, DualCPMetricParams):
        return params
    return DualCPMetricParams(
        nfft=int(params.get("nfft", 2048)),
        window=str(params.get("window", "hann")),
        early_window_ns=float(params.get("early_window_ns", 3.0)),
        early_window_sensitivity_ns=tuple(float(x) for x in params.get("early_window_sensitivity_ns", [])),
        tmax_ns=float(params.get("tmax_ns", 30.0)),
        noise_tail_ns=float(params.get("noise_tail_ns", 8.0)),
        threshold_db=float(params.get("threshold_db", 6.0)),
        detect_power=str(params.get("detect_power", "total")),
        delay_spread_source=str(params.get("delay_spread_source", "total")),
        power_floor=float(params.get("power_floor", 1e-18)),
        eval_basis=str(params.get("eval_basis", "circular")),
        convention=str(params.get("convention", "IEEE-RHCP")),
    )


def _dualcp_tap_powers(h_tau: np.ndarray, power_floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.abs(np.asarray(h_tau, dtype=np.complex128)) ** 2
    p_co = p[:, 0, 0].astype(float)
    p_cross = np.maximum(p[:, 1, 0].astype(float), float(power_floor))
    p_total = (p_co + p_cross).astype(float)
    return p_co, p_cross, p_total


def _estimate_tau0(
    tau_s: np.ndarray,
    p_detect: np.ndarray,
    noise_tail_ns: float,
    threshold_db: float,
) -> dict[str, Any]:
    tau = np.asarray(tau_s, dtype=float)
    p = np.asarray(p_detect, dtype=float)
    if len(tau) == 0:
        return {
            "idx": 0,
            "tau0_s": 0.0,
            "noise_floor_linear": 0.0,
            "threshold_linear": 0.0,
            "detected": False,
        }
    noise_tail_s = max(float(noise_tail_ns), 0.0) * 1e-9
    tail_start = float(max(0.0, tau[-1] - noise_tail_s))
    tail_mask = tau >= tail_start
    if not np.any(tail_mask):
        tail_mask = np.ones_like(tau, dtype=bool)
    noise_floor = float(np.median(p[tail_mask])) if np.any(tail_mask) else float(np.median(p))
    threshold = float(max(noise_floor * (10.0 ** (float(threshold_db) / 10.0)), EPS))
    hit = np.where(p > threshold)[0]
    if len(hit):
        idx = int(hit[0])
        detected = True
    else:
        idx = int(np.argmax(p))
        detected = False
    return {
        "idx": idx,
        "tau0_s": float(tau[idx]),
        "noise_floor_linear": noise_floor,
        "threshold_linear": threshold,
        "detected": detected,
    }


def _ratio_db(num: float, den: float, power_floor: float) -> float:
    return float(10.0 * np.log10((float(num) + EPS) / (max(float(den), float(power_floor)) + EPS)))


def _window_masks(tau_s: np.ndarray, tau0_s: float, early_window_ns: float, tmax_ns: float) -> tuple[np.ndarray, np.ndarray]:
    tau = np.asarray(tau_s, dtype=float)
    te = max(float(early_window_ns), 0.0) * 1e-9
    tmax = max(float(tmax_ns), te) * 1e-9
    early = (tau >= tau0_s) & (tau < (tau0_s + te))
    late = (tau >= (tau0_s + te)) & (tau <= (tau0_s + tmax))
    return early, late


def _delay_spread_tau_rms_ns(
    tau_s: np.ndarray,
    p_weight: np.ndarray,
    mask: np.ndarray,
) -> float:
    t = np.asarray(tau_s, dtype=float)[mask]
    w = np.asarray(p_weight, dtype=float)[mask]
    if len(t) == 0:
        return float("nan")
    s = float(np.sum(w))
    if s <= 0.0:
        return float("nan")
    mu = float(np.sum(t * w) / s)
    var = float(np.sum(((t - mu) ** 2) * w) / s)
    return float(np.sqrt(max(var, 0.0)) * 1e9)


def _resolve_floor_band(
    calibration_floor: dict[str, Any] | None,
    f_hz: np.ndarray,
) -> tuple[float | None, float | None]:
    if not calibration_floor:
        return None, None
    floor = calibration_floor.get("xpd_floor_db")
    if floor is None:
        return None, None
    floor_arr = np.asarray(floor, dtype=float)
    if floor_arr.ndim == 0:
        floor_band = float(floor_arr)
    else:
        f_ref = np.asarray(calibration_floor.get("frequency_hz", []), dtype=float)
        if len(f_ref) == len(floor_arr) and len(f_ref) > 1:
            f = np.asarray(f_hz, dtype=float)
            floor_i = np.interp(f, f_ref, floor_arr, left=float(floor_arr[0]), right=float(floor_arr[-1]))
            floor_band = float(np.median(floor_i))
        else:
            floor_band = float(np.median(floor_arr))
    uncert = calibration_floor.get("xpd_floor_uncert_db")
    if uncert is None:
        return floor_band, None
    u_arr = np.asarray(uncert, dtype=float)
    if u_arr.ndim == 0:
        u_band = float(u_arr)
    else:
        u_band = float(np.median(u_arr))
    return floor_band, u_band


def _compute_window_metrics(
    tau_s: np.ndarray,
    p_co: np.ndarray,
    p_cross: np.ndarray,
    p_total: np.ndarray,
    tau0_s: float,
    early_window_ns: float,
    tmax_ns: float,
    power_floor: float,
) -> dict[str, Any]:
    early_mask, late_mask = _window_masks(tau_s, tau0_s=tau0_s, early_window_ns=early_window_ns, tmax_ns=tmax_ns)
    if not np.any(early_mask):
        j = int(np.argmin(np.abs(np.asarray(tau_s, dtype=float) - float(tau0_s))))
        early_mask[j] = True

    e_co_early = float(np.sum(np.asarray(p_co, dtype=float)[early_mask]))
    e_cross_early = float(np.sum(np.asarray(p_cross, dtype=float)[early_mask]))
    e_co_late = float(np.sum(np.asarray(p_co, dtype=float)[late_mask]))
    e_cross_late = float(np.sum(np.asarray(p_cross, dtype=float)[late_mask]))
    e_total = float(np.sum(np.asarray(p_total, dtype=float)[early_mask | late_mask]))
    e_early_total = float(np.sum(np.asarray(p_total, dtype=float)[early_mask]))

    xpd_early = _ratio_db(e_co_early, e_cross_early, power_floor)
    xpd_late = _ratio_db(e_co_late, e_cross_late, power_floor)
    rho_lin = float((e_cross_early + EPS) / (e_co_early + EPS))
    rho_db = float(10.0 * np.log10(rho_lin + EPS))
    return {
        "early_window_ns": float(early_window_ns),
        "tmax_ns": float(tmax_ns),
        "early_bin_count": int(np.sum(early_mask)),
        "late_bin_count": int(np.sum(late_mask)),
        "energy_co_early": e_co_early,
        "energy_cross_early": e_cross_early,
        "energy_co_late": e_co_late,
        "energy_cross_late": e_cross_late,
        "xpd_early_db": xpd_early,
        "xpd_late_db": xpd_late,
        "l_pol_db": float(xpd_early - xpd_late),
        "rho_early_linear": rho_lin,
        "rho_early_db": rho_db,
        "early_energy_concentration": float((e_early_total + EPS) / (e_total + EPS)),
        "window_energy_total": e_total,
    }


def compute_dualcp_metrics_from_Hf(
    H_f: np.ndarray,
    f_hz: np.ndarray,
    params: dict[str, Any] | DualCPMetricParams | None = None,
    calibration_floor: dict[str, Any] | None = None,
    ground_truth_delay_s: float | None = None,
) -> dict[str, Any]:
    p = _to_params(params)
    freq = np.asarray(f_hz, dtype=float)
    h_f = np.asarray(H_f, dtype=np.complex128)
    if h_f.shape != (len(freq), 2, 2):
        raise ValueError(f"H_f shape must be {(len(freq), 2, 2)}")

    h_tau, tau_s = ctf_to_cir(h_f, freq, nfft=int(p.nfft), window=str(p.window))  # type: ignore[arg-type]
    p_co, p_cross, p_total = _dualcp_tap_powers(h_tau, power_floor=float(p.power_floor))
    p_detect = p_total if str(p.detect_power).lower() == "total" else p_co
    det = _estimate_tau0(
        tau_s=tau_s,
        p_detect=p_detect,
        noise_tail_ns=float(p.noise_tail_ns),
        threshold_db=float(p.threshold_db),
    )

    main = _compute_window_metrics(
        tau_s=tau_s,
        p_co=p_co,
        p_cross=p_cross,
        p_total=p_total,
        tau0_s=float(det["tau0_s"]),
        early_window_ns=float(p.early_window_ns),
        tmax_ns=float(p.tmax_ns),
        power_floor=float(p.power_floor),
    )
    delay_weight = p_total if str(p.delay_spread_source).lower() == "total" else p_co
    span_mask = (tau_s >= float(det["tau0_s"])) & (tau_s <= float(det["tau0_s"]) + max(float(p.tmax_ns), 0.0) * 1e-9)
    tau_rms_ns = _delay_spread_tau_rms_ns(tau_s, delay_weight, span_mask)

    out: dict[str, Any] = {
        **main,
        "tau0_s": float(det["tau0_s"]),
        "tau0_ns": float(det["tau0_s"]) * 1e9,
        "tau_rms_ns": float(tau_rms_ns),
        "toa_estimate_ns": float(det["tau0_s"]) * 1e9,
        "noise_floor_linear": float(det["noise_floor_linear"]),
        "threshold_linear": float(det["threshold_linear"]),
        "tau_resolution_ns": float((tau_s[1] - tau_s[0]) * 1e9) if len(tau_s) > 1 else 0.0,
        "detected_from_threshold": bool(det["detected"]),
        "detect_power": str(p.detect_power),
        "delay_spread_source": str(p.delay_spread_source),
        "nfft": int(p.nfft),
        "window": str(p.window),
        "noise_tail_ns": float(p.noise_tail_ns),
        "threshold_db": float(p.threshold_db),
        "power_floor": float(p.power_floor),
        "basis": str(p.eval_basis),
        "convention": str(p.convention),
    }
    if ground_truth_delay_s is not None:
        out["toa_bias_ns"] = float((float(det["tau0_s"]) - float(ground_truth_delay_s)) * 1e9)

    sens_vals = [float(x) for x in p.early_window_sensitivity_ns if float(x) > 0.0]
    if sens_vals:
        out["sensitivity"] = {
            f"Te_{te:.3f}ns": _compute_window_metrics(
                tau_s=tau_s,
                p_co=p_co,
                p_cross=p_cross,
                p_total=p_total,
                tau0_s=float(det["tau0_s"]),
                early_window_ns=te,
                tmax_ns=float(p.tmax_ns),
                power_floor=float(p.power_floor),
            )
            for te in sens_vals
        }

    floor_db, floor_uncert = _resolve_floor_band(calibration_floor, freq)
    if floor_db is not None:
        xpd_early_excess = float(apply_floor_excess(np.asarray([out["xpd_early_db"]], dtype=float), floor_db)[0])
        xpd_late_excess = float(apply_floor_excess(np.asarray([out["xpd_late_db"]], dtype=float), floor_db)[0])
        out["xpd_floor_db"] = float(floor_db)
        if floor_uncert is not None:
            out["xpd_floor_uncert_db"] = float(floor_uncert)
        out["xpd_early_excess_db"] = xpd_early_excess
        out["xpd_late_excess_db"] = xpd_late_excess
        if floor_uncert is not None:
            u = abs(float(floor_uncert))
            out["claim_caution_early"] = bool(abs(xpd_early_excess) <= u)
            out["claim_caution_late"] = bool(abs(xpd_late_excess) <= u)
        else:
            out["claim_caution_early"] = False
            out["claim_caution_late"] = False

    return out


def compute_dualcp_metrics_from_rt_paths(
    paths: list[dict[str, Any]],
    f_hz: np.ndarray,
    params: dict[str, Any] | DualCPMetricParams | None = None,
    matrix_source: str = "A",
    input_basis: str | None = None,
    eval_basis: str = "circular",
    convention: str = "IEEE-RHCP",
    calibration_floor: dict[str, Any] | None = None,
    ground_truth_delay_s: float | None = None,
) -> dict[str, Any]:
    freq = np.asarray(f_hz, dtype=float)
    H_f = synthesize_ctf_with_source(paths, f_hz=freq, matrix_source=str(matrix_source))
    src_b = str(input_basis).lower() if input_basis is not None else None
    dst_b = str(eval_basis).lower()
    if src_b in {"linear", "circular"} and dst_b in {"linear", "circular"} and src_b != dst_b:
        H_f = convert_basis(H_f, src=src_b, dst=dst_b, convention=str(convention))

    p = _to_params(params)
    merged = DualCPMetricParams(
        nfft=p.nfft,
        window=p.window,
        early_window_ns=p.early_window_ns,
        early_window_sensitivity_ns=p.early_window_sensitivity_ns,
        tmax_ns=p.tmax_ns,
        noise_tail_ns=p.noise_tail_ns,
        threshold_db=p.threshold_db,
        detect_power=p.detect_power,
        delay_spread_source=p.delay_spread_source,
        power_floor=p.power_floor,
        eval_basis=str(eval_basis),
        convention=str(convention),
    )
    out = compute_dualcp_metrics_from_Hf(
        H_f=H_f,
        f_hz=freq,
        params=merged,
        calibration_floor=calibration_floor,
        ground_truth_delay_s=ground_truth_delay_s,
    )
    out["matrix_source"] = str(matrix_source)
    out["input_basis"] = str(input_basis) if input_basis is not None else ""
    return out

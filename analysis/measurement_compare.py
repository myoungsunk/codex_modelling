"""Measurement bridge: import measured 2x2 S21 and compare with RT/Synth.

Example:
    >>> import numpy as np
    >>> from analysis.measurement_compare import MeasurementData, compare_measured_to_dataset
    >>> f = np.linspace(6e9, 6.1e9, 8)
    >>> H = np.zeros((len(f), 2, 2), dtype=np.complex128)
    >>> H[:, 0, 0] = 1.0
    >>> H[:, 1, 1] = 1.0
    >>> ds = {"meta": {"basis": "linear", "convention": "IEEE-RHCP"}, "frequency": f, "scenarios": {"S": {"cases": {"0": {"params": {}, "paths": []}}}}}
    >>> _ = compare_measured_to_dataset(ds, MeasurementData(frequency_hz=f, H_f=H), create_plots=False)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from analysis.ctf_cir import convert_basis, ctf_to_cir, pdp, synthesize_ctf_with_source


EPS = 1e-15


@dataclass(frozen=True)
class MeasurementData:
    frequency_hz: NDArray[np.float64]
    H_f: NDArray[np.complex128]
    source: str = "measurement"
    meta: dict[str, Any] = field(default_factory=dict)


def _norm_key(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "_")


def _pick_float(row: dict[str, str], keymap: dict[str, str], keys: list[str]) -> float | None:
    for k in keys:
        kk = _norm_key(k)
        if kk in keymap:
            raw = row.get(keymap[kk], "")
            if raw is None or str(raw).strip() == "":
                continue
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def _pick_complex_from_row(row: dict[str, str], keymap: dict[str, str], prefixes: list[str]) -> complex:
    for pfx in prefixes:
        p = _norm_key(pfx)
        re = _pick_float(row, keymap, [f"{p}_re", f"{p}_real"])
        im = _pick_float(row, keymap, [f"{p}_im", f"{p}_imag", f"{p}_imaginary"])
        if re is not None and im is not None:
            return complex(re, im)
        mag_db = _pick_float(row, keymap, [f"{p}_mag_db", f"{p}_db", f"{p}_magnitude_db"])
        ph_deg = _pick_float(row, keymap, [f"{p}_phase_deg", f"{p}_deg"])
        if mag_db is not None and ph_deg is not None:
            mag = 10.0 ** (mag_db / 20.0)
            ph = np.deg2rad(ph_deg)
            return complex(mag * np.cos(ph), mag * np.sin(ph))
    raise ValueError(f"missing complex columns for any of prefixes={prefixes}")


def _read_rows(path: str | Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {p}")
        rows = list(reader)
    keymap = {_norm_key(k): k for k in (reader.fieldnames or [])}
    return rows, keymap


def _parse_trace_csv(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    rows, keymap = _read_rows(path)
    fk = [k for k in ["f_hz", "freq_hz", "frequency_hz", "f"] if _norm_key(k) in keymap]
    if not fk:
        raise ValueError(f"missing frequency column in {path}")
    freq: list[float] = []
    vals: list[complex] = []
    for r in rows:
        f = _pick_float(r, keymap, fk)
        if f is None:
            continue
        z = _pick_complex_from_row(r, keymap, prefixes=["s21", "h"])
        freq.append(float(f))
        vals.append(complex(z))
    if not freq:
        raise ValueError(f"no valid rows in {path}")
    return np.asarray(freq, dtype=float), np.asarray(vals, dtype=np.complex128)


def load_measurement_matrix_csv(path: str | Path) -> MeasurementData:
    """Load 2x2 measurement from one CSV.

    Required columns:
      frequency: one of f_hz/freq_hz/frequency_hz/f
      complex entries for HH/HV/VH/VV (or H11/H12/H21/H22)
      each entry can be (re,im) or (mag_db,phase_deg).
    """

    rows, keymap = _read_rows(path)
    fk = [k for k in ["f_hz", "freq_hz", "frequency_hz", "f"] if _norm_key(k) in keymap]
    if not fk:
        raise ValueError(f"missing frequency column in {path}")

    freq: list[float] = []
    mats: list[np.ndarray] = []
    for r in rows:
        f = _pick_float(r, keymap, fk)
        if f is None:
            continue
        hh = _pick_complex_from_row(r, keymap, prefixes=["hh", "h11"])
        hv = _pick_complex_from_row(r, keymap, prefixes=["hv", "h12"])
        vh = _pick_complex_from_row(r, keymap, prefixes=["vh", "h21"])
        vv = _pick_complex_from_row(r, keymap, prefixes=["vv", "h22"])
        m = np.asarray([[hh, hv], [vh, vv]], dtype=np.complex128)
        freq.append(float(f))
        mats.append(m)
    if not freq:
        raise ValueError(f"no valid rows in {path}")
    return MeasurementData(
        frequency_hz=np.asarray(freq, dtype=float),
        H_f=np.asarray(mats, dtype=np.complex128),
        source=str(path),
    )


def _interp_complex(freq_src: NDArray[np.float64], z_src: NDArray[np.complex128], freq_dst: NDArray[np.float64]) -> NDArray[np.complex128]:
    xr = np.interp(freq_dst, freq_src, np.real(z_src), left=np.real(z_src[0]), right=np.real(z_src[-1]))
    xi = np.interp(freq_dst, freq_src, np.imag(z_src), left=np.imag(z_src[0]), right=np.imag(z_src[-1]))
    return xr + 1j * xi


def load_measurement_four_csv(
    hh_csv: str | Path,
    hv_csv: str | Path,
    vh_csv: str | Path,
    vv_csv: str | Path,
) -> MeasurementData:
    """Load 2x2 measurement from four per-polarization CSV traces.

    Each file must provide:
      frequency column: f_hz/freq_hz/frequency_hz/f
      one complex trace column as (re,im) or (mag_db,phase_deg)
      trace aliases: s21_* or h_* (e.g. s21_re,s21_im).
    """

    f_hh, z_hh = _parse_trace_csv(hh_csv)
    f_hv, z_hv = _parse_trace_csv(hv_csv)
    f_vh, z_vh = _parse_trace_csv(vh_csv)
    f_vv, z_vv = _parse_trace_csv(vv_csv)
    f = np.asarray(f_hh, dtype=float)
    z_hv_i = _interp_complex(f_hv, z_hv, f)
    z_vh_i = _interp_complex(f_vh, z_vh, f)
    z_vv_i = _interp_complex(f_vv, z_vv, f)
    mats = np.zeros((len(f), 2, 2), dtype=np.complex128)
    mats[:, 0, 0] = z_hh
    mats[:, 0, 1] = z_hv_i
    mats[:, 1, 0] = z_vh_i
    mats[:, 1, 1] = z_vv_i
    return MeasurementData(frequency_hz=f, H_f=mats, source=f"{hh_csv},{hv_csv},{vh_csv},{vv_csv}")


def load_measurement_dualcp_two_csv(
    co_csv: str | Path,
    cross_csv: str | Path,
    basis: str = "circular",
    convention: str = "IEEE-RHCP",
) -> MeasurementData:
    """Load dual-CP sequential measurement from two CSV traces.

    Mapping:
      H_f[:,0,0] = RHCP->RHCP (co)
      H_f[:,1,0] = RHCP->LHCP (cross)
      others are set to 0.
    """

    f_co, z_co = _parse_trace_csv(co_csv)
    f_cross, z_cross = _parse_trace_csv(cross_csv)
    f = np.asarray(f_co, dtype=float)
    z_cross_i = _interp_complex(np.asarray(f_cross, dtype=float), np.asarray(z_cross, dtype=np.complex128), f)
    mats = np.zeros((len(f), 2, 2), dtype=np.complex128)
    mats[:, 0, 0] = np.asarray(z_co, dtype=np.complex128)
    mats[:, 1, 0] = z_cross_i
    return MeasurementData(
        frequency_hz=f,
        H_f=mats,
        source=f"{co_csv},{cross_csv}",
        meta={
            "basis": str(basis),
            "convention": str(convention),
            "format": "dualcp_two_csv",
        },
    )


def load_measurement_dualcp_three_csv(
    co_pre_csv: str | Path,
    cross_csv: str | Path,
    co_post_csv: str | Path,
    basis: str = "circular",
    convention: str = "IEEE-RHCP",
) -> MeasurementData:
    """Load dual-CP sequential measurement with drift check.

    Uses co_pre as the co trace for H_f mapping:
      H_f[:,0,0] = co_pre
      H_f[:,1,0] = cross
    And computes drift from co_post against co_pre:
      drift_co_db = median(|20log10(|co_post|/|co_pre|)|)
    """

    f_pre, z_pre = _parse_trace_csv(co_pre_csv)
    f_cross, z_cross = _parse_trace_csv(cross_csv)
    f_post, z_post = _parse_trace_csv(co_post_csv)
    f = np.asarray(f_pre, dtype=float)
    z_cross_i = _interp_complex(np.asarray(f_cross, dtype=float), np.asarray(z_cross, dtype=np.complex128), f)
    z_post_i = _interp_complex(np.asarray(f_post, dtype=float), np.asarray(z_post, dtype=np.complex128), f)

    mats = np.zeros((len(f), 2, 2), dtype=np.complex128)
    mats[:, 0, 0] = np.asarray(z_pre, dtype=np.complex128)
    mats[:, 1, 0] = z_cross_i

    ratio_db = 20.0 * np.log10((np.abs(z_post_i) + EPS) / (np.abs(z_pre) + EPS))
    drift_abs = np.abs(np.asarray(ratio_db, dtype=float))
    drift_med = float(np.median(drift_abs)) if len(drift_abs) else float("nan")
    drift_p95 = float(np.percentile(drift_abs, 95.0)) if len(drift_abs) else float("nan")

    return MeasurementData(
        frequency_hz=f,
        H_f=mats,
        source=f"{co_pre_csv},{cross_csv},{co_post_csv}",
        meta={
            "basis": str(basis),
            "convention": str(convention),
            "format": "dualcp_three_csv",
            "co_pre_csv": str(co_pre_csv),
            "co_post_csv": str(co_post_csv),
            "drift_metric": "median_abs_delta_mag_db",
            "drift_co_db": drift_med,
            "drift_co_p95_db": drift_p95,
        },
    )


def _select_case(dataset: dict[str, Any], scenario_id: str | None = None, case_id: str | None = None) -> tuple[str, str, dict[str, Any]]:
    scenarios = dataset.get("scenarios", {})
    if scenario_id is not None:
        if scenario_id not in scenarios:
            raise ValueError(f"scenario_id not found: {scenario_id}")
        cases = scenarios[scenario_id].get("cases", {})
        if case_id is not None:
            if case_id not in cases:
                raise ValueError(f"case_id not found in {scenario_id}: {case_id}")
            return str(scenario_id), str(case_id), cases[case_id]
        # choose strongest among max-path cases
        best = None
        for cid, c in cases.items():
            paths = c.get("paths", [])
            n = len(paths)
            p = max(
                [float(np.mean(np.abs(np.asarray(pp.get("A_f", np.zeros((1, 2, 2))), dtype=np.complex128)) ** 2)) for pp in paths]
                or [-np.inf]
            )
            score = (n, p)
            if best is None or score > best[0]:
                best = (score, str(cid), c)
        if best is None:
            raise ValueError(f"no cases in scenario: {scenario_id}")
        return str(scenario_id), best[1], best[2]

    best2 = None
    for sid, sc in scenarios.items():
        for cid, c in sc.get("cases", {}).items():
            paths = c.get("paths", [])
            n = len(paths)
            p = max(
                [float(np.mean(np.abs(np.asarray(pp.get("A_f", np.zeros((1, 2, 2))), dtype=np.complex128)) ** 2)) for pp in paths]
                or [-np.inf]
            )
            score = (n, p)
            if best2 is None or score > best2[0]:
                best2 = (score, str(sid), str(cid), c)
    if best2 is None:
        raise ValueError("dataset has no cases")
    return best2[1], best2[2], best2[3]


def _xpd_over_frequency(H_f: NDArray[np.complex128]) -> NDArray[np.float64]:
    co = np.abs(H_f[:, 0, 0]) ** 2 + np.abs(H_f[:, 1, 1]) ** 2
    cr = np.abs(H_f[:, 0, 1]) ** 2 + np.abs(H_f[:, 1, 0]) ** 2
    return 10.0 * np.log10((co + EPS) / (np.maximum(cr, 1e-12) + EPS))


def _ecdf(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if len(x) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    xs = np.sort(np.asarray(x, dtype=float))
    ys = np.arange(1, len(xs) + 1, dtype=float) / float(len(xs))
    return xs, ys


def _save(fig: plt.Figure, out_dir: Path, name: str) -> str:
    fig.tight_layout()
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    return str(png)


def compare_measured_to_dataset(
    dataset: dict[str, Any],
    measurement: MeasurementData,
    channel_definition: str = "embedded",
    scenario_id: str | None = None,
    case_id: str | None = None,
    measurement_basis: str | None = None,
    measurement_convention: str | None = None,
    eval_basis: str | None = None,
    eval_convention: str | None = None,
    synth_paths: list[dict[str, Any]] | None = None,
    out_dir: str | Path | None = None,
    create_plots: bool = True,
) -> dict[str, Any]:
    """Compare measured H(f) against RT (and optional Synth) under a unified definition."""

    matrix_source = "A" if str(channel_definition).lower().startswith("emb") else "J"
    rt_basis = str(dataset.get("meta", {}).get("basis", "linear"))
    rt_conv = str(dataset.get("meta", {}).get("convention", "IEEE-RHCP"))
    meas_meta = measurement.meta if isinstance(measurement.meta, dict) else {}
    meas_basis = str(measurement_basis or meas_meta.get("basis", rt_basis))
    meas_conv = str(measurement_convention or meas_meta.get("convention", rt_conv))
    out_basis = str(eval_basis or rt_basis)
    out_conv = str(eval_convention or rt_conv)

    sid, cid, case = _select_case(dataset, scenario_id=scenario_id, case_id=case_id)
    rt_freq = np.asarray(dataset.get("frequency", []), dtype=float)
    if len(rt_freq) == 0:
        raise ValueError("dataset frequency is empty")
    paths = case.get("paths", [])
    H_rt = synthesize_ctf_with_source(paths, rt_freq, matrix_source=matrix_source)
    if rt_basis != out_basis:
        H_rt = convert_basis(H_rt, src=rt_basis, dst=out_basis, convention=out_conv)

    meas_f = np.asarray(measurement.frequency_hz, dtype=float)
    H_meas = np.asarray(measurement.H_f, dtype=np.complex128)
    if H_meas.shape != (len(meas_f), 2, 2):
        raise ValueError("measurement H_f must have shape (Nf,2,2)")
    if meas_basis != out_basis:
        H_meas = convert_basis(H_meas, src=meas_basis, dst=out_basis, convention=out_conv)

    H_meas_rt = np.zeros((len(rt_freq), 2, 2), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            H_meas_rt[:, i, j] = _interp_complex(meas_f, H_meas[:, i, j], rt_freq)

    H_sy = None
    if synth_paths:
        H_sy = synthesize_ctf_with_source(synth_paths, rt_freq, matrix_source=matrix_source)
        if rt_basis != out_basis:
            H_sy = convert_basis(H_sy, src=rt_basis, dst=out_basis, convention=out_conv)

    mag_rt = 20.0 * np.log10(np.abs(H_rt) + 1e-12)
    mag_me = 20.0 * np.log10(np.abs(H_meas_rt) + 1e-12)
    rmse_mag_db_rt = float(np.sqrt(np.mean((mag_me - mag_rt) ** 2)))

    xpd_me = _xpd_over_frequency(H_meas_rt)
    xpd_rt = _xpd_over_frequency(H_rt)
    ks_rt = stats.ks_2samp(xpd_me, xpd_rt, alternative="two-sided", method="auto")

    ks_sy_p = np.nan
    rmse_mag_db_sy = np.nan
    xpd_sy = np.asarray([], dtype=float)
    if H_sy is not None:
        mag_sy = 20.0 * np.log10(np.abs(H_sy) + 1e-12)
        rmse_mag_db_sy = float(np.sqrt(np.mean((mag_me - mag_sy) ** 2)))
        xpd_sy = _xpd_over_frequency(H_sy)
        ks_sy = stats.ks_2samp(xpd_me, xpd_sy, alternative="two-sided", method="auto")
        ks_sy_p = float(ks_sy.pvalue)

    plots: dict[str, str] = {}
    if create_plots and out_dir is not None:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        h_me, tau = ctf_to_cir(H_meas_rt, rt_freq, nfft=2048)
        h_rt, tau_rt = ctf_to_cir(H_rt, rt_freq, nfft=2048)
        p_me = pdp(h_me)["sum"]
        p_rt = pdp(h_rt)["sum"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(tau * 1e9, 10.0 * np.log10(p_me + 1e-18), label="Measured")
        ax.plot(tau_rt * 1e9, 10.0 * np.log10(p_rt + 1e-18), label=f"RT ({matrix_source})")
        if H_sy is not None:
            h_sy, tau_sy = ctf_to_cir(H_sy, rt_freq, nfft=2048)
            p_sy = pdp(h_sy)["sum"]
            ax.plot(tau_sy * 1e9, 10.0 * np.log10(p_sy + 1e-18), label="Synthetic")
        ax.set_title(
            "Measured vs RT/Synth PDP\n"
            f"basis={out_basis}, conv={out_conv}, channel_definition={channel_definition}, matrix={matrix_source}"
        )
        ax.set_xlabel("tau [ns]")
        ax.set_ylabel("power [dB]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plots["pdp_overlay"] = _save(fig, outp, "measurement_vs_rt_synth_pdp")

        fig, ax = plt.subplots(figsize=(7, 4))
        xm, ym = _ecdf(xpd_me)
        xr, yr = _ecdf(xpd_rt)
        ax.plot(xm, ym, label="Measured")
        ax.plot(xr, yr, label=f"RT ({matrix_source})")
        if len(xpd_sy):
            xs, ys = _ecdf(xpd_sy)
            ax.plot(xs, ys, label="Synthetic")
        ax.set_title("Measured vs RT/Synth XPD CDF")
        ax.set_xlabel("XPD [dB]")
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plots["xpd_cdf_overlay"] = _save(fig, outp, "measurement_vs_rt_synth_xpd_cdf")

    return {
        "scenario_id": sid,
        "case_id": cid,
        "channel_definition": str(channel_definition),
        "matrix_source": matrix_source,
        "measurement_source": str(measurement.source),
        "measurement_freq_count": int(len(meas_f)),
        "rt_freq_count": int(len(rt_freq)),
        "measurement_basis": meas_basis,
        "measurement_convention": meas_conv,
        "eval_basis": out_basis,
        "eval_convention": out_conv,
        "rmse_mag_db_meas_vs_rt": rmse_mag_db_rt,
        "rmse_mag_db_meas_vs_synth": float(rmse_mag_db_sy) if np.isfinite(rmse_mag_db_sy) else np.nan,
        "xpd_mu_measured_db": float(np.mean(xpd_me)),
        "xpd_mu_rt_db": float(np.mean(xpd_rt)),
        "xpd_mu_synth_db": float(np.mean(xpd_sy)) if len(xpd_sy) else np.nan,
        "xpd_ks2_p_meas_vs_rt": float(ks_rt.pvalue),
        "xpd_ks2_p_meas_vs_synth": float(ks_sy_p) if np.isfinite(ks_sy_p) else np.nan,
        "has_synth_compare": bool(H_sy is not None),
        "plots": plots,
    }

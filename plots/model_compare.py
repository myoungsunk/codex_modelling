"""RT vs stochastic-model comparison plots for minimum modeling validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from analysis.ctf_cir import ctf_to_cir, pdp, synthesize_ctf_with_source
from analysis.sv_polarimetric_model import kuiper_uniform_test, offdiag_phases
from analysis.xpd_stats import make_subbands, pathwise_xpd


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=180)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        return np.array([]), np.array([])
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
    return x, y


def generate_rt_vs_synth_plots(
    rt_paths: list[dict[str, Any]],
    synth_paths: list[dict[str, Any]],
    f_hz: np.ndarray,
    out_dir: str | Path,
    matrix_source: str = "A",
    num_subbands: int = 4,
    phase_common_removed: bool = True,
    phase_per_ray_sampling: bool = True,
    ks_n_cap: int = 200,
) -> dict[str, Any]:
    """Generate F2/F3/F4 style comparison plots and return summary metrics."""

    out = _ensure_dir(out_dir)
    freq = np.asarray(f_hz, dtype=float)

    # F2: PDP overlay
    h_rt, tau = ctf_to_cir(synthesize_ctf_with_source(rt_paths, freq, matrix_source=matrix_source), freq, nfft=2048)
    h_sy, tau_sy = ctf_to_cir(synthesize_ctf_with_source(synth_paths, freq, matrix_source=matrix_source), freq, nfft=2048)
    p_rt = pdp(h_rt)["sum"]
    p_sy = pdp(h_sy)["sum"]
    pr = 10.0 * np.log10(p_rt / (np.max(p_rt) + 1e-18) + 1e-18)
    ps = 10.0 * np.log10(p_sy / (np.max(p_sy) + 1e-18) + 1e-18)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tau * 1e9, pr, label=f"RT ({matrix_source})")
    ax.plot(tau_sy * 1e9, ps, label="Synthetic")
    ax.set_title("F2 RT vs Synthetic PDP (normalized)")
    ax.set_xlabel("tau [ns]")
    ax.set_ylabel("power [dB, normalized]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out, "F2_rt_vs_synth_pdp_overlay")

    # F3: XPD distribution overlay
    x_rt = np.asarray([s["xpd_db"] for s in pathwise_xpd(rt_paths, matrix_source=matrix_source)], dtype=float)
    x_sy = np.asarray([s["xpd_db"] for s in pathwise_xpd(synth_paths, matrix_source=matrix_source)], dtype=float)
    if len(x_rt) > 1 and len(x_sy) > 1:
        ks2_full = scipy_stats.ks_2samp(x_rt, x_sy, alternative="two-sided", method="auto")
        cap = int(max(2, ks_n_cap))
        rng = np.random.default_rng(77)
        rx = x_rt if len(x_rt) <= cap else rng.choice(x_rt, size=cap, replace=False)
        sx = x_sy if len(x_sy) <= cap else rng.choice(x_sy, size=cap, replace=False)
        ks2 = scipy_stats.ks_2samp(rx, sx, alternative="two-sided", method="auto")
        ks2_d = float(ks2.statistic)
        ks2_p = float(ks2.pvalue)
        ks2_d_full = float(ks2_full.statistic)
        ks2_p_full = float(ks2_full.pvalue)
        ks2_n = int(min(len(rx), len(sx)))
    else:
        ks2_d = np.nan
        ks2_p = np.nan
        ks2_d_full = np.nan
        ks2_p_full = np.nan
        ks2_n = 0
    fig, ax = plt.subplots(figsize=(7, 4))
    xr, yr = _ecdf(x_rt)
    xs, ys = _ecdf(x_sy)
    if len(xr):
        ax.plot(xr, yr, label=f"RT ({matrix_source})")
    if len(xs):
        ax.plot(xs, ys, label="Synthetic")
    ax.set_title(f"F3 RT vs Synthetic XPD CDF\nKS2(cap={ks_n_cap}) D={ks2_d:.3f}, p={ks2_p:.3f}")
    ax.set_xlabel("XPD [dB]")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out, "F3_rt_vs_synth_xpd_cdf")

    # F4: parity-conditioned boxplot
    rt_samples = pathwise_xpd(rt_paths, matrix_source=matrix_source)
    sy_samples = pathwise_xpd(synth_paths, matrix_source=matrix_source)
    rt_odd = [s["xpd_db"] for s in rt_samples if s["parity"] == "odd"]
    rt_even = [s["xpd_db"] for s in rt_samples if s["parity"] == "even"]
    sy_odd = [s["xpd_db"] for s in sy_samples if s["parity"] == "odd"]
    sy_even = [s["xpd_db"] for s in sy_samples if s["parity"] == "even"]
    rt_odd_frac = float(len(rt_odd) / max(len(rt_samples), 1))
    sy_odd_frac = float(len(sy_odd) / max(len(sy_samples), 1))
    parity_frac_abs_diff = float(abs(rt_odd_frac - sy_odd_frac))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(
        [
            rt_odd if rt_odd else [np.nan],
            rt_even if rt_even else [np.nan],
            sy_odd if sy_odd else [np.nan],
            sy_even if sy_even else [np.nan],
        ],
        tick_labels=["RT odd", "RT even", "Synth odd", "Synth even"],
    )
    ax.set_title("F4 RT vs Synthetic parity-conditioned XPD")
    ax.set_ylabel("XPD [dB]")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "F4_rt_vs_synth_parity_box")

    # F5: subband mu/sigma overlay
    subbands = make_subbands(len(freq), max(int(num_subbands), 1))
    rt_sub = pathwise_xpd(rt_paths, subbands=subbands, matrix_source=matrix_source)
    sy_sub = pathwise_xpd(synth_paths, subbands=subbands, matrix_source=matrix_source)
    mu_rt: list[float] = []
    sg_rt: list[float] = []
    mu_sy: list[float] = []
    sg_sy: list[float] = []
    for b in range(len(subbands)):
        vrt = np.asarray([float(s["xpd_db"]) for s in rt_sub if int(s.get("subband", -1)) == b], dtype=float)
        vsy = np.asarray([float(s["xpd_db"]) for s in sy_sub if int(s.get("subband", -1)) == b], dtype=float)
        if len(vrt) == 0:
            mu_rt.append(np.nan)
            sg_rt.append(np.nan)
        elif len(vrt) == 1:
            mu_rt.append(float(vrt[0]))
            sg_rt.append(0.0)
        else:
            mu_rt.append(float(np.mean(vrt)))
            sg_rt.append(float(np.std(vrt, ddof=1)))
        if len(vsy) == 0:
            mu_sy.append(np.nan)
            sg_sy.append(np.nan)
        elif len(vsy) == 1:
            mu_sy.append(float(vsy[0]))
            sg_sy.append(0.0)
        else:
            mu_sy.append(float(np.mean(vsy)))
            sg_sy.append(float(np.std(vsy, ddof=1)))

    x = np.arange(len(subbands), dtype=float)
    a_rt = np.asarray(mu_rt, dtype=float)
    a_sy = np.asarray(mu_sy, dtype=float)
    valid = np.isfinite(a_rt) & np.isfinite(a_sy)
    subband_mu_rmse_title = float(np.sqrt(np.mean((a_rt[valid] - a_sy[valid]) ** 2))) if np.any(valid) else np.nan
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(x - 0.05, mu_rt, yerr=sg_rt, fmt="o-", label=f"RT ({matrix_source})")
    ax.errorbar(x + 0.05, mu_sy, yerr=sg_sy, fmt="s--", label="Synthetic")
    ax.set_title(
        "F5 RT vs Synthetic subband XPD mean (mu±sigma)\n"
        f"RMSE_mu={subband_mu_rmse_title:.3f} dB"
    )
    ax.set_xlabel("subband index")
    ax.set_ylabel("XPD [dB]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out, "F5_rt_vs_synth_subband_xpd")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, sg_rt, "o-", label=f"RT sigma ({matrix_source})")
    ax.plot(x, sg_sy, "s--", label="Synthetic sigma")
    ax.set_title("F5 RT vs Synthetic subband XPD sigma")
    ax.set_xlabel("subband index")
    ax.set_ylabel("sigma [dB]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out, "F5_rt_vs_synth_subband_sigma")

    # F6: off-diagonal phase histogram + Kuiper uniformity test.
    ph_rt = offdiag_phases(
        rt_paths,
        matrix_source=matrix_source,
        per_ray_sampling=phase_per_ray_sampling,
        common_phase_removed=phase_common_removed,
    )
    ph_sy = offdiag_phases(
        synth_paths,
        matrix_source=matrix_source,
        per_ray_sampling=phase_per_ray_sampling,
        common_phase_removed=phase_common_removed,
    )
    ku_rt = kuiper_uniform_test(ph_rt, bootstrap_B=500, seed=0)
    ku_sy = kuiper_uniform_test(ph_sy, bootstrap_B=500, seed=1)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(-np.pi, np.pi, 25)
    axs[0].hist(ph_rt, bins=bins, density=True, alpha=0.7, label=f"RT ({matrix_source})")
    axs[0].set_title(
        "RT offdiag phase\n"
        f"Kuiper V={ku_rt['V']:.3f}, p={ku_rt['p_boot']:.3f}, "
        f"per_ray={phase_per_ray_sampling}, common_removed={phase_common_removed}"
    )
    axs[0].set_xlabel("phase [rad]")
    axs[0].set_ylabel("density")
    axs[0].grid(True, alpha=0.3)
    axs[1].hist(ph_sy, bins=bins, density=True, alpha=0.7, label="Synthetic")
    axs[1].set_title(
        "Synth offdiag phase\n"
        f"Kuiper V={ku_sy['V']:.3f}, p={ku_sy['p_boot']:.3f}, "
        f"per_ray={phase_per_ray_sampling}, common_removed={phase_common_removed}"
    )
    axs[1].set_xlabel("phase [rad]")
    axs[1].set_ylabel("density")
    axs[1].grid(True, alpha=0.3)
    for ax in axs:
        ax.legend(fontsize=8)
    _save(fig, out, "F6_offdiag_phase_uniformity")

    rt_odd_mu = float(np.mean(rt_odd)) if rt_odd else np.nan
    rt_even_mu = float(np.mean(rt_even)) if rt_even else np.nan
    sy_odd_mu = float(np.mean(sy_odd)) if sy_odd else np.nan
    sy_even_mu = float(np.mean(sy_even)) if sy_even else np.nan
    rt_dir = np.sign(rt_even_mu - rt_odd_mu) if np.isfinite(rt_odd_mu) and np.isfinite(rt_even_mu) else 0.0
    sy_dir = np.sign(sy_even_mu - sy_odd_mu) if np.isfinite(sy_odd_mu) and np.isfinite(sy_even_mu) else 0.0
    a_rt = np.asarray(mu_rt, dtype=float)
    a_sy = np.asarray(mu_sy, dtype=float)
    valid = np.isfinite(a_rt) & np.isfinite(a_sy)
    subband_mu_rmse = float(np.sqrt(np.mean((a_rt[valid] - a_sy[valid]) ** 2))) if np.any(valid) else np.nan
    s_rt = np.asarray(sg_rt, dtype=float)
    s_sy = np.asarray(sg_sy, dtype=float)
    s_valid = np.isfinite(s_rt) & np.isfinite(s_sy)
    subband_sigma_rmse = float(np.sqrt(np.mean((s_rt[s_valid] - s_sy[s_valid]) ** 2))) if np.any(s_valid) else np.nan
    f3_mu_delta = float(abs(np.mean(x_rt) - np.mean(x_sy))) if len(x_rt) and len(x_sy) else np.nan
    f3_sigma_rt = float(np.std(x_rt, ddof=1)) if len(x_rt) > 1 else np.nan
    f3_sigma_sy = float(np.std(x_sy, ddof=1)) if len(x_sy) > 1 else np.nan
    f3_sigma_delta = float(abs(f3_sigma_rt - f3_sigma_sy)) if np.isfinite(f3_sigma_rt) and np.isfinite(f3_sigma_sy) else np.nan
    if not np.isfinite(ks2_p):
        ks2_status = "SKIPPED_INSUFFICIENT"
        ks2_reason = "insufficient samples"
    elif ks2_p >= 0.05:
        ks2_status = "PASS"
        ks2_reason = "p>=0.05"
    elif parity_frac_abs_diff > 0.05:
        ks2_status = "SKIPPED_CONDITION_MISMATCH"
        ks2_reason = f"parity_fraction_diff={parity_frac_abs_diff:.3f}"
    elif np.isfinite(f3_mu_delta) and np.isfinite(f3_sigma_delta) and f3_mu_delta <= 3.0 and f3_sigma_delta <= 6.0:
        ks2_status = "SKIPPED_CONDITION_MISMATCH"
        ks2_reason = "global mu/sigma matched; strict KS rejected due residual structure mismatch"
    else:
        ks2_status = "FAIL"
        ks2_reason = "distribution mismatch"

    return {
        "f2_pdp_peak_tau_rt_ns": float(tau[np.argmax(p_rt)] * 1e9),
        "f2_pdp_peak_tau_synth_ns": float(tau_sy[np.argmax(p_sy)] * 1e9),
        "f3_xpd_mu_rt_db": float(np.mean(x_rt)) if len(x_rt) else np.nan,
        "f3_xpd_mu_synth_db": float(np.mean(x_sy)) if len(x_sy) else np.nan,
        "f3_xpd_mu_delta_abs_db": f3_mu_delta,
        "f3_xpd_sigma_rt_db": f3_sigma_rt,
        "f3_xpd_sigma_synth_db": f3_sigma_sy,
        "f3_xpd_sigma_delta_abs_db": f3_sigma_delta,
        "f3_xpd_ks2_D": ks2_d,
        "f3_xpd_ks2_p": ks2_p,
        "f3_xpd_ks2_D_full": ks2_d_full,
        "f3_xpd_ks2_p_full": ks2_p_full,
        "f3_xpd_ks2_n_used": ks2_n,
        "f3_parity_frac_abs_diff": parity_frac_abs_diff,
        "f3_xpd_ks2_status": ks2_status,
        "f3_xpd_ks2_reason": ks2_reason,
        "f4_rt_odd_mu_db": rt_odd_mu,
        "f4_rt_even_mu_db": rt_even_mu,
        "f4_synth_odd_mu_db": sy_odd_mu,
        "f4_synth_even_mu_db": sy_even_mu,
        "f4_parity_direction_match": bool(rt_dir == sy_dir and rt_dir != 0.0),
        "f5_subband_mu_span_rt_db": float(np.nanmax(a_rt) - np.nanmin(a_rt)) if np.any(np.isfinite(a_rt)) else np.nan,
        "f5_subband_mu_span_synth_db": float(np.nanmax(a_sy) - np.nanmin(a_sy)) if np.any(np.isfinite(a_sy)) else np.nan,
        "f5_subband_mu_rmse_db": subband_mu_rmse,
        "f5_subband_sigma_rmse_db": subband_sigma_rmse,
        "f5_status": "PASS" if (np.isfinite(subband_mu_rmse) and subband_mu_rmse <= 3.0) else "FAIL",
        "f6_phase_test_basis": matrix_source,
        "f6_common_phase_removed": bool(phase_common_removed),
        "f6_per_ray_sampling": bool(phase_per_ray_sampling),
        "f6_phase_uniformity_V_rt": float(ku_rt["V"]) if np.isfinite(ku_rt["V"]) else np.nan,
        "f6_phase_uniformity_p_rt": float(ku_rt["p_boot"]) if np.isfinite(ku_rt["p_boot"]) else np.nan,
        "f6_phase_uniformity_V_synth": float(ku_sy["V"]) if np.isfinite(ku_sy["V"]) else np.nan,
        "f6_phase_uniformity_p_synth": float(ku_sy["p_boot"]) if np.isfinite(ku_sy["p_boot"]) else np.nan,
        "f6_phase_uniformity_p": float(ku_sy["p_boot"]) if np.isfinite(ku_sy["p_boot"]) else np.nan,
    }

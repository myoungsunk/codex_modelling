"""RT vs stochastic-model comparison plots for minimum modeling validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from analysis.ctf_cir import ctf_to_cir, pdp, synthesize_ctf, synthesize_ctf_with_source
from analysis.xpd_stats import pathwise_xpd


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
) -> dict[str, Any]:
    """Generate F2/F3/F4 style comparison plots and return summary metrics."""

    out = _ensure_dir(out_dir)
    freq = np.asarray(f_hz, dtype=float)

    # F2: PDP overlay
    h_rt, tau = ctf_to_cir(synthesize_ctf_with_source(rt_paths, freq, matrix_source=matrix_source), freq, nfft=2048)
    h_sy, tau_sy = ctf_to_cir(synthesize_ctf(synth_paths, freq), freq, nfft=2048)
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
    x_sy = np.asarray([s["xpd_db"] for s in pathwise_xpd(synth_paths, matrix_source="A")], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    xr, yr = _ecdf(x_rt)
    xs, ys = _ecdf(x_sy)
    if len(xr):
        ax.plot(xr, yr, label=f"RT ({matrix_source})")
    if len(xs):
        ax.plot(xs, ys, label="Synthetic")
    ax.set_title("F3 RT vs Synthetic XPD CDF")
    ax.set_xlabel("XPD [dB]")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out, "F3_rt_vs_synth_xpd_cdf")

    # F4: parity-conditioned boxplot
    rt_samples = pathwise_xpd(rt_paths, matrix_source=matrix_source)
    sy_samples = pathwise_xpd(synth_paths, matrix_source="A")
    rt_odd = [s["xpd_db"] for s in rt_samples if s["parity"] == "odd"]
    rt_even = [s["xpd_db"] for s in rt_samples if s["parity"] == "even"]
    sy_odd = [s["xpd_db"] for s in sy_samples if s["parity"] == "odd"]
    sy_even = [s["xpd_db"] for s in sy_samples if s["parity"] == "even"]
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

    rt_odd_mu = float(np.mean(rt_odd)) if rt_odd else np.nan
    rt_even_mu = float(np.mean(rt_even)) if rt_even else np.nan
    sy_odd_mu = float(np.mean(sy_odd)) if sy_odd else np.nan
    sy_even_mu = float(np.mean(sy_even)) if sy_even else np.nan
    rt_dir = np.sign(rt_even_mu - rt_odd_mu) if np.isfinite(rt_odd_mu) and np.isfinite(rt_even_mu) else 0.0
    sy_dir = np.sign(sy_even_mu - sy_odd_mu) if np.isfinite(sy_odd_mu) and np.isfinite(sy_even_mu) else 0.0

    return {
        "f2_pdp_peak_tau_rt_ns": float(tau[np.argmax(p_rt)] * 1e9),
        "f2_pdp_peak_tau_synth_ns": float(tau_sy[np.argmax(p_sy)] * 1e9),
        "f3_xpd_mu_rt_db": float(np.mean(x_rt)) if len(x_rt) else np.nan,
        "f3_xpd_mu_synth_db": float(np.mean(x_sy)) if len(x_sy) else np.nan,
        "f4_rt_odd_mu_db": rt_odd_mu,
        "f4_rt_even_mu_db": rt_even_mu,
        "f4_synth_odd_mu_db": sy_odd_mu,
        "f4_synth_even_mu_db": sy_even_mu,
        "f4_parity_direction_match": bool(rt_dir == sy_dir and rt_dir != 0.0),
    }


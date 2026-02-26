"""Plot helpers for diagnostic/intermediate reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _finite(vals: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def _prep_out(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_cdf(values: list[float] | np.ndarray, out_png: str | Path, title: str, xlabel: str) -> str:
    p = _prep_out(out_png)
    x = np.sort(_finite(values))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    if len(x):
        y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
        ax.plot(x, y, lw=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def plot_multi_cdf(series: dict[str, list[float] | np.ndarray], out_png: str | Path, title: str, xlabel: str) -> str:
    p = _prep_out(out_png)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for k, vals in series.items():
        x = np.sort(_finite(vals))
        if len(x) == 0:
            continue
        y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
        ax.plot(x, y, lw=1.8, label=str(k))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    if series:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def plot_box_by_group(rows: list[dict[str, Any]], group_key: str, value_key: str, out_png: str | Path, title: str, ylabel: str) -> str:
    p = _prep_out(out_png)
    groups: dict[str, list[float]] = {}
    for r in rows:
        g = str(r.get(group_key, "NA"))
        groups.setdefault(g, []).append(float(r.get(value_key, np.nan)))
    labels = []
    data = []
    for g in sorted(groups.keys()):
        x = _finite(groups[g])
        if len(x) == 0:
            continue
        labels.append(g)
        data.append(x)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.0 + 0.6 * len(labels)), 4.5))
    if data:
        ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def plot_scatter(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    out_png: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    c: list[float] | np.ndarray | None = None,
    add_fit: bool = False,
) -> str:
    p = _prep_out(out_png)
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    m = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[m]
    yy = yy[m]
    cc = None
    if c is not None:
        cc_arr = np.asarray(c, dtype=float)
        if len(cc_arr) == len(m):
            cc = cc_arr[m]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if len(xx):
        if cc is not None:
            sc = ax.scatter(xx, yy, c=cc, s=28, alpha=0.85, cmap="viridis")
            fig.colorbar(sc, ax=ax, shrink=0.9)
        else:
            ax.scatter(xx, yy, s=28, alpha=0.85)
        if add_fit and len(xx) >= 2 and len(np.unique(np.round(xx, 12))) >= 2:
            try:
                z = np.polyfit(xx, yy, 1)
                xf = np.linspace(float(np.min(xx)), float(np.max(xx)), 100)
                yf = z[0] * xf + z[1]
                ax.plot(xf, yf, "--", lw=1.8)
            except Exception:
                pass
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def plot_pdp_overlay(delay_s: np.ndarray, p_co: np.ndarray, p_cross: np.ndarray, out_png: str | Path, title: str) -> str:
    p = _prep_out(out_png)
    d_ns = np.asarray(delay_s, dtype=float) * 1e9
    co_db = 10.0 * np.log10(np.maximum(np.asarray(p_co, dtype=float), 1e-20))
    cr_db = 10.0 * np.log10(np.maximum(np.asarray(p_cross, dtype=float), 1e-20))
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(d_ns, co_db, label="P_co")
    ax.plot(d_ns, cr_db, label="P_cross")
    ax.set_title(title)
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Power (dB)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def plot_heatmap_xy(
    rows: list[dict[str, Any]],
    value_key: str,
    out_png: str | Path,
    title: str,
    x_key: str = "rx_x",
    y_key: str = "rx_y",
) -> str:
    p = _prep_out(out_png)
    xs = np.asarray([float(r.get(x_key, np.nan)) for r in rows], dtype=float)
    ys = np.asarray([float(r.get(y_key, np.nan)) for r in rows], dtype=float)
    vs = np.asarray([float(r.get(value_key, np.nan)) for r in rows], dtype=float)
    m = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(vs)
    xs = xs[m]
    ys = ys[m]
    vs = vs[m]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    if len(xs):
        sc = ax.scatter(xs, ys, c=vs, s=90, cmap="coolwarm", edgecolor="k", linewidth=0.2)
        fig.colorbar(sc, ax=ax, shrink=0.9)
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)

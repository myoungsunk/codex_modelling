"""Standard plots for proxy-level validation outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _maybe_num(v: str) -> Any:
    s = str(v).strip()
    if s == "":
        return ""
    try:
        x = float(s)
        return int(x) if x.is_integer() else x
    except Exception:
        return s


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        return [{k: _maybe_num(v) for k, v in r.items()} for r in rd]


def _save(fig: plt.Figure, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def generate_standard_plots_from_rows(
    rows: list[dict[str, Any]],
    out_dir: str | Path,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    # C0 floor plots
    c0 = [r for r in rows if str(r.get("scenario_id", "")).upper() == "C0"]
    if c0:
        xpd = np.asarray([float(r.get("XPD_early_db", np.nan)) for r in c0], dtype=float)
        yaw = np.asarray([float(r.get("yaw_deg", np.nan)) for r in c0], dtype=float)
        d = np.asarray([float(r.get("d_m", np.nan)) for r in c0], dtype=float)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(yaw, xpd, c=d, cmap="viridis")
        ax.set_title("C0: XPD_floor vs yaw")
        ax.set_xlabel("yaw [deg]")
        ax.set_ylabel("XPD_early [dB]")
        ax.grid(True, alpha=0.3)
        saved["c0_floor_vs_yaw"] = _save(fig, out / "c0_floor_vs_yaw.png")

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        xv = np.sort(xpd[np.isfinite(xpd)])
        if len(xv):
            yv = np.arange(1, len(xv) + 1) / float(len(xv))
            ax2.plot(xv, yv)
        ax2.set_title("C0: XPD_floor CDF")
        ax2.set_xlabel("XPD_early [dB]")
        ax2.set_ylabel("CDF")
        ax2.grid(True, alpha=0.3)
        saved["c0_floor_cdf"] = _save(fig2, out / "c0_floor_cdf.png")

    # A2/A3 parity plot
    a23 = [r for r in rows if str(r.get("scenario_id", "")).upper() in {"A2", "A3"}]
    if a23:
        fig, ax = plt.subplots(figsize=(7, 4))
        for sid, marker in [("A2", "o"), ("A3", "s")]:
            rr = [r for r in a23 if str(r.get("scenario_id", "")).upper() == sid]
            if not rr:
                continue
            ang = np.arange(len(rr), dtype=int)
            xpd = np.asarray([float(r.get("XPD_early_db", np.nan)) for r in rr], dtype=float)
            ax.plot(ang, xpd, marker=marker, linestyle="-", label=sid)
        ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
        ax.set_title("A2/A3: XPD_early trend")
        ax.set_xlabel("sample index")
        ax.set_ylabel("XPD_early [dB]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        saved["a2_a3_xpd_early"] = _save(fig, out / "a2_a3_xpd_early.png")

    # A4 material scatter
    a4 = [r for r in rows if str(r.get("scenario_id", "")).upper() == "A4"]
    if a4:
        fig, ax = plt.subplots(figsize=(7, 4))
        mats = sorted(set(str(r.get("material_class", "NA")) for r in a4))
        for m in mats:
            rr = [r for r in a4 if str(r.get("material_class", "NA")) == m]
            el = np.asarray([float(r.get("EL_proxy_db", np.nan)) for r in rr], dtype=float)
            xpd = np.asarray([float(r.get("XPD_early_db", np.nan)) for r in rr], dtype=float)
            ax.scatter(el, xpd, label=m, alpha=0.8)
        ax.set_title("A4: XPD_early vs EL_proxy")
        ax.set_xlabel("EL_proxy [dB]")
        ax.set_ylabel("XPD_early [dB]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        saved["a4_material_scatter"] = _save(fig, out / "a4_material_scatter.png")

    # A5 stress comparison
    a5 = [r for r in rows if str(r.get("scenario_id", "")).upper() == "A5"]
    if a5:
        base = np.asarray(
            [float(r.get("XPD_early_db", np.nan)) for r in a5 if int(r.get("roughness_flag", 0)) == 0 and int(r.get("human_flag", 0)) == 0],
            dtype=float,
        )
        stress = np.asarray(
            [float(r.get("XPD_early_db", np.nan)) for r in a5 if int(r.get("roughness_flag", 0)) == 1 or int(r.get("human_flag", 0)) == 1],
            dtype=float,
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        for arr, label in [(base, "baseline"), (stress, "stress")]:
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                continue
            xs = np.sort(arr)
            ys = np.arange(1, len(xs) + 1) / float(len(xs))
            ax.plot(xs, ys, label=label)
        ax.set_title("A5: baseline vs stress CDF (XPD_early)")
        ax.set_xlabel("XPD_early [dB]")
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        saved["a5_stress_cdf"] = _save(fig, out / "a5_stress_cdf.png")

    # B-space: heatmap-like and scatter
    b = [r for r in rows if str(r.get("scenario_id", "")).upper().startswith("B")]
    if b:
        x = np.asarray([float(r.get("rx_x", np.nan)) for r in b], dtype=float)
        y = np.asarray([float(r.get("rx_y", np.nan)) for r in b], dtype=float)
        z = np.asarray([float(r.get("XPD_early_db", np.nan)) for r in b], dtype=float)
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
            xu = np.sort(np.unique(x))
            yu = np.sort(np.unique(y))
            Z = np.full((len(yu), len(xu)), np.nan, dtype=float)
            for r in b:
                ix = int(np.where(xu == float(r.get("rx_x", np.nan)))[0][0])
                iy = int(np.where(yu == float(r.get("rx_y", np.nan)))[0][0])
                Z[iy, ix] = float(r.get("XPD_early_db", np.nan))
            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(Z, origin="lower", aspect="auto", extent=[xu[0], xu[-1], yu[0], yu[-1]])
            fig.colorbar(im, ax=ax, label="XPD_early [dB]")
            ax.set_title("B-space: heatmap(XPD_early)")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            saved["b_heatmap_xpd_early"] = _save(fig, out / "b_heatmap_xpd_early.png")

        el = np.asarray([float(r.get("EL_proxy_db", np.nan)) for r in b], dtype=float)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.scatter(el, z, alpha=0.7)
        m = np.isfinite(el) & np.isfinite(z)
        if np.sum(m) >= 2:
            coef = np.polyfit(el[m], z[m], 1)
            xx = np.linspace(np.min(el[m]), np.max(el[m]), 50)
            yy = coef[0] * xx + coef[1]
            ax2.plot(xx, yy, "r--", linewidth=1.2)
        ax2.set_title("B-space: XPD_early vs EL_proxy")
        ax2.set_xlabel("EL_proxy [dB]")
        ax2.set_ylabel("XPD_early [dB]")
        ax2.grid(True, alpha=0.3)
        saved["b_scatter_xpd_vs_el"] = _save(fig2, out / "b_scatter_xpd_vs_el.png")

    return saved


def generate_standard_plots(
    link_metrics_csv: str | Path,
    out_dir: str | Path,
) -> dict[str, str]:
    rows = _load_rows(link_metrics_csv)
    return generate_standard_plots_from_rows(rows, out_dir=out_dir)

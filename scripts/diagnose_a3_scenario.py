"""A3 scenario diagnostics: 2-bounce existence/early viability + per-link PDP grids."""

from __future__ import annotations

import argparse
import csv
import json
from math import ceil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-18


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r.get(key, "")), []).append(r)
    return out


def _parse_window_json(row: dict[str, Any]) -> dict[str, float]:
    try:
        w = json.loads(str(row.get("window", "{}") or "{}"))
        return {
            "tau0_s": float(w.get("tau0_s", np.nan)),
            "Te_s": float(w.get("Te_s", np.nan)),
            "Tmax_s": float(w.get("Tmax_s", np.nan)),
        }
    except Exception:
        return {"tau0_s": float("nan"), "Te_s": float("nan"), "Tmax_s": float("nan")}


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _link_key_sort(link_id: str) -> tuple[int, str]:
    s = str(link_id)
    if "_" in s:
        tail = s.split("_")[-1]
        try:
            return int(tail), s
        except Exception:
            return 10**9, s
    return 10**9, s


def _plot_grid(
    link_ids: list[str],
    pdp_dir: Path,
    status: dict[str, dict[str, Any]],
    out_png: Path,
    title: str,
) -> None:
    n = len(link_ids)
    if n == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No links", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        return
    ncols = 4
    nrows = int(ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows), squeeze=False)
    for i, lid in enumerate(link_ids):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]
        npz = pdp_dir / f"pdp_{lid}.npz"
        if not npz.exists():
            ax.text(0.5, 0.5, f"{lid}\nmissing npz", ha="center", va="center", fontsize=8)
            ax.set_axis_off()
            continue
        d = np.load(npz)
        tau = np.asarray(d["delay_tau_s"], dtype=float) * 1e9
        pco = np.asarray(d["P_co"], dtype=float)
        pcr = np.asarray(d["P_cross"], dtype=float)
        ax.plot(tau, 10.0 * np.log10(pco + EPS), lw=0.9, label="co")
        ax.plot(tau, 10.0 * np.log10(pcr + EPS), lw=0.9, label="cross")
        st = status.get(lid, {})
        tag = f"2b={'Y' if st.get('has_2bounce') else 'N'}, early={'Y' if st.get('target_in_early') else 'N'}"
        ax.set_title(f"{lid} | {tag}", fontsize=8)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].set_axis_off()
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--link-metrics-csv", required=True, type=str)
    ap.add_argument("--rays-csv", required=True, type=str)
    ap.add_argument("--pdp-dir", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--target-bounce", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdp_dir = Path(args.pdp_dir)

    mrows = _read_csv(args.link_metrics_csv)
    rrows = _read_csv(args.rays_csv)

    m_by = {str(r.get("link_id", "")): r for r in mrows}
    r_by = _group_by(rrows, "link_id")

    link_ids = sorted(set(m_by.keys()) | set(r_by.keys()), key=_link_key_sort)

    status_rows: list[dict[str, Any]] = []
    status_by_link: dict[str, dict[str, Any]] = {}

    for lid in link_ids:
        rr = r_by.get(lid, [])
        m = m_by.get(lid, {})
        w = _parse_window_json(m)
        tau0 = float(w.get("tau0_s", np.nan))
        te = float(w.get("Te_s", np.nan))
        tgt_tau: list[float] = []
        los_ray_count = 0
        for r in rr:
            b = _safe_float(r.get("n_bounce", np.nan))
            tau = _safe_float(r.get("tau_s", np.nan))
            p = _safe_float(r.get("P_lin", np.nan))
            los = _safe_float(r.get("los_flag_ray", np.nan))
            if np.isfinite(los) and int(round(los)) == 1:
                los_ray_count += 1
            if np.isfinite(b) and int(round(b)) == int(args.target_bounce) and np.isfinite(tau) and np.isfinite(p) and p > 0.0:
                tgt_tau.append(float(tau))

        has_tgt = len(tgt_tau) > 0
        in_early = bool(has_tgt and np.isfinite(tau0) and np.isfinite(te) and min(tgt_tau) <= (tau0 + te))
        row = {
            "link_id": lid,
            "n_rays": int(len(rr)),
            "los_ray_count": int(los_ray_count),
            "has_2bounce": int(has_tgt),
            "target_tau_min_ns": float(min(tgt_tau) * 1e9) if has_tgt else float("nan"),
            "tau0_ns": float(tau0 * 1e9) if np.isfinite(tau0) else float("nan"),
            "Te_ns": float(te * 1e9) if np.isfinite(te) else float("nan"),
            "target_in_early": int(in_early),
            "dominant_parity_early": str(m.get("dominant_parity_early", "NA")),
            "XPD_early_db": _safe_float(m.get("XPD_early_db", np.nan)),
            "XPD_early_excess_db": _safe_float(m.get("XPD_early_excess_db", np.nan)),
        }
        status_rows.append(row)
        status_by_link[lid] = {"has_2bounce": bool(has_tgt), "target_in_early": bool(in_early)}

    has_ids = [r["link_id"] for r in status_rows if int(r["has_2bounce"]) == 1]
    no_ids = [r["link_id"] for r in status_rows if int(r["has_2bounce"]) == 0]

    _plot_grid(link_ids, pdp_dir, status_by_link, out_dir / "a3_all_links_grid_db.png", "A3 all links (co/cross PDP, dB)")
    _plot_grid(has_ids, pdp_dir, status_by_link, out_dir / "a3_has_2bounce_grid_db.png", "A3 links with 2-bounce target")
    _plot_grid(no_ids, pdp_dir, status_by_link, out_dir / "a3_no_2bounce_grid_db.png", "A3 links without 2-bounce target")

    out_csv = out_dir / "a3_link_status.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(status_rows[0].keys()) if status_rows else ["link_id"])
        wr.writeheader()
        for r in status_rows:
            wr.writerow(r)

    n = len(status_rows)
    n_has = int(sum(int(r["has_2bounce"]) for r in status_rows))
    n_early = int(sum(int(r["target_in_early"]) for r in status_rows if int(r["has_2bounce"]) == 1))
    n_los = int(sum(int(r["los_ray_count"]) > 0 for r in status_rows))

    summary = {
        "n_links": int(n),
        "n_has_2bounce": int(n_has),
        "has_2bounce_rate": float(n_has / n) if n > 0 else float("nan"),
        "n_target_in_early_given_has": int(n_early),
        "target_in_early_rate_given_has": float(n_early / n_has) if n_has > 0 else float("nan"),
        "n_links_with_los_ray": int(n_los),
        "plots": {
            "all": str(out_dir / "a3_all_links_grid_db.png"),
            "has_2bounce": str(out_dir / "a3_has_2bounce_grid_db.png"),
            "no_2bounce": str(out_dir / "a3_no_2bounce_grid_db.png"),
        },
        "status_csv": str(out_csv),
    }

    out_json = out_dir / "a3_diagnostic_report.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    out_md = out_dir / "a3_diagnostic_report.md"
    lines = [
        "# A3 Automatic Diagnostic Report",
        "",
        f"- n_links: {summary['n_links']}",
        f"- n_has_2bounce: {summary['n_has_2bounce']} ({summary['has_2bounce_rate']:.3f})",
        f"- target_in_early_rate_given_has: {summary['target_in_early_rate_given_has']:.3f}",
        f"- links_with_LOS_ray: {summary['n_links_with_los_ray']}",
        "",
        "## Artifacts",
        f"- status_csv: {out_csv}",
        f"- all_grid: {out_dir / 'a3_all_links_grid_db.png'}",
        f"- has_2bounce_grid: {out_dir / 'a3_has_2bounce_grid_db.png'}",
        f"- no_2bounce_grid: {out_dir / 'a3_no_2bounce_grid_db.png'}",
        "",
        "## Note",
        "- Viable-subset interpretation is allowed: report both full-set rate and target-existing subset rate.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out_md)
    print(out_json)


if __name__ == "__main__":
    main()

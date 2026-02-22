"""Sweep parity hold/collapse map for rich multipath settings.

This script runs a bounded grid of runner configurations, then computes
odd/even parity separation statistics in circular basis using propagation-only
matrices (J). It writes:
  - parity_collapse_map.json
  - P26_parity_hold_collapse_heatmap.png/pdf
  - P27_parity_subband_delta_heatmap.png/pdf
  - P28_collapse_probability_vs_roughness.png/pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Allow running as "python scripts/..." from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.xpd_stats import make_subbands, pathwise_xpd
from rt_io.hdf5_io import load_rt_dataset


def _run_case(
    python: str,
    out_h5: Path,
    out_report: Path,
    out_plots: Path,
    *,
    scenario_ids: str,
    nf: int,
    max_bounce: int,
    diffuse_factor: float,
    roughness_sigma_mm: float,
    dispersion: str,
    seed: int,
    materials_db: str | None,
) -> None:
    cmd = [
        python,
        "-m",
        "scenarios.runner",
        "--no-model-compare",
        "--basis",
        "circular",
        "--xpd-matrix-source",
        "J",
        "--physics-validation-mode",
        "--scenario-ids",
        str(scenario_ids),
        "--nf",
        str(int(nf)),
        "--max-bounce-override",
        str(int(max_bounce)),
        "--diffuse",
        "on" if float(diffuse_factor) > 0.0 else "off",
        "--diffuse-factor",
        str(float(diffuse_factor)),
        "--diffuse-model",
        "directive",
        "--diffuse-lobe-alpha",
        "8.0",
        "--diffuse-rays-per-hit",
        "2",
        "--diffuse-seed",
        str(int(seed)),
        "--material-dispersion",
        str(dispersion),
        "--output",
        str(out_h5),
        "--plots-dir",
        str(out_plots),
        "--report",
        str(out_report),
    ]
    if materials_db:
        cmd.extend(["--materials-db", str(materials_db)])
    # roughness is currently a tagged parameter for map bookkeeping;
    # diffuse model uses diffuse_factor as the active stress parameter.
    _ = roughness_sigma_mm
    subprocess.run(cmd, check=True)


def _collect_parity_stats(
    ds: dict[str, Any],
    matrix_source: str = "J",
    min_n: int = 20,
    num_subbands: int = 4,
) -> dict[str, Any]:
    all_paths: list[dict[str, Any]] = []
    for sc in ds.get("scenarios", {}).values():
        for case in sc.get("cases", {}).values():
            all_paths.extend(case.get("paths", []))

    samples = pathwise_xpd(
        all_paths,
        matrix_source=matrix_source,
        input_basis="circular",
        eval_basis="circular",
        convention=str(ds.get("meta", {}).get("convention", "IEEE-RHCP")),
        power_floor=1e-12,
    )
    odd = np.asarray([float(s["xpd_db"]) for s in samples if str(s.get("parity")) == "odd"], dtype=float)
    even = np.asarray([float(s["xpd_db"]) for s in samples if str(s.get("parity")) == "even"], dtype=float)
    n_odd = int(len(odd))
    n_even = int(len(even))
    if n_odd >= 2 and n_even >= 2:
        ks = stats.ks_2samp(odd, even, alternative="two-sided", method="auto")
        ks_p = float(ks.pvalue)
    else:
        ks_p = float("nan")
    mu_odd = float(np.nanmean(odd)) if n_odd > 0 else float("nan")
    mu_even = float(np.nanmean(even)) if n_even > 0 else float("nan")
    sg_odd = float(np.nanstd(odd, ddof=1)) if n_odd > 1 else float("nan")
    sg_even = float(np.nanstd(even, ddof=1)) if n_even > 1 else float("nan")

    subbands = make_subbands(len(np.asarray(ds.get("frequency", []), dtype=float)), max(1, int(num_subbands)))
    sb_samples = pathwise_xpd(
        all_paths,
        matrix_source=matrix_source,
        input_basis="circular",
        eval_basis="circular",
        convention=str(ds.get("meta", {}).get("convention", "IEEE-RHCP")),
        power_floor=1e-12,
        subbands=subbands,
    )
    sb_delta: list[float] = []
    for b in range(len(subbands)):
        odd_b = [float(s["xpd_db"]) for s in sb_samples if int(s.get("subband", -1)) == b and str(s.get("parity")) == "odd"]
        even_b = [float(s["xpd_db"]) for s in sb_samples if int(s.get("subband", -1)) == b and str(s.get("parity")) == "even"]
        if len(odd_b) and len(even_b):
            sb_delta.append(float(abs(np.mean(even_b) - np.mean(odd_b))))
        else:
            sb_delta.append(float("nan"))

    return {
        "n_odd": n_odd,
        "n_even": n_even,
        "mu_odd_db": mu_odd,
        "mu_even_db": mu_even,
        "sigma_odd_db": sg_odd,
        "sigma_even_db": sg_even,
        "delta_mu_abs_db": float(abs(mu_even - mu_odd)) if np.isfinite(mu_even) and np.isfinite(mu_odd) else float("nan"),
        "ks_pvalue": ks_p,
        "sufficient_n": bool(n_odd >= int(min_n) and n_even >= int(min_n)),
        "subband_delta_mu_abs_db": sb_delta,
    }


def _label_hold_collapse(stats_row: dict[str, Any], threshold_db: float, p_th: float) -> str:
    if not bool(stats_row.get("sufficient_n", False)):
        return "INSUFFICIENT"
    dmu = float(stats_row.get("delta_mu_abs_db", float("nan")))
    ks_p = float(stats_row.get("ks_pvalue", float("nan")))
    if np.isfinite(dmu) and np.isfinite(ks_p) and dmu >= float(threshold_db) and ks_p <= float(p_th):
        return "HOLD"
    return "COLLAPSE"


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=180)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def _plot_p26(rows: list[dict[str, Any]], out_dir: Path, roughness_values: list[float], s_values: list[float], bounce_values: list[int]) -> None:
    code = {"INSUFFICIENT": -1.0, "COLLAPSE": 0.0, "HOLD": 1.0}
    fig, axs = plt.subplots(1, len(roughness_values), figsize=(4.8 * max(1, len(roughness_values)), 4), squeeze=False)
    for i, rough in enumerate(roughness_values):
        mat = np.full((len(s_values), len(bounce_values)), np.nan, dtype=float)
        for r in rows:
            if float(r["roughness_sigma_mm"]) != float(rough):
                continue
            si = s_values.index(float(r["diffuse_factor"]))
            bi = bounce_values.index(int(r["max_bounce"]))
            mat[si, bi] = code.get(str(r["label"]), np.nan)
        ax = axs[0, i]
        im = ax.imshow(mat, origin="lower", aspect="auto", vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_title(f"P26 roughness={rough:.2f} mm")
        ax.set_xticks(np.arange(len(bounce_values)))
        ax.set_xticklabels([str(b) for b in bounce_values])
        ax.set_yticks(np.arange(len(s_values)))
        ax.set_yticklabels([f"{s:.2f}" for s in s_values])
        ax.set_xlabel("max_bounce")
        ax.set_ylabel("diffuse_factor")
        for si in range(len(s_values)):
            for bi in range(len(bounce_values)):
                v = mat[si, bi]
                if np.isfinite(v):
                    txt = "H" if v > 0.5 else ("C" if v >= -0.5 else "I")
                    ax.text(bi, si, txt, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("label code (I=-1, C=0, H=1)")
    _save_fig(fig, out_dir, "P26_parity_hold_collapse_heatmap")


def _plot_p27(rows: list[dict[str, Any]], out_dir: Path, roughness_values: list[float], num_subbands: int) -> None:
    mat = np.full((len(roughness_values), num_subbands), np.nan, dtype=float)
    for i, rough in enumerate(roughness_values):
        vals = [[] for _ in range(num_subbands)]
        for r in rows:
            if float(r["roughness_sigma_mm"]) != float(rough):
                continue
            sb = r.get("subband_delta_mu_abs_db", [])
            for b in range(min(num_subbands, len(sb))):
                if np.isfinite(float(sb[b])):
                    vals[b].append(float(sb[b]))
        for b in range(num_subbands):
            if len(vals[b]) > 0:
                mat[i, b] = float(np.mean(vals[b]))
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap="magma")
    ax.set_title("P27 |Δmu| vs roughness/subband")
    ax.set_yticks(np.arange(len(roughness_values)))
    ax.set_yticklabels([f"{r:.2f}" for r in roughness_values])
    ax.set_xticks(np.arange(num_subbands))
    ax.set_xticklabels([str(i) for i in range(num_subbands)])
    ax.set_ylabel("roughness_sigma_mm")
    ax.set_xlabel("subband index")
    fig.colorbar(im, ax=ax, label="|Δmu| [dB]")
    _save_fig(fig, out_dir, "P27_parity_subband_delta_heatmap")


def _plot_p28(rows: list[dict[str, Any]], out_dir: Path, roughness_values: list[float]) -> None:
    x = []
    y = []
    for rough in roughness_values:
        rr = [r for r in rows if float(r["roughness_sigma_mm"]) == float(rough) and str(r["label"]) != "INSUFFICIENT"]
        if len(rr) == 0:
            continue
        p_c = float(np.mean([1.0 if str(r["label"]) == "COLLAPSE" else 0.0 for r in rr]))
        x.append(float(rough))
        y.append(p_c)
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(x):
        ax.plot(x, y, "o-")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("roughness_sigma_mm")
    ax.set_ylabel("collapse probability")
    ax.set_title("P28 collapse probability vs roughness")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out_dir, "P28_collapse_probability_vs_roughness")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=str, default="outputs/parity_collapse_map.json")
    parser.add_argument("--out-dir", type=str, default="outputs/plots_parity_map")
    parser.add_argument("--tmp-dir", type=str, default="outputs/parity_map_tmp")
    parser.add_argument("--materials-db", type=str, default="materials/materials_db.json")
    parser.add_argument("--scenario-ids", type=str, default="A2,A3,A6,B0")
    parser.add_argument("--nf", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold-db", type=float, default=3.0)
    parser.add_argument("--p-th", type=float, default=0.05)
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--num-subbands", type=int, default=4)
    parser.add_argument("--max-bounces", type=str, default="2,4,6,8")
    parser.add_argument("--diffuse-factors", type=str, default="0.0,0.2,0.4,0.6")
    parser.add_argument("--roughness-sigma-mm", type=str, default="0.0,0.2,0.5,1.0")
    parser.add_argument("--dispersion-modes", type=str, default="off,on")
    args = parser.parse_args()

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    bounces = [int(x) for x in str(args.max_bounces).split(",") if str(x).strip()]
    s_vals = [float(x) for x in str(args.diffuse_factors).split(",") if str(x).strip()]
    rough_vals = [float(x) for x in str(args.roughness_sigma_mm).split(",") if str(x).strip()]
    disp_vals = [str(x).strip() for x in str(args.dispersion_modes).split(",") if str(x).strip()]
    scenario_ids = str(args.scenario_ids)

    rows: list[dict[str, Any]] = []
    run_id = 0
    for disp in disp_vals:
        for rough in rough_vals:
            for max_b in bounces:
                for s in s_vals:
                    run_id += 1
                    tag = f"d{disp}_r{rough:.2f}_b{max_b}_s{s:.2f}"
                    h5 = tmp_dir / f"{tag}.h5"
                    rep = tmp_dir / f"{tag}.md"
                    plo = tmp_dir / f"{tag}_plots"
                    _run_case(
                        python=sys.executable,
                        out_h5=h5,
                        out_report=rep,
                        out_plots=plo,
                        scenario_ids=scenario_ids,
                        nf=int(args.nf),
                        max_bounce=int(max_b),
                        diffuse_factor=float(s),
                        roughness_sigma_mm=float(rough),
                        dispersion=disp,
                        seed=int(args.seed + run_id),
                        materials_db=args.materials_db,
                    )
                    ds = load_rt_dataset(h5)
                    st = _collect_parity_stats(
                        ds,
                        matrix_source="J",
                        min_n=int(args.min_n),
                        num_subbands=int(args.num_subbands),
                    )
                    row = {
                        "dispersion_mode": disp,
                        "roughness_sigma_mm": float(rough),
                        "max_bounce": int(max_b),
                        "diffuse_factor": float(s),
                        **st,
                    }
                    row["label"] = _label_hold_collapse(row, threshold_db=float(args.threshold_db), p_th=float(args.p_th))
                    rows.append(row)

    summary = {
        "threshold_db": float(args.threshold_db),
        "p_th": float(args.p_th),
        "min_n": int(args.min_n),
        "scenario_ids": scenario_ids,
        "grid": {
            "max_bounces": bounces,
            "diffuse_factors": s_vals,
            "roughness_sigma_mm": rough_vals,
            "dispersion_modes": disp_vals,
        },
        "rows": rows,
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_p26(rows, out_dir=out_dir, roughness_values=rough_vals, s_values=s_vals, bounce_values=bounces)
    _plot_p27(rows, out_dir=out_dir, roughness_values=rough_vals, num_subbands=int(args.num_subbands))
    _plot_p28(rows, out_dir=out_dir, roughness_values=rough_vals)

    n_hold = int(sum(1 for r in rows if str(r.get("label")) == "HOLD"))
    n_col = int(sum(1 for r in rows if str(r.get("label")) == "COLLAPSE"))
    n_suf = int(sum(1 for r in rows if str(r.get("label")) != "INSUFFICIENT"))
    print(f"parity_map rows={len(rows)}, sufficient={n_suf}, HOLD={n_hold}, COLLAPSE={n_col}")


if __name__ == "__main__":
    main()

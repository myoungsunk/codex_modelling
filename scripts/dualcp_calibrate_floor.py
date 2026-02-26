"""Estimate dual-CP XPD floor from LOS/free-space measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.dualcp_calibration import dualcp_xpd_db_from_Hf, estimate_xpd_floor_from_cases
from analysis.measurement_compare import load_measurement_dualcp_two_csv
from analysis.xpd_stats import make_subbands
from rt_io.measurement_hdf5 import iter_measurement_cases


def _load_cases_from_h5(path: str | Path, scenario_id: str) -> list[dict[str, Any]]:
    raw = iter_measurement_cases(path, scenario_id=scenario_id)
    out: list[dict[str, Any]] = []
    for c in raw:
        out.append(
            {
                "scenario_id": str(c.get("scenario_id", scenario_id)),
                "case_id": str(c.get("case_id", "0")),
                "frequency_hz": np.asarray(c["frequency_hz"], dtype=float),
                "H_f": np.asarray(c["H_f"], dtype=np.complex128),
            }
        )
    return out


def _load_cases_from_csv_pairs(co_csv: list[str], cross_csv: list[str]) -> list[dict[str, Any]]:
    if len(co_csv) != len(cross_csv):
        raise ValueError("--co-csv and --cross-csv counts must match")
    out: list[dict[str, Any]] = []
    for i, (co, cr) in enumerate(zip(co_csv, cross_csv)):
        m = load_measurement_dualcp_two_csv(co_csv=co, cross_csv=cr)
        out.append(
            {
                "scenario_id": "CSV",
                "case_id": f"csv_{i}",
                "frequency_hz": np.asarray(m.frequency_hz, dtype=float),
                "H_f": np.asarray(m.H_f, dtype=np.complex128),
            }
        )
    return out


def _align(freq_src: np.ndarray, y_src: np.ndarray, freq_dst: np.ndarray) -> np.ndarray:
    return np.interp(
        np.asarray(freq_dst, dtype=float),
        np.asarray(freq_src, dtype=float),
        np.asarray(y_src, dtype=float),
        left=float(y_src[0]),
        right=float(y_src[-1]),
    )


def _save_plots(
    out_dir: str | Path,
    result: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, str]:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    freq = np.asarray(result["frequency_hz"], dtype=float)
    floor_db = np.asarray(result["xpd_floor_db"], dtype=float)
    uncert_db = np.asarray(result["xpd_floor_uncert_db"], dtype=float)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(freq * 1e-9, floor_db, color="tab:blue", linewidth=1.8, label="XPD floor")
    if len(uncert_db) == len(floor_db):
        ax1.fill_between(freq * 1e-9, floor_db - uncert_db, floor_db + uncert_db, color="tab:blue", alpha=0.2, label="uncert")
    ax1.set_xlabel("Frequency [GHz]")
    ax1.set_ylabel("XPD floor [dB]")
    ax1.set_title("Dual-CP XPD Floor Calibration")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    floor_png = p / "dualcp_floor_curve.png"
    fig1.savefig(floor_png, dpi=180)
    plt.close(fig1)

    per_case = []
    medians = []
    for c in cases:
        xpd = dualcp_xpd_db_from_Hf(np.asarray(c["H_f"], dtype=np.complex128))
        xpd_i = _align(np.asarray(c["frequency_hz"], dtype=float), xpd, freq)
        per_case.append(xpd_i)
        medians.append(float(np.median(xpd_i)))

    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(10, 4))
    if per_case:
        ax2.boxplot(np.asarray(per_case, dtype=float).T, showfliers=False)
    ax2.set_title("Per-frequency XPD spread")
    ax2.set_xlabel("Frequency bin")
    ax2.set_ylabel("XPD [dB]")
    ax2.grid(True, alpha=0.3)

    flat = np.sort(np.asarray(per_case, dtype=float).reshape(-1)) if per_case else np.asarray([], dtype=float)
    if len(flat):
        cdf = np.arange(1, len(flat) + 1, dtype=float) / float(len(flat))
        ax3.plot(flat, cdf, color="tab:green", linewidth=1.8)
    ax3.set_title("XPD CDF (all LOS samples)")
    ax3.set_xlabel("XPD [dB]")
    ax3.set_ylabel("CDF")
    ax3.grid(True, alpha=0.3)
    fig2.tight_layout()
    summary_png = p / "dualcp_floor_summary.png"
    fig2.savefig(summary_png, dpi=180)
    plt.close(fig2)

    return {
        "floor_curve_png": str(floor_png),
        "summary_png": str(summary_png),
        "case_median_xpd_db": ",".join(f"{m:.3f}" for m in medians),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-h5", type=str, default=None)
    parser.add_argument("--scenario-id", type=str, default="C0")
    parser.add_argument("--co-csv", action="append", default=[])
    parser.add_argument("--cross-csv", action="append", default=[])
    parser.add_argument("--out-json", type=str, default="outputs/calibration_floor.json")
    parser.add_argument("--plots-dir", type=str, default="outputs/plots")
    parser.add_argument("--num-subbands", type=int, default=4)
    parser.add_argument("--percentile-low", type=float, default=5.0)
    parser.add_argument("--percentile-high", type=float, default=95.0)
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    if args.measurement_h5:
        cases.extend(_load_cases_from_h5(args.measurement_h5, scenario_id=args.scenario_id))
    if args.co_csv or args.cross_csv:
        cases.extend(_load_cases_from_csv_pairs(list(args.co_csv), list(args.cross_csv)))
    if not cases:
        raise SystemExit("No input cases. Use --measurement-h5 or --co-csv/--cross-csv.")

    nf = len(np.asarray(cases[0]["frequency_hz"], dtype=float))
    subbands = make_subbands(nf, max(1, int(args.num_subbands)))
    result = estimate_xpd_floor_from_cases(
        cases=cases,
        method="median",
        subbands=subbands,
        percentiles=(float(args.percentile_low), float(args.percentile_high)),
        alignment_sweep={"scenario_id": str(args.scenario_id)},
    )
    result["inputs"] = {
        "measurement_h5": str(args.measurement_h5) if args.measurement_h5 else "",
        "scenario_id": str(args.scenario_id),
        "n_cases": int(len(cases)),
    }

    plots = _save_plots(args.plots_dir, result=result, cases=cases)
    result["plots"] = plots

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()

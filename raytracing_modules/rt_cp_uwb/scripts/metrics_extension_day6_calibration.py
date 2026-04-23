from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from metrics_extension_day6_immediate import (
    ROOT,
    bootstrap_sample,
    bootstrap_scheme,
    load_csv,
    normalize_oof,
    resolve_input,
    stage2_subset_definitions,
)


OUT_DIR = ROOT / "results" / "metrics_extension" / "day6_calibration"
BOOTSTRAP_N = 2000
GLOBAL_SEED = 20260423 + 600


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def clip_probs(p: np.ndarray | pd.Series, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    return np.clip(arr, eps, 1.0 - eps)


def brier_score(y: pd.Series, p: pd.Series) -> float:
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    return float(np.mean((p_arr - y_arr) ** 2))


def prevalence_brier(y: pd.Series) -> float:
    prev = float(np.mean(np.asarray(y, dtype=float)))
    return float(np.mean((np.asarray(y, dtype=float) - prev) ** 2))


def brier_skill(y: pd.Series, p: pd.Series) -> float:
    base = prevalence_brier(y)
    if base <= 0:
        return np.nan
    return float(1.0 - brier_score(y, p) / base)


def percentile_ci(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if not pd.isna(v)], dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def subset_stability(n_total: int, n_pos: int, n_neg: int) -> str:
    if n_total >= 100 and n_pos >= 30 and n_neg >= 30:
        return "stable"
    if n_total >= 40 and n_pos >= 15 and n_neg >= 15:
        return "unstable_small_subset"
    return "insufficient"


def bootstrap_brier_metrics(df: pd.DataFrame, score_col: str, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    strata = bootstrap_scheme(df)
    brier_dist: list[float] = []
    skill_dist: list[float] = []
    for _ in range(BOOTSTRAP_N):
        sample = bootstrap_sample(df, rng, strata)
        brier_dist.append(brier_score(sample["y_true"], sample[score_col]))
        skill_dist.append(brier_skill(sample["y_true"], sample[score_col]))
    brier_lo, brier_hi = percentile_ci(brier_dist)
    skill_lo, skill_hi = percentile_ci(skill_dist)
    return {
        "brier_ci_low": brier_lo,
        "brier_ci_high": brier_hi,
        "brier_skill_ci_low": skill_lo,
        "brier_skill_ci_high": skill_hi,
    }


def build_equal_mass_bins(y: pd.Series, p: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y.astype(int), "score": p.astype(float)}).copy()
    df["score_rank"] = df["score"].rank(method="first")
    unique_bins = min(n_bins, len(df))
    df["bin"] = pd.qcut(df["score_rank"], q=unique_bins, labels=False, duplicates="drop")
    grouped = (
        df.groupby("bin", dropna=False)
        .agg(n_bin=("score", "size"), pred_mean=("score", "mean"), observed_rate=("y_true", "mean"), score_min=("score", "min"), score_max=("score", "max"))
        .reset_index()
    )
    grouped["binning"] = "equal_mass_10"
    grouped["bin_index"] = np.arange(1, len(grouped) + 1)
    return grouped


def build_equal_width_bins(y: pd.Series, p: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y.astype(int), "score": p.astype(float)}).copy()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin"] = pd.cut(df["score"], bins=bins, labels=False, include_lowest=True, right=True)
    grouped = (
        df.groupby("bin", dropna=False)
        .agg(n_bin=("score", "size"), pred_mean=("score", "mean"), observed_rate=("y_true", "mean"), score_min=("score", "min"), score_max=("score", "max"))
        .reset_index()
    )
    grouped = grouped[grouped["n_bin"] > 0].copy()
    grouped["binning"] = "equal_width_10"
    grouped["bin_index"] = np.arange(1, len(grouped) + 1)
    return grouped


def compute_ece(bin_df: pd.DataFrame, n_total: int) -> float:
    if n_total == 0 or bin_df.empty:
        return np.nan
    return float(np.sum(np.abs(bin_df["observed_rate"] - bin_df["pred_mean"]) * bin_df["n_bin"] / n_total))


def calibration_bins(df: pd.DataFrame, score_col: str, score_name: str, subset_name: str, stage: str) -> tuple[pd.DataFrame, dict[str, float]]:
    equal_mass = build_equal_mass_bins(df["y_true"], df[score_col], 10)
    equal_width = build_equal_width_bins(df["y_true"], df[score_col], 10)
    stability = subset_stability(len(df), int(df["y_true"].sum()), int(len(df) - df["y_true"].sum()))
    for frame in (equal_mass, equal_width):
        frame["stage"] = stage
        frame["subset_name"] = subset_name
        frame["score_name"] = score_name
        frame["subset_stability"] = stability
    metrics = {
        "ece_equal_mass_10": compute_ece(equal_mass, len(df)),
        "ece_equal_width_10": compute_ece(equal_width, len(df)),
    }
    return pd.concat([equal_mass, equal_width], ignore_index=True), metrics


def fit_calibration_model(y: pd.Series, p: pd.Series) -> tuple[float, float, str]:
    y_arr = np.asarray(y, dtype=float)
    if pd.Series(y_arr).nunique() < 2:
        return np.nan, np.nan, "single_class"
    x = np.log(clip_probs(p) / (1.0 - clip_probs(p)))

    def objective(params: np.ndarray) -> float:
        intercept, slope = params
        pred = sigmoid(intercept + slope * x)
        pred = np.clip(pred, 1e-9, 1.0 - 1e-9)
        return float(-np.sum(y_arr * np.log(pred) + (1.0 - y_arr) * np.log(1.0 - pred)))

    methods = ["BFGS", "L-BFGS-B", "Powell", "Nelder-Mead"]
    best_result = None
    for method in methods:
        try:
            result = minimize(objective, x0=np.array([0.0, 1.0]), method=method)
        except Exception:
            continue
        if result.success and np.all(np.isfinite(result.x)):
            return float(result.x[0]), float(result.x[1]), f"ok_{method}"
        if best_result is None or (np.isfinite(result.fun) and result.fun < best_result.fun):
            best_result = result
    if best_result is not None and np.all(np.isfinite(best_result.x)):
        return float(best_result.x[0]), float(best_result.x[1]), "fallback_best_effort"
    return np.nan, np.nan, "optimization_failed"


def calibration_row(df: pd.DataFrame, stage: str, subset_name: str, condition: str, seed: int) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    n_total = int(len(df))
    n_pos = int(df["y_true"].sum())
    n_neg = n_total - n_pos
    prevalence = float(df["y_true"].mean()) if n_total else np.nan
    row: dict[str, object] = {
        "stage": stage,
        "subset_name": subset_name,
        "condition": condition,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": prevalence,
        "subset_stability": subset_stability(n_total, n_pos, n_neg),
    }
    cir_bins, cir_ece = calibration_bins(df, "cir_score", "CIR", subset_name, stage)
    joint_bins, joint_ece = calibration_bins(df, "joint_score", "Joint", subset_name, stage)
    row["brier_cir"] = brier_score(df["y_true"], df["cir_score"])
    row["brier_joint"] = brier_score(df["y_true"], df["joint_score"])
    row["brier_skill_cir"] = brier_skill(df["y_true"], df["cir_score"])
    row["brier_skill_joint"] = brier_skill(df["y_true"], df["joint_score"])
    row.update({f"{k}_cir": v for k, v in bootstrap_brier_metrics(df, "cir_score", seed).items()})
    row.update({f"{k}_joint": v for k, v in bootstrap_brier_metrics(df, "joint_score", seed + 50).items()})
    row["ece_equal_mass_10_cir"] = cir_ece["ece_equal_mass_10"]
    row["ece_equal_mass_10_joint"] = joint_ece["ece_equal_mass_10"]
    row["ece_equal_width_10_cir"] = cir_ece["ece_equal_width_10"]
    row["ece_equal_width_10_joint"] = joint_ece["ece_equal_width_10"]
    cir_intercept, cir_slope, cir_status = fit_calibration_model(df["y_true"], df["cir_score"])
    joint_intercept, joint_slope, joint_status = fit_calibration_model(df["y_true"], df["joint_score"])
    row["calibration_intercept_cir"] = cir_intercept
    row["calibration_slope_cir"] = cir_slope
    row["calibration_status_cir"] = cir_status
    row["calibration_intercept_joint"] = joint_intercept
    row["calibration_slope_joint"] = joint_slope
    row["calibration_status_joint"] = joint_status
    return row, cir_bins, joint_bins


def plot_brier_ci(metrics_df: pd.DataFrame, out_path: Path) -> None:
    subset_order = ["Stage1_ALL", "Stage2_ALL", "Stage2_room_A", "Stage2_room_B", "Stage2_room_C", "Stage2_low_confidence", "Stage2_M1", "Stage2_C4"]
    plot_df = metrics_df[metrics_df["subset_name"].isin(subset_order)].copy()
    plot_df["subset_name"] = pd.Categorical(plot_df["subset_name"], categories=subset_order, ordered=True)
    plot_df = plot_df.sort_values("subset_name")
    x = np.arange(len(plot_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, plot_df["brier_cir"], width, color="#4c78a8", label="CIR")
    ax.bar(x + width / 2, plot_df["brier_joint"], width, color="#f58518", label="Joint")
    ax.errorbar(
        x - width / 2,
        plot_df["brier_cir"],
        yerr=[plot_df["brier_cir"] - plot_df["brier_ci_low_cir"], plot_df["brier_ci_high_cir"] - plot_df["brier_cir"]],
        fmt="none",
        color="black",
        capsize=3,
    )
    ax.errorbar(
        x + width / 2,
        plot_df["brier_joint"],
        yerr=[plot_df["brier_joint"] - plot_df["brier_ci_low_joint"], plot_df["brier_ci_high_joint"] - plot_df["brier_joint"]],
        fmt="none",
        color="black",
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["subset_name"], rotation=35, ha="right")
    ax.set_ylabel("Brier score")
    ax.set_title("Brier score with bootstrap CI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_ece(metrics_df: pd.DataFrame, out_path: Path) -> None:
    subset_order = ["Stage1_ALL", "Stage2_ALL", "Stage2_room_A", "Stage2_room_B", "Stage2_room_C", "Stage2_low_confidence", "Stage2_M1", "Stage2_C4"]
    plot_df = metrics_df[metrics_df["subset_name"].isin(subset_order)].copy()
    plot_df["subset_name"] = pd.Categorical(plot_df["subset_name"], categories=subset_order, ordered=True)
    plot_df = plot_df.sort_values("subset_name")
    x = np.arange(len(plot_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, plot_df["ece_equal_mass_10_cir"], width, color="#4c78a8", label="CIR equal-mass")
    ax.bar(x + width / 2, plot_df["ece_equal_mass_10_joint"], width, color="#f58518", label="Joint equal-mass")
    ax.scatter(x - width / 2, plot_df["ece_equal_width_10_cir"], color="#1f4e79", marker="x")
    ax.scatter(x + width / 2, plot_df["ece_equal_width_10_joint"], color="#a64b00", marker="x")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["subset_name"], rotation=35, ha="right")
    ax.set_ylabel("ECE")
    ax.set_title("ECE by subset (bars: equal-mass, x: equal-width)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_reliability(cir_bins_df: pd.DataFrame, joint_bins_df: pd.DataFrame, out_path: Path) -> None:
    subset_order = ["Stage1_ALL", "Stage2_ALL", "Stage2_room_C", "Stage2_low_confidence", "Stage2_M1", "Stage2_C4"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for ax, subset_name in zip(axes.ravel(), subset_order):
        cir = cir_bins_df[(cir_bins_df["subset_name"] == subset_name) & (cir_bins_df["binning"] == "equal_mass_10")].copy()
        joint = joint_bins_df[(joint_bins_df["subset_name"] == subset_name) & (joint_bins_df["binning"] == "equal_mass_10")].copy()
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.plot(cir["pred_mean"], cir["observed_rate"], marker="o", color="#4c78a8", label="CIR")
        ax.plot(joint["pred_mean"], joint["observed_rate"], marker="o", color="#f58518", label="Joint")
        stability = "stable"
        if not joint.empty:
            stability = str(joint["subset_stability"].iloc[0])
        ax.set_title(f"{subset_name}\n{stability}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0, 0].legend()
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed NLOS rate")
    for ax in axes[-1, :]:
        ax.set_xlabel("Predicted mean")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_slope_intercept(metrics_df: pd.DataFrame, out_path: Path) -> None:
    subset_order = ["Stage1_ALL", "Stage2_ALL", "Stage2_room_A", "Stage2_room_B", "Stage2_room_C", "Stage2_low_confidence", "Stage2_M1", "Stage2_C4"]
    plot_df = metrics_df[metrics_df["subset_name"].isin(subset_order)].copy()
    plot_df["subset_name"] = pd.Categorical(plot_df["subset_name"], categories=subset_order, ordered=True)
    plot_df = plot_df.sort_values("subset_name")
    x = np.arange(len(plot_df))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].plot(x, plot_df["calibration_slope_cir"], marker="o", color="#4c78a8", label="CIR")
    axes[0].plot(x, plot_df["calibration_slope_joint"], marker="o", color="#f58518", label="Joint")
    axes[0].set_title("Calibration slope")
    axes[0].set_ylabel("Slope")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].plot(x, plot_df["calibration_intercept_cir"], marker="o", color="#4c78a8", label="CIR")
    axes[1].plot(x, plot_df["calibration_intercept_joint"], marker="o", color="#f58518", label="Joint")
    axes[1].set_title("Calibration intercept")
    axes[1].set_ylabel("Intercept")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["subset_name"], rotation=35, ha="right")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def missing_inputs_rows(stage1_df: pd.DataFrame, stage2_df: pd.DataFrame) -> list[dict[str, str]]:
    required = ["y_true", "cir_score", "joint_score", "room_type", "dominant_wall_material", "k_factor_estimate", "fp_to_total_ratio"]
    rows = []
    for name, df in [("Stage1_OOF", stage1_df), ("Stage2_OOF", stage2_df)]:
        missing = [col for col in required if col not in df.columns]
        rows.append(
            {
                "input_name": name,
                "status": "OK" if not missing else "MISSING_COLUMNS",
                "details": "None" if not missing else ", ".join(missing),
            }
        )
    rows.append(
        {
            "input_name": "Posthoc_recalibration_model",
            "status": "NOT_USED",
            "details": "Calibration audit uses raw OOF scores only; no posthoc Platt or isotonic recalibration was fitted.",
        }
    )
    return rows


def write_missing_inputs_md(rows: list[dict[str, str]], out_path: Path, resolved_paths: dict[str, Path]) -> None:
    lines = ["# Missing Inputs Calibration", "", "## Resolved Inputs", ""]
    for key, value in resolved_paths.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Availability", ""])
    for row in rows:
        lines.append(f"- `{row['input_name']}` | `{row['status']}`")
        lines.append(f"  details: {row['details']}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resolved_paths = {
        "stage1_oof": resolve_input(
            "results/recovery_mini/day5/requested_material_copies_core/day1_oof_predictions_stage1_copy.csv",
            "/mnt/data/day1_oof_predictions_stage1_copy.csv",
        ),
        "stage2_oof": resolve_input(
            "results/recovery_mini/day5/requested_material_copies_core/day1_oof_predictions_stage2_copy.csv",
            "/mnt/data/day1_oof_predictions_stage2_copy.csv",
        ),
        "day2_candidates": resolve_input(
            "results/recovery_mini/day5/requested_material_copies_core/day2_candidate_regimes_copy.csv",
            "/mnt/data/day2_candidate_regimes_copy.csv",
        ),
        "day4_condition_bins": resolve_input(
            "results/recovery_mini/day5/requested_material_copies/day4_stage2_condition_bins_copy.csv",
            "/mnt/data/day4_stage2_condition_bins_copy.csv",
        ),
    }

    stage1_df = normalize_oof(load_csv(resolved_paths["stage1_oof"]), "Stage1")
    stage2_df = normalize_oof(load_csv(resolved_paths["stage2_oof"]), "Stage2")
    day2_candidates = load_csv(resolved_paths["day2_candidates"])
    day4_bins = load_csv(resolved_paths["day4_condition_bins"])
    subset_defs, _, _ = stage2_subset_definitions(stage2_df, day2_candidates, day4_bins)

    all_specs = [{"stage": "Stage1", "subset_name": "Stage1_ALL", "condition": "All Stage 1 OOF rows", "df": stage1_df.copy()}]
    keep_stage2 = {"Stage2_ALL", "Stage2_low_confidence", "Stage2_M1", "Stage2_C4", "Stage2_room_A", "Stage2_room_B", "Stage2_room_C"}
    all_specs.extend({"stage": "Stage2", **spec} for spec in subset_defs if spec["subset_name"] in keep_stage2)

    calibration_rows: list[dict[str, object]] = []
    cir_bins_all: list[pd.DataFrame] = []
    joint_bins_all: list[pd.DataFrame] = []
    for i, spec in enumerate(all_specs):
        row, cir_bins, joint_bins = calibration_row(spec["df"], spec["stage"], spec["subset_name"], spec["condition"], GLOBAL_SEED + 100 * i)
        calibration_rows.append(row)
        cir_bins_all.append(cir_bins)
        joint_bins_all.append(joint_bins)

    metrics_df = pd.DataFrame(calibration_rows)
    metrics_df.to_csv(OUT_DIR / "calibration_metrics.csv", index=False)
    metrics_df[metrics_df["subset_name"].str.startswith("Stage2_room_")].copy().to_csv(OUT_DIR / "calibration_metrics_by_room.csv", index=False)
    cir_bins_df = pd.concat(cir_bins_all, ignore_index=True)
    joint_bins_df = pd.concat(joint_bins_all, ignore_index=True)
    cir_bins_df.to_csv(OUT_DIR / "reliability_bins_cir.csv", index=False)
    joint_bins_df.to_csv(OUT_DIR / "reliability_bins_joint.csv", index=False)

    missing_rows = missing_inputs_rows(stage1_df, stage2_df)
    write_missing_inputs_md(missing_rows, OUT_DIR / "missing_inputs_calibration.md", resolved_paths)

    plot_brier_ci(metrics_df, OUT_DIR / "plot_brier_ci.png")
    plot_ece(metrics_df, OUT_DIR / "plot_ece_by_subset.png")
    plot_reliability(cir_bins_df, joint_bins_df, OUT_DIR / "plot_reliability_joint_vs_cir.png")
    plot_slope_intercept(metrics_df, OUT_DIR / "plot_calibration_slope_intercept.png")

    lookup = metrics_df.set_index("subset_name")

    def fmt_ci(val: float, lo: float, hi: float) -> str:
        if any(pd.isna(x) for x in [val, lo, hi]):
            return "NA"
        return f"{val:.4f} [{lo:.4f}, {hi:.4f}]"

    stage2_all_joint_brier = lookup.loc["Stage2_ALL", "brier_joint"]
    stage2_all_joint_slope = lookup.loc["Stage2_ALL", "calibration_slope_joint"]
    stage2_all_joint_ece = lookup.loc["Stage2_ALL", "ece_equal_mass_10_joint"]
    stage2_all_cir_brier = lookup.loc["Stage2_ALL", "brier_cir"]
    stage2_all_cir_slope = lookup.loc["Stage2_ALL", "calibration_slope_cir"]

    go_nogo = "NO-GO"
    rationale = "Joint score probability quality is not consistently better than CIR across Stage 2 ALL and the main subset family, so soft weighting should stay supplemental until Stage 3 preflight confirms stability."
    if (
        stage2_all_joint_brier <= stage2_all_cir_brier
        and abs(stage2_all_joint_slope - 1.0) <= abs(stage2_all_cir_slope - 1.0)
        and stage2_all_joint_ece <= lookup.loc["Stage2_ALL", "ece_equal_mass_10_cir"]
    ):
        go_nogo = "GO_WITH_CAUTION"
        rationale = "Joint score is at least as good as CIR on the Stage 2 global calibration metrics, but subset drift still requires caution."

    lines = [
        "# Calibration Summary",
        "",
        f"- Decision: `{go_nogo}`",
        f"- Rationale: {rationale}",
        "",
        "## Core Metrics",
        "",
        f"- Stage1_ALL Joint Brier: `{fmt_ci(lookup.loc['Stage1_ALL', 'brier_joint'], lookup.loc['Stage1_ALL', 'brier_ci_low_joint'], lookup.loc['Stage1_ALL', 'brier_ci_high_joint'])}`",
        f"- Stage2_ALL CIR Brier: `{fmt_ci(lookup.loc['Stage2_ALL', 'brier_cir'], lookup.loc['Stage2_ALL', 'brier_ci_low_cir'], lookup.loc['Stage2_ALL', 'brier_ci_high_cir'])}`",
        f"- Stage2_ALL Joint Brier: `{fmt_ci(lookup.loc['Stage2_ALL', 'brier_joint'], lookup.loc['Stage2_ALL', 'brier_ci_low_joint'], lookup.loc['Stage2_ALL', 'brier_ci_high_joint'])}`",
        f"- Stage2_ALL CIR Brier skill: `{fmt_ci(lookup.loc['Stage2_ALL', 'brier_skill_cir'], lookup.loc['Stage2_ALL', 'brier_skill_ci_low_cir'], lookup.loc['Stage2_ALL', 'brier_skill_ci_high_cir'])}`",
        f"- Stage2_ALL Joint Brier skill: `{fmt_ci(lookup.loc['Stage2_ALL', 'brier_skill_joint'], lookup.loc['Stage2_ALL', 'brier_skill_ci_low_joint'], lookup.loc['Stage2_ALL', 'brier_skill_ci_high_joint'])}`",
        f"- Stage2_ALL CIR ECE equal-mass: `{lookup.loc['Stage2_ALL', 'ece_equal_mass_10_cir']:.4f}`",
        f"- Stage2_ALL Joint ECE equal-mass: `{lookup.loc['Stage2_ALL', 'ece_equal_mass_10_joint']:.4f}`",
        f"- Stage2_ALL CIR slope/intercept: `{lookup.loc['Stage2_ALL', 'calibration_slope_cir']:.3f}` / `{lookup.loc['Stage2_ALL', 'calibration_intercept_cir']:.3f}`",
        f"- Stage2_ALL Joint slope/intercept: `{lookup.loc['Stage2_ALL', 'calibration_slope_joint']:.3f}` / `{lookup.loc['Stage2_ALL', 'calibration_intercept_joint']:.3f}`",
        "",
        "## Priority Subsets",
        "",
        f"- Ambiguity [0.4,0.6] Joint Brier: `{fmt_ci(lookup.loc['Stage2_low_confidence', 'brier_joint'], lookup.loc['Stage2_low_confidence', 'brier_ci_low_joint'], lookup.loc['Stage2_low_confidence', 'brier_ci_high_joint'])}`; slope `{lookup.loc['Stage2_low_confidence', 'calibration_slope_joint']:.3f}`; stability `{lookup.loc['Stage2_low_confidence', 'subset_stability']}`.",
        f"- M1 Joint Brier: `{fmt_ci(lookup.loc['Stage2_M1', 'brier_joint'], lookup.loc['Stage2_M1', 'brier_ci_low_joint'], lookup.loc['Stage2_M1', 'brier_ci_high_joint'])}`; slope `{lookup.loc['Stage2_M1', 'calibration_slope_joint']:.3f}`; stability `{lookup.loc['Stage2_M1', 'subset_stability']}`.",
        f"- C4 Joint Brier: `{fmt_ci(lookup.loc['Stage2_C4', 'brier_joint'], lookup.loc['Stage2_C4', 'brier_ci_low_joint'], lookup.loc['Stage2_C4', 'brier_ci_high_joint'])}`; slope `{lookup.loc['Stage2_C4', 'calibration_slope_joint']:.3f}`; stability `{lookup.loc['Stage2_C4', 'subset_stability']}`.",
        "",
        "## Interpretation",
        "",
        "1. Brier is the primary probability-quality metric here. ECE is only a supporting diagnostic.",
        "2. If Joint improves discrimination but not Brier or slope/intercept, it should not yet be used as a reliability weight.",
        "3. If subset gains are confined to unstable pockets such as C4, they stay supplemental.",
        "4. For downstream weighting or covariance inflation, the Stage 2 global calibration trend matters more than a single positive regime.",
        "",
        f"## Final Conclusion\n\n- soft score use as weight/reliability score: `{go_nogo}`\n- if `NO-GO`, postpone to after Stage 3 preflight: `{'yes' if go_nogo == 'NO-GO' else 'conditional'}`",
    ]
    (OUT_DIR / "calibration_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

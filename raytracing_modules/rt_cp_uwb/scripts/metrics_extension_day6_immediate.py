from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "metrics_extension" / "day6_immediate"
BOOTSTRAP_N = 2000
GLOBAL_SEED = 20260423


def resolve_input(relative_repo_path: str, fallback_path: str) -> Path:
    repo_path = ROOT / relative_repo_path
    if repo_path.exists():
        return repo_path
    fallback = Path(fallback_path)
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing input: {relative_repo_path} and {fallback_path}")


def parse_interval(label: str) -> tuple[float, float] | None:
    if not isinstance(label, str):
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", label.replace("E", "e"))
    if len(nums) < 2:
        return None
    return float(nums[0]), float(nums[1])


def percentile_ci(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if not pd.isna(v)], dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def sanitize_numeric(df: pd.DataFrame, skip: set[str] | None = None) -> pd.DataFrame:
    skip = skip or set()
    out = df.copy()
    for col in out.columns:
        if col in skip:
            continue
        if out[col].dtype == object:
            try:
                out[col] = pd.to_numeric(out[col])
            except (TypeError, ValueError):
                pass
    return out


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def canonical_feature_groups(feature_list_path: Path) -> tuple[list[str], list[str], list[str]]:
    feature_df = pd.read_csv(feature_list_path)
    features = feature_df.iloc[:, 0].dropna().astype(str).tolist()
    cp_features = features[:6]
    cir_features = features[6:]
    return features, cp_features, cir_features


def normalize_oof(df: pd.DataFrame, stage_name: str) -> pd.DataFrame:
    out = df.copy()
    skip = {
        "dataset",
        "label_name",
        "antenna_type",
        "material_name",
        "slab_placement",
        "room_type",
        "grid_layer",
        "dominant_wall_material",
        "label_component",
    }
    out = sanitize_numeric(out, skip)
    out["stage"] = stage_name
    if "cp_save_0p5" in out.columns:
        out["cp_save"] = out["cp_save_0p5"].astype(float)
    if "cp_harm_0p5" in out.columns:
        out["cp_harm"] = out["cp_harm_0p5"].astype(float)
    if "cir_correct_0p5" in out.columns:
        out["cir_correct"] = out["cir_correct_0p5"].astype(float)
    else:
        out["cir_correct"] = ((out["cir_score"] >= 0.5).astype(int) == out["y_true"].astype(int)).astype(float)
    out["joint_correct"] = ((out["joint_score"] >= 0.5).astype(int) == out["y_true"].astype(int)).astype(float)
    out["low_conf_cir"] = ((out["cir_score"] >= 0.4) & (out["cir_score"] <= 0.6)).astype(int)
    out["ambiguity_wide"] = ((out["cir_score"] >= 0.3) & (out["cir_score"] <= 0.7)).astype(int)
    return out


def normalize_day4_per_case(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    skip = {
        "room_type",
        "grid_layer",
        "dominant_wall_material",
        "label_rule",
        "cv_scheme",
        "expected_candidate_id",
        "expected_condition",
        "label_component",
    }
    out = sanitize_numeric(out, skip)
    out["stage"] = "Stage2_Day4Mini"
    out["cp_save"] = out["cp_save"].astype(float)
    out["cp_harm"] = out["cp_harm"].astype(float)
    out["cir_correct"] = ((out["cir_score"] >= 0.5).astype(int) == out["y_true"].astype(int)).astype(float)
    out["joint_correct"] = ((out["joint_score"] >= 0.5).astype(int) == out["y_true"].astype(int)).astype(float)
    out["low_conf_cir"] = out["low_conf_cir"].astype(float)
    out["ambiguity_wide"] = ((out["cir_score"] >= 0.3) & (out["cir_score"] <= 0.7)).astype(int)
    return out


def derive_label_component(df: pd.DataFrame) -> pd.Series:
    if {"is_nlos_geo", "is_nlos_bounce_0p33"}.issubset(df.columns):
        geo = df["is_nlos_geo"].fillna(0).astype(int)
        bounce = df["is_nlos_bounce_0p33"].fillna(0).astype(int)
        labels = np.where(
            (geo == 1) & (bounce == 1),
            "geo_and_bounce",
            np.where(
                geo == 1,
                "geo_only",
                np.where(bounce == 1, "bounce_only", "negative"),
            ),
        )
        return pd.Series(labels, index=df.index)
    if "label_component" in df.columns:
        return df["label_component"].fillna("unknown").astype(str)
    return pd.Series(["unknown"] * len(df), index=df.index)


def safe_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = pd.Series(y_true).astype(int)
    if y.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y, pd.Series(score).astype(float)))


def safe_pr_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = pd.Series(y_true).astype(int)
    if y.sum() == 0:
        return np.nan
    return float(average_precision_score(y, pd.Series(score).astype(float)))


def recall_at_fixed_fpr(y_true: pd.Series, score: pd.Series, targets: list[float]) -> dict[str, float]:
    y = pd.Series(y_true).astype(int)
    if y.nunique() < 2:
        return {f"recall_at_fpr_{int(t * 100)}": np.nan for t in targets}
    fpr, tpr, _ = roc_curve(y, pd.Series(score).astype(float))
    out = {}
    for target in targets:
        valid = tpr[fpr <= target]
        out[f"recall_at_fpr_{int(target * 100)}"] = float(valid.max()) if valid.size else np.nan
    return out


def fpr_at_fixed_recall(y_true: pd.Series, score: pd.Series, targets: list[float]) -> dict[str, float]:
    y = pd.Series(y_true).astype(int)
    if y.nunique() < 2:
        return {f"los_fpr_at_recall_{int(t * 100)}": np.nan for t in targets}
    fpr, tpr, _ = roc_curve(y, pd.Series(score).astype(float))
    out = {}
    for target in targets:
        valid = fpr[tpr >= target]
        out[f"los_fpr_at_recall_{int(target * 100)}"] = float(valid.min()) if valid.size else np.nan
    return out


def precision_at_fixed_recall(y_true: pd.Series, score: pd.Series, targets: list[float]) -> dict[str, float]:
    y = pd.Series(y_true).astype(int)
    if y.sum() == 0:
        return {f"precision_at_recall_{int(t * 100)}": np.nan for t in targets}
    precision, recall, _ = precision_recall_curve(y, pd.Series(score).astype(float))
    out = {}
    for target in targets:
        valid = precision[recall >= target]
        out[f"precision_at_recall_{int(target * 100)}"] = float(valid.max()) if valid.size else np.nan
    return out


def bootstrap_sample(df: pd.DataFrame, rng: np.random.Generator, strata: list[str]) -> pd.DataFrame:
    sampled_parts = []
    if strata:
        for _, group in df.groupby(strata, dropna=False):
            if group.empty:
                continue
            idx = rng.choice(group.index.to_numpy(), size=len(group), replace=True)
            sampled_parts.append(df.loc[idx])
    else:
        idx = rng.choice(df.index.to_numpy(), size=len(df), replace=True)
        sampled_parts.append(df.loc[idx])
    return pd.concat(sampled_parts, ignore_index=True) if sampled_parts else df.iloc[0:0].copy()


def bootstrap_delta_auc(df: pd.DataFrame, strata: list[str], seed: int) -> tuple[float, float]:
    if df["y_true"].nunique() < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    dist: list[float] = []
    for _ in range(BOOTSTRAP_N):
        sample = bootstrap_sample(df, rng, strata)
        if sample["y_true"].nunique() < 2:
            continue
        dist.append(safe_auc(sample["y_true"], sample["joint_score"]) - safe_auc(sample["y_true"], sample["cir_score"]))
    return percentile_ci(dist)


def bootstrap_recovery(df: pd.DataFrame, strata: list[str], seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    save_dist: list[float] = []
    harm_dist: list[float] = []
    net_dist: list[float] = []
    fail_rate_dist: list[float] = []
    recover_fail_dist: list[float] = []
    harm_correct_dist: list[float] = []
    for _ in range(BOOTSTRAP_N):
        sample = bootstrap_sample(df, rng, strata)
        cir_correct = sample["cir_correct"].astype(int)
        cir_fail = 1 - cir_correct
        save_rate = float(sample["cp_save"].mean())
        harm_rate = float(sample["cp_harm"].mean())
        save_dist.append(save_rate)
        harm_dist.append(harm_rate)
        net_dist.append(save_rate - harm_rate)
        fail_rate = float(cir_fail.mean())
        fail_rate_dist.append(fail_rate)
        fail_den = int(cir_fail.sum())
        correct_den = int(cir_correct.sum())
        recover_fail_dist.append(float(sample["cp_save"].sum() / fail_den) if fail_den > 0 else np.nan)
        harm_correct_dist.append(float(sample["cp_harm"].sum() / correct_den) if correct_den > 0 else np.nan)
    save_lo, save_hi = percentile_ci(save_dist)
    harm_lo, harm_hi = percentile_ci(harm_dist)
    net_lo, net_hi = percentile_ci(net_dist)
    fail_lo, fail_hi = percentile_ci(fail_rate_dist)
    rec_lo, rec_hi = percentile_ci(recover_fail_dist)
    harmc_lo, harmc_hi = percentile_ci(harm_correct_dist)
    return {
        "cp_save_rate_ci_low": save_lo,
        "cp_save_rate_ci_high": save_hi,
        "cp_harm_rate_ci_low": harm_lo,
        "cp_harm_rate_ci_high": harm_hi,
        "net_recovery_ci_low": net_lo,
        "net_recovery_ci_high": net_hi,
        "cir_fail_rate_ci_low": fail_lo,
        "cir_fail_rate_ci_high": fail_hi,
        "recovery_given_cir_fail_ci_low": rec_lo,
        "recovery_given_cir_fail_ci_high": rec_hi,
        "harm_given_cir_correct_ci_low": harmc_lo,
        "harm_given_cir_correct_ci_high": harmc_hi,
    }


def bootstrap_scheme(df: pd.DataFrame) -> list[str]:
    if "room_type" in df.columns and df["room_type"].notna().any() and df["room_type"].astype(str).nunique() > 1:
        return ["room_type", "y_true"]
    return ["y_true"]


def paired_auc_row(df: pd.DataFrame, stage: str, subset_name: str, condition: str, seed: int) -> dict[str, object]:
    n_pos = int(df["y_true"].sum())
    n_total = int(len(df))
    n_neg = n_total - n_pos
    strata = bootstrap_scheme(df)
    ci_low, ci_high = bootstrap_delta_auc(df, strata, seed)
    auc_cir = safe_auc(df["y_true"], df["cir_score"])
    auc_joint = safe_auc(df["y_true"], df["joint_score"])
    return {
        "stage": stage,
        "subset_name": subset_name,
        "condition": condition,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auc_cir": auc_cir,
        "auc_joint": auc_joint,
        "delta_auc": auc_joint - auc_cir if not np.isnan(auc_cir) and not np.isnan(auc_joint) else np.nan,
        "delta_auc_ci_low": ci_low,
        "delta_auc_ci_high": ci_high,
        "bootstrap_scheme": "x".join(strata),
        "delong_status": "DeLong unavailable",
        "delong_p_value": np.nan,
    }


def pr_fixed_row(df: pd.DataFrame, stage: str, subset_name: str, condition: str) -> dict[str, object]:
    y = df["y_true"]
    n_pos = int(y.sum())
    n_total = int(len(df))
    n_neg = n_total - n_pos
    out = {
        "stage": stage,
        "subset_name": subset_name,
        "condition": condition,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "reliability": "reliable" if n_pos >= 20 and n_neg >= 20 else "unreliable_small_class",
        "pr_auc_cir": safe_pr_auc(y, df["cir_score"]),
        "pr_auc_joint": safe_pr_auc(y, df["joint_score"]),
    }
    for prefix, score_col in [("cir", "cir_score"), ("joint", "joint_score")]:
        for key, value in recall_at_fixed_fpr(y, df[score_col], [0.01, 0.05, 0.10]).items():
            out[f"{key}_{prefix}"] = value
        for key, value in fpr_at_fixed_recall(y, df[score_col], [0.80, 0.90, 0.95]).items():
            out[f"{key}_{prefix}"] = value
        for key, value in precision_at_fixed_recall(y, df[score_col], [0.80, 0.90]).items():
            out[f"{key}_{prefix}"] = value
    return out


def confidence_bin_rows(df: pd.DataFrame, stage: str, seed_base: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    bin_edges = np.round(np.linspace(0.0, 1.0, 11), 1)
    for i in range(10):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i < 9:
            sub = df[(df["cir_score"] >= left) & (df["cir_score"] < right)].copy()
        else:
            sub = df[(df["cir_score"] >= left) & (df["cir_score"] <= right)].copy()
        rows.append(confidence_row(sub, stage, "decile", f"[{left:.1f},{right:.1f}]", seed_base + i))
    rows.append(confidence_row(df[(df["cir_score"] >= 0.4) & (df["cir_score"] <= 0.6)].copy(), stage, "ambiguity", "[0.4,0.6]", seed_base + 200))
    rows.append(confidence_row(df[(df["cir_score"] >= 0.3) & (df["cir_score"] <= 0.7)].copy(), stage, "ambiguity_wide", "[0.3,0.7]", seed_base + 300))
    return rows


def confidence_row(df: pd.DataFrame, stage: str, bin_family: str, bin_label: str, seed: int) -> dict[str, object]:
    n_total = int(len(df))
    n_pos = int(df["y_true"].sum()) if n_total else 0
    n_neg = n_total - n_pos
    auc_cir = safe_auc(df["y_true"], df["cir_score"]) if n_total else np.nan
    auc_joint = safe_auc(df["y_true"], df["joint_score"]) if n_total else np.nan
    claim_tier = "invalid"
    ci_low = np.nan
    ci_high = np.nan
    if n_total >= 50 and n_pos >= 20 and n_neg >= 20:
        claim_tier = "main"
        ci_low, ci_high = bootstrap_delta_auc(df, bootstrap_scheme(df), seed)
    elif n_total >= 30:
        claim_tier = "supplemental"
    return {
        "stage": stage,
        "bin_family": bin_family,
        "bin_label": bin_label,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "claim_tier": claim_tier,
        "auc_cir": auc_cir,
        "auc_joint": auc_joint,
        "delta_auc": auc_joint - auc_cir if not np.isnan(auc_cir) and not np.isnan(auc_joint) else np.nan,
        "delta_auc_ci_low": ci_low,
        "delta_auc_ci_high": ci_high,
        "cp_save_count": int(df["cp_save"].sum()) if n_total else 0,
        "cp_harm_count": int(df["cp_harm"].sum()) if n_total else 0,
        "cp_save_rate": float(df["cp_save"].mean()) if n_total else np.nan,
        "cp_harm_rate": float(df["cp_harm"].mean()) if n_total else np.nan,
        "net_recovery_all": float(df["cp_save"].mean() - df["cp_harm"].mean()) if n_total else np.nan,
    }


def recovery_row(df: pd.DataFrame, stage: str, subset_name: str, condition: str, seed: int) -> dict[str, object]:
    n_total = int(len(df))
    n_pos = int(df["y_true"].sum())
    n_neg = n_total - n_pos
    cir_correct = df["cir_correct"].astype(int)
    cir_fail = 1 - cir_correct
    fail_den = int(cir_fail.sum())
    correct_den = int(cir_correct.sum())
    row = {
        "stage": stage,
        "subset_name": subset_name,
        "condition": condition,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "cp_save_rate_all": float(df["cp_save"].mean()) if n_total else np.nan,
        "cp_harm_rate_all": float(df["cp_harm"].mean()) if n_total else np.nan,
        "net_recovery_all": float(df["cp_save"].mean() - df["cp_harm"].mean()) if n_total else np.nan,
        "cir_fail_rate": float(cir_fail.mean()) if n_total else np.nan,
        "recovery_given_cir_fail": float(df["cp_save"].sum() / fail_den) if fail_den > 0 else np.nan,
        "harm_given_cir_correct": float(df["cp_harm"].sum() / correct_den) if correct_den > 0 else np.nan,
        "denominator_cir_fail": fail_den,
        "denominator_cir_correct": correct_den,
        "bootstrap_scheme": "x".join(bootstrap_scheme(df)),
    }
    row.update(bootstrap_recovery(df, bootstrap_scheme(df), seed))
    return row


def stratum_metrics(df: pd.DataFrame) -> dict[str, float]:
    n_total = int(len(df))
    n_pos = int(df["y_true"].sum())
    n_neg = n_total - n_pos
    cir_correct = df["cir_correct"].astype(int)
    cir_fail = 1 - cir_correct
    fail_den = int(cir_fail.sum())
    correct_den = int(cir_correct.sum())
    return {
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auc_cir": safe_auc(df["y_true"], df["cir_score"]),
        "auc_joint": safe_auc(df["y_true"], df["joint_score"]),
        "delta_auc": safe_auc(df["y_true"], df["joint_score"]) - safe_auc(df["y_true"], df["cir_score"]) if df["y_true"].nunique() == 2 else np.nan,
        "pr_auc_cir": safe_pr_auc(df["y_true"], df["cir_score"]),
        "pr_auc_joint": safe_pr_auc(df["y_true"], df["joint_score"]),
        "cp_save_rate_all": float(df["cp_save"].mean()) if n_total else np.nan,
        "cp_harm_rate_all": float(df["cp_harm"].mean()) if n_total else np.nan,
        "net_recovery_all": float(df["cp_save"].mean() - df["cp_harm"].mean()) if n_total else np.nan,
        "recovery_given_cir_fail": float(df["cp_save"].sum() / fail_den) if fail_den > 0 else np.nan,
        "harm_given_cir_correct": float(df["cp_harm"].sum() / correct_den) if correct_den > 0 else np.nan,
    }


def claim_tier(n_total: int, n_pos: int, n_neg: int) -> str:
    if n_total >= 40 and n_pos >= 15 and n_neg >= 15:
        return "main"
    if n_total >= 30:
        return "supplemental"
    return "invalid"


def find_best_bin(day4_bins: pd.DataFrame, analysis_name: str) -> tuple[float, float]:
    sub = day4_bins[(day4_bins["scope"] == "ALL") & (day4_bins["analysis_name"] == analysis_name)].copy()
    sub = sanitize_numeric(sub)
    sub = sub[sub["claim_tier"] == "main"].copy()
    sub = sub.sort_values(["net_recovery", "delta_auc"], ascending=[False, False])
    if sub.empty:
        raise ValueError(f"No main bin for {analysis_name}")
    interval = parse_interval(str(sub.iloc[0]["axis1_label"]))
    if interval is None:
        raise ValueError(f"Could not parse interval for {analysis_name}")
    return interval


def add_quantile_bins(series: pd.Series, labels: list[str]) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    if clean.dropna().nunique() < len(labels):
        return pd.Series(["not_available"] * len(series), index=series.index)
    return pd.qcut(clean, q=len(labels), labels=labels, duplicates="drop").astype(str)


def stage2_subset_definitions(stage2_df: pd.DataFrame, day2_candidates: pd.DataFrame, day4_bins: pd.DataFrame) -> tuple[list[dict[str, object]], tuple[float, float], tuple[float, float]]:
    c3 = day2_candidates[day2_candidates["candidate_id"] == "C3"].iloc[0]
    c4 = day2_candidates[day2_candidates["candidate_id"] == "C4"].iloc[0]
    c3_interval = parse_interval(str(c3["axis1_label"]))
    if c3_interval is None:
        raise ValueError("Could not parse C3 snr interval")
    k_interval = find_best_bin(day4_bins, "k_factor_estimate")
    fp_interval = find_best_bin(day4_bins, "fp_to_total_ratio")
    defs = [
        {"subset_name": "Stage2_ALL", "condition": "All Stage 2 OOF rows", "df": stage2_df.copy()},
        {"subset_name": "Stage2_low_confidence", "condition": "cir_score in [0.4,0.6]", "df": stage2_df[(stage2_df["cir_score"] >= 0.4) & (stage2_df["cir_score"] <= 0.6)].copy()},
        {"subset_name": "Stage2_M1", "condition": f"k_factor_estimate in [{k_interval[0]:.4f},{k_interval[1]:.4f}] and fp_to_total_ratio in [{fp_interval[0]:.4f},{fp_interval[1]:.4f}]", "df": stage2_df[(stage2_df["k_factor_estimate"] >= k_interval[0]) & (stage2_df["k_factor_estimate"] <= k_interval[1]) & (stage2_df["fp_to_total_ratio"] >= fp_interval[0]) & (stage2_df["fp_to_total_ratio"] <= fp_interval[1])].copy()},
        {"subset_name": "Stage2_C3", "condition": f"snr_db in [{c3_interval[0]:.2f},{c3_interval[1]:.2f}]", "df": stage2_df[(stage2_df["snr_db"] >= c3_interval[0]) & (stage2_df["snr_db"] <= c3_interval[1])].copy()},
        {"subset_name": "Stage2_C4", "condition": "room_type == A and dominant_wall_material == concrete", "df": stage2_df[(stage2_df["room_type"].astype(str) == str(c4["axis1_label"])) & (stage2_df["dominant_wall_material"].astype(str) == str(c4["axis2_label"]))].copy()},
    ]
    for room in sorted(stage2_df["room_type"].dropna().astype(str).unique()):
        defs.append({"subset_name": f"Stage2_room_{room}", "condition": f"room_type == {room}", "df": stage2_df[stage2_df["room_type"].astype(str) == room].copy()})
    return defs, k_interval, fp_interval


def plot_paired_delta_auc(paired_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = paired_df.copy()
    plot_df = plot_df[plot_df["subset_name"].isin(["Stage1_ALL", "Stage2_ALL", "Stage2_low_confidence", "Stage2_M1", "Stage2_C3", "Stage2_C4", "Stage2_room_A", "Stage2_room_B", "Stage2_room_C"])]
    plot_df = plot_df.sort_values("delta_auc")
    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(plot_df))
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.errorbar(
        plot_df["delta_auc"],
        y,
        xerr=[plot_df["delta_auc"] - plot_df["delta_auc_ci_low"], plot_df["delta_auc_ci_high"] - plot_df["delta_auc"]],
        fmt="o",
        color="#1f77b4",
        ecolor="#7aa6d8",
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["subset_name"])
    ax.set_xlabel("Paired delta AUC (Joint - CIR)")
    ax.set_title("Paired delta AUC with bootstrap CI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pr_auc(pr_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = pr_df[pr_df["subset_name"].isin(["Stage1_ALL", "Stage2_ALL", "Stage2_low_confidence", "Stage2_M1", "Stage2_C3", "Stage2_C4", "Stage2_room_A", "Stage2_room_B", "Stage2_room_C"])].copy()
    plot_df = plot_df.reset_index(drop=True)
    x = np.arange(len(plot_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, plot_df["pr_auc_cir"], width, label="CIR", color="#4c78a8")
    ax.bar(x + width / 2, plot_df["pr_auc_joint"], width, label="Joint", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["subset_name"], rotation=35, ha="right")
    ax.set_ylabel("PR-AUC (NLOS positive)")
    ax.set_title("PR-AUC by subset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fixed_fpr(pr_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = pr_df[pr_df["subset_name"].isin(["Stage1_ALL", "Stage2_ALL", "Stage2_low_confidence", "Stage2_M1", "Stage2_C3", "Stage2_C4"])].copy()
    targets = [1, 5, 10]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, target in zip(axes, targets):
        cir_col = f"recall_at_fpr_{target}_cir"
        joint_col = f"recall_at_fpr_{target}_joint"
        x = np.arange(len(plot_df))
        ax.plot(x, plot_df[cir_col], marker="o", label="CIR", color="#4c78a8")
        ax.plot(x, plot_df[joint_col], marker="o", label="Joint", color="#f58518")
        ax.set_title(f"Recall at FPR={target}%")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["subset_name"], rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("NLOS recall")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_confidence_bin_delta(conf_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = conf_df[conf_df["bin_family"] == "decile"].copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    for stage, color in [("Stage1", "#4c78a8"), ("Stage2", "#f58518")]:
        sub = plot_df[plot_df["stage"] == stage].copy()
        ax.plot(sub["bin_label"], sub["delta_auc"], marker="o", label=stage, color=color)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Delta AUC (Joint - CIR)")
    ax.set_xlabel("CIR score decile bin")
    ax.set_title("Confidence-bin delta AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_confidence_bin_recovery(conf_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = conf_df[conf_df["bin_family"] == "decile"].copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    for stage, color in [("Stage1", "#4c78a8"), ("Stage2", "#f58518")]:
        sub = plot_df[plot_df["stage"] == stage].copy()
        ax.plot(sub["bin_label"], sub["net_recovery_all"], marker="o", label=stage, color=color)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Net recovery")
    ax.set_xlabel("CIR score decile bin")
    ax.set_title("Confidence-bin net recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_save_harm_by_room(recovery_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = recovery_df[recovery_df["subset_name"].isin(["Stage2_room_A", "Stage2_room_B", "Stage2_room_C"])].copy()
    x = np.arange(len(plot_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, plot_df["cp_save_rate_all"], width, label="CP-save", color="#54a24b")
    ax.bar(x + width / 2, plot_df["cp_harm_rate_all"], width, label="CP-harm", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["subset_name"])
    ax.set_ylabel("Absolute rate")
    ax.set_title("CP-save vs CP-harm by room")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_save_harm_by_label_component(label_metrics_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = label_metrics_df[label_metrics_df["audit_type"] == "save_harm_by_label_component"].copy()
    x = np.arange(len(plot_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, plot_df["cp_save_rate"], width, label="CP-save", color="#54a24b")
    ax.bar(x + width / 2, plot_df["cp_harm_rate"], width, label="CP-harm", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["row_value"], rotation=20, ha="right")
    ax.set_ylabel("Absolute rate")
    ax.set_title("CP-save vs CP-harm by label component")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_m1_vs_baseline(paired_df: pd.DataFrame, recovery_df: pd.DataFrame, out_path: Path) -> None:
    auc_rows = paired_df[paired_df["subset_name"].isin(["Stage2_ALL", "Stage2_M1"])].copy()
    rec_rows = recovery_df[recovery_df["subset_name"].isin(["Stage2_ALL", "Stage2_M1"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(auc_rows))
    axes[0].bar(x, auc_rows["delta_auc"], color=["#9ecae9", "#3182bd"])
    axes[0].errorbar(
        x,
        auc_rows["delta_auc"],
        yerr=[auc_rows["delta_auc"] - auc_rows["delta_auc_ci_low"], auc_rows["delta_auc_ci_high"] - auc_rows["delta_auc"]],
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(auc_rows["subset_name"])
    axes[0].set_ylabel("Delta AUC")
    axes[0].set_title("Stage2 ALL vs M1")
    x2 = np.arange(len(rec_rows))
    axes[1].bar(x2, rec_rows["net_recovery_all"], color=["#c7e9c0", "#31a354"])
    axes[1].errorbar(
        x2,
        rec_rows["net_recovery_all"],
        yerr=[rec_rows["net_recovery_all"] - rec_rows["net_recovery_ci_low"], rec_rows["net_recovery_ci_high"] - rec_rows["net_recovery_all"]],
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(rec_rows["subset_name"])
    axes[1].set_ylabel("Net recovery")
    axes[1].set_title("Net recovery")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_missing_input_rows(stage1_df: pd.DataFrame, stage2_df: pd.DataFrame, day4_df: pd.DataFrame, feature_names: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    required_oof_cols = [
        "case_id",
        "y_true",
        "cir_score",
        "joint_score",
        "cp_save_0p5",
        "cp_harm_0p5",
        "cir_correct_0p5",
        "room_type",
        "gamma_cp_3_fp_only",
        "fp_to_total_ratio",
        "k_factor_estimate",
        "rms_delay_spread",
        "snr_db",
        "xpol_coupling_db",
        "dominant_wall_material",
        "los_angle_from_anchor_bore_deg",
        "is_nlos_geo",
        "is_nlos_bounce_0p33",
        "is_nlos_mixed_0p33",
        "has_los_path",
        "bounce_to_los_ratio_mid",
    ]
    for stage_name, df in [("Stage1", stage1_df), ("Stage2", stage2_df)]:
        missing = [col for col in required_oof_cols if col not in df.columns]
        rows.append(
            {
                "input_name": f"{stage_name}_OOF_required_columns",
                "status": "OK" if not missing else "MISSING_COLUMNS",
                "details": "None" if not missing else ", ".join(missing),
                "impact": "Full immediate metric set available" if not missing else "Some immediate metrics unavailable",
            }
        )
    missing_canonical_raw = [feature for feature in feature_names if feature not in stage2_df.columns]
    rows.append(
        {
            "input_name": "OOF_full_18_feature_values",
            "status": "PARTIAL",
            "details": "Missing raw feature columns in uploaded OOF copies: " + ", ".join(missing_canonical_raw),
            "impact": "No retraining needed, but per-feature stratification is limited to uploaded audit columns.",
        }
    )
    rows.append(
        {
            "input_name": "Stage2_expected_candidate_id_in_canonical_oof",
            "status": "MISSING_IN_CANONICAL_OOF" if "expected_candidate_id" not in stage2_df.columns else "OK",
            "details": "expected_candidate_id is only available in Day4 per-case copy." if "expected_candidate_id" not in stage2_df.columns else "Available",
            "impact": "expected_candidate_id stratification uses Day4 mini per-case table only.",
        }
    )
    rows.append(
        {
            "input_name": "Residual_or_ranging_error_label",
            "status": "MISSING_INPUT",
            "details": "No residual/ranging-error auxiliary label was uploaded.",
            "impact": "Residual-label disagreement audit cannot be computed.",
        }
    )
    rows.append(
        {
            "input_name": "Validated_DeLong_implementation",
            "status": "MISSING_INPUT",
            "details": "Repo search found only prior notes explicitly saying no validated DeLong implementation was available.",
            "impact": "Paired bootstrap CI is primary; DeLong p-value is reported as unavailable.",
        }
    )
    rows.append(
        {
            "input_name": "Day4_required_per_case_columns",
            "status": "OK" if set(day4_df.columns).issuperset({"case_id", "room_type", "y_true", "cir_score", "joint_score", "cp_save", "cp_harm", "k_factor_estimate", "fp_to_total_ratio"}) else "MISSING_COLUMNS",
            "details": "Loaded Day4 required-columns copy.",
            "impact": "Day4-derived M1 and expected_candidate_id checks available.",
        }
    )
    return rows


def write_missing_inputs_md(rows: list[dict[str, object]], out_path: Path, resolved_paths: dict[str, Path]) -> None:
    lines = ["# Missing Inputs Immediate", "", "## Resolved Inputs", ""]
    for key, value in resolved_paths.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Availability / Gaps", ""])
    for row in rows:
        lines.append(f"- `{row['input_name']}` | `{row['status']}` | {row['impact']}")
        lines.append(f"  details: {row['details']}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def label_disagreement_table(stage2_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if {"is_nlos_geo", "is_nlos_bounce_0p33", "is_nlos_mixed_0p33"}.issubset(stage2_df.columns):
        for name, row_col, col_col in [
            ("geo_vs_bounce", "is_nlos_geo", "is_nlos_bounce_0p33"),
            ("geo_vs_mixed", "is_nlos_geo", "is_nlos_mixed_0p33"),
            ("bounce_vs_mixed", "is_nlos_bounce_0p33", "is_nlos_mixed_0p33"),
        ]:
            cross = pd.crosstab(stage2_df[row_col].fillna(0).astype(int), stage2_df[col_col].fillna(0).astype(int))
            total = int(cross.to_numpy().sum())
            for r in cross.index:
                for c in cross.columns:
                    count = int(cross.loc[r, c])
                    rows.append(
                        {
                            "audit_type": name,
                            "row_value": str(r),
                            "col_value": str(c),
                            "count": count,
                            "proportion": count / total if total else np.nan,
                        }
                    )
        mixed_expected = ((stage2_df["is_nlos_geo"].fillna(0).astype(int) == 1) | (stage2_df["is_nlos_bounce_0p33"].fillna(0).astype(int) == 1)).astype(int)
        mixed_actual = stage2_df["is_nlos_mixed_0p33"].fillna(0).astype(int)
        mismatch = int((mixed_expected != mixed_actual).sum())
        rows.append(
            {
                "audit_type": "mixed_or_check",
                "row_value": "mixed_equals_or(geo,bounce)",
                "col_value": "status",
                "count": len(stage2_df) - mismatch,
                "proportion": (len(stage2_df) - mismatch) / len(stage2_df),
                "status": "PASS" if mismatch == 0 else "FAIL",
                "mismatch_count": mismatch,
            }
        )
    comp = stage2_df["label_component_derived"]
    for component, sub in stage2_df.groupby(comp):
        rows.append(
            {
                "audit_type": "save_harm_by_label_component",
                "row_value": component,
                "col_value": "",
                "count": len(sub),
                "proportion": len(sub) / len(stage2_df),
                "cp_save_rate": float(sub["cp_save"].mean()),
                "cp_harm_rate": float(sub["cp_harm"].mean()),
                "net_recovery_all": float(sub["cp_save"].mean() - sub["cp_harm"].mean()),
            }
        )
    rows.append(
        {
            "audit_type": "missing_input",
            "row_value": "residual_label_disagreement",
            "col_value": "",
            "count": np.nan,
            "proportion": np.nan,
            "status": "MISSING_INPUT",
        }
    )
    return pd.DataFrame(rows)


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
        "canonical_feature_list": resolve_input(
            "results/recovery_mini/day5/requested_material_copies_core/canonical_feature_names_list.csv",
            "/mnt/data/canonical_feature_names_list.csv",
        ),
        "canonical_feature_m": resolve_input(
            "results/recovery_mini/day5/requested_material_copies_core/canonicalFeatureNames_copy.m",
            "/mnt/data/canonicalFeatureNames_copy.m",
        ),
        "day4_metrics_by_room": resolve_input(
            "results/recovery_mini/day5/requested_material_copies_core/day4_stage2_room_mini_metrics_by_room_copy.csv",
            "/mnt/data/day4_stage2_room_mini_metrics_by_room_copy.csv",
        ),
        "day4_per_case": resolve_input(
            "results/recovery_mini/day5/requested_material_copies/stage2_recovery_room_mini_per_case_required_columns_copy.csv",
            "/mnt/data/stage2_recovery_room_mini_per_case_required_columns_copy.csv",
        ),
        "day4_condition_bins": resolve_input(
            "results/recovery_mini/day5/requested_material_copies/day4_stage2_condition_bins_copy.csv",
            "/mnt/data/day4_stage2_condition_bins_copy.csv",
        ),
        "day4_cv_comparison": resolve_input(
            "results/recovery_mini/day5/requested_material_copies/day4_stage2_cv_comparison_copy.csv",
            "/mnt/data/day4_stage2_cv_comparison_copy.csv",
        ),
        "hfss_case_list": resolve_input(
            "results/recovery_mini/day5/requested_material_copies/hfss_case_list_det_copy.csv",
            "/mnt/data/hfss_case_list_det_copy.csv",
        ),
        "stage3_reselection": resolve_input(
            "results/recovery_mini/day5/requested_material_copies/stage3_reselection_cells_det_copy.csv",
            "/mnt/data/stage3_reselection_cells_det_copy.csv",
        ),
    }

    feature_names, cp_features, cir_features = canonical_feature_groups(resolved_paths["canonical_feature_list"])

    stage1_df = normalize_oof(load_csv(resolved_paths["stage1_oof"]), "Stage1")
    stage2_df = normalize_oof(load_csv(resolved_paths["stage2_oof"]), "Stage2")
    stage2_df["label_component_derived"] = derive_label_component(stage2_df)
    day2_candidates = load_csv(resolved_paths["day2_candidates"])
    day4_per_case = normalize_day4_per_case(load_csv(resolved_paths["day4_per_case"]))
    day4_bins = load_csv(resolved_paths["day4_condition_bins"])
    day4_cv = load_csv(resolved_paths["day4_cv_comparison"])
    day4_metrics_by_room = load_csv(resolved_paths["day4_metrics_by_room"])

    missing_rows = build_missing_input_rows(stage1_df, stage2_df, day4_per_case, feature_names)
    write_missing_inputs_md(missing_rows, OUT_DIR / "missing_inputs_immediate.md", resolved_paths)

    subset_defs, m1_k_interval, m1_fp_interval = stage2_subset_definitions(stage2_df, day2_candidates, day4_bins)

    paired_rows = []
    paired_rows.append(paired_auc_row(stage1_df, "Stage1", "Stage1_ALL", "All Stage 1 OOF rows", GLOBAL_SEED + 1))
    for i, spec in enumerate(subset_defs):
        paired_rows.append(paired_auc_row(spec["df"], "Stage2", spec["subset_name"], spec["condition"], GLOBAL_SEED + 10 + i))
    paired_df = pd.DataFrame(paired_rows)
    paired_by_room_df = paired_df[paired_df["subset_name"].str.startswith("Stage2_room_")].copy()
    paired_df.to_csv(OUT_DIR / "paired_auc_metrics.csv", index=False)
    paired_by_room_df.to_csv(OUT_DIR / "paired_auc_metrics_by_room.csv", index=False)

    pr_rows = [pr_fixed_row(stage1_df, "Stage1", "Stage1_ALL", "All Stage 1 OOF rows")]
    for spec in subset_defs:
        pr_rows.append(pr_fixed_row(spec["df"], "Stage2", spec["subset_name"], spec["condition"]))
    pr_df = pd.DataFrame(pr_rows)
    pr_df.to_csv(OUT_DIR / "pr_fixed_fpr_metrics.csv", index=False)

    conf_rows = confidence_bin_rows(stage1_df, "Stage1", GLOBAL_SEED + 1000)
    conf_rows.extend(confidence_bin_rows(stage2_df, "Stage2", GLOBAL_SEED + 2000))
    conf_df = pd.DataFrame(conf_rows)
    conf_df.to_csv(OUT_DIR / "confidence_bin_metrics.csv", index=False)

    recovery_rows = [recovery_row(stage1_df, "Stage1", "Stage1_ALL", "All Stage 1 OOF rows", GLOBAL_SEED + 3000)]
    for i, spec in enumerate(subset_defs):
        recovery_rows.append(recovery_row(spec["df"], "Stage2", spec["subset_name"], spec["condition"], GLOBAL_SEED + 3100 + i))
    for i, room in enumerate(sorted(stage2_df["room_type"].dropna().astype(str).unique())):
        sub = stage2_df[stage2_df["room_type"].astype(str) == room].copy()
        recovery_rows.append(recovery_row(sub, "Stage2", f"Stage2_room_{room}", f"room_type == {room}", GLOBAL_SEED + 3200 + i))
    for i, component in enumerate(["geo_pos", "bounce_pos", "mixed_pos"]):
        if component == "geo_pos":
            sub = stage2_df[stage2_df["is_nlos_geo"].fillna(0).astype(int) == 1].copy()
            condition = "is_nlos_geo == 1"
        elif component == "bounce_pos":
            sub = stage2_df[stage2_df["is_nlos_bounce_0p33"].fillna(0).astype(int) == 1].copy()
            condition = "is_nlos_bounce_0p33 == 1"
        else:
            sub = stage2_df[stage2_df["is_nlos_mixed_0p33"].fillna(0).astype(int) == 1].copy()
            condition = "is_nlos_mixed_0p33 == 1"
        recovery_rows.append(recovery_row(sub, "Stage2", component, condition, GLOBAL_SEED + 3300 + i))
    recovery_rows.append(recovery_row(stage2_df[(stage2_df["cir_score"] >= 0.3) & (stage2_df["cir_score"] <= 0.7)].copy(), "Stage2", "Stage2_ambiguity_wide", "cir_score in [0.3,0.7]", GLOBAL_SEED + 3401))
    recovery_df = pd.DataFrame(recovery_rows).drop_duplicates(subset=["stage", "subset_name", "condition"])
    recovery_df.to_csv(OUT_DIR / "recovery_metrics.csv", index=False)

    regime_rows: list[dict[str, object]] = []
    stage2_df["xpol_bin"] = add_quantile_bins(stage2_df["xpol_coupling_db"], ["low", "mid", "high"])
    stage2_df["los_angle_bin"] = add_quantile_bins(stage2_df["los_angle_from_anchor_bore_deg"], ["low", "mid", "high"])
    k_labels = []
    for low, high in [parse_interval("[0.02312, 0.08303]"), parse_interval("[0.08303, 0.1446]"), parse_interval("[0.1446, 0.3373]")]:
        if low is not None and high is not None:
            k_labels.append((low, high))
    stage2_df["k_factor_bin"] = pd.cut(
        stage2_df["k_factor_estimate"],
        bins=[-np.inf, 0.08303, 0.1446, np.inf],
        labels=["low", "mid", "high"],
        include_lowest=True,
    ).astype(str)
    stage2_df["fp_ratio_bin"] = pd.cut(
        stage2_df["fp_to_total_ratio"],
        bins=[-np.inf, 0.07666, 0.1264, np.inf],
        labels=["low", "mid", "high"],
        include_lowest=True,
    ).astype(str)
    stage2_df["m1_membership"] = np.where(
        (stage2_df["k_factor_estimate"] >= m1_k_interval[0])
        & (stage2_df["k_factor_estimate"] <= m1_k_interval[1])
        & (stage2_df["fp_to_total_ratio"] >= m1_fp_interval[0])
        & (stage2_df["fp_to_total_ratio"] <= m1_fp_interval[1]),
        "M1_in_band",
        "baseline_out_of_band",
    )

    def append_regime_rows(source_df: pd.DataFrame, source_name: str, axis_name: str, group_col: str, caveat: str = "") -> None:
        for level, sub in source_df.groupby(group_col, dropna=False):
            metrics = stratum_metrics(sub)
            metrics.update(
                {
                    "source_dataset": source_name,
                    "axis_name": axis_name,
                    "stratum_value": str(level),
                    "claim_tier": claim_tier(metrics["n_total"], metrics["n_pos"], metrics["n_neg"]),
                    "caveat": caveat,
                }
            )
            regime_rows.append(metrics)

    append_regime_rows(stage2_df, "Stage2_canonical_oof", "room_type", "room_type")
    append_regime_rows(stage2_df, "Stage2_canonical_oof", "dominant_wall_material", "dominant_wall_material")
    append_regime_rows(stage2_df, "Stage2_canonical_oof", "xpol_coupling_db_bin", "xpol_bin")
    append_regime_rows(stage2_df, "Stage2_canonical_oof", "los_angle_from_anchor_bore_deg_bin", "los_angle_bin")
    append_regime_rows(stage2_df, "Stage2_canonical_oof", "k_factor_estimate_bin", "k_factor_bin", "Strong collinearity with fp_to_total_ratio; do not claim as independent mechanism.")
    append_regime_rows(stage2_df, "Stage2_canonical_oof", "fp_to_total_ratio_bin", "fp_ratio_bin", "Strong collinearity with k_factor_estimate; same signal family.")
    append_regime_rows(stage2_df, "Stage2_canonical_oof", "combined_M1_band", "m1_membership", "M1 uses both k_factor_estimate and fp_to_total_ratio from the same collinear family.")
    append_regime_rows(stage2_df, "Stage2_canonical_oof", "label_component", "label_component_derived")
    if "expected_candidate_id" in day4_per_case.columns:
        append_regime_rows(day4_per_case, "Stage2_day4_mini", "expected_candidate_id", "expected_candidate_id")
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(OUT_DIR / "regime_stratified_metrics.csv", index=False)

    label_audit_df = label_disagreement_table(stage2_df)
    label_audit_df.to_csv(OUT_DIR / "label_disagreement_audit.csv", index=False)

    plot_paired_delta_auc(paired_df, OUT_DIR / "plot_paired_delta_auc_ci.png")
    plot_pr_auc(pr_df, OUT_DIR / "plot_pr_auc_by_subset.png")
    plot_fixed_fpr(pr_df, OUT_DIR / "plot_fixed_fpr_recall.png")
    plot_confidence_bin_delta(conf_df, OUT_DIR / "plot_confidence_bin_delta_auc.png")
    plot_confidence_bin_recovery(conf_df, OUT_DIR / "plot_confidence_bin_recovery.png")
    plot_save_harm_by_room(recovery_df, OUT_DIR / "plot_save_harm_by_room.png")
    plot_save_harm_by_label_component(label_audit_df, OUT_DIR / "plot_save_harm_by_label_component.png")
    plot_m1_vs_baseline(paired_df, recovery_df, OUT_DIR / "plot_regime_m1_vs_baseline.png")

    paired_lookup = paired_df.set_index("subset_name")
    pr_lookup = pr_df.set_index("subset_name")
    recovery_lookup = recovery_df.set_index("subset_name")
    conf_lookup = conf_df.set_index(["stage", "bin_family", "bin_label"])
    day4_group = day4_cv[(day4_cv["cv_scheme"] == "group_stratified") & (day4_cv["scope"] == "ALL")].iloc[0]

    def fmt_ci(val: float, lo: float, hi: float) -> str:
        if any(pd.isna(x) for x in [val, lo, hi]):
            return "NA"
        return f"{val:+.4f} [{lo:+.4f}, {hi:+.4f}]"

    summary_lines = [
        "# Metrics Immediate Summary",
        "",
        "## A1. Canonical Status",
        "",
        f"- Canonical feature count: `{len(feature_names)}`",
        f"- CP features (6): `{', '.join(cp_features)}`",
        f"- CIR features (12): `{', '.join(cir_features)}`",
        f"- Stage 1 label policy in uploaded OOF: `{stage1_df['label_name'].dropna().astype(str).mode().iat[0]}`",
        f"- Stage 2 label policy in uploaded OOF: `{stage2_df['label_name'].dropna().astype(str).mode().iat[0]}`",
        "- OOF-only evaluation was used throughout. No retraining or train-on-test AUC was introduced here.",
        "- DeLong p-value: `unavailable`. Prior repo audit notes already stated no validated implementation was available, so paired bootstrap CI is the primary inference path.",
        "",
        "## A2-A6. Immediate Findings",
        "",
        f"- Stage 1 ALL paired delta AUC: `{fmt_ci(paired_lookup.loc['Stage1_ALL', 'delta_auc'], paired_lookup.loc['Stage1_ALL', 'delta_auc_ci_low'], paired_lookup.loc['Stage1_ALL', 'delta_auc_ci_high'])}`",
        f"- Stage 2 ALL paired delta AUC: `{fmt_ci(paired_lookup.loc['Stage2_ALL', 'delta_auc'], paired_lookup.loc['Stage2_ALL', 'delta_auc_ci_low'], paired_lookup.loc['Stage2_ALL', 'delta_auc_ci_high'])}`",
        f"- Stage 2 low-confidence paired delta AUC: `{fmt_ci(paired_lookup.loc['Stage2_low_confidence', 'delta_auc'], paired_lookup.loc['Stage2_low_confidence', 'delta_auc_ci_low'], paired_lookup.loc['Stage2_low_confidence', 'delta_auc_ci_high'])}`",
        f"- Stage 2 M1 paired delta AUC: `{fmt_ci(paired_lookup.loc['Stage2_M1', 'delta_auc'], paired_lookup.loc['Stage2_M1', 'delta_auc_ci_low'], paired_lookup.loc['Stage2_M1', 'delta_auc_ci_high'])}`",
        f"- Stage 2 C3 paired delta AUC: `{fmt_ci(paired_lookup.loc['Stage2_C3', 'delta_auc'], paired_lookup.loc['Stage2_C3', 'delta_auc_ci_low'], paired_lookup.loc['Stage2_C3', 'delta_auc_ci_high'])}`",
        f"- Stage 2 C4 paired delta AUC: `{fmt_ci(paired_lookup.loc['Stage2_C4', 'delta_auc'], paired_lookup.loc['Stage2_C4', 'delta_auc_ci_low'], paired_lookup.loc['Stage2_C4', 'delta_auc_ci_high'])}`",
        f"- Stage 2 ambiguity-band net recovery: `{fmt_ci(recovery_lookup.loc['Stage2_low_confidence', 'net_recovery_all'], recovery_lookup.loc['Stage2_low_confidence', 'net_recovery_ci_low'], recovery_lookup.loc['Stage2_low_confidence', 'net_recovery_ci_high'])}`",
        f"- Stage 2 M1 net recovery: `{fmt_ci(recovery_lookup.loc['Stage2_M1', 'net_recovery_all'], recovery_lookup.loc['Stage2_M1', 'net_recovery_ci_low'], recovery_lookup.loc['Stage2_M1', 'net_recovery_ci_high'])}`",
        f"- Stage 2 C3 net recovery: `{fmt_ci(recovery_lookup.loc['Stage2_C3', 'net_recovery_all'], recovery_lookup.loc['Stage2_C3', 'net_recovery_ci_low'], recovery_lookup.loc['Stage2_C3', 'net_recovery_ci_high'])}`",
        f"- Stage 2 C4 net recovery: `{fmt_ci(recovery_lookup.loc['Stage2_C4', 'net_recovery_all'], recovery_lookup.loc['Stage2_C4', 'net_recovery_ci_low'], recovery_lookup.loc['Stage2_C4', 'net_recovery_ci_high'])}`",
        f"- Day4 mini grouped-CV context from copied table: `delta_auc={day4_group['delta_auc']:+.4f}`, `net_recovery={day4_group['net_recovery']:+.4f}`.",
        "",
        "## A7. Label Disagreement",
        "",
        f"- mixed == OR(geo,bounce): `{label_audit_df[label_audit_df['audit_type'] == 'mixed_or_check']['status'].iloc[0]}`",
        f"- residual/ranging-error disagreement audit: `MISSING_INPUT`",
        "",
        "## Immediate Interpretation",
        "",
        "1. Which stage or subset survives when global delta AUC is small?",
        "   Stage 1 ALL survives cleanly: delta AUC is positive with a fully positive bootstrap CI. Stage 2 ALL does not. Inside Stage 2, the direct survivors are C3 and C4; both have positive paired delta AUC with CI above zero. M1 is weaker: positive mean delta AUC but CI still crosses zero. The broad low-confidence band does not survive on paired delta AUC.",
        "2. Is LoS harm actually large under PR and fixed-FPR views?",
        f"   Global Stage 2 harm does not look like an explosion. On Stage2_ALL, Joint raises PR-AUC slightly and lowers LoS FPR at recall 80/90/95 from `{pr_lookup.loc['Stage2_ALL', 'los_fpr_at_recall_80_cir']:.3f}/{pr_lookup.loc['Stage2_ALL', 'los_fpr_at_recall_90_cir']:.3f}/{pr_lookup.loc['Stage2_ALL', 'los_fpr_at_recall_95_cir']:.3f}` to `{pr_lookup.loc['Stage2_ALL', 'los_fpr_at_recall_80_joint']:.3f}/{pr_lookup.loc['Stage2_ALL', 'los_fpr_at_recall_90_joint']:.3f}/{pr_lookup.loc['Stage2_ALL', 'los_fpr_at_recall_95_joint']:.3f}`. The tradeoff is subset-specific: C3 improves LoS FPR at every fixed-recall target, M1 improves LoS FPR but not recall at 1 percent FPR, and C4 shows stronger ranking gain but worse fixed-FPR recall at 5 and 10 percent FPR.",
        "3. Does CP rescue survive in the ambiguity band?",
        f"   Stage 1 yes. Stage 2 no, at least not in the broad thresholded sense. The Stage2 [0.4,0.6] ambiguity band has delta AUC `{paired_lookup.loc['Stage2_low_confidence', 'delta_auc']:+.4f}` with CI crossing zero and net recovery `{recovery_lookup.loc['Stage2_low_confidence', 'net_recovery_all']:+.4f}` with CI crossing zero. The decile table shows more of a ranking effect in the 0.2 to 0.6 CIR-score region than a clean 0.5-threshold rescue effect.",
        "4. Is M1 really the main candidate, or are C3 and C4 more direct?",
        "   C3 and C4 are more direct immediate candidates on the canonical Stage 2 OOF rows. M1 still matters because it beats the Stage2 baseline-out-of-band rows on delta AUC and net recovery, but its CI does not close the case and it must carry the k-factor/fp-ratio collinearity caveat. C4 is the strongest mean effect, but it is also the most caution-heavy because its fixed-FPR recall degrades. C3 is the cleaner balanced candidate.",
        "5. What should be the primary Stage 3 preflight gate?",
        "   Use net_recovery_all and recovery_given_cir_fail as the primary gate, not raw delta AUC alone. Keep paired delta AUC with bootstrap CI as a secondary ranking signal. Add PR-AUC and fixed-recall LoS FPR as reject gates, so a candidate that improves ranking but increases LoS harm too much does not pass.",
        "",
        "## Resolved Inputs",
        "",
    ]
    for key, value in resolved_paths.items():
        summary_lines.append(f"- `{key}`: `{value}`")
    (OUT_DIR / "metrics_immediate_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

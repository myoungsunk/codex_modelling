from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "theta0_peak_aligned_cp16_feature_interaction"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20260425
N_BOOT = 1000

CIR12 = [
    "rms_delay_spread",
    "mean_excess_delay",
    "max_excess_delay",
    "fp_to_total_ratio",
    "rise_time_fp",
    "fp_kurtosis",
    "kurtosis_total",
    "skewness_total",
    "energy_concentration_50ns",
    "num_significant_peaks",
    "peak_to_avg_ratio",
    "k_factor_estimate",
]
CP16 = [
    "xpr_fp_db",
    "xpr_late_db",
    "xpr_all_db",
    "s3_fp",
    "s3_late",
    "s3_all",
    "delta_tau_l_given_r_s",
    "delta_p_l_given_r_db",
    "f_r_fp",
    "f_l_fp",
    "delta_f_l_minus_r_fp",
    "lambda_l_late_fraction",
    "gamma_anchor_linear",
    "gamma_delay_linear",
    "cp_phase_slope_delay_s",
    "cp_phase_residual_circvar",
]

SOURCES = {
    "stage1": ROOT
    / "results"
    / "stage1_theta0_peak_aligned_cp16_phi_fastest"
    / "stage1_3000_theta0_peak_aligned_cp16_phi_fastest_det.csv",
    "stage2": ROOT
    / "results"
    / "stage2_theta0_peak_aligned_cp16_phi_fastest"
    / "stage2_900_theta0_peak_aligned_cp16_phi_fastest_det.csv",
}


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def load_stage(stage: str) -> pd.DataFrame:
    df = pd.read_csv(SOURCES[stage])
    failed = as_bool(df["failed"]) if "failed" in df else pd.Series(False, index=df.index)
    df = df.loc[~failed].copy()
    for col in CIR12 + CP16:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if stage == "stage1":
        df["y_true"] = as_bool(df["is_nlos"]).astype(int)
        df["room_type"] = "stage1"
    else:
        has_los = as_bool(df["has_los_path"])
        ratio = pd.to_numeric(df["bounce_to_los_ratio_mid"], errors="coerce")
        df["y_true"] = ((~has_los) | (has_los & (ratio >= 0.33))).astype(int)
        room_col = "room_type" if "room_type" in df.columns else "room_type_cases"
        df["room_type"] = df[room_col].astype(str)
    finite = np.isfinite(df[CIR12 + CP16].to_numpy(float)).all(axis=1)
    return df.loc[finite].reset_index(drop=True)


def make_model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, solver="lbfgs", random_state=SEED),
    )


def build_splits(df: pd.DataFrame, stage: str, scheme: str) -> list[tuple[np.ndarray, np.ndarray]]:
    y = df["y_true"].to_numpy(int)
    if scheme == "fit_all":
        idx = np.arange(len(df))
        return [(idx, idx)]
    if scheme == "stratified5":
        return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    if scheme == "room_label_stratified5":
        strata = df["room_type"].astype(str).to_numpy() + "|" + y.astype(str)
        return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), strata))
    if scheme == "leave_one_room_out":
        return list(LeaveOneGroupOut().split(np.zeros(len(y)), y, df["room_type"].astype(str).to_numpy()))
    raise ValueError(scheme)


def oof_score(df: pd.DataFrame, features: list[str], stage: str, scheme: str) -> np.ndarray:
    y = df["y_true"].to_numpy(int)
    X = df[features].to_numpy(float)
    pred = np.full(len(df), np.nan)
    for train_idx, test_idx in build_splits(df, stage, scheme):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = make_model()
        model.fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return pred


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    if mask.sum() < 3 or len(np.unique(y[mask])) < 2:
        return np.nan
    return float(roc_auc_score(y[mask], score[mask]))


def safe_pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    if mask.sum() < 3 or len(np.unique(y[mask])) < 2:
        return np.nan
    return float(average_precision_score(y[mask], score[mask]))


def threshold_stats(y: np.ndarray, cir: np.ndarray, cp: np.ndarray, joint: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(cir) & np.isfinite(cp) & np.isfinite(joint)
    yv, cir, cp, joint = y[valid], cir[valid], cp[valid], joint[valid]
    cir_hat = cir >= 0.5
    cp_hat = cp >= 0.5
    joint_hat = joint >= 0.5
    cir_ok = cir_hat == yv
    cp_ok = cp_hat == yv
    joint_ok = joint_hat == yv
    save = (~cir_ok) & joint_ok
    harm = cir_ok & (~joint_ok)
    cir_nlos = cir_hat
    cp_nlos = cp_hat
    both_nlos = cir_nlos & cp_nlos
    union_nlos = cir_nlos | cp_nlos
    return {
        "n": int(valid.sum()),
        "cir_positive_rate": float(cir_nlos.mean()),
        "cp_positive_rate": float(cp_nlos.mean()),
        "joint_positive_rate": float(joint_hat.mean()),
        "cir_cp_predicted_nlos_jaccard": float(both_nlos.sum() / max(union_nlos.sum(), 1)),
        "cp_nlos_given_cir_nlos": float(both_nlos.sum() / max(cir_nlos.sum(), 1)),
        "cir_nlos_given_cp_nlos": float(both_nlos.sum() / max(cp_nlos.sum(), 1)),
        "both_correct_rate": float((cir_ok & cp_ok).mean()),
        "cir_only_correct_rate": float((cir_ok & ~cp_ok).mean()),
        "cp_only_correct_rate": float((~cir_ok & cp_ok).mean()),
        "both_wrong_rate": float((~cir_ok & ~cp_ok).mean()),
        "cp_save": float(save.mean()),
        "cp_harm": float(harm.mean()),
        "net_recovery": float(save.mean() - harm.mean()),
        "recovery_given_cir_fail": float(save.sum() / max((~cir_ok).sum(), 1)),
        "harm_given_cir_correct": float(harm.sum() / max(cir_ok.sum(), 1)),
    }


def bootstrap_ci(y: np.ndarray, cir: np.ndarray, joint: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(cir) & np.isfinite(joint)
    y, cir, joint = y[valid], cir[valid], joint[valid]
    if len(np.unique(y)) < 2:
        return {}
    rng = np.random.default_rng(SEED)
    groups = [np.where(y == val)[0] for val in np.unique(y)]
    deltas = []
    nets = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], joint[idx]) - roc_auc_score(y[idx], cir[idx]))
        stats = threshold_stats(y[idx], cir[idx], cir[idx], joint[idx])
        nets.append(stats["net_recovery"])
    return {
        "delta_auc_ci_low": float(np.percentile(deltas, 2.5)),
        "delta_auc_ci_high": float(np.percentile(deltas, 97.5)),
        "net_recovery_ci_low": float(np.percentile(nets, 2.5)),
        "net_recovery_ci_high": float(np.percentile(nets, 97.5)),
    }


def correlation_outputs(stage: str, df: pd.DataFrame) -> dict[str, float]:
    all_features = CIR12 + CP16
    pearson = df[all_features].corr(method="pearson")
    spearman = df[all_features].corr(method="spearman")
    pearson.to_csv(OUT_DIR / f"{stage}_all_feature_pearson_corr.csv")
    spearman.to_csv(OUT_DIR / f"{stage}_all_feature_spearman_corr.csv")

    rows = []
    for cp in CP16:
        for cir in CIR12:
            rows.append(
                {
                    "stage": stage,
                    "cp_feature": cp,
                    "cir_feature": cir,
                    "pearson": pearson.loc[cp, cir],
                    "spearman": spearman.loc[cp, cir],
                    "abs_pearson": abs(pearson.loc[cp, cir]),
                    "abs_spearman": abs(spearman.loc[cp, cir]),
                }
            )
    cross = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)
    cross.to_csv(OUT_DIR / f"{stage}_cp_cir_cross_correlation.csv", index=False)

    pair_rows = []
    for a, b in combinations(all_features, 2):
        if a in CIR12 and b in CIR12:
            pair_type = "cir_cir"
        elif a in CP16 and b in CP16:
            pair_type = "cp_cp"
        else:
            pair_type = "cp_cir"
        pair_rows.append(
            {
                "stage": stage,
                "feature_a": a,
                "feature_b": b,
                "pair_type": pair_type,
                "pearson": pearson.loc[a, b],
                "spearman": spearman.loc[a, b],
                "abs_pearson": abs(pearson.loc[a, b]),
                "abs_spearman": abs(spearman.loc[a, b]),
            }
        )
    pairs = pd.DataFrame(pair_rows).sort_values("abs_spearman", ascending=False)
    pairs.to_csv(OUT_DIR / f"{stage}_single_feature_pair_correlation.csv", index=False)

    return {
        "mean_abs_cp_cir_spearman": float(cross["abs_spearman"].mean()),
        "median_abs_cp_cir_spearman": float(cross["abs_spearman"].median()),
        "max_abs_cp_cir_spearman": float(cross["abs_spearman"].max()),
        "mean_abs_cp_cp_spearman": float(pairs.loc[pairs["pair_type"] == "cp_cp", "abs_spearman"].mean()),
        "mean_abs_cir_cir_spearman": float(pairs.loc[pairs["pair_type"] == "cir_cir", "abs_spearman"].mean()),
    }


def single_feature_auc(stage: str, df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    y = df["y_true"].to_numpy(int)
    rows = []
    for feature in CIR12 + CP16:
        score = oof_score(df, [feature], stage, scheme)
        rows.append(
            {
                "stage": stage,
                "cv_scheme": scheme,
                "feature": feature,
                "feature_family": "cir" if feature in CIR12 else "cp",
                "AUC": safe_auc(y, score),
                "PR_AUC": safe_pr_auc(y, score),
            }
        )
    out = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    out.to_csv(OUT_DIR / f"{stage}_single_feature_auc.csv", index=False)
    return out


def greedy_joint(stage: str, df: pd.DataFrame, scheme: str, max_cp: int = 6) -> pd.DataFrame:
    y = df["y_true"].to_numpy(int)
    cir_score = oof_score(df, CIR12, stage, scheme)
    baseline_auc = safe_auc(y, cir_score)
    selected: list[str] = []
    remaining = CP16.copy()
    rows = [
        {
            "stage": stage,
            "cv_scheme": scheme,
            "step": 0,
            "added_feature": "",
            "selected_cp_features": "",
            "n_cp": 0,
            "AUC": baseline_auc,
            "PR_AUC": safe_pr_auc(y, cir_score),
            "delta_vs_cir": 0.0,
        }
    ]
    for step in range(1, max_cp + 1):
        best = None
        for feature in remaining:
            features = CIR12 + selected + [feature]
            score = oof_score(df, features, stage, scheme)
            auc = safe_auc(y, score)
            pr = safe_pr_auc(y, score)
            candidate = (auc, pr, feature, score)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            break
        auc, pr, feature, _ = best
        selected.append(feature)
        remaining.remove(feature)
        rows.append(
            {
                "stage": stage,
                "cv_scheme": scheme,
                "step": step,
                "added_feature": feature,
                "selected_cp_features": ";".join(selected),
                "n_cp": len(selected),
                "AUC": auc,
                "PR_AUC": pr,
                "delta_vs_cir": auc - baseline_auc,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / f"{stage}_greedy_joint_cp_added_to_cir.csv", index=False)
    return out


def evaluate_feature_sets(stage: str, df: pd.DataFrame, scheme: str, selected_cp: list[str]) -> pd.DataFrame:
    y = df["y_true"].to_numpy(int)
    sets = {
        "CIR_only_12": CIR12,
        "CP_only_16": CP16,
        "Joint_CIR_plus_CP16": CIR12 + CP16,
        "Joint_CIR_plus_greedy_CP": CIR12 + selected_cp,
    }
    scores = {name: oof_score(df, feats, stage, scheme) for name, feats in sets.items()}
    rows = []
    for name, score in scores.items():
        rows.append(
            {
                "stage": stage,
                "cv_scheme": scheme,
                "feature_set": name,
                "n_features": len(sets[name]),
                "features": ";".join(sets[name]),
                "AUC": safe_auc(y, score),
                "PR_AUC": safe_pr_auc(y, score),
            }
        )
    stats = threshold_stats(y, scores["CIR_only_12"], scores["CP_only_16"], scores["Joint_CIR_plus_greedy_CP"])
    paired = {
        "stage": stage,
        "cv_scheme": scheme,
        "joint_definition": "Joint_CIR_plus_greedy_CP",
        "AUC_CIR": safe_auc(y, scores["CIR_only_12"]),
        "AUC_CP": safe_auc(y, scores["CP_only_16"]),
        "AUC_Joint": safe_auc(y, scores["Joint_CIR_plus_greedy_CP"]),
        "Delta_AUC": safe_auc(y, scores["Joint_CIR_plus_greedy_CP"]) - safe_auc(y, scores["CIR_only_12"]),
        **stats,
        **bootstrap_ci(y, scores["CIR_only_12"], scores["Joint_CIR_plus_greedy_CP"]),
    }
    return pd.DataFrame(rows), pd.DataFrame([paired])


def write_markdown(summary: pd.DataFrame, set_metrics: pd.DataFrame, paired: pd.DataFrame, greedy: pd.DataFrame) -> None:
    md = OUT_DIR / "theta0_cp_cir_feature_interaction_summary.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Theta0 CP/CIR Feature Interaction Audit\n\n")
        f.write("Inputs are the full theta0 peak-aligned CP16 Stage 1 and Stage 2 rerun CSVs. Stage 1 uses stratified 5-fold CV; Stage 2 uses room-label stratified 5-fold CV as the primary room-aware estimate.\n\n")
        f.write("## Correlation Summary\n\n")
        f.write("| stage | mean_abs_cp_cir_spearman | median_abs_cp_cir_spearman | max_abs_cp_cir_spearman | mean_abs_cp_cp_spearman | mean_abs_cir_cir_spearman |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for _, r in summary.iterrows():
            f.write(
                f"| {r.stage} | {r.mean_abs_cp_cir_spearman:.6f} | {r.median_abs_cp_cir_spearman:.6f} | {r.max_abs_cp_cir_spearman:.6f} | {r.mean_abs_cp_cp_spearman:.6f} | {r.mean_abs_cir_cir_spearman:.6f} |\n"
            )
        f.write("\n## Main Feature-Set Metrics\n\n")
        f.write("| stage | cv_scheme | feature_set | n_features | AUC | PR_AUC |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for _, r in set_metrics.iterrows():
            f.write(f"| {r.stage} | {r.cv_scheme} | {r.feature_set} | {int(r.n_features)} | {r.AUC:.6f} | {r.PR_AUC:.6f} |\n")
        f.write("\n## Save / Harm Using Greedy Joint\n\n")
        f.write("| stage | cv_scheme | AUC_CIR | AUC_CP | AUC_Joint | Delta_AUC | Delta_AUC_CI_low | cp_save | cp_harm | net_recovery | net_recovery_CI_low | nlos_jaccard |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in paired.iterrows():
            f.write(
                f"| {r.stage} | {r.cv_scheme} | {r.AUC_CIR:.6f} | {r.AUC_CP:.6f} | {r.AUC_Joint:.6f} | {r.Delta_AUC:.6f} | {r.delta_auc_ci_low:.6f} | {r.cp_save:.6f} | {r.cp_harm:.6f} | {r.net_recovery:.6f} | {r.net_recovery_ci_low:.6f} | {r.cir_cp_predicted_nlos_jaccard:.6f} |\n"
            )
        f.write("\n## Greedy Joint Selection\n\n")
        for stage in greedy["stage"].unique():
            f.write(f"\n### {stage}\n\n")
            g = greedy[greedy["stage"] == stage]
            f.write("| step | added_feature | n_cp | AUC | delta_vs_cir |\n")
            f.write("|---:|---|---:|---:|---:|\n")
            for _, r in g.iterrows():
                f.write(f"| {int(r.step)} | {r.added_feature} | {int(r.n_cp)} | {r.AUC:.6f} | {r.delta_vs_cir:.6f} |\n")


def main() -> None:
    summaries = []
    all_set_metrics = []
    all_paired = []
    all_greedy = []
    for stage in ["stage1", "stage2"]:
        df = load_stage(stage)
        scheme = "stratified5" if stage == "stage1" else "room_label_stratified5"
        corr_summary = correlation_outputs(stage, df)
        summaries.append({"stage": stage, "n": len(df), "n_pos": int(df["y_true"].sum()), **corr_summary})
        single_feature_auc(stage, df, scheme)
        greedy = greedy_joint(stage, df, scheme, max_cp=6)
        best_idx = greedy["AUC"].astype(float).idxmax()
        best_features = str(greedy.loc[best_idx, "selected_cp_features"])
        selected = best_features.split(";") if best_features else []
        set_metrics, paired = evaluate_feature_sets(stage, df, scheme, selected)
        all_set_metrics.append(set_metrics)
        all_paired.append(paired)
        all_greedy.append(greedy)

    summary = pd.DataFrame(summaries)
    set_metrics = pd.concat(all_set_metrics, ignore_index=True)
    paired = pd.concat(all_paired, ignore_index=True)
    greedy = pd.concat(all_greedy, ignore_index=True)
    summary.to_csv(OUT_DIR / "correlation_summary.csv", index=False)
    set_metrics.to_csv(OUT_DIR / "main_feature_set_metrics.csv", index=False)
    paired.to_csv(OUT_DIR / "save_harm_overlap_metrics.csv", index=False)
    greedy.to_csv(OUT_DIR / "greedy_joint_selection.csv", index=False)
    write_markdown(summary, set_metrics, paired, greedy)
    print(f"Saved feature interaction audit to {OUT_DIR}")


if __name__ == "__main__":
    main()

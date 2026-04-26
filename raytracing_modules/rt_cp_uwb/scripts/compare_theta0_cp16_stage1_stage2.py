from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "theta0_peak_aligned_cp16_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20260425
N_BOOT = 1000

CP6 = [
    "gamma_cp_1_freq_avg",
    "gamma_cp_2_freq_db",
    "gamma_cp_3_fp_only",
    "gamma_cp_6_phase_circvar",
    "a_fp_2_peak_to_total",
    "a_fp_6_fp_to_2nd_peak",
]
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
CP_PRUNED_6 = [
    "delta_tau_l_given_r_s",
    "gamma_anchor_linear",
    "gamma_delay_linear",
    "lambda_l_late_fraction",
    "cp_phase_slope_delay_s",
    "cp_phase_residual_circvar",
]
CP_LOW_OVERLAP_4 = [
    "delta_tau_l_given_r_s",
    "gamma_delay_linear",
    "lambda_l_late_fraction",
    "cp_phase_slope_delay_s",
]
FEATURE_SETS = {
    "CIR_only_12": CIR12,
    "CP6_canonical": CP6,
    "Joint_CIR_plus_CP6": CIR12 + CP6,
    "CP_all_RH_LH_16": CP16,
    "CP_pruned_6": CP_PRUNED_6,
    "CP_low_overlap_4": CP_LOW_OVERLAP_4,
    "Joint_CIR_plus_CP_all": CIR12 + CP16,
    "Joint_CIR_plus_CP_pruned": CIR12 + CP_PRUNED_6,
    "Joint_CIR_plus_CP_low_overlap": CIR12 + CP_LOW_OVERLAP_4,
}

VARIANTS = {
    "stage1": {
        "old_local_negx": ROOT
        / "results"
        / "stage1_peak_aligned_ffd_phi_fastest"
        / "stage1_3000_ffd_peak_aligned_phi_fastest_det.csv",
        "new_theta0_posz_cp16": ROOT
        / "results"
        / "stage1_theta0_peak_aligned_cp16_phi_fastest"
        / "stage1_3000_theta0_peak_aligned_cp16_phi_fastest_det.csv",
    },
    "stage2": {
        "old_local_negx": ROOT
        / "results"
        / "stage2_peak_aligned_phi_fastest"
        / "stage2_900_ffd_peak_aligned_phi_fastest_det.csv",
        "new_theta0_posz_cp16": ROOT
        / "results"
        / "stage2_theta0_peak_aligned_cp16_phi_fastest"
        / "stage2_900_theta0_peak_aligned_cp16_phi_fastest_det.csv",
    },
}


def resolve_col(df: pd.DataFrame, base: str) -> str:
    for col in [base, f"{base}_cases", f"{base}_results", f"{base}_cases_tbl", f"{base}_results_tbl"]:
        if col in df.columns:
            return col
    raise KeyError(base)


def as_str(df: pd.DataFrame, base: str, default: str = "") -> pd.Series:
    try:
        return df[resolve_col(df, base)].astype(str)
    except KeyError:
        return pd.Series(default, index=df.index, dtype=object)


def as_num(df: pd.DataFrame, base: str, default=np.nan) -> pd.Series:
    try:
        return pd.to_numeric(df[resolve_col(df, base)], errors="coerce")
    except KeyError:
        return pd.Series(default, index=df.index, dtype=float)


def as_bool(df: pd.DataFrame, base: str, default=False) -> pd.Series:
    try:
        raw = df[resolve_col(df, base)]
    except KeyError:
        return pd.Series(default, index=df.index, dtype=bool)
    if raw.dtype == bool:
        return raw
    return raw.astype(str).str.lower().isin(["true", "1", "yes"])


def load_stage(stage: str, variant: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    failed = as_bool(df, "failed", False)
    df = df.loc[~failed].copy()
    df["stage"] = stage
    df["variant"] = variant
    df["room_type"] = as_str(df, "room_type", "ALL")
    for col in set(CP6 + CIR12 + CP16):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if stage == "stage1":
        df["y_true"] = as_bool(df, "is_nlos", False).astype(int)
        df["scope"] = "ALL"
        df["label_component"] = "current_0p20"
    else:
        has_los = as_bool(df, "has_los_path", True)
        ratio = as_num(df, "bounce_to_los_ratio_mid", 0.0)
        geo = ~has_los
        bounce = has_los & (ratio >= 0.33)
        df["y_true"] = (geo | bounce).astype(int)
        df["is_nlos_geo"] = geo
        df["is_nlos_bounce_0p33"] = bounce
        df["label_component"] = "negative"
        df.loc[bounce & ~geo, "label_component"] = "bounce"
        df.loc[geo, "label_component"] = "geo"
    return df


def usable(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [f for f in features if f not in df.columns]
    if missing:
        return pd.DataFrame()
    cols = ["y_true", "stage", "variant", "room_type", "label_component"] + features
    out = df[cols].copy()
    mask = np.isfinite(out[features].to_numpy(float)).all(axis=1)
    return out.loc[mask].copy()


def model_scores(df: pd.DataFrame, features: list[str], scheme: str) -> np.ndarray:
    X = df[features].to_numpy(float)
    y = df["y_true"].to_numpy(int)
    if len(np.unique(y)) < 2:
        return np.full(len(df), np.nan)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))
    pred = np.full(len(df), np.nan)
    if scheme == "fit_all":
        model.fit(X, y)
        return model.predict_proba(X)[:, 1]
    if scheme == "stratified5":
        splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)
    elif scheme == "room_label_stratified5":
        strata = df["room_type"].astype(str).to_numpy() + "|" + y.astype(str)
        splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, strata)
    elif scheme == "leave_one_room_out":
        splits = LeaveOneGroupOut().split(X, y, df["room_type"].astype(str).to_numpy())
    else:
        raise ValueError(scheme)
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        model.fit(X[tr], y[tr])
        pred[te] = model.predict_proba(X[te])[:, 1]
    return pred


def safe_auc(y, score) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    if mask.sum() < 3 or len(np.unique(y[mask])) < 2:
        return np.nan
    return float(roc_auc_score(y[mask], score[mask]))


def safe_pr_auc(y, score) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    if mask.sum() < 3 or len(np.unique(y[mask])) < 2:
        return np.nan
    return float(average_precision_score(y[mask], score[mask]))


def threshold_metrics(y, cir_score, joint_score) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    cir_hat = np.asarray(cir_score) >= 0.5
    joint_hat = np.asarray(joint_score) >= 0.5
    valid = np.isfinite(cir_score) & np.isfinite(joint_score)
    if not valid.any():
        return {"CP_save": np.nan, "CP_harm": np.nan, "Net_recovery": np.nan, "Recovery_given_CIR_fail": np.nan, "Harm_given_CIR_correct": np.nan}
    cir_ok = cir_hat[valid] == y[valid]
    joint_ok = joint_hat[valid] == y[valid]
    save = (~cir_ok) & joint_ok
    harm = cir_ok & (~joint_ok)
    return {
        "CP_save": float(save.mean()),
        "CP_harm": float(harm.mean()),
        "Net_recovery": float(save.mean() - harm.mean()),
        "Recovery_given_CIR_fail": float(save.sum() / max((~cir_ok).sum(), 1)),
        "Harm_given_CIR_correct": float(harm.sum() / max(cir_ok.sum(), 1)),
    }


def operating(y, score, prefix: str) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    out = {}
    if mask.sum() < 3 or len(np.unique(y[mask])) < 2:
        for v in [1, 5, 10]:
            out[f"{prefix}_Recall_at_FPR_{v}pct"] = np.nan
        for v in [80, 90, 95]:
            out[f"{prefix}_LoS_FPR_at_NLoS_recall_{v}pct"] = np.nan
        return out
    fpr, tpr, _ = roc_curve(y[mask], score[mask])
    for target in [0.01, 0.05, 0.10]:
        valid = tpr[fpr <= target]
        out[f"{prefix}_Recall_at_FPR_{int(target * 100)}pct"] = float(np.max(valid)) if valid.size else np.nan
    for target in [0.80, 0.90, 0.95]:
        valid = fpr[tpr >= target]
        out[f"{prefix}_LoS_FPR_at_NLoS_recall_{int(target * 100)}pct"] = float(np.min(valid)) if valid.size else np.nan
    return out


def bootstrap_delta(df: pd.DataFrame, cir_score, joint_score) -> tuple[float, float, float, float]:
    y = df["y_true"].to_numpy(int)
    cir_score = np.asarray(cir_score, dtype=float)
    joint_score = np.asarray(joint_score, dtype=float)
    valid = np.isfinite(cir_score) & np.isfinite(joint_score)
    y, cir_score, joint_score = y[valid], cir_score[valid], joint_score[valid]
    if len(y) < 5 or len(np.unique(y)) < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    rng = np.random.default_rng(SEED)
    groups = pd.Series(y.astype(str))
    idx = np.arange(len(y))
    group_idx = [idx[groups.to_numpy() == g] for g in pd.unique(groups)]
    deltas, nets = [], []
    for _ in range(N_BOOT):
        sidx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in group_idx])
        if len(np.unique(y[sidx])) < 2:
            continue
        deltas.append(roc_auc_score(y[sidx], joint_score[sidx]) - roc_auc_score(y[sidx], cir_score[sidx]))
        nets.append(threshold_metrics(y[sidx], cir_score[sidx], joint_score[sidx])["Net_recovery"])
    return (
        float(np.percentile(deltas, 2.5)) if deltas else np.nan,
        float(np.percentile(deltas, 97.5)) if deltas else np.nan,
        float(np.percentile(nets, 2.5)) if nets else np.nan,
        float(np.percentile(nets, 97.5)) if nets else np.nan,
    )


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    sub = df[cols].copy()
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in sub.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float) or isinstance(val, np.floating):
                vals.append("nan" if not np.isfinite(val) else f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def eval_feature_set(df: pd.DataFrame, stage: str, variant: str, feature_set: str, features: list[str], scheme: str, scope: str) -> dict:
    u = usable(df, features)
    if u.empty:
        return {
            "stage": stage,
            "variant": variant,
            "scope": scope,
            "cv_scheme": scheme,
            "feature_set": feature_set,
            "n": 0,
            "n_pos": 0,
            "status": "missing_or_nonfinite_features",
        }
    if scope.startswith("room:"):
        room = scope.split(":", 1)[1]
        u = u.loc[u["room_type"].astype(str) == room].copy()
    elif scope.startswith("component:"):
        comp = scope.split(":", 1)[1]
        u = u.loc[(u["label_component"].astype(str) == comp) | (u["y_true"] == 0)].copy()
    if len(u) < 5 or u["y_true"].nunique() < 2:
        return {"stage": stage, "variant": variant, "scope": scope, "cv_scheme": scheme, "feature_set": feature_set, "n": len(u), "n_pos": int(u["y_true"].sum()) if len(u) else 0}
    score = model_scores(u, features, scheme)
    row = {
        "stage": stage,
        "variant": variant,
        "scope": scope,
        "cv_scheme": scheme,
        "feature_set": feature_set,
        "n": len(u),
        "n_pos": int(u["y_true"].sum()),
        "AUC": safe_auc(u["y_true"], score),
        "PR_AUC": safe_pr_auc(u["y_true"], score),
    }
    row.update(operating(u["y_true"], score, feature_set))
    return row


def main() -> None:
    rows = []
    single_rows = []
    coverage_rows = []
    for stage, variants in VARIANTS.items():
        for variant, path in variants.items():
            df = load_stage(stage, variant, path)
            if df.empty:
                coverage_rows.append({"stage": stage, "variant": variant, "source": str(path), "status": "missing"})
                continue
            coverage_rows.append({
                "stage": stage,
                "variant": variant,
                "source": str(path),
                "status": "loaded",
                "n_valid": len(df),
                "n_pos": int(df["y_true"].sum()),
                "cp16_present": int(all(c in df.columns for c in CP16)),
                "cp16_nan_count": int(df[[c for c in CP16 if c in df.columns]].isna().sum().sum()) if any(c in df.columns for c in CP16) else np.nan,
                "cp16_inf_count": int(np.isinf(df[[c for c in CP16 if c in df.columns]].to_numpy(float)).sum()) if all(c in df.columns for c in CP16) else np.nan,
            })
            schemes = ["stratified5"] if stage == "stage1" else ["fit_all", "stratified5", "room_label_stratified5", "leave_one_room_out"]
            scopes = ["ALL"] if stage == "stage1" else ["ALL"] + [f"room:{r}" for r in sorted(df["room_type"].astype(str).unique())] + ["component:geo", "component:bounce"]
            for scheme in schemes:
                for scope in scopes:
                    for name, feats in FEATURE_SETS.items():
                        if scheme == "leave_one_room_out" and scope != "ALL":
                            continue
                        rows.append(eval_feature_set(df, stage, variant, name, feats, scheme, scope))
                    if all(c in df.columns for c in CP16):
                        for feat in CP16:
                            if scheme == "leave_one_room_out" and scope != "ALL":
                                continue
                            single_rows.append(eval_feature_set(df, stage, variant, f"CP16_single:{feat}", [feat], scheme, scope))

            if all(c in df.columns for c in CIR12 + CP16):
                u = usable(df, CIR12 + CP16)
                if stage == "stage1":
                    scheme = "stratified5"
                    cir_score = model_scores(u, CIR12, scheme)
                    cp_score = model_scores(u, CP16, scheme)
                    joint_score = model_scores(u, CIR12 + CP16, scheme)
                    base_row = {"stage": stage, "variant": variant, "scope": "ALL", "cv_scheme": scheme, "feature_set": "Joint_CIR_plus_CP_all"}
                    base_row.update({
                        "AUC_CIR": safe_auc(u["y_true"], cir_score),
                        "AUC_CP": safe_auc(u["y_true"], cp_score),
                        "AUC_Joint": safe_auc(u["y_true"], joint_score),
                        "Delta_AUC": safe_auc(u["y_true"], joint_score) - safe_auc(u["y_true"], cir_score),
                    })
                    base_row.update(threshold_metrics(u["y_true"].to_numpy(int), cir_score, joint_score))
                    lo, hi, nlo, nhi = bootstrap_delta(u, cir_score, joint_score)
                    base_row.update({"Delta_AUC_boot_CI_low": lo, "Delta_AUC_boot_CI_high": hi, "Net_recovery_CI_low": nlo, "Net_recovery_CI_high": nhi})
                    rows.append(base_row)
                else:
                    for scheme in ["stratified5", "room_label_stratified5", "leave_one_room_out"]:
                        cir_score = model_scores(u, CIR12, scheme)
                        cp_score = model_scores(u, CP16, scheme)
                        joint_score = model_scores(u, CIR12 + CP16, scheme)
                        base_row = {"stage": stage, "variant": variant, "scope": "ALL", "cv_scheme": scheme, "feature_set": "paired_primary_joint_cp16"}
                        base_row.update({
                            "AUC_CIR": safe_auc(u["y_true"], cir_score),
                            "AUC_CP": safe_auc(u["y_true"], cp_score),
                            "AUC_Joint": safe_auc(u["y_true"], joint_score),
                            "Delta_AUC": safe_auc(u["y_true"], joint_score) - safe_auc(u["y_true"], cir_score),
                        })
                        base_row.update(threshold_metrics(u["y_true"].to_numpy(int), cir_score, joint_score))
                        lo, hi, nlo, nhi = bootstrap_delta(u, cir_score, joint_score)
                        base_row.update({"Delta_AUC_boot_CI_low": lo, "Delta_AUC_boot_CI_high": hi, "Net_recovery_CI_low": nlo, "Net_recovery_CI_high": nhi})
                        rows.append(base_row)

    metrics = pd.DataFrame(rows)
    singles = pd.DataFrame(single_rows)
    coverage = pd.DataFrame(coverage_rows)
    metrics.to_csv(OUT_DIR / "theta0_cp16_stage1_stage2_metrics.csv", index=False)
    singles.to_csv(OUT_DIR / "theta0_cp16_single_feature_auc.csv", index=False)
    coverage.to_csv(OUT_DIR / "theta0_cp16_coverage.csv", index=False)

    primary = metrics[
        (metrics["scope"] == "ALL")
        & (metrics["feature_set"].isin([
            "CIR_only_12",
            "CP6_canonical",
            "Joint_CIR_plus_CP6",
            "CP_all_RH_LH_16",
            "CP_pruned_6",
            "CP_low_overlap_4",
            "Joint_CIR_plus_CP_all",
            "paired_primary_joint_cp16",
        ]))
    ].copy()
    primary.to_csv(OUT_DIR / "theta0_cp16_primary_metrics.csv", index=False)

    md = OUT_DIR / "theta0_cp16_stage1_stage2_comparison_report.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Theta0 Peak-Aligned CP16 Stage 1/2 Comparison\n\n")
        f.write("The new rerun uses local `theta=0, phi=90`, i.e. local `+z`, as the nominal peak and maps it to Tx `-z` and Rx `+z`.\n\n")
        f.write("## Sources\n\n")
        for _, r in coverage.iterrows():
            f.write(f"- {r['stage']} / {r['variant']}: {r['status']} / `{r['source']}`\n")
        f.write("\n## Primary Metrics\n\n")
        cols = [c for c in ["stage", "variant", "scope", "cv_scheme", "feature_set", "n", "n_pos", "AUC", "PR_AUC", "AUC_CIR", "AUC_CP", "AUC_Joint", "Delta_AUC", "Delta_AUC_boot_CI_low", "Delta_AUC_boot_CI_high", "Net_recovery", "Net_recovery_CI_low", "Net_recovery_CI_high"] if c in primary.columns]
        f.write(markdown_table(primary, cols))
        f.write("\n\n## CP16 Single Features\n\n")
        if not singles.empty:
            top = singles[(singles["variant"] == "new_theta0_posz_cp16") & (singles["scope"] == "ALL")].sort_values(["stage", "AUC"], ascending=[True, False])
            f.write(markdown_table(top, ["stage", "cv_scheme", "feature_set", "n", "n_pos", "AUC", "PR_AUC"]))
        f.write("\n")
    print(f"Saved reports to {OUT_DIR}")


if __name__ == "__main__":
    main()

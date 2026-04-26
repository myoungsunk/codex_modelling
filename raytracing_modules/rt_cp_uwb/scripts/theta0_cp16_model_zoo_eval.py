from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "theta0_peak_aligned_cp16_model_zoo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20260426

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
CP3_STAGE2_GREEDY = ["lambda_l_late_fraction", "gamma_delay_linear", "delta_f_l_minus_r_fp"]
CP6_STAGE1_GREEDY = ["delta_tau_l_given_r_s", "s3_fp", "s3_all", "delta_f_l_minus_r_fp", "cp_phase_slope_delay_s", "s3_late"]
CP_PRUNED_6 = [
    "delta_tau_l_given_r_s",
    "gamma_anchor_linear",
    "gamma_delay_linear",
    "lambda_l_late_fraction",
    "cp_phase_slope_delay_s",
    "cp_phase_residual_circvar",
]
CP_LOW_OVERLAP_4 = ["delta_tau_l_given_r_s", "gamma_delay_linear", "lambda_l_late_fraction", "cp_phase_slope_delay_s"]
CP_LATE_ENERGY_5 = ["xpr_late_db", "s3_late", "lambda_l_late_fraction", "gamma_delay_linear", "xpr_all_db"]
CP_PHASE_DELAY_4 = ["delta_tau_l_given_r_s", "cp_phase_slope_delay_s", "cp_phase_residual_circvar", "delta_p_l_given_r_db"]
CP_RATIO_FP_6 = ["xpr_fp_db", "s3_fp", "f_r_fp", "f_l_fp", "delta_f_l_minus_r_fp", "gamma_anchor_linear"]

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


def feature_sets_for_stage(stage: str) -> dict[str, list[str]]:
    cp_greedy = CP6_STAGE1_GREEDY if stage == "stage1" else CP3_STAGE2_GREEDY
    return {
        "CIR12_only": CIR12,
        "CP16_all_only": CP16,
        "CP_greedy_only": cp_greedy,
        "CP_pruned6_only": CP_PRUNED_6,
        "CP_low_overlap4_only": CP_LOW_OVERLAP_4,
        "CP_late_energy5_only": CP_LATE_ENERGY_5,
        "CP_phase_delay4_only": CP_PHASE_DELAY_4,
        "CP_ratio_fp6_only": CP_RATIO_FP_6,
        "Joint_CIR12_CP16_all": CIR12 + CP16,
        "Joint_CIR12_CP_greedy": CIR12 + cp_greedy,
        "Joint_CIR12_CP_pruned6": CIR12 + CP_PRUNED_6,
        "Joint_CIR12_CP_low_overlap4": CIR12 + CP_LOW_OVERLAP_4,
        "Joint_CIR12_CP_late_energy5": CIR12 + CP_LATE_ENERGY_5,
        "Joint_CIR12_CP_phase_delay4": CIR12 + CP_PHASE_DELAY_4,
        "Joint_CIR12_CP_ratio_fp6": CIR12 + CP_RATIO_FP_6,
    }


def model_zoo() -> dict[str, object]:
    return {
        "logistic_l2": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=5000, solver="lbfgs", random_state=SEED),
        ),
        "logistic_l1": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=5000, solver="liblinear", penalty="l1", C=0.5, random_state=SEED),
        ),
        "svm_rbf": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED),
        ),
        "knn_15_distance": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=15, weights="distance"),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        "gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingClassifier(random_state=SEED, n_estimators=250, learning_rate=0.035, max_depth=2),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(random_state=SEED, max_iter=250, learning_rate=0.035, l2_regularization=0.05),
        ),
    }


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def resolve_col(df: pd.DataFrame, base: str) -> str:
    for name in [base, f"{base}_cases", f"{base}_results", f"{base}_cases_tbl", f"{base}_results_tbl"]:
        if name in df.columns:
            return name
    raise KeyError(base)


def load_stage(stage: str) -> pd.DataFrame:
    df = pd.read_csv(SOURCES[stage])
    if "failed" in df:
        df = df.loc[~as_bool(df["failed"])].copy()
    for col in CIR12 + CP16:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if stage == "stage1":
        df["y_true"] = as_bool(df["is_nlos"]).astype(int)
        df["room_type_eval"] = "stage1"
    else:
        has_los = as_bool(df["has_los_path"])
        ratio = pd.to_numeric(df["bounce_to_los_ratio_mid"], errors="coerce")
        df["y_true"] = ((~has_los) | (has_los & (ratio >= 0.33))).astype(int)
        df["room_type_eval"] = df[resolve_col(df, "room_type")].astype(str)
    return df.reset_index(drop=True)


def cv_splits(df: pd.DataFrame, stage: str, scheme: str) -> list[tuple[np.ndarray, np.ndarray]]:
    y = df["y_true"].to_numpy(int)
    if scheme == "stratified5":
        return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    if scheme == "room_label_stratified5":
        strata = df["room_type_eval"].astype(str).to_numpy() + "|" + y.astype(str)
        return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), strata))
    if scheme == "leave_one_room_out":
        return list(LeaveOneGroupOut().split(np.zeros(len(y)), y, df["room_type_eval"].astype(str).to_numpy()))
    raise ValueError(scheme)


def predict_proba_or_score(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    score = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-score))


def eval_one(df: pd.DataFrame, stage: str, scheme: str, feature_set: str, features: list[str], model_name: str, model_template: object) -> tuple[dict, pd.DataFrame]:
    y = df["y_true"].to_numpy(int)
    X = df[features].to_numpy(float)
    pred = np.full(len(df), np.nan)
    for fold, (tr, te) in enumerate(cv_splits(df, stage, scheme), start=1):
        if len(np.unique(y[tr])) < 2:
            continue
        model = clone_model(model_template)
        model.fit(X[tr], y[tr])
        pred[te] = predict_proba_or_score(model, X[te])
    mask = np.isfinite(pred)
    yv = y[mask]
    pv = pred[mask]
    yhat = (pv >= 0.5).astype(int)
    metrics = {
        "stage": stage,
        "cv_scheme": scheme,
        "feature_set": feature_set,
        "feature_family": family_name(feature_set),
        "n_features": len(features),
        "model": model_name,
        "n": int(mask.sum()),
        "n_pos": int(yv.sum()),
        "AUC": float(roc_auc_score(yv, pv)) if len(np.unique(yv)) == 2 else np.nan,
        "PR_AUC": float(average_precision_score(yv, pv)) if len(np.unique(yv)) == 2 else np.nan,
        "accuracy_at_0p5": float(accuracy_score(yv, yhat)),
        "balanced_accuracy_at_0p5": float(balanced_accuracy_score(yv, yhat)),
        "f1_at_0p5": float(f1_score(yv, yhat, zero_division=0)),
        "features": ";".join(features),
    }
    pred_df = pd.DataFrame(
        {
            "stage": stage,
            "cv_scheme": scheme,
            "feature_set": feature_set,
            "model": model_name,
            "case_id": df.loc[mask, "case_id"].to_numpy(),
            "y_true": yv,
            "score": pv,
            "pred_0p5": yhat,
        }
    )
    return metrics, pred_df


def clone_model(model: object) -> object:
    import copy

    return copy.deepcopy(model)


def family_name(feature_set: str) -> str:
    if feature_set.startswith("CIR"):
        return "cir_only"
    if feature_set.startswith("CP"):
        return "cp_only"
    return "joint"


def main() -> None:
    metrics_rows = []
    prediction_frames = []
    models = model_zoo()
    for stage in ["stage1", "stage2"]:
        df = load_stage(stage)
        schemes = ["stratified5"] if stage == "stage1" else ["room_label_stratified5", "leave_one_room_out"]
        fsets = feature_sets_for_stage(stage)
        for scheme in schemes:
            for feature_set, features in fsets.items():
                finite = np.isfinite(df[features].to_numpy(float)).all(axis=1)
                stage_df = df.loc[finite].reset_index(drop=True)
                for model_name, model in models.items():
                    metrics, preds = eval_one(stage_df, stage, scheme, feature_set, features, model_name, model)
                    metrics_rows.append(metrics)
                    if feature_set in {"CIR12_only", "CP_greedy_only", "CP16_all_only", "Joint_CIR12_CP_greedy", "Joint_CIR12_CP16_all"}:
                        prediction_frames.append(preds)
                    print(stage, scheme, feature_set, model_name, f"AUC={metrics['AUC']:.4f}", f"PR={metrics['PR_AUC']:.4f}")
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["stage", "cv_scheme", "AUC"], ascending=[True, True, False])
    metrics_df.to_csv(OUT_DIR / "model_zoo_metrics.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(OUT_DIR / "model_zoo_selected_predictions.csv", index=False)
    write_summary(metrics_df)
    print(f"Saved model zoo outputs to {OUT_DIR}")


def write_summary(metrics: pd.DataFrame) -> None:
    md = OUT_DIR / "model_zoo_summary.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Theta0 CP16 Model Zoo Evaluation\n\n")
        f.write("Feature families are evaluated as CIR-only, CP-only variants, and joint variants. Stage 1 uses stratified 5-fold CV; Stage 2 uses room-label stratified 5-fold CV plus leave-one-room-out.\n\n")
        for stage in metrics["stage"].unique():
            for scheme in metrics.loc[metrics["stage"] == stage, "cv_scheme"].unique():
                sub = metrics[(metrics["stage"] == stage) & (metrics["cv_scheme"] == scheme)]
                f.write(f"## {stage} / {scheme}\n\n")
                top = sub.head(15)
                f.write("| rank | family | feature_set | model | n_features | AUC | PR_AUC | bal_acc@0.5 | F1@0.5 |\n")
                f.write("|---:|---|---|---|---:|---:|---:|---:|---:|\n")
                for rank, (_, r) in enumerate(top.iterrows(), start=1):
                    f.write(
                        f"| {rank} | {r.feature_family} | {r.feature_set} | {r.model} | {int(r.n_features)} | {r.AUC:.6f} | {r.PR_AUC:.6f} | {r.balanced_accuracy_at_0p5:.6f} | {r.f1_at_0p5:.6f} |\n"
                    )
                f.write("\n")
                best_by_family = sub.sort_values("AUC", ascending=False).groupby("feature_family", as_index=False).head(1)
                f.write("Best by family:\n\n")
                f.write("| family | feature_set | model | AUC | PR_AUC |\n")
                f.write("|---|---|---|---:|---:|\n")
                for _, r in best_by_family.iterrows():
                    f.write(f"| {r.feature_family} | {r.feature_set} | {r.model} | {r.AUC:.6f} | {r.PR_AUC:.6f} |\n")
                f.write("\n")


if __name__ == "__main__":
    main()

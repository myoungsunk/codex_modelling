from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
STAGE2_CSV = ROOT / "results" / "stage2_theta0_peak_aligned_cp16_phi_fastest" / "stage2_900_theta0_peak_aligned_cp16_phi_fastest_det.csv"
OUT_DIR = ROOT / "results" / "stage3_theta0_peak_aligned_cp16"
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
CP3 = ["lambda_l_late_fraction", "gamma_delay_linear", "delta_f_l_minus_r_fp"]
JOINT = CIR12 + CP3


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def col(df: pd.DataFrame, base: str) -> str:
    for name in [base, f"{base}_cases", f"{base}_results", f"{base}_cases_tbl", f"{base}_results_tbl"]:
        if name in df.columns:
            return name
    raise KeyError(base)


def load_stage2() -> pd.DataFrame:
    df = pd.read_csv(STAGE2_CSV)
    df = df.loc[~as_bool(df["failed"])].copy()
    for name in CIR12 + CP3 + ["bounce_to_los_ratio_mid", "snr_db", "xpol_coupling_db"]:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df["los_angle_from_anchor_bore_deg"] = pd.to_numeric(df[col(df, "los_angle_from_anchor_bore_deg")], errors="coerce")
    df["room_type"] = df[col(df, "room_type")].astype(str)
    df["grid_layer"] = pd.to_numeric(df[col(df, "grid_layer")], errors="coerce")
    df["dominant_wall_material"] = df[col(df, "dominant_wall_material")].astype(str)
    has_los = as_bool(df["has_los_path"])
    ratio = df["bounce_to_los_ratio_mid"]
    df["is_nlos_geo"] = ~has_los
    df["is_nlos_bounce_0p33"] = has_los & (ratio >= 0.33)
    df["is_nlos_mixed_0p33"] = df["is_nlos_geo"] | df["is_nlos_bounce_0p33"]
    df["label_component"] = "negative"
    df.loc[df["is_nlos_bounce_0p33"] & ~df["is_nlos_geo"], "label_component"] = "bounce"
    df.loc[df["is_nlos_geo"], "label_component"] = "geo"
    finite = np.isfinite(df[CIR12 + CP3].to_numpy(float)).all(axis=1)
    return df.loc[finite].reset_index(drop=True)


def score_auc(df: pd.DataFrame, features: list[str], label: str) -> tuple[float, float]:
    y = df[label].astype(int).to_numpy()
    if len(df) < 20 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    X = df[features].to_numpy(float)
    pred = np.full(len(df), np.nan)
    strata = df["room_type"].astype(str).to_numpy() + "|" + y.astype(str)
    n_min = pd.Series(strata).value_counts().min()
    if n_min < 2:
        return np.nan, np.nan
    n_splits = max(2, min(5, int(n_min)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in cv.split(np.zeros(len(y)), strata):
        if len(np.unique(y[tr])) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))
        model.fit(X[tr], y[tr])
        pred[te] = model.predict_proba(X[te])[:, 1]
    mask = np.isfinite(pred)
    if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(y[mask], pred[mask])), float(average_precision_score(y[mask], pred[mask]))


def bin_label(s: pd.Series, n_bins: int = 5) -> pd.Series:
    try:
        return pd.qcut(s, q=n_bins, duplicates="drop").astype(str)
    except ValueError:
        return pd.Series(["all"] * len(s), index=s.index)


def make_cells(df: pd.DataFrame) -> pd.DataFrame:
    work = add_regime_bins(df)
    axes = ["room_type", "los_angle_bin", "snr_bin", "xpol_bin", "grid_layer_bin", "dominant_wall_material_bin"]
    labels = {"GEO": "is_nlos_geo", "BOUNCE": "is_nlos_bounce_0p33"}
    rows = []
    for group, label in labels.items():
        subset = work if group == "GEO" else work.loc[as_bool(work["has_los_path"])].copy()
        for axis in axes:
            for value, cell in subset.groupby(axis, dropna=False):
                y = cell[label].astype(int)
                if len(cell) < 20 or y.nunique() < 2 or y.sum() < 5 or (len(y) - y.sum()) < 5:
                    continue
                auc_cir, pr_cir = score_auc(cell, CIR12, label)
                auc_joint, pr_joint = score_auc(cell, JOINT, label)
                if not np.isfinite(auc_cir) or not np.isfinite(auc_joint):
                    continue
                rows.append(
                    {
                        "group": group,
                        "label_name": label,
                        "axis": axis,
                        "axis_value": str(value),
                        "regime_desc": f"{axis}={value}",
                        "n": len(cell),
                        "n_pos": int(y.sum()),
                        "n_neg": int(len(y) - y.sum()),
                        "auc_cir": auc_cir,
                        "auc_joint": auc_joint,
                        "delta_auc": auc_joint - auc_cir,
                        "pr_auc_cir": pr_cir,
                        "pr_auc_joint": pr_joint,
                        "delta_pr_auc": pr_joint - pr_cir,
                    }
                )
    cells = pd.DataFrame(rows)
    cells = cells.sort_values(["delta_auc", "n"], ascending=[False, False]).reset_index(drop=True)
    return cells


def select_regimes(cells: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for group in ["GEO", "BOUNCE"]:
        g = cells[(cells["group"] == group) & (cells["delta_auc"] > 0)].copy()
        used_axis = set()
        for _, row in g.iterrows():
            if row["axis"] in used_axis:
                continue
            selected.append(row)
            used_axis.add(row["axis"])
            if sum(r["group"] == group for r in selected) >= 3:
                break
    return pd.DataFrame(selected).reset_index(drop=True)


def representative_cases(df: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    used_case_ids: set[int] = set()
    df = add_regime_bins(df)
    for _, regime in regimes.iterrows():
        group = regime["group"]
        axis = regime["axis"]
        value = str(regime["axis_value"])
        label = regime["label_name"]
        work = df if group == "GEO" else df.loc[as_bool(df["has_los_path"])].copy()
        mask = work[axis].astype(str) == value
        pool = work.loc[mask & ~work["case_id"].isin(used_case_ids)].copy()
        if pool.empty:
            continue
        pool["target_label"] = pool[label].astype(bool)
        pos = pool.loc[pool["target_label"]].copy()
        neg = pool.loc[~pool["target_label"]].copy()
        chosen = pd.concat([nearest_cases(pos, 3), nearest_cases(neg, 2)], ignore_index=True)
        if len(chosen) < 5:
            rest = pool.loc[~pool["case_id"].isin(chosen["case_id"])].copy()
            chosen = pd.concat([chosen, nearest_cases(rest, 5 - len(chosen))], ignore_index=True)
        for _, row in chosen.head(5).iterrows():
            used_case_ids.add(int(row["case_id"]))
            rows.append(
                {
                    "group": group,
                    "reason": "theta0_cp16_stage3_candidate",
                    "regime_desc": regime["regime_desc"],
                    "case_id": int(row["case_id"]),
                    "room_type": row["room_type"],
                    "grid_layer": row["grid_layer"],
                    "anchor_x": row["anchor_x"],
                    "anchor_y": row["anchor_y"],
                    "anchor_z": row["anchor_z"],
                    "tag_x": row["tag_x"],
                    "tag_y": row["tag_y"],
                    "tag_z": row["tag_z"],
                    "los_angle_deg": row["los_angle_from_anchor_bore_deg"],
                    "snr_db": row["snr_db"],
                    "xpol_coupling_db_expected": row["xpol_coupling_db"],
                    "label_component": row["label_component"],
                    "label_positive": bool(row["target_label"]),
                    "lambda_l_late_fraction": row["lambda_l_late_fraction"],
                    "gamma_delay_linear": row["gamma_delay_linear"],
                    "delta_f_l_minus_r_fp": row["delta_f_l_minus_r_fp"],
                }
            )
    out = pd.DataFrame(rows).head(30).copy()
    out.insert(0, "hfss_rank", np.arange(1, len(out) + 1))
    return out


def nearest_cases(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if df.empty or k <= 0:
        return df.head(0)
    cols = ["tag_x", "tag_y", "tag_z", "los_angle_from_anchor_bore_deg", "snr_db", "xpol_coupling_db"] + CP3
    X = df[cols].to_numpy(float)
    mu = np.nanmedian(X, axis=0)
    sig = np.nanstd(X, axis=0)
    sig[sig < 1e-9] = 1.0
    d = np.sqrt(np.nansum(((X - mu) / sig) ** 2, axis=1))
    return df.iloc[np.argsort(d)[:k]].copy()


def add_regime_bins(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["los_angle_bin"] = bin_label(work["los_angle_from_anchor_bore_deg"])
    work["snr_bin"] = bin_label(work["snr_db"])
    work["xpol_bin"] = bin_label(work["xpol_coupling_db"])
    work["grid_layer_bin"] = work["grid_layer"].astype(str)
    work["dominant_wall_material_bin"] = work["dominant_wall_material"].astype(str)
    return work


def write_summary(df: pd.DataFrame, cells: pd.DataFrame, regimes: pd.DataFrame, cases: pd.DataFrame) -> None:
    md = OUT_DIR / "stage3_theta0_cp16_selection_summary.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Stage 3 Theta0 CP16 Selection\n\n")
        f.write("This Stage 3 selection is regenerated from the theta0 peak-aligned CP16 Stage 2 full rerun. It does not overwrite the legacy `results/stage3` deterministic list.\n\n")
        f.write(f"- source_csv: `{STAGE2_CSV}`\n")
        f.write(f"- valid_stage2_cases: {len(df)}\n")
        f.write(f"- selected_cases: {len(cases)}\n")
        f.write("- primary_joint: `CIR12 + lambda_l_late_fraction + gamma_delay_linear + delta_f_l_minus_r_fp`\n")
        f.write("- groups: GEO and BOUNCE\n\n")
        f.write("## Selected Regimes\n\n")
        f.write("| group | regime | n | n_pos | n_neg | auc_cir | auc_joint | delta_auc | pr_auc_cir | pr_auc_joint |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in regimes.iterrows():
            f.write(
                f"| {r.group} | {r.regime_desc} | {int(r.n)} | {int(r.n_pos)} | {int(r.n_neg)} | {r.auc_cir:.6f} | {r.auc_joint:.6f} | {r.delta_auc:.6f} | {r.pr_auc_cir:.6f} | {r.pr_auc_joint:.6f} |\n"
            )
        f.write("\n## Case Distribution\n\n")
        dist = cases.groupby(["group", "room_type"]).size().reset_index(name="n")
        f.write("| group | room | n |\n")
        f.write("|---|---|---:|\n")
        for _, r in dist.iterrows():
            f.write(f"| {r.group} | {r.room_type} | {int(r.n)} |\n")


def main() -> None:
    df = load_stage2()
    cells = make_cells(df)
    regimes = select_regimes(cells)
    cases = representative_cases(df, regimes)
    cells.to_csv(OUT_DIR / "stage3_theta0_cp16_all_cells.csv", index=False)
    regimes.to_csv(OUT_DIR / "stage3_theta0_cp16_reselection_cells.csv", index=False)
    cases.to_csv(OUT_DIR / "hfss_case_list_theta0_cp16.csv", index=False)
    write_summary(df, cells, regimes, cases)
    print(f"Saved Stage 3 theta0 CP16 selection to {OUT_DIR}")
    print(f"selected_cases={len(cases)}")


if __name__ == "__main__":
    main()

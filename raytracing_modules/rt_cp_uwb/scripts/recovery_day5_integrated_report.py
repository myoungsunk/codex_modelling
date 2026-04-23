from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import textwrap

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


ROOT = Path(r"D:\codex\raytracing_modules\rt_cp_uwb")
RESULTS = ROOT / "results"
RECOVERY = RESULTS / "recovery_mini"
DAY5 = RECOVERY / "day5"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_auc(y_true: pd.Series, scores: pd.Series) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if y.size == 0 or len(np.unique(y)) < 2:
        return math.nan
    ranks = pd.Series(s).rank(method="average").to_numpy()
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_delta_and_net(
    df: pd.DataFrame,
    *,
    y_col: str = "y_true",
    cir_col: str = "cir_score",
    joint_col: str = "joint_score",
    save_col: str = "cp_save",
    harm_col: str = "cp_harm",
    n_boot: int = 2000,
    seed: int = 20260423,
) -> tuple[float, float, float, float]:
    if df.empty:
        return math.nan, math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    nets: list[float] = []
    n = len(df)
    y = df[y_col].to_numpy()
    cir = df[cir_col].to_numpy()
    joint = df[joint_col].to_numpy()
    save = df[save_col].to_numpy()
    harm = df[harm_col].to_numpy()
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if np.unique(y_b).size < 2:
            continue
        delta = safe_auc(pd.Series(y_b), pd.Series(joint[idx])) - safe_auc(pd.Series(y_b), pd.Series(cir[idx]))
        net = float(np.mean(save[idx]) - np.mean(harm[idx]))
        deltas.append(delta)
        nets.append(net)
    if not deltas:
        return math.nan, math.nan, math.nan, math.nan
    return (
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
        float(np.quantile(nets, 0.025)),
        float(np.quantile(nets, 0.975)),
    )


def fmt_ci(low: float, high: float) -> str:
    if pd.isna(low) or pd.isna(high):
        return "NA"
    return f"[{low:.4f}, {high:.4f}]"


def format_md_cell(value) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def df_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_md_cell(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def normalize_day1_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cp_save"] = out["cp_save_0p5"]
    out["cp_harm"] = out["cp_harm_0p5"]
    out["low_confidence_rescue"] = out["low_confidence_rescue_0p5"]
    out["cir_wrong"] = 1 - out["cir_correct_0p5"]
    out["joint_wrong"] = 1 - out["joint_correct_0p5"]
    return out


def normalize_day3_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cp_save"] = out["cp_save_0p5"]
    out["cp_harm"] = out["cp_harm_0p5"]
    out["low_confidence_rescue"] = out["low_confidence_rescue_0p5"]
    out["cir_wrong"] = 1 - out["cir_correct_0p5"]
    out["joint_wrong"] = 1 - out["joint_correct_0p5"]
    return out


def summarize_predictions(df: pd.DataFrame, *, delta_ci: tuple[float, float] | None = None, net_ci: tuple[float, float] | None = None) -> dict:
    auc_cir = safe_auc(df["y_true"], df["cir_score"])
    auc_cp = safe_auc(df["y_true"], df["cp_score"]) if "cp_score" in df.columns else math.nan
    auc_joint = safe_auc(df["y_true"], df["joint_score"])
    delta_auc = auc_joint - auc_cir if not pd.isna(auc_cir) and not pd.isna(auc_joint) else math.nan
    cp_save_rate = float(df["cp_save"].mean())
    cp_harm_rate = float(df["cp_harm"].mean())
    net_recovery = cp_save_rate - cp_harm_rate
    low_conf = int(df["low_confidence"].sum()) if "low_confidence" in df.columns else 0
    if delta_ci is None or net_ci is None:
        d_lo, d_hi, n_lo, n_hi = bootstrap_delta_and_net(df)
    else:
        d_lo, d_hi = delta_ci
        n_lo, n_hi = net_ci
    return {
        "auc_cir": auc_cir,
        "auc_cp": auc_cp,
        "auc_joint": auc_joint,
        "delta_auc": delta_auc,
        "delta_auc_ci_low": d_lo,
        "delta_auc_ci_high": d_hi,
        "delta_auc_ci": fmt_ci(d_lo, d_hi),
        "cir_fail_rate": float(df["cir_wrong"].mean()),
        "cp_save_rate": cp_save_rate,
        "cp_harm_rate": cp_harm_rate,
        "net_recovery": net_recovery,
        "net_recovery_ci_low": n_lo,
        "net_recovery_ci_high": n_hi,
        "net_recovery_ci": fmt_ci(n_lo, n_hi),
        "low_confidence_rescue_rate": float(df.loc[df["low_confidence"] == 1, "low_confidence_rescue"].mean()) if low_conf else math.nan,
        "n_total": int(len(df)),
        "n_pos": int(df["y_true"].sum()),
        "n_neg": int((1 - df["y_true"]).sum()),
    }


def take_first(series: pd.Series) -> float | str:
    vals = series.dropna()
    return vals.iloc[0] if not vals.empty else np.nan


def extract_stage1_stage2_rows(day1_metrics: pd.DataFrame, paper_auc: pd.DataFrame) -> tuple[dict, dict]:
    stage1 = day1_metrics[(day1_metrics["dataset"] == "Stage1") & (day1_metrics["scope"] == "ALL") & (day1_metrics["threshold_type"] == "threshold_0p5")].iloc[0]
    stage2 = day1_metrics[(day1_metrics["dataset"] == "Stage2") & (day1_metrics["scope"] == "ALL") & (day1_metrics["threshold_type"] == "threshold_0p5")].iloc[0]
    stage1_ci = paper_auc[(paper_auc["scope"] == "Stage1") & (paper_auc["subset"] == "ALL")].iloc[0]
    stage2_ci = paper_auc[(paper_auc["scope"] == "Stage2 mixed@0.33") & (paper_auc["subset"] == "ALL")].iloc[0]
    s1 = {
        "row_id": "Stage1 canonical",
        "source_stage": "Stage1 canonical",
        "analysis_scope": "ALL",
        "auc_cir": float(stage1["auc_cir"]),
        "auc_cp": float(stage1["auc_cp"]),
        "auc_joint": float(stage1["auc_joint"]),
        "delta_auc": float(stage1["delta_auc"]),
        "delta_auc_ci_low": float(stage1_ci["delta_ci_low"]),
        "delta_auc_ci_high": float(stage1_ci["delta_ci_high"]),
        "delta_auc_ci": fmt_ci(float(stage1_ci["delta_ci_low"]), float(stage1_ci["delta_ci_high"])),
        "cp_save_rate": float(stage1["cp_save_rate"]),
        "cp_harm_rate": float(stage1["cp_harm_rate"]),
        "net_recovery": float(stage1["net_recovery"]),
        "net_recovery_ci_low": float(stage1["net_recovery_ci_low"]),
        "net_recovery_ci_high": float(stage1["net_recovery_ci_high"]),
        "net_recovery_ci": stage1["net_recovery_ci"],
        "n_total": int(stage1["n"]),
        "n_pos": int(stage1["n_pos"]),
        "n_neg": int(stage1["n_neg"]),
        "room_coverage": "single-slab",
        "patch_ffd_relevance": "high",
        "confound_risk": "low",
    }
    s2 = {
        "row_id": "Stage2 canonical",
        "source_stage": "Stage2 canonical mixed@0.33",
        "analysis_scope": "ALL",
        "auc_cir": float(stage2["auc_cir"]),
        "auc_cp": float(stage2["auc_cp"]),
        "auc_joint": float(stage2["auc_joint"]),
        "delta_auc": float(stage2["delta_auc"]),
        "delta_auc_ci_low": float(stage2_ci["delta_ci_low"]),
        "delta_auc_ci_high": float(stage2_ci["delta_ci_high"]),
        "delta_auc_ci": fmt_ci(float(stage2_ci["delta_ci_low"]), float(stage2_ci["delta_ci_high"])),
        "cp_save_rate": float(stage2["cp_save_rate"]),
        "cp_harm_rate": float(stage2["cp_harm_rate"]),
        "net_recovery": float(stage2["net_recovery"]),
        "net_recovery_ci_low": float(stage2["net_recovery_ci_low"]),
        "net_recovery_ci_high": float(stage2["net_recovery_ci_high"]),
        "net_recovery_ci": stage2["net_recovery_ci"],
        "n_total": int(stage2["n"]),
        "n_pos": int(stage2["n_pos"]),
        "n_neg": int(stage2["n_neg"]),
        "room_coverage": "A/B/C",
        "patch_ffd_relevance": "high",
        "confound_risk": "moderate",
    }
    return s1, s2


def pick_day3_rows(day3_metrics: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    patch_all = day3_metrics[(day3_metrics["antenna_type"] == "patch_ffd") & (day3_metrics["candidate_id"] == "ALL")].iloc[0]
    summary_row = {
        "row_id": "Day3 Stage1 mini",
        "source_stage": "Day3 Stage1 mini",
        "analysis_scope": "patch_ffd ALL",
        "auc_cir": float(patch_all["auc_cir"]),
        "auc_cp": float(patch_all["auc_cp"]),
        "auc_joint": float(patch_all["auc_joint"]),
        "delta_auc": float(patch_all["delta_auc"]),
        "delta_auc_ci_low": float(patch_all["delta_auc_ci_low"]),
        "delta_auc_ci_high": float(patch_all["delta_auc_ci_high"]),
        "delta_auc_ci": fmt_ci(float(patch_all["delta_auc_ci_low"]), float(patch_all["delta_auc_ci_high"])),
        "cp_save_rate": float(patch_all["cp_save_rate"]),
        "cp_harm_rate": float(patch_all["cp_harm_rate"]),
        "net_recovery": float(patch_all["net_recovery"]),
        "net_recovery_ci_low": float(patch_all["net_recovery_ci_low"]),
        "net_recovery_ci_high": float(patch_all["net_recovery_ci_high"]),
        "net_recovery_ci": fmt_ci(float(patch_all["net_recovery_ci_low"]), float(patch_all["net_recovery_ci_high"])),
        "n_total": int(patch_all["n_total"]),
        "n_pos": int(patch_all["n_pos"]),
        "n_neg": int(patch_all["n_neg"]),
        "room_coverage": "single-slab mini",
        "patch_ffd_relevance": "high",
        "confound_risk": "moderate",
    }
    patch_candidates = day3_metrics[(day3_metrics["antenna_type"] == "patch_ffd") & (day3_metrics["candidate_id"].isin(["C1", "C2", "C5"]))]
    return summary_row, patch_candidates


def derive_day4_candidate_metrics(day4_preds: pd.DataFrame) -> tuple[dict, list[dict], pd.DataFrame]:
    overall = day4_preds.copy()
    overall_metrics = summarize_predictions(overall)
    overall_row = {
        "row_id": "Day4 Stage2 mini",
        "source_stage": "Day4 Stage2 mini",
        "analysis_scope": "group_stratified ALL",
        "auc_cir": overall_metrics["auc_cir"],
        "auc_cp": overall_metrics["auc_cp"],
        "auc_joint": overall_metrics["auc_joint"],
        "delta_auc": overall_metrics["delta_auc"],
        "delta_auc_ci_low": overall_metrics["delta_auc_ci_low"],
        "delta_auc_ci_high": overall_metrics["delta_auc_ci_high"],
        "delta_auc_ci": overall_metrics["delta_auc_ci"],
        "cp_save_rate": overall_metrics["cp_save_rate"],
        "cp_harm_rate": overall_metrics["cp_harm_rate"],
        "net_recovery": overall_metrics["net_recovery"],
        "net_recovery_ci_low": overall_metrics["net_recovery_ci_low"],
        "net_recovery_ci_high": overall_metrics["net_recovery_ci_high"],
        "net_recovery_ci": overall_metrics["net_recovery_ci"],
        "n_total": overall_metrics["n_total"],
        "n_pos": overall_metrics["n_pos"],
        "n_neg": overall_metrics["n_neg"],
        "room_coverage": "A/B/C",
        "patch_ffd_relevance": "high",
        "confound_risk": "moderate",
    }

    room_metrics: list[dict] = []
    for room, sub in day4_preds.groupby("room_type"):
        m = summarize_predictions(sub)
        room_metrics.append(
            {
                "row_id": f"Day4 Room {room}",
                "source_stage": "Day4 Stage2 mini",
                "analysis_scope": f"room {room}",
                "auc_cir": m["auc_cir"],
                "auc_cp": m["auc_cp"],
                "auc_joint": m["auc_joint"],
                "delta_auc": m["delta_auc"],
                "delta_auc_ci_low": m["delta_auc_ci_low"],
                "delta_auc_ci_high": m["delta_auc_ci_high"],
                "delta_auc_ci": m["delta_auc_ci"],
                "cp_save_rate": m["cp_save_rate"],
                "cp_harm_rate": m["cp_harm_rate"],
                "net_recovery": m["net_recovery"],
                "net_recovery_ci_low": m["net_recovery_ci_low"],
                "net_recovery_ci_high": m["net_recovery_ci_high"],
                "net_recovery_ci": m["net_recovery_ci"],
                "n_total": m["n_total"],
                "n_pos": m["n_pos"],
                "n_neg": m["n_neg"],
                "room_coverage": room,
                "patch_ffd_relevance": "high",
                "confound_risk": "moderate",
            }
        )

    day4_main_mask = day4_preds["k_factor_estimate"].between(0.1446, 0.3373) & day4_preds["fp_to_total_ratio"].between(0.1264, 0.2522)
    day4_main = day4_preds.loc[day4_main_mask].copy()
    m1 = summarize_predictions(day4_main)
    m1_row = {
        "candidate_id": "M1",
        "regime_name": "mid k_factor + mid fp_to_total_ratio band",
        "condition_definition": "k_factor_estimate in [0.1446, 0.3373] and fp_to_total_ratio in [0.1264, 0.2522]",
        "source_stage": "Day4 derived condition",
        "auc_cir": m1["auc_cir"],
        "auc_cp": m1["auc_cp"],
        "auc_joint": m1["auc_joint"],
        "delta_auc": m1["delta_auc"],
        "delta_auc_ci_low": m1["delta_auc_ci_low"],
        "delta_auc_ci_high": m1["delta_auc_ci_high"],
        "delta_auc_ci": m1["delta_auc_ci"],
        "cp_save_rate": m1["cp_save_rate"],
        "cp_harm_rate": m1["cp_harm_rate"],
        "net_recovery": m1["net_recovery"],
        "net_recovery_ci_low": m1["net_recovery_ci_low"],
        "net_recovery_ci_high": m1["net_recovery_ci_high"],
        "net_recovery_ci": m1["net_recovery_ci"],
        "n_total": m1["n_total"],
        "n_pos": m1["n_pos"],
        "n_neg": m1["n_neg"],
        "room_coverage": "/".join(sorted(day4_main["room_type"].unique())),
        "patch_ffd_relevance": "high",
        "confound_risk": "caution: fp_to_total_ratio vs k_factor_estimate r=0.998",
    }

    p2_rooma = day4_preds[(day4_preds["expected_candidate_id"] == "P2_SENS") & (day4_preds["room_type"] == "A")].copy()
    p2 = summarize_predictions(p2_rooma)
    p2_row = {
        "candidate_id": "P2A",
        "regime_name": "Room A P2_SENS targeted band",
        "condition_definition": "expected_candidate_id=P2_SENS and room_type=A",
        "source_stage": "Day4 targeted candidate",
        "auc_cir": p2["auc_cir"],
        "auc_cp": p2["auc_cp"],
        "auc_joint": p2["auc_joint"],
        "delta_auc": p2["delta_auc"],
        "delta_auc_ci_low": p2["delta_auc_ci_low"],
        "delta_auc_ci_high": p2["delta_auc_ci_high"],
        "delta_auc_ci": p2["delta_auc_ci"],
        "cp_save_rate": p2["cp_save_rate"],
        "cp_harm_rate": p2["cp_harm_rate"],
        "net_recovery": p2["net_recovery"],
        "net_recovery_ci_low": p2["net_recovery_ci_low"],
        "net_recovery_ci_high": p2["net_recovery_ci_high"],
        "net_recovery_ci": p2["net_recovery_ci"],
        "n_total": p2["n_total"],
        "n_pos": p2["n_pos"],
        "n_neg": p2["n_neg"],
        "room_coverage": "A only",
        "patch_ffd_relevance": "high",
        "confound_risk": "room-conditional",
    }

    return overall_row, [m1_row, p2_row], day4_main


def build_final_candidate_table(
    day2_candidates: pd.DataFrame,
    day3_patch_candidates: pd.DataFrame,
    day4_rows: list[dict],
) -> pd.DataFrame:
    final_rows: list[dict] = []
    day2_idx = {row["candidate_id"]: row for _, row in day2_candidates.iterrows()}
    day3_idx = {row["candidate_id"]: row for _, row in day3_patch_candidates.iterrows()}

    def add_row(
        *,
        candidate_id: str,
        regime_name: str,
        final_status: str,
        condition_definition: str,
        source_stage: str,
        supporting_metric: str,
        n_total: int,
        n_pos: int,
        n_neg: int,
        auc_cir: float,
        auc_cp: float,
        auc_joint: float,
        delta_auc: float,
        delta_auc_ci_low: float,
        delta_auc_ci_high: float,
        cp_save_rate: float,
        cp_harm_rate: float,
        net_recovery: float,
        net_recovery_ci_low: float,
        net_recovery_ci_high: float,
        room_coverage: str,
        patch_ffd_relevance: str,
        confound_risk: str,
        recommended_stage3_action: str,
        notes: str,
    ) -> None:
        final_rows.append(
            {
                "candidate_id": candidate_id,
                "regime_name": regime_name,
                "final_status": final_status,
                "condition_definition": condition_definition,
                "source_stage": source_stage,
                "supporting_metric": supporting_metric,
                "n_total": n_total,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "auc_cir": auc_cir,
                "auc_cp": auc_cp,
                "auc_joint": auc_joint,
                "delta_auc": delta_auc,
                "delta_auc_ci_low": delta_auc_ci_low,
                "delta_auc_ci_high": delta_auc_ci_high,
                "delta_auc_ci": fmt_ci(delta_auc_ci_low, delta_auc_ci_high),
                "cp_save_rate": cp_save_rate,
                "cp_harm_rate": cp_harm_rate,
                "net_recovery": net_recovery,
                "net_recovery_ci_low": net_recovery_ci_low,
                "net_recovery_ci_high": net_recovery_ci_high,
                "net_recovery_ci": fmt_ci(net_recovery_ci_low, net_recovery_ci_high),
                "room_coverage": room_coverage,
                "patch_ffd_relevance": patch_ffd_relevance,
                "confound_risk": confound_risk,
                "recommended_stage3_action": recommended_stage3_action,
                "notes": notes,
            }
        )

    c1_d2 = day2_idx["C1"]
    c1_d3 = day3_idx["C1"]
    add_row(
        candidate_id="C1",
        regime_name="low eps broad Stage1 patch regime",
        final_status="conditional_candidate",
        condition_definition="Stage1 eps_r=[2.0026, 3.9999]; Day3 survivor narrows toward eps_r=2.5",
        source_stage="Day2 + Day3",
        supporting_metric="Day3 patch mini net CI stays positive, but Stage4 transfer fails",
        n_total=int(c1_d3["n_total"]),
        n_pos=int(c1_d3["n_pos"]),
        n_neg=int(c1_d3["n_neg"]),
        auc_cir=float(c1_d3["auc_cir"]),
        auc_cp=float(c1_d3["auc_cp"]),
        auc_joint=float(c1_d3["auc_joint"]),
        delta_auc=float(c1_d3["delta_auc"]),
        delta_auc_ci_low=float(c1_d3["delta_auc_ci_low"]),
        delta_auc_ci_high=float(c1_d3["delta_auc_ci_high"]),
        cp_save_rate=float(c1_d3["cp_save_rate"]),
        cp_harm_rate=float(c1_d3["cp_harm_rate"]),
        net_recovery=float(c1_d3["net_recovery"]),
        net_recovery_ci_low=float(c1_d3["net_recovery_ci_low"]),
        net_recovery_ci_high=float(c1_d3["net_recovery_ci_high"]),
        room_coverage="Stage1 only",
        patch_ffd_relevance="high",
        confound_risk="Stage1-only transfer risk",
        recommended_stage3_action="Use only as Stage1-to-HFSS proxy inspiration, not direct 30-case headline",
        notes="verified: Day2 positive and Day3 weak-positive; verified: Day4 P1_MAIN does not transfer",
    )

    c2_d3 = day3_idx["C2"]
    add_row(
        candidate_id="C2",
        regime_name="low eps + mid xpol hotspot",
        final_status="rejected",
        condition_definition="Stage1 eps_r=[2.0026, 3.9999] and xpol_coupling_db=[30, 35]",
        source_stage="Day2 + Day3",
        supporting_metric="Day3 patch mini net recovery turns negative",
        n_total=int(c2_d3["n_total"]),
        n_pos=int(c2_d3["n_pos"]),
        n_neg=int(c2_d3["n_neg"]),
        auc_cir=float(c2_d3["auc_cir"]),
        auc_cp=float(c2_d3["auc_cp"]),
        auc_joint=float(c2_d3["auc_joint"]),
        delta_auc=float(c2_d3["delta_auc"]),
        delta_auc_ci_low=float(c2_d3["delta_auc_ci_low"]),
        delta_auc_ci_high=float(c2_d3["delta_auc_ci_high"]),
        cp_save_rate=float(c2_d3["cp_save_rate"]),
        cp_harm_rate=float(c2_d3["cp_harm_rate"]),
        net_recovery=float(c2_d3["net_recovery"]),
        net_recovery_ci_low=float(c2_d3["net_recovery_ci_low"]),
        net_recovery_ci_high=float(c2_d3["net_recovery_ci_high"]),
        room_coverage="Stage1 only",
        patch_ffd_relevance="high",
        confound_risk="narrow hotspot and CI overlap",
        recommended_stage3_action="Do not prioritize in Stage3",
        notes="verified: Day2 hotspot did not survive Day3 patch validation",
    )

    c3 = day2_idx["C3"]
    add_row(
        candidate_id="C3",
        regime_name="Stage2 mid-SNR band",
        final_status="unresolved",
        condition_definition=str(c3["condition_definition"]),
        source_stage="Day2",
        supporting_metric="Day2 canonical positive, but Day4 targeted replication did not isolate this axis cleanly",
        n_total=int(c3["n_total"]),
        n_pos=int(c3["n_pos"]),
        n_neg=int(c3["n_neg"]),
        auc_cir=math.nan,
        auc_cp=math.nan,
        auc_joint=math.nan,
        delta_auc=float(c3["delta_auc"]),
        delta_auc_ci_low=float(c3["delta_auc_ci_low"]),
        delta_auc_ci_high=float(c3["delta_auc_ci_high"]),
        cp_save_rate=float(c3["cp_save_rate"]),
        cp_harm_rate=float(c3["cp_harm_rate"]),
        net_recovery=float(c3["net_recovery"]),
        net_recovery_ci_low=float(c3["net_recovery_ci_low"]),
        net_recovery_ci_high=float(c3["net_recovery_ci_high"]),
        room_coverage="A/B/C in canonical Stage2 only",
        patch_ffd_relevance="high",
        confound_risk="not directly revalidated in Day4 mini",
        recommended_stage3_action="Keep as supplemental hypothesis only",
        notes="verified: Day2 canonical positive; caution: Day4 did not independently confirm",
    )

    c4 = day2_idx["C4"]
    add_row(
        candidate_id="C4",
        regime_name="Room A concrete pocket",
        final_status="conditional_candidate",
        condition_definition=str(c4["condition_definition"]),
        source_stage="Day2",
        supporting_metric="Day2 room-conditional positive pocket",
        n_total=int(c4["n_total"]),
        n_pos=int(c4["n_pos"]),
        n_neg=int(c4["n_neg"]),
        auc_cir=math.nan,
        auc_cp=math.nan,
        auc_joint=math.nan,
        delta_auc=float(c4["delta_auc"]),
        delta_auc_ci_low=float(c4["delta_auc_ci_low"]),
        delta_auc_ci_high=float(c4["delta_auc_ci_high"]),
        cp_save_rate=float(c4["cp_save_rate"]),
        cp_harm_rate=float(c4["cp_harm_rate"]),
        net_recovery=float(c4["net_recovery"]),
        net_recovery_ci_low=float(c4["net_recovery_ci_low"]),
        net_recovery_ci_high=float(c4["net_recovery_ci_high"]),
        room_coverage="A only",
        patch_ffd_relevance="high",
        confound_risk="room-only and material-specific",
        recommended_stage3_action="Use only as conditional Room A control if extra slots exist",
        notes="verified: positive in Day2 canonical only; caution: not a cross-room claim",
    )

    c5_d3 = day3_idx["C5"]
    add_row(
        candidate_id="C5",
        regime_name="low-confidence CIR posterior band",
        final_status="rejected",
        condition_definition="Stage1 low_confidence flag in [0.4, 0.6]",
        source_stage="Day2 + Day3",
        supporting_metric="AUC gain persists, but patch net recovery turns negative in mini validation",
        n_total=int(c5_d3["n_total"]),
        n_pos=int(c5_d3["n_pos"]),
        n_neg=int(c5_d3["n_neg"]),
        auc_cir=float(c5_d3["auc_cir"]),
        auc_cp=float(c5_d3["auc_cp"]),
        auc_joint=float(c5_d3["auc_joint"]),
        delta_auc=float(c5_d3["delta_auc"]),
        delta_auc_ci_low=float(c5_d3["delta_auc_ci_low"]),
        delta_auc_ci_high=float(c5_d3["delta_auc_ci_high"]),
        cp_save_rate=float(c5_d3["cp_save_rate"]),
        cp_harm_rate=float(c5_d3["cp_harm_rate"]),
        net_recovery=float(c5_d3["net_recovery"]),
        net_recovery_ci_low=float(c5_d3["net_recovery_ci_low"]),
        net_recovery_ci_high=float(c5_d3["net_recovery_ci_high"]),
        room_coverage="Stage1 only",
        patch_ffd_relevance="high",
        confound_risk="decision-boundary artifact risk",
        recommended_stage3_action="Do not use as paper-facing regime",
        notes="verified: Day2 promising; verified: Day3 patch validation rejected it",
    )

    m1 = next(row for row in day4_rows if row["candidate_id"] == "M1")
    add_row(
        candidate_id="M1",
        regime_name=m1["regime_name"],
        final_status="main_candidate",
        condition_definition=m1["condition_definition"],
        source_stage="Day4",
        supporting_metric="Cross-room positive derived band with CP-save > CP-harm",
        n_total=int(m1["n_total"]),
        n_pos=int(m1["n_pos"]),
        n_neg=int(m1["n_neg"]),
        auc_cir=float(m1["auc_cir"]),
        auc_cp=float(m1["auc_cp"]),
        auc_joint=float(m1["auc_joint"]),
        delta_auc=float(m1["delta_auc"]),
        delta_auc_ci_low=float(m1["delta_auc_ci_low"]),
        delta_auc_ci_high=float(m1["delta_auc_ci_high"]),
        cp_save_rate=float(m1["cp_save_rate"]),
        cp_harm_rate=float(m1["cp_harm_rate"]),
        net_recovery=float(m1["net_recovery"]),
        net_recovery_ci_low=float(m1["net_recovery_ci_low"]),
        net_recovery_ci_high=float(m1["net_recovery_ci_high"]),
        room_coverage=str(m1["room_coverage"]),
        patch_ffd_relevance="high",
        confound_risk=str(m1["confound_risk"]),
        recommended_stage3_action="Use as first HFSS preflight positive regime, but keep collinearity caveat",
        notes="verified: repeated positive A/B/C; caution: fp_to_total_ratio and k_factor are effectively one axis",
    )

    p2 = next(row for row in day4_rows if row["candidate_id"] == "P2A")
    add_row(
        candidate_id="P2A",
        regime_name=p2["regime_name"],
        final_status="conditional_candidate",
        condition_definition=p2["condition_definition"],
        source_stage="Day4",
        supporting_metric="Room A-only targeted positive control",
        n_total=int(p2["n_total"]),
        n_pos=int(p2["n_pos"]),
        n_neg=int(p2["n_neg"]),
        auc_cir=float(p2["auc_cir"]),
        auc_cp=float(p2["auc_cp"]),
        auc_joint=float(p2["auc_joint"]),
        delta_auc=float(p2["delta_auc"]),
        delta_auc_ci_low=float(p2["delta_auc_ci_low"]),
        delta_auc_ci_high=float(p2["delta_auc_ci_high"]),
        cp_save_rate=float(p2["cp_save_rate"]),
        cp_harm_rate=float(p2["cp_harm_rate"]),
        net_recovery=float(p2["net_recovery"]),
        net_recovery_ci_low=float(p2["net_recovery_ci_low"]),
        net_recovery_ci_high=float(p2["net_recovery_ci_high"]),
        room_coverage="A only",
        patch_ffd_relevance="high",
        confound_risk="room-conditional",
        recommended_stage3_action="Use only as conditional positive control",
        notes="verified: Day4 positive in Room A; verified: negative in B/C",
    )

    add_row(
        candidate_id="RCGEO",
        regime_name="Room C geo-only pocket",
        final_status="conditional_candidate",
        condition_definition="Stage2 geo_only positives concentrated in Room C; no room-general inference",
        source_stage="Day1 + Stage3 readiness",
        supporting_metric="Geo result exists, but canonical negatives are absent in the focused pocket",
        n_total=35,
        n_pos=35,
        n_neg=0,
        auc_cir=math.nan,
        auc_cp=math.nan,
        auc_joint=math.nan,
        delta_auc=math.nan,
        delta_auc_ci_low=math.nan,
        delta_auc_ci_high=math.nan,
        cp_save_rate=math.nan,
        cp_harm_rate=math.nan,
        net_recovery=math.nan,
        net_recovery_ci_low=math.nan,
        net_recovery_ci_high=math.nan,
        room_coverage="C only",
        patch_ffd_relevance="high",
        confound_risk="Room C-only and geo-only",
        recommended_stage3_action="If kept, write claim as Room C conditional only",
        notes="verified: no main AUC claim possible because class balance is missing",
    )

    add_row(
        candidate_id="IDEAL",
        regime_name="ideal antenna wall/angle pockets",
        final_status="ideal_only",
        condition_definition="ideal x wall_x / wall_y / angle bins from paper_facing_conditional_cells_ci.csv",
        source_stage="paper_facing_conditional_cells_ci.csv",
        supporting_metric="Strong ideal-only delta AUC around 0.20 but not patch-relevant",
        n_total=371,
        n_pos=207,
        n_neg=164,
        auc_cir=math.nan,
        auc_cp=math.nan,
        auc_joint=math.nan,
        delta_auc=0.210027100271003,
        delta_auc_ci_low=0.143072,
        delta_auc_ci_high=0.292035,
        cp_save_rate=math.nan,
        cp_harm_rate=math.nan,
        net_recovery=math.nan,
        net_recovery_ci_low=math.nan,
        net_recovery_ci_high=math.nan,
        room_coverage="Stage1 ideal only",
        patch_ffd_relevance="low",
        confound_risk="antenna-limited",
        recommended_stage3_action="Exclude from main Stage3 patch narrative",
        notes="verified: ideal upper bound exists; verified: not a patch headline",
    )

    return pd.DataFrame(final_rows)


def build_integrated_metrics(
    stage1_row: dict,
    stage2_row: dict,
    day3_summary: dict,
    day4_summary: dict,
    final_candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows = [stage1_row, stage2_row, day3_summary, day4_summary]
    for _, row in final_candidates.iterrows():
        rows.append(
            {
                "row_id": row["candidate_id"],
                "source_stage": row["source_stage"],
                "analysis_scope": row["regime_name"],
                "auc_cir": row["auc_cir"],
                "auc_cp": row["auc_cp"],
                "auc_joint": row["auc_joint"],
                "delta_auc": row["delta_auc"],
                "delta_auc_ci_low": row["delta_auc_ci_low"],
                "delta_auc_ci_high": row["delta_auc_ci_high"],
                "delta_auc_ci": row["delta_auc_ci"],
                "cp_save_rate": row["cp_save_rate"],
                "cp_harm_rate": row["cp_harm_rate"],
                "net_recovery": row["net_recovery"],
                "net_recovery_ci_low": row["net_recovery_ci_low"],
                "net_recovery_ci_high": row["net_recovery_ci_high"],
                "net_recovery_ci": row["net_recovery_ci"],
                "n_total": row["n_total"],
                "n_pos": row["n_pos"],
                "n_neg": row["n_neg"],
                "room_coverage": row["room_coverage"],
                "patch_ffd_relevance": row["patch_ffd_relevance"],
                "confound_risk": row["confound_risk"],
            }
        )
    return pd.DataFrame(rows)


def stage3_audit(
    stage2_det: pd.DataFrame,
    hfss_cases: pd.DataFrame,
    day1_stage2_preds: pd.DataFrame,
    main_candidate_mask_fn,
) -> tuple[dict, pd.DataFrame]:
    merged = hfss_cases.merge(stage2_det, on="case_id", how="left", suffixes=("_hfss", ""))
    preds = normalize_day1_predictions(day1_stage2_preds).merge(stage2_det, on="case_id", how="left", suffixes=("_pred", ""))
    merged["main_candidate_band"] = main_candidate_mask_fn(merged)
    overlap_count = int(merged["main_candidate_band"].sum())
    overlap_by_room = merged.loc[merged["main_candidate_band"]].groupby("room_type").size().to_dict()
    rooms = merged["room_type"].value_counts().to_dict()
    groups = merged.groupby(["group", "room_type"]).size().reset_index(name="n")

    preflight_pool = preds.copy()
    preflight_pool["main_candidate_band"] = main_candidate_mask_fn(preflight_pool)
    preflight_pool["priority"] = (
        preflight_pool["cp_save"].astype(int) * 4
        + preflight_pool["low_confidence_rescue"].astype(int) * 2
        + (preflight_pool["joint_minus_cir"] > 0).astype(int)
    )
    positive_band = preflight_pool[(preflight_pool["main_candidate_band"]) & (preflight_pool["cp_save"] == 1)]
    selected_rows = []
    for room in ["A", "B", "C"]:
        room_rows = positive_band[positive_band["room_type"] == room].sort_values(["priority", "joint_minus_cir"], ascending=[False, False]).head(2)
        selected_rows.append(room_rows)
    negative_control = preflight_pool[(preflight_pool["cp_harm"] == 1)].sort_values("joint_minus_cir").head(2)
    preflight = pd.concat(selected_rows + [negative_control], ignore_index=True)
    if not preflight.empty:
        preflight = preflight[
            [
                "case_id",
                "room_type",
                "snr_db",
                "xpol_coupling_db",
                "fp_to_total_ratio",
                "k_factor_estimate",
                "gamma_cp_3_fp_only",
                "cp_save",
                "cp_harm",
                "joint_minus_cir",
                "is_nlos_geo",
                "is_nlos_bounce_0p33",
                "is_nlos_mixed_0p33",
            ]
        ].drop_duplicates()

    audit = {
        "n_cases": int(len(merged)),
        "room_counts": rooms,
        "group_room_counts": groups,
        "main_overlap_count": overlap_count,
        "main_overlap_by_room": overlap_by_room,
        "has_room_b": bool(rooms.get("B", 0)),
        "room_c_geo_only_count": int(((merged["group"] == "GEO") & (merged["room_type"] == "C")).sum()),
        "xpol_header_ok": "xpol_coupling_db_expected" in hfss_cases.columns,
    }
    return audit, preflight


def plot_delta_dot(integrated: pd.DataFrame, out_path: Path) -> None:
    rows = integrated[integrated["row_id"].isin(["Stage1 canonical", "Stage2 canonical", "Day3 Stage1 mini", "Day4 Stage2 mini"])].copy()
    labels = rows["row_id"].tolist()
    y = np.arange(len(rows))[::-1]
    plt.figure(figsize=(8.0, 4.5))
    plt.errorbar(
        rows["delta_auc"],
        y,
        xerr=[
            rows["delta_auc"] - rows["delta_auc_ci_low"],
            rows["delta_auc_ci_high"] - rows["delta_auc"],
        ],
        fmt="o",
        color="#1f5a8a",
        ecolor="#7aa5c7",
        capsize=4,
        lw=1.5,
    )
    plt.axvline(0, color="black", lw=1, linestyle="--")
    plt.yticks(y, labels)
    plt.xlabel("Delta AUC (Joint - CIR)")
    plt.title("Canonical vs Mini Delta AUC")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_candidate_net(final_candidates: pd.DataFrame, out_path: Path) -> None:
    df = final_candidates[final_candidates["final_status"].isin(["main_candidate", "conditional_candidate", "rejected", "unresolved"])].copy()
    y = np.arange(len(df))[::-1]
    plt.figure(figsize=(8.2, 5.2))
    plt.errorbar(
        df["net_recovery"],
        y,
        xerr=[
            df["net_recovery"] - df["net_recovery_ci_low"],
            df["net_recovery_ci_high"] - df["net_recovery"],
        ],
        fmt="o",
        color="#8a2d2d",
        ecolor="#d39f9f",
        capsize=4,
        lw=1.5,
    )
    plt.axvline(0, color="black", lw=1, linestyle="--")
    plt.yticks(y, df["candidate_id"] + " | " + df["final_status"])
    plt.xlabel("Net recovery")
    plt.title("Candidate Net Recovery with 95% CI")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_save_harm(final_candidates: pd.DataFrame, out_path: Path) -> None:
    df = final_candidates[final_candidates["cp_save_rate"].notna() & final_candidates["cp_harm_rate"].notna()].copy()
    x = np.arange(len(df))
    width = 0.38
    plt.figure(figsize=(9.2, 4.8))
    plt.bar(x - width / 2, df["cp_save_rate"], width, label="CP-save", color="#2c7a7b")
    plt.bar(x + width / 2, df["cp_harm_rate"], width, label="CP-harm", color="#b55239")
    plt.xticks(x, df["candidate_id"], rotation=25, ha="right")
    plt.ylabel("Rate")
    plt.title("CP-save vs CP-harm per Candidate")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_stage3_overlay(stage2_det: pd.DataFrame, hfss_cases: pd.DataFrame, out_path: Path) -> None:
    merged = hfss_cases.merge(stage2_det, on="case_id", how="left", suffixes=("_hfss", ""))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    ax = axes[0]
    ax.scatter(stage2_det["fp_to_total_ratio"], stage2_det["k_factor_estimate"], s=12, c="#c7ccd1", alpha=0.55, edgecolors="none")
    for group, color in [("GEO", "#1f5a8a"), ("BOUNCE", "#b55239")]:
        sub = merged[merged["group"] == group]
        ax.scatter(sub["fp_to_total_ratio"], sub["k_factor_estimate"], s=36, c=color, label=f"HFSS {group}", alpha=0.9)
    rect = patches.Rectangle((0.1264, 0.1446), 0.2522 - 0.1264, 0.3373 - 0.1446, fill=False, edgecolor="#0f6e2d", lw=2)
    ax.add_patch(rect)
    ax.set_xlabel("fp_to_total_ratio")
    ax.set_ylabel("k_factor_estimate")
    ax.set_title("Main Derived Band Overlay")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.scatter(stage2_det["xpol_coupling_db"], stage2_det["los_angle_from_anchor_bore_deg_cases"], s=12, c="#c7ccd1", alpha=0.55, edgecolors="none")
    for group, color in [("GEO", "#1f5a8a"), ("BOUNCE", "#b55239")]:
        sub = merged[merged["group"] == group]
        ax.scatter(sub["xpol_coupling_db"], sub["los_angle_from_anchor_bore_deg_cases"], s=36, c=color, label=f"HFSS {group}", alpha=0.9)
    ax.set_xlabel("xpol_coupling_db expected")
    ax.set_ylabel("los_angle_from_anchor_bore_deg")
    ax.set_title("Current Stage3 Cases on Stage2 Cloud")
    ax.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_claim_support(out_path: Path) -> None:
    claims = [
        "CP is globally dominant",
        "CP is a conditional CIR blind-spot complement",
        "Stage1 low-eps patch regime is positive",
        "Derived mid-k/fp band repeats across rooms",
        "Room C GEO is room-general",
        "Stage3 validates trend/ranking, not 30-case AUC",
    ]
    cols = ["Stage1 canonical", "Stage2 canonical", "Day3", "Day4", "Stage3 expected"]
    matrix = np.array(
        [
            [-1, -1, -1, -1, -1],
            [0, 1, 1, 1, 1],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, -1, -1],
            [0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    cmap = plt.get_cmap("RdYlGn", 3)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(claims)))
    ax.set_yticklabels(claims)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            label = {1: "support", 0: "mixed", -1: "no"}[int(matrix[i, j])]
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="black")
    ax.set_title("Claim Support Matrix")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_markdown(
    integrated: pd.DataFrame,
    final_candidates: pd.DataFrame,
    stage3_audit_info: dict,
    preflight: pd.DataFrame,
) -> None:
    source_lines = [
        "- Generated by `scripts/recovery_day5_integrated_report.py`.",
        "- Primary sources: `results/recovery_mini/day1/day1_metrics_summary.csv`, `results/recovery_mini/day2/day2_candidate_regimes.csv`, `results/recovery_mini/day3/day3_stage1_recovery_mini_metrics.csv`, `results/recovery_mini/day4/day4_stage2_room_stratified_predictions.csv`, `results/stage2/stage2_900_ffd_relabel_det.csv`, `results/stage3/hfss_case_list_det.csv`, `results/code_audit/stage3_readiness_report.md`, `results/code_audit/action_update_20260422.md`, `results/code_audit/paper_facing_auc_ci.md`.",
    ]

    summary_md = [
        "# Day 5 Integrated Summary",
        "",
        *source_lines,
        "",
        "## One-line decision",
        "",
        "- `verified`: Stage 3 should be `PREFLIGHT_FIRST`, not `KEEP`.",
        "",
        "## Blocker",
        "",
        "- `verified`: current Stage 3 deterministic list covers `A=5, B=0, C=25`, so the Day 4 cross-room main positive band cannot be claimed as already covered.",
        "",
        "## Caution",
        "",
        "- `verified`: Day 4 main_candidate is a derived `k_factor_estimate x fp_to_total_ratio` band, but those two features are almost perfectly collinear and should be treated as one signal family.",
        "- `verified`: Room C GEO evidence remains conditional only; negative class support is insufficient inside the focused GEO pocket.",
        "- `verified`: `xpol_coupling_db_expected` in `hfss_case_list_det.csv` is an expected MATLAB proxy range, not an HFSS input parameter.",
        "",
        "## Supplemental",
        "",
        "- `verified`: Stage 1 low-eps patch regime survives Day 3 as a weak-positive controlled result, but it does not transfer cleanly to Day 4 room multipath.",
        "- `verified`: ideal-only wall/angle pockets remain upper-bound only and are not patch-facing claims.",
        "",
        "## No issue",
        "",
        "- `verified`: Day 1 canonical Stage 1/2 values remain intact and distinct from the Day 3/4 mini experiments.",
        "- `verified`: current Stage 3 protocol framing already says feature-trend / ranking validation instead of 30-case AUC reproduction.",
        "",
        "## Final candidate verdicts",
        "",
        df_to_markdown(final_candidates[[
            "candidate_id",
            "final_status",
            "regime_name",
            "delta_auc_ci",
            "net_recovery_ci",
            "room_coverage",
            "confound_risk",
        ]]),
        "",
    ]
    (DAY5 / "day5_integrated_summary.md").write_text("\n".join(summary_md), encoding="utf-8")

    guardrails_md = [
        "# Day 5 Paper Claim Guardrails",
        "",
        *source_lines,
        "",
        "## Allowed claims",
        "",
        "1. `verified`: CP is not a global replacement for CIR, but it can act as a conditional complementary cue when CIR enters specific blind-spot regimes.",
        "2. `verified`: In controlled Stage 1 patch_ffd mini sweeps, low-eps conditions can produce positive CP-save > CP-harm behavior, although the effect narrows relative to the broader Day 2 shortlist.",
        "3. `verified`: Stage 3 HFSS should be framed as regime / feature-trend / ranking validation, not as a 30-case AUC reproduction exercise.",
        "",
        "## Conditional claims",
        "",
        "1. `verified`: A derived mid `k_factor_estimate` and mid `fp_to_total_ratio` band shows positive Day 4 net recovery across A/B/C, but the two axes are collinear and should be reported as one family of conditions.",
        "2. `verified`: Room C GEO behavior can be discussed only as `Room C conditional`; it is not a room-general multipath claim.",
        "3. `verified`: Room A targeted positive pockets such as `P2_SENS` are acceptable only as room-conditional controls.",
        "",
        "## Not allowed claims",
        "",
        "1. `verified`: CP is a globally dominant feature over CIR.",
        "2. `verified`: Multipath richness increases CP utility monotonically from Room A to B to C.",
        "3. `verified`: The retired `eps_r x xpol +0.2666` top cell is the paper headline.",
        "4. `verified`: Room C GEO results generalize across all rooms.",
        "",
        "## Required caveats",
        "",
        "1. `verified`: Stage 2 mini and leave-one-room-out results do not support a room-general transfer of the Day 3 Stage 1 survivor.",
        "2. `verified`: `xpol_coupling_db_expected` is a MATLAB proxy / expected depolarization range only, not an HFSS input variable.",
        "3. `verified`: Day 4 main positive evidence comes from a derived condition; its interpretation must mention the `fp_to_total_ratio` vs `k_factor_estimate` collinearity.",
        "4. `verified`: Any GEO result concentrated in Room C must be labeled `Room C conditional`.",
    ]
    (DAY5 / "day5_paper_claim_guardrails.md").write_text("\n".join(guardrails_md), encoding="utf-8")

    update_md = [
        "# Day 5 Stage 3 Case Update Recommendation",
        "",
        *source_lines,
        "",
        "## Decision",
        "",
        "- `verified`: `PREFLIGHT_FIRST`.",
        "",
        "## Why not KEEP",
        "",
        f"- `verified`: only `{stage3_audit_info['main_overlap_count']} / {stage3_audit_info['n_cases']}` current HFSS cases land inside the Day 4 main derived band.",
        f"- `verified`: overlap by room inside that band is `{stage3_audit_info['main_overlap_by_room']}`; Room B coverage is missing.",
        f"- `verified`: current deterministic room counts are `{stage3_audit_info['room_counts']}` and GEO Room C-only count is `{stage3_audit_info['room_c_geo_only_count']}`.",
        "",
        "## Why not MAJOR_RESELECT",
        "",
        "- `verified`: the current list still contains useful Room C GEO and Room A bounce controls, so full reselection is not required before a smaller validation pass.",
        "",
        "## Preflight recommendation",
        "",
        "- `verified`: run `6-8` HFSS cases first: `2` positive-band cases each from Rooms `A/B/C`, plus `2` negative controls.",
        "- `verified`: keep current Room C GEO slots only as conditional probes; do not use them to stand in for the new cross-room candidate.",
        "- `추측`: after preflight, move to `MINOR_UPDATE` only if the main derived band preserves sign / ranking consistency in at least two rooms.",
        "",
        "## Suggested preflight pool",
        "",
        (df_to_markdown(preflight) if not preflight.empty else "- No preflight pool was generated."),
    ]
    (DAY5 / "day5_stage3_case_update_recommendation.md").write_text("\n".join(update_md), encoding="utf-8")

    risks_md = [
        "# Day 5 Remaining Risks",
        "",
        *source_lines,
        "",
        "## blocker",
        "",
        "- `verified`: Stage 4 overall room-stratified delta AUC is effectively null, so Stage 3 cannot be justified as a broad transfer-confirmation stage.",
        "",
        "## caution",
        "",
        "- `verified`: main positive Day 4 evidence is derived-condition based and partly overlaps with current Stage 3 cases only in Rooms A/C, not B.",
        "- `verified`: the current 30-case deterministic list is heavily Room C weighted.",
        "- `verified`: Room C GEO and Room A control pockets can be useful, but both require conditional wording.",
        "",
        "## supplemental",
        "",
        "- `verified`: Stage 1 low-eps patch sweep remains useful as a physical mechanism hint.",
        "- `verified`: ideal-only wall/angle cells remain an upper-bound antenna limitation finding.",
        "",
        "## no_issue",
        "",
        "- `verified`: canonical deterministic citation numbers remain unchanged.",
        "- `verified`: Stage 3 documentation already reflects the post-0.2666 downgraded framing.",
    ]
    (DAY5 / "day5_remaining_risks.md").write_text("\n".join(risks_md), encoding="utf-8")

    go_md = [
        "# Day 5 GO / NO-GO",
        "",
        *source_lines,
        "",
        "- Stage 3 deterministic update decision: `PREFLIGHT_FIRST`.",
        "- Main candidate: `verified` cross-room derived `mid k_factor_estimate + mid fp_to_total_ratio` band.",
        "- Conditional candidate: `verified` Room A targeted `P2_SENS` and Room C GEO-only pockets.",
        "- Rejected regime: `verified` Day 3 `C2` and `C5` patch mini candidates.",
        "- Paper-facing global statement: `NO-GO`.",
        "- Conditional regime-validation statement: `GO`.",
    ]
    (DAY5 / "day5_go_nogo.md").write_text("\n".join(go_md), encoding="utf-8")


def main() -> None:
    ensure_dir(DAY5)

    day1_metrics = pd.read_csv(RECOVERY / "day1" / "day1_metrics_summary.csv")
    day1_stage2_preds = pd.read_csv(RECOVERY / "day1" / "day1_oof_predictions_stage2.csv")
    day2_candidates = pd.read_csv(RECOVERY / "day2" / "day2_candidate_regimes.csv")
    day3_metrics = pd.read_csv(RECOVERY / "day3" / "day3_stage1_recovery_mini_metrics.csv")
    day4_preds = pd.read_csv(RECOVERY / "day4" / "day4_stage2_room_stratified_predictions.csv")
    day4_preds["cp_save"] = day4_preds["cp_save"].astype(int)
    day4_preds["cp_harm"] = day4_preds["cp_harm"].astype(int)
    day4_preds["low_confidence"] = day4_preds["low_confidence"].astype(int)
    day4_preds["low_confidence_rescue"] = day4_preds["low_confidence_rescue"].astype(int)
    stage2_det = pd.read_csv(RESULTS / "stage2" / "stage2_900_ffd_relabel_det.csv")
    hfss_cases = pd.read_csv(RESULTS / "stage3" / "hfss_case_list_det.csv")
    paper_auc = pd.read_csv(RESULTS / "code_audit" / "paper_facing_auc_ci.csv")

    stage1_row, stage2_row = extract_stage1_stage2_rows(day1_metrics, paper_auc)
    day3_summary, day3_patch_candidates = pick_day3_rows(day3_metrics)
    day4_summary, derived_day4_rows, _ = derive_day4_candidate_metrics(day4_preds)

    final_candidates = build_final_candidate_table(day2_candidates, day3_patch_candidates, derived_day4_rows)
    integrated = build_integrated_metrics(stage1_row, stage2_row, day3_summary, day4_summary, final_candidates)
    integrated.to_csv(DAY5 / "day5_integrated_metrics.csv", index=False)
    final_candidates.to_csv(DAY5 / "day5_final_candidate_regimes.csv", index=False)

    def main_mask_fn(df: pd.DataFrame) -> pd.Series:
        return df["k_factor_estimate"].between(0.1446, 0.3373) & df["fp_to_total_ratio"].between(0.1264, 0.2522)

    stage3_info, preflight = stage3_audit(stage2_det, hfss_cases, day1_stage2_preds, main_mask_fn)
    if not preflight.empty:
        preflight.to_csv(DAY5 / "day5_stage3_preflight_pool.csv", index=False)

    plot_delta_dot(integrated, DAY5 / "day5_plot_delta_auc_dot.png")
    plot_candidate_net(final_candidates, DAY5 / "day5_plot_candidate_net_recovery_ci.png")
    plot_save_harm(final_candidates, DAY5 / "day5_plot_candidate_save_vs_harm.png")
    plot_stage3_overlay(stage2_det, hfss_cases, DAY5 / "day5_plot_stage3_overlay_candidate_maps.png")
    plot_claim_support(DAY5 / "day5_plot_claim_support_matrix.png")

    write_markdown(integrated, final_candidates, stage3_info, preflight)


if __name__ == "__main__":
    main()

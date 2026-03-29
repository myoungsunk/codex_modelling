#!/usr/bin/env python3
"""Generate Figure 1-7 and Table 1-5 from the Plan 3 campaign outputs.

Default inputs:
  - campaign: outputs/indoor_campaign_fullrun_phase3_fix
  - calibration: outputs/phase2_realistic_compare_fullrun

Example:
  python phase3_materials/plot_plan3_results.py --root D:/codex/ISAC_communication_verification --out D:/codex/ISAC_communication_verification/outputs/phase3_materials_report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter

matplotlib.use("Agg")


SCENARIO_ORDER = [
    "open_office",
    "mixed_office",
    "lecture_room",
    "straight_corridor",
    "factory_lite",
]
PROXY_ORDER = ["l_corridor_proxy", "lobby_blocker_proxy"]
SCENARIO_LABELS = {
    "open_office": "Open office",
    "mixed_office": "Mixed office",
    "lecture_room": "Lecture room",
    "straight_corridor": "Straight corridor",
    "factory_lite": "Factory-lite",
    "l_corridor_proxy": "L-corridor proxy",
    "lobby_blocker_proxy": "Lobby blocker proxy",
}
CLUTTER_DESC = {
    "open_office": "2 desks, 1 cabinet, 1 printer island",
    "mixed_office": "2 low partitions, 5 desk clusters",
    "lecture_room": "5 desk rows, 1 lectern",
    "straight_corridor": "3 metal-plate pairs on corridor walls",
    "factory_lite": "3 metal shelf rows, 1 workbench",
    "l_corridor_proxy": "1 concrete corner block, 1 metal plate",
    "lobby_blocker_proxy": "2 concrete columns",
}


def scientific_axis(ax, which: str = "y") -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    if which == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def save_df(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    try:
        markdown = df.to_markdown(index=False)
    except Exception:
        markdown = df.to_csv(index=False)
    md_path.write_text(markdown, encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(root: Path, campaign_dir: Path, calibration_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, np.lib.npyio.NpzFile, dict]:
    sys.path.insert(0, str(root))
    summary = load_json(campaign_dir / "summary.json")
    scene_metrics = pd.read_csv(campaign_dir / "scene_metrics.csv")
    pose_metrics = pd.read_csv(campaign_dir / "pose_metrics.csv")
    maps = np.load(campaign_dir / "maps.npz", allow_pickle=True)
    phase2_summary = load_json(calibration_dir / "summary.json")
    return summary, scene_metrics, pose_metrics, maps, phase2_summary


def selected_scene_order(summary: dict) -> list[str]:
    reps = summary["selection"]["representatives"]
    return [str(reps[bucket]["scenario_name"]) for bucket in ("mild", "moderate", "rich")]


def table_scene_definition(canonical_scenarios, out_dir: Path) -> None:
    rows = []
    for spec in canonical_scenarios(include_proxy=True):
        rows.append(
            {
                "scenario": SCENARIO_LABELS[spec.name],
                "room_size_m": " x ".join(f"{value:g}" for value in spec.room_size),
                "tx_pos_m": f"({spec.tx_pos[0]:.1f}, {spec.tx_pos[1]:.1f}, {spec.tx_pos[2]:.1f})",
                "tx_height_m": float(spec.tx_pos[2]),
                "material_preset": spec.material_preset,
                "clutter_objects": CLUTTER_DESC[spec.name],
                "proxy": bool(spec.proxy),
            }
        )
    save_df(pd.DataFrame(rows), out_dir, "table1_scene_definition")


def table_richness_summary(summary: dict, out_dir: Path) -> None:
    rows = []
    for entry in summary["scenarios"]:
        rows.append(
            {
                "scenario": SCENARIO_LABELS[entry["scenario_name"]],
                "proxy": bool(entry["proxy"]),
                "points": int(entry["point_count"]),
                "median_MRS": float(entry["median_mrs_score"]),
                "median_N_eff": float(entry["median_n_eff"]),
                "median_tau_rms_ns": float(entry["median_tau_rms_ns"]),
                "median_AoA_spread_deg": float(entry["median_mean_aoa_spread_deg"]),
                "median_DPR_dB": float(entry["median_dpr_db"]),
                "median_PMI_env": float(entry["median_pmi_env"]),
            }
        )
    df = pd.DataFrame(rows)
    categories = [SCENARIO_LABELS[name] for name in SCENARIO_ORDER + PROXY_ORDER]
    df["scenario"] = pd.Categorical(df["scenario"], categories=categories, ordered=True)
    df = df.sort_values(["proxy", "scenario"]).reset_index(drop=True)
    save_df(df, out_dir, "table2_richness_summary")


def table_headline_performance(summary: dict, pose_metrics: pd.DataFrame, out_dir: Path) -> None:
    reps = summary["selection"]["representatives"]
    rows = []
    for bucket in ("mild", "moderate", "rich"):
        info = reps[bucket]
        scenario_name = str(info["scenario_name"])
        row = pose_metrics[
            (pose_metrics["scenario_name"] == scenario_name)
            & (~pose_metrics["proxy"])
            & (pose_metrics["protocol"] == "P0")
            & (pose_metrics["snr_db"] == 10.0)
        ].iloc[0]
        rows.append(
            {
                "bucket": bucket,
                "scenario": SCENARIO_LABELS[scenario_name],
                "representative_rx_pos_m": f"({info['rx_pos'][0]:.1f}, {info['rx_pos'][1]:.1f}, {info['rx_pos'][2]:.1f})",
                "mean_delta_R_EP_norm": float(row["delta_rate_mean"]),
                "P_NI_0": float(row["noninferiority_prob_0"]),
                "P_NI_epsilon": float(row["noninferiority_prob_epsilon"]),
                "LP_R_5pct": float(row["lp_rate_p05"]),
                "CP_R_5pct": float(row["cp_rate_p05"]),
                "delta_R_5pct": float(row["delta_rate_p05"]),
                "outage_LP": float(row["outage_lp"]),
                "outage_CP": float(row["outage_cp"]),
            }
        )
    save_df(pd.DataFrame(rows), out_dir, "table3_headline_normalized_performance_10dB")


def table_pose_robustness(summary: dict, pose_metrics: pd.DataFrame, out_dir: Path) -> None:
    selected = set(selected_scene_order(summary))
    df = pose_metrics[
        (~pose_metrics["proxy"])
        & (pose_metrics["snr_db"] == 10.0)
        & (pose_metrics["scenario_name"].isin(selected))
        & (pose_metrics["protocol"].isin(["P0", "P1", "P2"]))
    ].copy()
    df["scenario"] = df["scenario_name"].map(SCENARIO_LABELS)
    df = df[
        [
            "scenario",
            "protocol",
            "samples",
            "delta_rate_mean",
            "noninferiority_prob_0",
            "noninferiority_prob_epsilon",
            "delta_rate_p05",
            "delta_outage",
        ]
    ].sort_values(["scenario", "protocol"]).reset_index(drop=True)
    save_df(df, out_dir, "table4_pose_robustness_10dB")


def table_secondary_raw_context(summary: dict, scene_metrics: pd.DataFrame, out_dir: Path) -> None:
    reps = summary["selection"]["representatives"]
    rows = []
    has_raw_capacity = "delta_capacity_raw_snr_10" in scene_metrics.columns
    has_norm_capacity = "delta_capacity_norm_snr_10" in scene_metrics.columns
    for bucket in ("mild", "moderate", "rich"):
        scenario_name = str(reps[bucket]["scenario_name"])
        group = scene_metrics[(scene_metrics["scenario_name"] == scenario_name) & (~scene_metrics["proxy"])]
        rows.append(
            {
                "bucket": bucket,
                "scenario": SCENARIO_LABELS[scenario_name],
                "raw_mean_delta_R_EP": float(group["delta_rate_raw_snr_10"].mean()),
                "raw_mean_delta_C_WF": float(group["delta_capacity_raw_snr_10"].mean()) if has_raw_capacity else np.nan,
                "normalized_mean_delta_R_EP": float(group["delta_rate_norm_snr_10"].mean()),
                "normalized_mean_delta_C_WF": float(group["delta_capacity_norm_snr_10"].mean()) if has_norm_capacity else np.nan,
                "note": "" if has_raw_capacity and has_norm_capacity else "C_WF not exported in Plan 3 campaign output",
            }
        )
    save_df(pd.DataFrame(rows), out_dir, "table5_secondary_raw_family_context_10dB")


def figure_calibration(fig_dir: Path, phase2_summary: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    snr_values = np.array(
        sorted(float(key) for key in phase2_summary["raw"].keys() if key not in ["delta_xpr_db", "delta_condition_number"])
    )
    raw_rate = np.array([phase2_summary["raw"][str(snr)]["delta_rate_mean"] for snr in snr_values])
    raw_capacity = np.array([phase2_summary["raw"][str(snr)]["delta_capacity_mean"] for snr in snr_values])
    norm_rate = np.array([phase2_summary["normalized"][str(snr)]["delta_rate_mean"] for snr in snr_values])
    norm_capacity = np.array([phase2_summary["normalized"][str(snr)]["delta_capacity_mean"] for snr in snr_values])

    axes[0].plot(snr_values, raw_rate, marker="o", label="Raw")
    axes[0].plot(snr_values, norm_rate, marker="s", label="Family-power normalized")
    axes[0].axhline(0.0, linewidth=1)
    axes[0].set_title("Shoebox mean Delta R_EP = CP - LP")
    axes[0].set_xlabel("SNR [dB]")
    axes[0].set_ylabel("Mean Delta R_EP")
    axes[0].legend()
    scientific_axis(axes[0], "y")

    axes[1].plot(snr_values, raw_capacity, marker="o", label="Raw")
    axes[1].plot(snr_values, norm_capacity, marker="s", label="Family-power normalized")
    axes[1].axhline(0.0, linewidth=1)
    axes[1].set_title("Shoebox mean Delta C_WF = CP - LP")
    axes[1].set_xlabel("SNR [dB]")
    axes[1].set_ylabel("Mean Delta C_WF")
    axes[1].legend()
    scientific_axis(axes[1], "y")

    figure.suptitle("Figure 1. Calibration control")
    figure.savefig(fig_dir / "figure1_calibration.png", dpi=200)
    plt.close(figure)


def figure_mrs_distribution(fig_dir: Path, scene_metrics: pd.DataFrame) -> None:
    figure, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    data = [
        scene_metrics[(scene_metrics["scenario_name"] == scenario_name) & (~scene_metrics["proxy"])]["mrs_score"].to_numpy()
        for scenario_name in SCENARIO_ORDER
    ]
    ax.boxplot(data, tick_labels=[SCENARIO_LABELS[name] for name in SCENARIO_ORDER], showfliers=False)
    ax.set_ylabel("MRS score [0-100]")
    ax.set_title("Figure 2. Screening MRS distribution across non-proxy scenarios")
    ax.tick_params(axis="x", rotation=20)
    figure.savefig(fig_dir / "figure2_mrs_distribution.png", dpi=200)
    plt.close(figure)


def figure_representative_layouts(fig_dir: Path, summary: dict, maps, get_scenario_spec, build_scenario_layout) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    reps = summary["selection"]["representatives"]
    for ax, bucket in zip(axes, ("mild", "moderate", "rich")):
        info = reps[bucket]
        scenario_name = str(info["scenario_name"])
        spec = get_scenario_spec(scenario_name)
        layout = build_scenario_layout(spec)
        lx, ly, _ = spec.room_size
        ax.add_patch(Rectangle((0, 0), lx, ly, fill=False, linewidth=1.5))
        for box in layout.solid_boxes:
            x0, y0, _ = box.min_corner
            x1, y1, _ = box.max_corner
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, alpha=0.4))
        for surface in layout.scene.surfaces:
            if surface.name.startswith(("wall_", "floor", "ceiling")) or abs(surface.normal[2]) > 1e-9:
                continue
            if abs(surface.normal[0]) > 0.5:
                xpos = surface.point[0]
                y0 = surface.point[1] - surface.half_u
                y1 = surface.point[1] + surface.half_u
                ax.plot([xpos, xpos], [y0, y1], linewidth=2)
            elif abs(surface.normal[1]) > 0.5:
                ypos = surface.point[1]
                x0 = surface.point[0] - surface.half_u
                x1 = surface.point[0] + surface.half_u
                ax.plot([x0, x1], [ypos, ypos], linewidth=2)
        prefix = f"full_{scenario_name}"
        if f"{prefix}_rx_x" in maps and f"{prefix}_valid_mask" in maps:
            rx_x = maps[f"{prefix}_rx_x"]
            rx_y = maps[f"{prefix}_rx_y"]
            valid_mask = maps[f"{prefix}_valid_mask"].astype(bool)
            xx, yy = np.meshgrid(rx_x, rx_y)
            ax.scatter(xx[valid_mask], yy[valid_mask], s=2, alpha=0.25)
        ax.plot(spec.tx_pos[0], spec.tx_pos[1], marker="*", markersize=12)
        ax.plot(info["rx_pos"][0], info["rx_pos"][1], marker="o", markersize=7)
        ax.set_xlim(-0.1, lx + 0.1)
        ax.set_ylim(-0.1, ly + 0.1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{bucket.title()}: {SCENARIO_LABELS[scenario_name]}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    figure.suptitle("Figure 3. Representative mild/moderate/rich scene layouts")
    figure.savefig(fig_dir / "figure3_representative_layouts.png", dpi=200)
    plt.close(figure)


def figure_normalized_heatmaps(fig_dir: Path, summary: dict, maps) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    for row, scenario_name in enumerate(selected_scene_order(summary)):
        rx_x = maps[f"full_{scenario_name}_rx_x"]
        rx_y = maps[f"full_{scenario_name}_rx_y"]
        valid_mask = maps[f"full_{scenario_name}_valid_mask"].astype(bool)
        lp = np.where(valid_mask, maps[f"full_{scenario_name}_norm_lp_rate_snr_10"], np.nan)
        cp = np.where(valid_mask, maps[f"full_{scenario_name}_norm_cp_rate_snr_10"], np.nan)
        delta = np.where(valid_mask, maps[f"full_{scenario_name}_norm_delta_rate_snr_10"], np.nan)
        rate_vmin = np.nanmin(np.stack([lp, cp]))
        rate_vmax = np.nanmax(np.stack([lp, cp]))
        delta_limit = np.nanmax(np.abs(delta))
        for col, (array, title) in enumerate(
            [
                (lp, "LP R_EP"),
                (cp, "CP R_EP"),
                (delta, "Delta R_EP = CP - LP"),
            ]
        ):
            ax = axes[row, col]
            if col < 2:
                image = ax.imshow(
                    array,
                    origin="lower",
                    aspect="auto",
                    extent=[rx_x.min(), rx_x.max(), rx_y.min(), rx_y.max()],
                    vmin=rate_vmin,
                    vmax=rate_vmax,
                )
            else:
                image = ax.imshow(
                    array,
                    origin="lower",
                    aspect="auto",
                    extent=[rx_x.min(), rx_x.max(), rx_y.min(), rx_y.max()],
                    vmin=-delta_limit,
                    vmax=delta_limit,
                )
            ax.set_title(f"{SCENARIO_LABELS[scenario_name]}\n{title}")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            colorbar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            colorbar.formatter.set_powerlimits((-2, 2))
            colorbar.update_ticks()
    figure.suptitle("Figure 4. Normalized rate heatmaps at 10 dB")
    figure.savefig(fig_dir / "figure4_normalized_rate_heatmaps_10dB.png", dpi=200)
    plt.close(figure)


def figure_richness_relation(fig_dir: Path, summary: dict, scene_metrics: pd.DataFrame, pose_metrics: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    nonproxy = scene_metrics[~scene_metrics["proxy"]]
    axes[0].scatter(nonproxy["mrs_score"], nonproxy["delta_rate_norm_snr_10"], s=8, alpha=0.35)
    corr = float(np.corrcoef(nonproxy["mrs_score"], nonproxy["delta_rate_norm_snr_10"])[0, 1])
    axes[0].set_title(f"Pointwise MRS vs Delta R_EP at 10 dB\n(corr={corr:.3f})")
    axes[0].set_xlabel("MRS score")
    axes[0].set_ylabel("Delta R_EP = CP - LP")
    scientific_axis(axes[0], "y")

    scene_med_mrs = {
        SCENARIO_LABELS[entry["scenario_name"]]: float(entry["median_mrs_score"])
        for entry in summary["scenarios"]
        if not entry["proxy"]
    }
    p0_10 = pose_metrics[
        (~pose_metrics["proxy"])
        & (pose_metrics["protocol"] == "P0")
        & (pose_metrics["snr_db"] == 10.0)
    ].copy()
    p0_10["scenario"] = p0_10["scenario_name"].map(SCENARIO_LABELS)
    p0_10["median_mrs"] = p0_10["scenario"].map(scene_med_mrs)
    axes[1].scatter(p0_10["median_mrs"], p0_10["delta_rate_p05"], s=40)
    for _, row in p0_10.iterrows():
        axes[1].annotate(
            row["scenario"],
            (row["median_mrs"], row["delta_rate_p05"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    axes[1].set_title("Scene-wise median MRS vs Delta R_5pct at 10 dB (P0)")
    axes[1].set_xlabel("Median MRS score")
    axes[1].set_ylabel("Delta R_5pct = CP - LP")
    scientific_axis(axes[1], "y")

    figure.suptitle("Figure 5. Richness-performance relation")
    figure.savefig(fig_dir / "figure5_richness_performance_relation.png", dpi=200)
    plt.close(figure)


def figure_pose_robustness(fig_dir: Path, summary: dict, pose_metrics: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    metrics = [
        ("delta_rate_mean", "Mean Delta R_EP"),
        ("noninferiority_prob_0", "P_NI(0)"),
        ("noninferiority_prob_epsilon", "P_NI(epsilon)"),
        ("delta_outage", "Delta outage"),
    ]
    for ax, (column, title) in zip(axes.flat, metrics):
        for scenario_name in selected_scene_order(summary):
            subset = pose_metrics[
                (pose_metrics["scenario_name"] == scenario_name)
                & (~pose_metrics["proxy"])
                & (pose_metrics["snr_db"] == 10.0)
                & (pose_metrics["protocol"].isin(["P0", "P1", "P2"]))
            ].sort_values("protocol")
            ax.plot(subset["protocol"], subset[column], marker="o", label=SCENARIO_LABELS[scenario_name])
        ax.set_title(title)
        ax.set_xlabel("Pose protocol")
        ax.set_ylabel(title)
        if column in ("delta_rate_mean", "delta_outage"):
            scientific_axis(ax, "y")
    axes[0, 0].legend(loc="best", fontsize=8)
    figure.suptitle("Figure 6. Pose robustness on selected scenes at 10 dB")
    figure.savefig(fig_dir / "figure6_pose_robustness_10dB.png", dpi=200)
    plt.close(figure)


def figure_proxy_appendix(fig_dir: Path, maps) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for row, scenario_name in enumerate(PROXY_ORDER):
        rx_x = maps[f"screening_{scenario_name}_rx_x"]
        rx_y = maps[f"screening_{scenario_name}_rx_y"]
        valid_mask = maps[f"screening_{scenario_name}_valid_mask"].astype(bool)
        mrs = np.where(valid_mask, maps[f"screening_{scenario_name}_mrs_score"], np.nan)
        delta = np.where(valid_mask, maps[f"screening_{scenario_name}_delta_rate_norm_snr_10"], np.nan)

        image_mrs = axes[row, 0].imshow(
            mrs,
            origin="lower",
            aspect="auto",
            extent=[rx_x.min(), rx_x.max(), rx_y.min(), rx_y.max()],
        )
        axes[row, 0].set_title(f"{SCENARIO_LABELS[scenario_name]}\nMRS screening map")
        axes[row, 0].set_xlabel("x [m]")
        axes[row, 0].set_ylabel("y [m]")
        figure.colorbar(image_mrs, ax=axes[row, 0], fraction=0.046, pad=0.04)

        delta_limit = np.nanmax(np.abs(delta))
        image_delta = axes[row, 1].imshow(
            delta,
            origin="lower",
            aspect="auto",
            extent=[rx_x.min(), rx_x.max(), rx_y.min(), rx_y.max()],
            vmin=-delta_limit,
            vmax=delta_limit,
        )
        axes[row, 1].set_title(f"{SCENARIO_LABELS[scenario_name]}\nNormalized Delta R_EP at 10 dB")
        axes[row, 1].set_xlabel("x [m]")
        axes[row, 1].set_ylabel("y [m]")
        colorbar = figure.colorbar(image_delta, ax=axes[row, 1], fraction=0.046, pad=0.04)
        colorbar.formatter.set_powerlimits((-2, 2))
        colorbar.update_ticks()
    figure.suptitle("Figure 7. Proxy appendix (exploratory only)")
    figure.savefig(fig_dir / "figure7_proxy_appendix.png", dpi=200)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Repository root that contains dualpol_rt/ and outputs/.",
    )
    parser.add_argument("--campaign-dir", type=Path, default=None)
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    campaign_dir = (args.campaign_dir or (root / "outputs" / "indoor_campaign_fullrun_phase3_fix")).resolve()
    calibration_dir = (args.calibration_dir or (root / "outputs" / "phase2_realistic_compare_fullrun")).resolve()
    out_dir = args.out.resolve()
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    summary, scene_metrics, pose_metrics, maps, phase2_summary = load_inputs(root, campaign_dir, calibration_dir)

    from dualpol_rt.scene.scenarios import build_scenario_layout, canonical_scenarios, get_scenario_spec

    table_scene_definition(canonical_scenarios, tab_dir)
    table_richness_summary(summary, tab_dir)
    table_headline_performance(summary, pose_metrics, tab_dir)
    table_pose_robustness(summary, pose_metrics, tab_dir)
    table_secondary_raw_context(summary, scene_metrics, tab_dir)

    figure_calibration(fig_dir, phase2_summary)
    figure_mrs_distribution(fig_dir, scene_metrics)
    figure_representative_layouts(fig_dir, summary, maps, get_scenario_spec, build_scenario_layout)
    figure_normalized_heatmaps(fig_dir, summary, maps)
    figure_richness_relation(fig_dir, summary, scene_metrics, pose_metrics)
    figure_pose_robustness(fig_dir, summary, pose_metrics)
    figure_proxy_appendix(fig_dir, maps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

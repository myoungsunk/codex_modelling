from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "metrics_extension" / "day6_stage_compare"
STAGE3_CASE_LIST = ROOT / "results" / "stage3" / "hfss_case_list_det.csv"


def build_stage2_grid(room_size=(5.0, 4.0, 3.0), n_positions=60):
    L, W, H = room_size
    anchor = np.array([L / 2.0, W / 2.0, H - 0.30])
    n_layer1 = int(np.ceil(n_positions / 2.0))
    n_layer2 = n_positions - n_layer1

    z_grid = [0.20, 1.20]
    n_xy = int(np.ceil(n_layer1 / len(z_grid)))
    x_count = max(3, int(np.ceil(np.sqrt(n_xy * L / W))))
    y_count = max(3, int(np.ceil(n_xy / x_count)))
    while x_count * y_count * len(z_grid) < n_layer1:
        if x_count <= y_count:
            x_count += 1
        else:
            y_count += 1
    x_grid = np.linspace(0.50, L - 0.50, x_count)
    y_grid = np.linspace(0.50, W - 0.50, y_count)
    layer1 = []
    for z in z_grid:
        for y in y_grid:
            for x in x_grid:
                layer1.append((x, y, z, 1))
    layer1 = pd.DataFrame(layer1, columns=["tag_x", "tag_y", "tag_z", "grid_layer"]).iloc[:n_layer1].copy()

    wall_offset = 0.30
    z_candidates = [0.20, 0.65, 1.20]
    fixed_points = np.array(
        [
            [wall_offset, W / 2.0, 0.20],
            [L - wall_offset, W / 2.0, 1.20],
            [L / 2.0, wall_offset, 0.20],
            [L / 2.0, W - wall_offset, 1.20],
            [wall_offset, wall_offset, 0.20],
            [wall_offset, W - wall_offset, 1.20],
            [L - wall_offset, wall_offset, 1.20],
            [L - wall_offset, W - wall_offset, 0.20],
            [0.20 * L, 0.50 * W, 0.65],
            [0.80 * L, 0.50 * W, 0.65],
            [0.50 * L, 0.20 * W, 0.65],
            [0.50 * L, 0.80 * W, 0.65],
            [0.30 * L, 0.78 * W, 0.40],
            [0.62 * L, 0.60 * W, 0.95],
        ]
    )
    rng = np.random.default_rng(20260423)
    layer2 = fixed_points.tolist()
    while len(layer2) < n_layer2:
        side = int(rng.integers(1, 5))
        z = z_candidates[int(rng.integers(0, len(z_candidates)))]
        if side == 1:
            tag = [wall_offset + 0.25 * rng.random(), 0.25 + (W - 0.50) * rng.random(), z]
        elif side == 2:
            tag = [L - wall_offset - 0.25 * rng.random(), 0.25 + (W - 0.50) * rng.random(), z]
        elif side == 3:
            tag = [0.25 + (L - 0.50) * rng.random(), wall_offset + 0.25 * rng.random(), z]
        else:
            tag = [0.25 + (L - 0.50) * rng.random(), W - wall_offset - 0.25 * rng.random(), z]
        layer2.append(tag)
    layer2 = pd.DataFrame(layer2[:n_layer2], columns=["tag_x", "tag_y", "tag_z"])
    layer2["grid_layer"] = 2

    all_tags = pd.concat([layer1, layer2], ignore_index=True)
    return anchor, all_tags


def build_stage1_examples():
    rng = np.random.default_rng(20260423)
    n = 80
    df = pd.DataFrame(
        {
            "tag_x": rng.uniform(-2.0, 2.0, n),
            "tag_y": rng.uniform(-2.0, 2.0, n),
            "tag_z": rng.uniform(0.1, 1.5, n),
            "anchor_z": rng.uniform(2.0, 3.0, n),
        }
    )
    return df


def draw_stage1_top(ax, df):
    ax.scatter([0], [0], c="black", s=90, marker="^", label="Tx antenna")
    ax.scatter(df["tag_x"], df["tag_y"], s=18, alpha=0.6, color="#4c78a8", label="Tag candidates")
    ax.axhline(0, color="0.85", linewidth=1)
    ax.axvline(0, color="0.85", linewidth=1)
    ax.add_patch(plt.Rectangle((-2, -2), 4, 4, fill=False, linestyle="--", edgecolor="0.5"))
    ax.text(0.05, 0.95, "Local frame\nTx at origin\nTag x/y in [-2, 2] m", transform=ax.transAxes, va="top", fontsize=9)
    ax.text(0.62, 0.08, "single slab\nplacement varies", transform=ax.transAxes, fontsize=8, color="0.35")
    ax.set_title("Stage 1 top view")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.set_aspect("equal")


def draw_stage1_side(ax, df):
    ax.scatter([0], [df["anchor_z"].mean()], c="black", s=90, marker="^", label="Tx antenna")
    ax.vlines(0, 2.0, 3.0, color="black", linewidth=3, alpha=0.25)
    x_side = np.abs(df["tag_x"])
    ax.scatter(x_side, df["tag_z"], s=18, alpha=0.6, color="#4c78a8", label="Tag candidates")
    ax.fill_between([0, 2.2], 0, 0.03, color="#b0b0b0", alpha=0.5)
    ax.text(0.05, 0.95, "anchor height: 2.0 to 3.0 m\ntag height: 0.1 to 1.5 m", transform=ax.transAxes, va="top", fontsize=9)
    ax.plot([0.7, 1.3], [1.2, 1.2], color="#f58518", linewidth=6, solid_capstyle="round")
    ax.text(1.0, 1.34, "slab between Tx/Rx", ha="center", fontsize=8, color="#7a3e00")
    ax.set_title("Stage 1 side view")
    ax.set_xlabel("projected x distance (m)")
    ax.set_ylabel("z (m)")
    ax.set_xlim(-0.1, 2.4)
    ax.set_ylim(0, 3.2)


def draw_stage2_top(ax, anchor, tags):
    ax.add_patch(plt.Rectangle((0, 0), 5.0, 4.0, fill=False, edgecolor="0.3", linewidth=1.5))
    layer1 = tags[tags["grid_layer"] == 1]
    layer2 = tags[tags["grid_layer"] == 2]
    ax.scatter(anchor[0], anchor[1], c="black", s=110, marker="^", label="Ceiling anchor")
    ax.scatter(layer1["tag_x"], layer1["tag_y"], s=26, color="#4c78a8", alpha=0.7, label="Layer 1 tags")
    ax.scatter(layer2["tag_x"], layer2["tag_y"], s=30, color="#f58518", alpha=0.8, label="Layer 2 tags")
    ax.text(0.04, 0.95, "Room A/B/C share\nsame anchor convention", transform=ax.transAxes, va="top", fontsize=9)
    ax.text(0.56, 0.09, "Layer 2:\nwall-near / clutter-probing", transform=ax.transAxes, fontsize=8, color="0.35")
    ax.set_title("Stage 2 top view")
    ax.set_xlabel("room x (m)")
    ax.set_ylabel("room y (m)")
    ax.set_xlim(-0.1, 5.1)
    ax.set_ylim(-0.1, 4.1)
    ax.set_aspect("equal")


def draw_stage2_side(ax, anchor, tags):
    ax.scatter([0], [anchor[2]], c="black", s=110, marker="^")
    ax.scatter(np.linspace(0.6, 1.8, len(tags)), tags["tag_z"], s=22, c=np.where(tags["grid_layer"] == 1, "#4c78a8", "#f58518"), alpha=0.7)
    ax.text(0.05, 0.95, "anchor z = 2.7 m\ntag z = 0.2 / 0.65 / 1.2 m", transform=ax.transAxes, va="top", fontsize=9)
    ax.fill_between([0, 2.0], 0, 0.03, color="#b0b0b0", alpha=0.5)
    ax.set_title("Stage 2 side view")
    ax.set_xlabel("illustrative horizontal spread")
    ax.set_ylabel("z (m)")
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(0, 3.2)


def draw_stage3_top(ax, df):
    ax.add_patch(plt.Rectangle((0, 0), 5.0, 4.0, fill=False, edgecolor="0.3", linewidth=1.5))
    markers = {"GEO": "o", "BOUNCE": "s"}
    colors = {"A": "#54a24b", "B": "#e45756", "C": "#4c78a8"}
    for (group, room), sub in df.groupby(["group", "room_type"]):
        ax.scatter(sub["tag_x"], sub["tag_y"], s=58, marker=markers.get(group, "o"), color=colors.get(room, "0.4"), alpha=0.85, label=f"{group}-{room}")
    if not df.empty:
        anchor_x = float(df["anchor_x"].mode().iloc[0])
        anchor_y = float(df["anchor_y"].mode().iloc[0])
        ax.scatter([anchor_x], [anchor_y], c="black", s=120, marker="^", label="HFSS anchor")
    ax.text(0.04, 0.95, "Selected HFSS subset\ncurrent list: A and C only", transform=ax.transAxes, va="top", fontsize=9)
    ax.text(0.53, 0.09, "GEO currently\nRoom C only", transform=ax.transAxes, fontsize=8, color="0.35")
    ax.set_title("Stage 3 top view")
    ax.set_xlabel("room x (m)")
    ax.set_ylabel("room y (m)")
    ax.set_xlim(-0.1, 5.1)
    ax.set_ylim(-0.1, 4.1)
    ax.set_aspect("equal")


def draw_stage3_side(ax, df):
    if df.empty:
        return
    anchor_z = float(df["anchor_z"].mode().iloc[0])
    ax.scatter([0], [anchor_z], c="black", s=120, marker="^")
    x_spread = np.linspace(0.5, 1.9, len(df))
    colors = df["room_type"].map({"A": "#54a24b", "B": "#e45756", "C": "#4c78a8"}).fillna("0.5")
    ax.scatter(x_spread, df["tag_z"], s=40, c=colors, alpha=0.85)
    ax.text(0.05, 0.95, f"anchor z = {anchor_z:.1f} m\nselected tag z from case list", transform=ax.transAxes, va="top", fontsize=9)
    ax.fill_between([0, 2.0], 0, 0.03, color="#b0b0b0", alpha=0.5)
    ax.set_title("Stage 3 side view")
    ax.set_xlabel("selected-case spread")
    ax.set_ylabel("z (m)")
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(0, 3.2)


def save_figure(stage1_df, stage2_anchor, stage2_tags, stage3_df, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    draw_stage1_top(axes[0, 0], stage1_df)
    draw_stage2_top(axes[0, 1], stage2_anchor, stage2_tags)
    draw_stage3_top(axes[0, 2], stage3_df)
    draw_stage1_side(axes[1, 0], stage1_df)
    draw_stage2_side(axes[1, 1], stage2_anchor, stage2_tags)
    draw_stage3_side(axes[1, 2], stage3_df)
    handles, labels = axes[0, 2].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=5, frameon=False)
    fig.suptitle("Stage 1 vs Stage 2 vs Stage 3: antenna and tag position schematics", y=0.98, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_markdown(stage3_df, out_path: Path, figure_path: Path):
    room_counts = stage3_df.groupby("room_type").size().to_dict()
    group_room_counts = stage3_df.groupby(["group", "room_type"]).size().to_dict()
    lines = [
        "# Stage 1 / 2 / 3 comparison",
        "",
        f"![stage compare]({figure_path.as_posix()})",
        "",
        "## Stage-by-stage comparison",
        "",
        "| Stage | Scene type | Antenna / anchor position | Tag position policy | Main interpretation |",
        "|---|---|---|---|---|",
        "| Stage 1 | Single bounded slab, no room enclosure | Tx antenna is the local reference. In the canonical Stage 1 sweep the Tx is built around the origin-side local frame and Tx height varies. | Tag position is sampled in a local x/y box and tag height varies. The slab is placed between Tx and Rx with placement modes such as floor / wall_x / wall_y / ceiling. | Controlled material-and-geometry sweep for single-slab propagation effects. |",
        "| Stage 2 | Full room A/B/C indoor scene | One ceiling anchor is fixed near the room center top: approximately (2.5, 2.0, 2.7) m in a 5 x 4 x 3 m room. | Tag positions are generated as a two-layer room grid: Layer 1 interior grid, Layer 2 wall-near / clutter-probing points. | Room multipath and clutter effects are added on top of the same anchor convention. |",
        "| Stage 3 | HFSS subset exported from Stage 2 candidates | Same room-anchor convention is inherited from Stage 2 for exported HFSS cases. | Only selected candidate tag locations are kept. Current deterministic list is not a full room grid; it is a filtered subset. | HFSS trend / regime validation, not a new free-form geometry stage. |",
        "",
        "## What changes across stages",
        "",
        "- Stage 1 is a local coordinate experiment around a single slab. It is not a room deployment geometry.",
        "- Stage 2 switches to a room coordinate system with a fixed ceiling anchor and many candidate tag points.",
        "- Stage 3 does not invent a new placement rule. It inherits Stage 2 room geometry and only selects a small HFSS-ready subset.",
        "",
        "## Stage 3 current coverage from the deterministic HFSS list",
        "",
        f"- room counts: `{room_counts}`",
        f"- group x room counts: `{group_room_counts}`",
        "- GEO coverage is currently Room C only, while BOUNCE covers Room A and Room C.",
        "",
        "## Source files used",
        "",
        f"- [designLhsSweep.m]({(ROOT / '+sweep' / 'designLhsSweep.m').as_posix()}:1)",
        f"- [makeSingleSlabScene.m]({(ROOT / '+scenes' / 'makeSingleSlabScene.m').as_posix()}:1)",
        f"- [generateRoomTxRxGrid.m]({(ROOT / '+scenes' / 'generateRoomTxRxGrid.m').as_posix()}:1)",
        f"- [makeRoomABCScene.m]({(ROOT / '+scenes' / 'makeRoomABCScene.m').as_posix()}:1)",
        f"- [hfss_case_list_det.csv]({(ROOT / 'results' / 'stage3' / 'hfss_case_list_det.csv').as_posix()}:1)",
        f"- [stage3_readiness_report.md]({(ROOT / 'results' / 'code_audit' / 'stage3_readiness_report.md').as_posix()}:1)",
        "",
        "## Practical read",
        "",
        "- If you compare Stage 1 and Stage 2 directly, the biggest geometry change is not antenna type but the coordinate system: local slab frame versus room-fixed ceiling-anchor frame.",
        "- If you compare Stage 2 and Stage 3 directly, the biggest change is not placement logic but subset selection: full room grid versus filtered HFSS candidate list.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage1_df = build_stage1_examples()
    stage2_anchor, stage2_tags = build_stage2_grid()
    stage3_df = pd.read_csv(STAGE3_CASE_LIST)
    figure_path = OUT_DIR / "stage123_layout_compare.png"
    md_path = OUT_DIR / "stage123_layout_compare.md"
    save_figure(stage1_df, stage2_anchor, stage2_tags, stage3_df, figure_path)
    write_markdown(stage3_df, md_path, figure_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "metrics_extension" / "day6_stage_compare_3d"
STAGE3_SPEC_PATH = Path(r"D:\codex\ansys_automation\outputs\stage3_room_abc_from_rt_cp_uwb\stage3_room_abc_spec.json")
STAGE3_POINTS_PATH = Path(r"D:\codex\ansys_automation\outputs\stage3_room_abc_from_rt_cp_uwb\stage3_scene_point_proposals_hfss_safe.json")


ROOM_SIZE = (5.0, 4.0, 3.0)
ANCHOR_POS = np.array([2.5, 2.0, 2.7], dtype=float)
ANCHOR_BORE_STAGE23 = np.array([0.0, 0.0, -1.0], dtype=float)
TAG_BORE_STAGE23 = np.array([0.0, 0.0, 1.0], dtype=float)


def set_axes_equal(ax, limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None) -> None:
    if limits is not None:
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def cuboid_vertices(origin: np.ndarray, size: np.ndarray) -> list[list[tuple[float, float, float]]]:
    x, y, z = origin
    dx, dy, dz = size
    p = np.array(
        [
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz],
        ]
    )
    faces = [
        [p[0], p[1], p[2], p[3]],
        [p[4], p[5], p[6], p[7]],
        [p[0], p[1], p[5], p[4]],
        [p[2], p[3], p[7], p[6]],
        [p[1], p[2], p[6], p[5]],
        [p[4], p[7], p[3], p[0]],
    ]
    return [[tuple(v) for v in face] for face in faces]


def add_cuboid(ax, origin, size, color, alpha=0.15, edgecolor=None, linewidth=0.8):
    faces = cuboid_vertices(np.asarray(origin, dtype=float), np.asarray(size, dtype=float))
    pc = Poly3DCollection(faces, facecolors=color, edgecolors=edgecolor or color, linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(pc)


def add_floor_outline(ax, room_size=ROOM_SIZE, color="0.45"):
    L, W, H = room_size
    corners = np.array([[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0], [0, 0, 0]])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color=color, linewidth=1.2)


def add_room_wireframe(ax, room_size=ROOM_SIZE, color="0.55"):
    L, W, H = room_size
    corners = np.array(
        [
            [0, 0, 0],
            [L, 0, 0],
            [L, W, 0],
            [0, W, 0],
            [0, 0, 0],
            [0, 0, H],
            [L, 0, H],
            [L, W, H],
            [0, W, H],
            [0, 0, H],
        ]
    )
    ax.plot(corners[:5, 0], corners[:5, 1], corners[:5, 2], color=color, linewidth=1.0)
    ax.plot(corners[5:, 0], corners[5:, 1], corners[5:, 2], color=color, linewidth=1.0)
    for x, y in [(0, 0), (L, 0), (L, W), (0, W)]:
        ax.plot([x, x], [y, y], [0, H], color=color, linewidth=1.0)


def add_arrow(ax, origin, direction, length=0.45, color="black", linewidth=2.0):
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / max(np.linalg.norm(d), 1e-12)
    d = d * length
    ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2], color=color, linewidth=linewidth, arrow_length_ratio=0.22)


def style_3d_ax(ax, title: str, limits=((0, 5), (0, 4), (0, 3)), elev=22, azim=-60):
    ax.set_title(title, pad=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    set_axes_equal(ax, limits)
    try:
        ax.set_box_aspect((limits[0][1] - limits[0][0], limits[1][1] - limits[1][0], limits[2][1] - limits[2][0]))
    except Exception:
        pass


def representative_stage1():
    tx = np.array([0.0, 0.0, 2.5])
    rx = np.array([1.35, 0.0, 0.55])
    return {
        "tx_pos": tx,
        "tx_bore": np.array([1.0, 0.0, 0.0]),
        "rx_pos": rx,
        "rx_bore": np.array([0.0, 0.0, 1.0]),
        "slabs": [
            {"origin": np.array([0.60, -0.65, 0.00]), "size": np.array([0.80, 1.30, 0.03]), "color": "#f58518", "label": "floor"},
            {"origin": np.array([0.60, -0.65, 1.70]), "size": np.array([0.80, 1.30, 0.03]), "color": "#f58518", "label": "ceiling"},
            {"origin": np.array([0.92, -0.65, 0.20]), "size": np.array([0.03, 1.30, 1.60]), "color": "#e45756", "label": "wall_x"},
        ],
        "tags": np.array(
            [
                [1.35, 0.00, 0.55],
                [1.05, -0.45, 0.45],
                [1.20, 0.35, 0.80],
                [0.95, 0.55, 0.35],
                [1.55, -0.25, 0.95],
            ]
        ),
    }


def generate_stage2_tags(room_type: str, n_positions=50, seed=20260423):
    L, W, H = ROOM_SIZE
    anchor = ANCHOR_POS.copy()
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
    layer1 = layer1[:n_layer1]

    wall_offset = 0.30
    z_candidates = [0.20, 0.65, 1.20]
    fixed_points = [
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
    ]
    if room_type.upper() == "C":
        fixed_points.extend([[0.30 * L, 0.78 * W, 0.40], [0.62 * L, 0.60 * W, 0.95]])
    elif room_type.upper() == "B":
        fixed_points.extend([[0.70 * L, 0.62 * W, 0.35]])

    rng = np.random.default_rng(seed + ord(room_type.upper()))
    layer2 = [tuple(p) + (2,) for p in fixed_points]
    while len(layer2) < n_layer2:
        side = int(rng.integers(1, 5))
        if side == 1:
            tag = [wall_offset + 0.25 * rng.random(), 0.25 + (W - 0.50) * rng.random(), z_candidates[int(rng.integers(0, len(z_candidates)))]]
        elif side == 2:
            tag = [L - wall_offset - 0.25 * rng.random(), 0.25 + (W - 0.50) * rng.random(), z_candidates[int(rng.integers(0, len(z_candidates)))]]
        elif side == 3:
            tag = [0.25 + (L - 0.50) * rng.random(), wall_offset + 0.25 * rng.random(), z_candidates[int(rng.integers(0, len(z_candidates)))]]
        else:
            tag = [0.25 + (L - 0.50) * rng.random(), W - wall_offset - 0.25 * rng.random(), z_candidates[int(rng.integers(0, len(z_candidates)))]]
        layer2.append(tuple(tag) + (2,))
    tags = pd.DataFrame(layer1 + layer2[:n_layer2], columns=["x", "y", "z", "layer"])
    return anchor, tags


def stage2_objects(room_type: str):
    objs = []
    if room_type.upper() in {"B", "C"}:
        objs.extend(
            [
                {"origin": [4.98, 1.60, 0.95], "size": [0.02, 0.80, 1.00], "color": "#9ecae9", "alpha": 0.18, "name": "window_glass"},
                {"origin": [2.80, 2.13, 0.73], "size": [1.20, 0.70, 0.04], "color": "#c49c94", "alpha": 0.28, "name": "desk_top"},
                {"origin": [4.73, 2.13, 0.00], "size": [0.04, 0.70, 0.75], "color": "#c49c94", "alpha": 0.30, "name": "desk_front"},
            ]
        )
    if room_type.upper() == "C":
        objs.extend(
            [
                {"origin": [1.00, 3.63, 0.00], "size": [0.80, 0.04, 1.80], "color": "#7f7f7f", "alpha": 0.25, "name": "cabinet_front"},
                {"origin": [0.88, 3.05, 0.00], "size": [0.04, 0.60, 1.80], "color": "#7f7f7f", "alpha": 0.25, "name": "cabinet_side"},
                {"origin": [1.00, 3.05, 1.78], "size": [0.80, 0.60, 0.04], "color": "#7f7f7f", "alpha": 0.25, "name": "cabinet_top"},
                {"origin": [1.65, 1.10, 0.00], "size": [1.20, 0.04, 2.00], "color": "#b58963", "alpha": 0.22, "name": "wood_partition"},
            ]
        )
    return objs


def load_stage3_json():
    with STAGE3_SPEC_PATH.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    with STAGE3_POINTS_PATH.open("r", encoding="utf-8") as f:
        points = json.load(f)
    return spec, points


def stage3_scenarios():
    spec, points = load_stage3_json()
    spec_map = {item["design_name"]: item for item in spec["scenarios"]}
    out = {}
    for item in points["scenarios"]:
        design = item["design_name"]
        room_key = design.split("_")[-1]
        tags = pd.DataFrame([tag["origin_m"] for tag in item["tags"]], columns=["x", "y", "z"])
        objs = []
        for obj in spec_map[design]["objects"]:
            objs.append(
                {
                    "origin": obj["origin_m"],
                    "size": obj["size_m"],
                    "color": "#9ecae9" if "glass" in obj["name"] else "#7f7f7f" if "cabinet" in obj["name"] or obj["material"] == "pec" else "#c49c94",
                    "alpha": 0.18 if "glass" in obj["name"] else 0.26,
                    "name": obj["name"],
                }
            )
        out[room_key] = {"anchor": np.array(item["anchors"][0]["origin_m"], dtype=float), "tags": tags, "objects": objs}
    return out


def sample_tag_rows(df: pd.DataFrame, max_arrows: int = 8) -> pd.DataFrame:
    if len(df) <= max_arrows:
        return df
    idx = np.linspace(0, len(df) - 1, max_arrows).round().astype(int)
    return df.iloc[idx].copy()


def draw_stage1_figure(out_path: Path):
    stage1 = representative_stage1()
    fig = plt.figure(figsize=(7.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    style_3d_ax(ax, "Stage 1 single-slab local frame", limits=((-0.2, 2.2), (-1.1, 1.1), (0.0, 3.0)), elev=20, azim=-58)
    ax.scatter(stage1["tx_pos"][0], stage1["tx_pos"][1], stage1["tx_pos"][2], c="black", s=110, marker="^")
    add_arrow(ax, stage1["tx_pos"], stage1["tx_bore"], length=0.45, color="black")
    ax.text(stage1["tx_pos"][0], stage1["tx_pos"][1], stage1["tx_pos"][2] + 0.12, "Tx", fontsize=9)
    ax.scatter(stage1["rx_pos"][0], stage1["rx_pos"][1], stage1["rx_pos"][2], c="#4c78a8", s=90, marker="o")
    add_arrow(ax, stage1["rx_pos"], stage1["rx_bore"], length=0.30, color="#4c78a8")
    ax.text(stage1["rx_pos"][0], stage1["rx_pos"][1], stage1["rx_pos"][2] + 0.12, "Rx/tag", fontsize=9, color="#1f4e79")
    ax.scatter(stage1["tags"][:, 0], stage1["tags"][:, 1], stage1["tags"][:, 2], c="#4c78a8", s=28, alpha=0.7)
    for slab in stage1["slabs"]:
        add_cuboid(ax, slab["origin"], slab["size"], slab["color"], alpha=0.18, edgecolor=slab["color"], linewidth=0.8)
    ax.text(0.45, -0.95, 2.7, "Tx broadside: +x", fontsize=9)
    ax.text(1.05, 0.42, 1.45, "Representative slab options:\nfloor / ceiling / wall_x", fontsize=8, color="#7a3e00")
    ax.plot([0, stage1["rx_pos"][0]], [0, stage1["rx_pos"][1]], [stage1["tx_pos"][2], stage1["rx_pos"][2]], linestyle="--", color="0.45", linewidth=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_stage23_grid(out_path: Path, stage_name: str, scenario_data: dict[str, dict], inferred_boresight_note: str | None = None):
    fig = plt.figure(figsize=(16, 5.8))
    room_order = ["A", "B", "C"]
    for i, room_type in enumerate(room_order, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        style_3d_ax(ax, f"{stage_name} scenario {room_type}", limits=((0, 5), (0, 4), (0, 3)), elev=22, azim=-58)
        add_room_wireframe(ax)
        add_floor_outline(ax)
        scene = scenario_data[room_type]
        for obj in scene["objects"]:
            add_cuboid(ax, obj["origin"], obj["size"], obj["color"], alpha=obj["alpha"], edgecolor=obj["color"], linewidth=0.7)
        anchor = np.asarray(scene["anchor"], dtype=float)
        ax.scatter(anchor[0], anchor[1], anchor[2], c="black", s=130, marker="^")
        add_arrow(ax, anchor, ANCHOR_BORE_STAGE23, length=0.45, color="black")
        ax.text(anchor[0], anchor[1], anchor[2] + 0.10, "ceiling anchor", fontsize=8)

        tags = scene["tags"].copy()
        if "layer" in tags.columns:
            layer1 = tags[tags["layer"] == 1]
            layer2 = tags[tags["layer"] == 2]
            ax.scatter(layer1["x"], layer1["y"], layer1["z"], c="#4c78a8", s=18, alpha=0.55)
            ax.scatter(layer2["x"], layer2["y"], layer2["z"], c="#f58518", s=22, alpha=0.70)
            arrow_tags = sample_tag_rows(tags[["x", "y", "z"]], max_arrows=8)
            ax.text(0.15, 3.55, 2.8, f"{len(tags)} candidate tags\nblue: Layer 1\norange: Layer 2", fontsize=8)
        else:
            ax.scatter(tags["x"], tags["y"], tags["z"], c="#4c78a8", s=34, alpha=0.82)
            arrow_tags = tags
            ax.text(0.15, 3.55, 2.8, f"{len(tags)} selected HFSS tags", fontsize=8)
        for _, row in arrow_tags.iterrows():
            add_arrow(ax, np.array([row["x"], row["y"], row["z"]]), TAG_BORE_STAGE23, length=0.18, color="#4c78a8", linewidth=1.2)
        if room_type == "A":
            ax.text(0.10, 0.18, 0.10, "tag broadside: +z", fontsize=8, color="#1f4e79")
            ax.text(2.95, 2.10, 2.92, "anchor broadside: -z", fontsize=8)
    if inferred_boresight_note:
        fig.text(0.5, 0.01, inferred_boresight_note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_markdown(out_path: Path, stage1_img: Path, stage2_img: Path, stage3_img: Path):
    lines = [
        "# Stage 1 / 2 / 3 3D spatial visualization",
        "",
        "## Stage 1",
        "",
        f"![stage1_3d]({stage1_img.as_posix()})",
        "",
        "- `verified`: Stage 1 uses the local single-slab frame from [designLhsSweep.m](" + (ROOT / "+sweep" / "designLhsSweep.m").as_posix() + ":1) and antenna boresight from [runOneCase.m](" + (ROOT / "+sweep" / "runOneCase.m").as_posix() + ":283).",
        "- `verified`: representative boresight convention is `Tx = +x`, `Rx/tag = +z`.",
        "- `caution`: the slab placement varies over `floor / wall_x / wall_y / ceiling`; the 3D figure shows a representative set of slab options rather than every sampled case.",
        "",
        "## Stage 2 scenarios A / B / C",
        "",
        f"![stage2_3d]({stage2_img.as_posix()})",
        "",
        "- `verified`: room geometry and clutter definitions come from [makeRoomABCScene.m](" + (ROOT / "+scenes" / "makeRoomABCScene.m").as_posix() + ":1).",
        "- `verified`: anchor/tag coordinate convention comes from [generateRoomTxRxGrid.m](" + (ROOT / "+scenes" / "generateRoomTxRxGrid.m").as_posix() + ":1) and [runOneCase.m](" + (ROOT / "+sweep" / "runOneCase.m").as_posix() + ":209).",
        "- `verified`: ceiling anchor broadside is downward `-z`; tag broadside is upward `+z`.",
        "- `verified`: Scenario A has no extra clutter, B adds `window + desk`, C adds `window + desk + cabinet + partition`.",
        "",
        "## Stage 3 scenarios A / B / C",
        "",
        f"![stage3_3d]({stage3_img.as_posix()})",
        "",
        "- `verified`: room/object geometry is loaded from [stage3_room_abc_spec.json](" + STAGE3_SPEC_PATH.as_posix() + ") and selected tags from [stage3_scene_point_proposals_hfss_safe.json](" + STAGE3_POINTS_PATH.as_posix() + ").",
        "- `verified`: Stage 3 A/B/C are the HFSS export scenarios corresponding to Room A/B/C.",
        "- `inference`: the Stage 3 JSON stores anchor positions but not explicit boresight vectors, so the broadside arrows are drawn using the inherited Stage 2 ceiling-anchor / tag convention from `rt_cp_uwb`.",
        "",
        "## Practical comparison",
        "",
        "- Stage 1 is a local slab-controlled geometry, not a room deployment.",
        "- Stage 2 is the full room candidate-cloud stage.",
        "- Stage 3 is the selected HFSS subset stage inside the same room geometry family.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stage1_img = OUT_DIR / "plot_stage1_3d.png"
    stage2_img = OUT_DIR / "plot_stage2_room_abc_3d.png"
    stage3_img = OUT_DIR / "plot_stage3_room_abc_3d.png"
    md_path = OUT_DIR / "stage123_3d_compare.md"

    draw_stage1_figure(stage1_img)

    stage2_data = {}
    for room in ["A", "B", "C"]:
        anchor, tags = generate_stage2_tags(room, n_positions=50, seed=20260423)
        stage2_data[room] = {"anchor": anchor, "tags": tags, "objects": stage2_objects(room)}
    draw_stage23_grid(stage2_img, "Stage 2", stage2_data)

    stage3_data = stage3_scenarios()
    draw_stage23_grid(
        stage3_img,
        "Stage 3",
        stage3_data,
        inferred_boresight_note="Stage 3 boresight arrows are inferred from the inherited Stage 2 ceiling-anchor / tag convention; the HFSS JSON itself stores positions, not explicit boresight vectors.",
    )

    write_markdown(md_path, stage1_img, stage2_img, stage3_img)


if __name__ == "__main__":
    main()

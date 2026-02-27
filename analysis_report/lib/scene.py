"""Scene debug JSON loading, validation, and plotting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _to_float(v: Any, default: float = float("nan")) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _rect_poly(cx: float, cy: float, hx: float, hy: float) -> list[list[float]]:
    return [
        [float(cx - hx), float(cy - hy)],
        [float(cx + hx), float(cy - hy)],
        [float(cx + hx), float(cy + hy)],
        [float(cx - hx), float(cy + hy)],
    ]


def _arrow_xy(theta_deg: float) -> list[float]:
    t = np.deg2rad(float(theta_deg))
    return [float(np.cos(t)), float(np.sin(t))]


def _extract_link_params(row: dict[str, Any]) -> dict[str, Any]:
    prov = row.get("provenance_json", {})
    if isinstance(prov, str):
        s = prov.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                prov = json.loads(s)
            except Exception:
                prov = {}
        else:
            prov = {}
    if isinstance(prov, dict):
        lp = prov.get("link_params", {})
        if isinstance(lp, dict):
            return lp
    return {}


def _layout_bounds(tx: tuple[float, float], rx: tuple[float, float], objects: list[dict[str, Any]]) -> dict[str, float]:
    xs = [float(tx[0]), float(rx[0])]
    ys = [float(tx[1]), float(rx[1])]
    for o in objects:
        for pt in o.get("poly_xy", []):
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
    xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
    ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    padx = 0.08 * max(1.0, xmax - xmin)
    pady = 0.08 * max(1.0, ymax - ymin)
    return {
        "xmin": float(xmin - padx),
        "xmax": float(xmax + padx),
        "ymin": float(ymin - pady),
        "ymax": float(ymax + pady),
    }


def build_fallback_scene_from_link_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Build a 2D layout-only scene from link row/provenance when scene_debug is absent."""

    sid = str(row.get("scenario_id", "NA"))
    cid = str(row.get("case_id", ""))
    params = _extract_link_params(row)
    warns: list[str] = []
    objects: list[dict[str, Any]] = []
    tx = (0.0, 0.0)
    rx = (float(_to_float(row.get("d_m", 1.0), 1.0)), 0.0)
    tx_bore = [1.0, 0.0]
    rx_bore = [-1.0, 0.0]

    if sid == "C0":
        d = _to_float(params.get("distance_m", row.get("d_m", 1.0)), 1.0)
        yaw = _to_float(params.get("yaw_deg", row.get("yaw_deg", 0.0)), 0.0)
        tx = (0.0, 0.0)
        rx = (float(d), 0.0)
        tx_bore = [1.0, 0.0]
        rx_bore = _arrow_xy(180.0 + yaw)

    elif sid in {"A2", "A4"}:
        d = _to_float(params.get("distance_m", row.get("d_m", 6.0)), 6.0)
        y_plane = _to_float(params.get("y_plane", 2.0), 2.0)
        tx = (0.0, 0.0)
        rx = (float(d), 0.0)
        x_mid = 0.5 * (tx[0] + rx[0])
        objects.append(
            {
                "name": "reflector_plane",
                "type": "reflector",
                "material": str(params.get("material", row.get("material_class", "PEC"))),
                "poly_xy": _rect_poly(cx=x_mid, cy=y_plane, hx=max(2.5, 0.6 * d), hy=0.06),
                "closed": True,
            }
        )
        if "physical_occluder" in str(row.get("los_block_method", "")):
            objects.append(
                {
                    "name": "los_blocker",
                    "type": "absorber",
                    "material": "absorber_proxy",
                    "poly_xy": _rect_poly(cx=x_mid, cy=0.0, hx=0.08, hy=0.5),
                    "closed": True,
                }
            )

    elif sid in {"A3", "A5"}:
        off = _to_float(params.get("offset", 3.5), 3.5)
        rx_x = _to_float(params.get("rx_x", row.get("rx_x", off)), off)
        rx_y = _to_float(params.get("rx_y", row.get("rx_y", off + 1.0)), off + 1.0)
        tx = (0.0, 0.0)
        rx = (float(rx_x), float(rx_y))
        objects.extend(
            [
                {
                    "name": "plane_y",
                    "type": "reflector",
                    "material": "PEC",
                    "poly_xy": _rect_poly(cx=0.5 * max(rx_x, off), cy=off, hx=max(2.5, 0.6 * max(rx_x, off)), hy=0.06),
                    "closed": True,
                },
                {
                    "name": "plane_x",
                    "type": "reflector",
                    "material": "PEC",
                    "poly_xy": _rect_poly(cx=off, cy=0.5 * max(rx_y, off), hx=0.06, hy=max(2.5, 0.6 * max(rx_y, off))),
                    "closed": True,
                },
            ]
        )
        if "physical_occluder" in str(row.get("los_block_method", "")):
            x_mid = 0.5 * (tx[0] + rx[0])
            y_mid = 0.5 * (tx[1] + rx[1])
            objects.append(
                {
                    "name": "los_blocker",
                    "type": "absorber",
                    "material": "absorber_proxy",
                    "poly_xy": _rect_poly(cx=x_mid, cy=y_mid, hx=0.08, hy=0.5),
                    "closed": True,
                }
            )
        if sid == "A5" and (int(_to_float(row.get("roughness_flag", 0), 0)) == 1 or int(_to_float(row.get("human_flag", 0), 0)) == 1):
            objects.append(
                {
                    "name": "stress_scatter_region",
                    "type": "furniture",
                    "material": "stress_region",
                    "poly_xy": _rect_poly(cx=0.65 * rx_x, cy=0.45 * rx_y, hx=0.45, hy=0.35),
                    "closed": True,
                }
            )

    elif sid in {"B1", "B2", "B3"}:
        tx = (2.0, 0.0)
        rx = (
            _to_float(params.get("rx_x", row.get("rx_x", 2.0)), 2.0),
            _to_float(params.get("rx_y", row.get("rx_y", 0.0)), 0.0),
        )
        # Room boundary in top view: x in [0,10], y in [-4,4].
        objects.append(
            {
                "name": "room_boundary",
                "type": "wall",
                "material": "PEC",
                "poly_xy": _rect_poly(cx=5.0, cy=0.0, hx=5.0, hy=4.0),
                "closed": True,
            }
        )
        if sid == "B2":
            objects.append(
                {
                    "name": "partition",
                    "type": "obstacle",
                    "material": "PEC",
                    "poly_xy": _rect_poly(cx=5.0, cy=0.0, hx=0.05, hy=1.8),
                    "closed": True,
                }
            )
        if sid == "B3":
            objects.extend(
                [
                    {
                        "name": "corner_wall_x",
                        "type": "obstacle",
                        "material": "PEC",
                        "poly_xy": _rect_poly(cx=5.5, cy=1.5, hx=0.05, hy=1.5),
                        "closed": True,
                    },
                    {
                        "name": "corner_wall_y",
                        "type": "obstacle",
                        "material": "PEC",
                        "poly_xy": _rect_poly(cx=5.5, cy=1.5, hx=1.5, hy=0.05),
                        "closed": True,
                    },
                ]
            )
    else:
        warns.append(f"fallback_layout_unknown_scenario:{sid}")

    bounds = _layout_bounds(tx=tx, rx=rx, objects=objects)
    scene = {
        "scene_schema": "scene_debug_v1",
        "scenario_id": sid,
        "case_id": cid,
        "case_label": str(row.get("case_label", row.get("link_id", cid))),
        "coord_frame": {"units": "m", "plane": "xy", "z_up": True},
        "bounds": bounds,
        "tx": {"x": float(tx[0]), "y": float(tx[1]), "z": 1.5, "boresight_xy": tx_bore},
        "rx": {"x": float(rx[0]), "y": float(rx[1]), "z": 1.5, "boresight_xy": rx_bore},
        "objects": objects,
        "rays_topk": [],
        "meta": {
            "source": "fallback_layout_from_link_row",
            "note": "Ray polylines unavailable without scene_debug export.",
        },
    }
    return scene, warns


def load_scene_debug(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"scene_debug must be a JSON object: {p}")
    obj["_path"] = str(p)
    return obj


def validate_scene_debug(scene: dict[str, Any]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if str(scene.get("scene_schema", "")) != "scene_debug_v1":
        problems.append("scene_schema!=scene_debug_v1")
    tx = scene.get("tx", {})
    rx = scene.get("rx", {})
    for k in ["x", "y", "z"]:
        if k not in tx:
            problems.append(f"tx.{k} missing")
        if k not in rx:
            problems.append(f"rx.{k} missing")
    objs = scene.get("objects", [])
    if not isinstance(objs, list):
        problems.append("objects missing or not list")
    else:
        for i, o in enumerate(objs):
            if not isinstance(o, dict):
                problems.append(f"objects[{i}] not object")
                continue
            if "poly_xy" not in o:
                problems.append(f"objects[{i}].poly_xy missing")
    rays = scene.get("rays_topk", [])
    if not isinstance(rays, list):
        problems.append("rays_topk missing or not list")
    else:
        for i, r in enumerate(rays):
            if not isinstance(r, dict):
                problems.append(f"rays_topk[{i}] not object")
                continue
            if "vertices_xyz" not in r:
                problems.append(f"rays_topk[{i}].vertices_xyz missing")
    return len(problems) == 0, problems


def _object_style(obj_type: str) -> tuple[str, str]:
    t = str(obj_type).lower()
    if t == "wall":
        return ("#e0e0e0", "#555555")
    if t == "floor_ceiling":
        return ("#f5f5f5", "#999999")
    if t in {"reflector", "obstacle"}:
        return ("#ffe5cc", "#cc6600")
    if t == "absorber":
        return ("#d8f0d8", "#2d7f2d")
    if t == "furniture":
        return ("#dfe7ff", "#3558a7")
    return ("#eeeeee", "#777777")


def _plot_object(ax: Any, obj: dict[str, Any]) -> None:
    pts = np.asarray(obj.get("poly_xy", []), dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) == 0:
        return
    fill, edge = _object_style(str(obj.get("type", "object")))
    unique_xy = np.unique(np.round(pts[:, :2], 9), axis=0)
    if len(unique_xy) < 3:
        ax.plot(pts[:, 0], pts[:, 1], color=edge, lw=2.0, alpha=0.9)
    else:
        ax.fill(pts[:, 0], pts[:, 1], facecolor=fill, edgecolor=edge, alpha=0.55, linewidth=1.2)


def plot_scene(scene: dict[str, Any], out_png: str | Path, figure_size: tuple[float, float] = (10.0, 6.0)) -> str:
    p = Path(out_png)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figure_size)

    objects = scene.get("objects", [])
    if isinstance(objects, list):
        for obj in objects:
            if isinstance(obj, dict):
                _plot_object(ax, obj)

    tx = scene.get("tx", {})
    rx = scene.get("rx", {})
    tx_xy = (float(tx.get("x", np.nan)), float(tx.get("y", np.nan)))
    rx_xy = (float(rx.get("x", np.nan)), float(rx.get("y", np.nan)))
    if np.isfinite(tx_xy[0]) and np.isfinite(tx_xy[1]):
        ax.scatter([tx_xy[0]], [tx_xy[1]], s=120, marker="^", color="#d62728", label="Tx")
        ax.text(tx_xy[0], tx_xy[1], " Tx", fontsize=9)
    if np.isfinite(rx_xy[0]) and np.isfinite(rx_xy[1]):
        ax.scatter([rx_xy[0]], [rx_xy[1]], s=120, marker="o", color="#1f77b4", label="Rx")
        ax.text(rx_xy[0], rx_xy[1], " Rx", fontsize=9)

    tx_b = np.asarray(tx.get("boresight_xy", []), dtype=float)
    if tx_b.shape == (2,) and np.all(np.isfinite(tx_b)) and np.isfinite(tx_xy[0]) and np.isfinite(tx_xy[1]):
        n = float(np.linalg.norm(tx_b))
        if n > 0:
            tx_b = tx_b / n
            ax.arrow(
                tx_xy[0],
                tx_xy[1],
                0.5 * tx_b[0],
                0.5 * tx_b[1],
                head_width=0.12,
                head_length=0.14,
                fc="#d62728",
                ec="#d62728",
                alpha=0.85,
                length_includes_head=True,
            )
    rx_b = np.asarray(rx.get("boresight_xy", []), dtype=float)
    if rx_b.shape == (2,) and np.all(np.isfinite(rx_b)) and np.isfinite(rx_xy[0]) and np.isfinite(rx_xy[1]):
        n = float(np.linalg.norm(rx_b))
        if n > 0:
            rx_b = rx_b / n
            ax.arrow(
                rx_xy[0],
                rx_xy[1],
                0.5 * rx_b[0],
                0.5 * rx_b[1],
                head_width=0.12,
                head_length=0.14,
                fc="#1f77b4",
                ec="#1f77b4",
                alpha=0.85,
                length_includes_head=True,
            )

    rays = scene.get("rays_topk", [])
    if isinstance(rays, list):
        for r in rays:
            if not isinstance(r, dict):
                continue
            v = np.asarray(r.get("vertices_xyz", []), dtype=float)
            if v.ndim != 2 or v.shape[1] < 2 or len(v) < 2:
                continue
            n_bounce = int(r.get("n_bounce", 0))
            color = "#2ca02c" if (n_bounce % 2 == 0) else "#9467bd"
            ax.plot(v[:, 0], v[:, 1], "-", color=color, alpha=0.6, lw=1.0)

    bounds = scene.get("bounds", {})
    try:
        xmin = float(bounds.get("xmin", np.nan))
        xmax = float(bounds.get("xmax", np.nan))
        ymin = float(bounds.get("ymin", np.nan))
        ymax = float(bounds.get("ymax", np.nan))
    except Exception:
        xmin = xmax = ymin = ymax = float("nan")
    if np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    title = f"Scene: {scene.get('scenario_id', 'NA')} / {scene.get('case_id', 'NA')}"
    label = str(scene.get("case_label", "")).strip()
    if label:
        title += f" ({label})"
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Keep legend deterministic and deduplicated.
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h
        ax.legend(list(uniq.values()), list(uniq.keys()), loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def plot_scene_global(
    base_scene: dict[str, Any],
    rx_points: list[tuple[float, float]],
    out_png: str | Path,
    figure_size: tuple[float, float] = (10.0, 6.0),
) -> str:
    scene = dict(base_scene)
    scene["rays_topk"] = []
    p = Path(out_png)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figure_size)

    objects = scene.get("objects", [])
    if isinstance(objects, list):
        for obj in objects:
            if isinstance(obj, dict):
                _plot_object(ax, obj)

    tx = scene.get("tx", {})
    tx_xy = (float(tx.get("x", np.nan)), float(tx.get("y", np.nan)))
    if np.isfinite(tx_xy[0]) and np.isfinite(tx_xy[1]):
        ax.scatter([tx_xy[0]], [tx_xy[1]], s=110, marker="^", color="#d62728", label="Tx")

    if rx_points:
        xx = np.asarray([p[0] for p in rx_points], dtype=float)
        yy = np.asarray([p[1] for p in rx_points], dtype=float)
        m = np.isfinite(xx) & np.isfinite(yy)
        ax.scatter(xx[m], yy[m], s=45, marker="o", alpha=0.75, color="#1f77b4", label="Rx grid")

    bounds = scene.get("bounds", {})
    if isinstance(bounds, dict):
        xmin = float(bounds.get("xmin", np.nan))
        xmax = float(bounds.get("xmax", np.nan))
        ymin = float(bounds.get("ymin", np.nan))
        ymax = float(bounds.get("ymax", np.nan))
        if np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax):
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    sid = str(scene.get("scenario_id", "NA"))
    ax.set_title(f"Scene Global Layout: {sid}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)

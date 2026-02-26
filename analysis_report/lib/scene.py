"""Scene debug JSON loading, validation, and plotting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    ax.legend(loc="best", fontsize=8)
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

"""Run standardized proxy simulations and export schema v1 outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.el_proxy import compute_el_proxy
from analysis.excess_loss import add_el_db
from analysis.link_conditions import build_link_U_from_scenario
from analysis.link_metrics import compute_link_metrics
from analysis.pdp_synthesis import synthesize_dualcp_pdp
from analysis.ray_table import build_ray_table_from_rt
from analysis.windowing import estimate_tau0, make_early_late_masks
from analysis.xpd_stats import make_subbands
from calibration.floor_model import AngleSensitiveFloorXPD, ConstantFloorXPD, FloorXPDModel, FreqDependentFloorXPD
from polarization.xpr_models import BaseXPRModel, BinnedXPR, ConditionalLinearXPR, ConstantXPR
from rt_core.geometry import Material, Plane
from rt_core.materials import DEFAULT_MATERIAL_SPECS
from rt_core.tracer import trace_paths
from rt_io.standard_outputs_hdf5 import export_csv, save_run
from rt_types.standard_outputs import RayTable, SCHEMA_VERSION, StandardOutputBundle
from scenarios import A2_pec_plane, A3_corner_2bounce, A4_dielectric_plane, A5_depol_stress, B0_room_box, C0_free_space
from scenarios.common import default_antennas, make_los_blocker_plane, paths_to_records, uwb_frequency


def _pctl(x: np.ndarray, q: float) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, float(q)))


def _parse_float_list(raw: str, default: list[float]) -> list[float]:
    out = []
    for tok in str(raw).split(","):
        s = tok.strip()
        if not s:
            continue
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out or list(default)


def _parse_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return bool(raw)
    s = str(raw).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _sanitize_token(s: Any) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))
    out = out.strip("._")
    return out or "na"


def _git_meta() -> tuple[str, str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = "unknown"
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        branch = "unknown"
    return commit, branch


def _load_json(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise ValueError(f"JSON not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _make_xpr_model(scenario: str, cfg: dict[str, Any]) -> BaseXPRModel:
    if cfg:
        t = str(cfg.get("type", "constant")).lower()
        if t == "linear":
            return ConditionalLinearXPR(
                a0=float(cfg.get("a0", 8.0)),
                a_el=float(cfg.get("a_el", 0.0)),
                a_late=float(cfg.get("a_late", 0.0)),
                a_incidence=float(cfg.get("a_incidence", 0.0)),
                a_rough=float(cfg.get("a_rough", -3.0)),
                a_human=float(cfg.get("a_human", -3.0)),
                sigma_db=float(cfg.get("sigma_db", 0.0)),
                material_bias=dict(cfg.get("material_bias", {})),
            )
        if t == "binned":
            return BinnedXPR(default_xpr_db=float(cfg.get("default_xpr_db", 8.0)), bins=list(cfg.get("bins", [])))
        return ConstantXPR(float(cfg.get("xpr_db", 8.0)))

    s = str(scenario).upper()
    if s == "A4":
        return ConditionalLinearXPR(a0=3.0, a_el=-0.03, sigma_db=0.6)
    if s == "A5":
        return ConditionalLinearXPR(a0=6.0, a_rough=-4.0, a_human=-4.0, sigma_db=0.8)
    if s.startswith("B"):
        return ConditionalLinearXPR(a0=7.0, a_el=-0.02, a_late=-1.2, sigma_db=1.0)
    return ConstantXPR(8.0)


def _make_floor_model(cfg: dict[str, Any]) -> FloorXPDModel:
    if not cfg:
        return ConstantFloorXPD(25.0, sigma_db=0.5)
    t = str(cfg.get("type", "constant")).lower()
    if t == "freq":
        return FreqDependentFloorXPD(
            freq_hz=np.asarray(cfg.get("freq_hz", [6e9, 8e9, 10e9]), dtype=float),
            xpd_floor_db=np.asarray(cfg.get("xpd_floor_db", [23.0, 25.0, 27.0]), dtype=float),
            sigma_db=float(cfg.get("sigma_db", 0.0)),
        )
    if t == "angle":
        return AngleSensitiveFloorXPD(
            base_db=float(cfg.get("base_db", 25.0)),
            yaw_slope_db_per_deg=float(cfg.get("yaw_slope_db_per_deg", -0.05)),
            pitch_slope_db_per_deg=float(cfg.get("pitch_slope_db_per_deg", -0.02)),
            sigma_db=float(cfg.get("sigma_db", 0.0)),
        )
    return ConstantFloorXPD(float(cfg.get("xpd_floor_db", 25.0)), sigma_db=float(cfg.get("sigma_db", 0.0)))


def _build_floor_reference(bundles: list[StandardOutputBundle], bin_keys: list[str] | None = None) -> dict[str, Any]:
    rows = []
    for b in bundles:
        if str(b.scenario_id).upper() != "C0":
            continue
        u = b.conditions.to_dict()
        rows.append(
            {
                "xpd_early_db": float(b.metrics.XPD_early_db),
                "yaw_deg": float(u.get("yaw_deg", np.nan)),
                "pitch_deg": float(u.get("pitch_deg", np.nan)),
                "d_m": float(u.get("d_m", np.nan)),
            }
        )
    x = np.asarray([r["xpd_early_db"] for r in rows], dtype=float)
    out = {
        "version": "floor_reference_v1",
        "source_scenario": "C0",
        "count": int(len(x)),
        "xpd_floor_db": float(np.nanmedian(x)) if len(x) else float("nan"),
        "p5_db": _pctl(x, 5.0),
        "p95_db": _pctl(x, 95.0),
        # Use half-width uncertainty (p95-p5)/2 for consistency with curve/subband uncertainty.
        "delta_floor_db": float(0.5 * (_pctl(x, 95.0) - _pctl(x, 5.0))) if len(x) else float("nan"),
        "bin_keys": list(bin_keys or ["yaw_deg", "pitch_deg"]),
        "groups": [],
    }
    if not rows:
        return out
    gkeys = list(out["bin_keys"])
    buckets: dict[str, list[float]] = {}
    centers: dict[str, dict[str, float]] = {}
    for r in rows:
        vals = [round(float(r.get(k, np.nan)), 6) for k in gkeys]
        key = "|".join(str(v) for v in vals)
        buckets.setdefault(key, []).append(float(r["xpd_early_db"]))
        if key not in centers:
            centers[key] = {k: float(r.get(k, np.nan)) for k in gkeys}
    groups = []
    for k in sorted(buckets.keys()):
        vals = np.asarray(buckets[k], dtype=float)
        groups.append(
            {
                **centers[k],
                "count": int(len(vals)),
                "xpd_floor_db": float(np.nanmedian(vals)),
                "p5_db": _pctl(vals, 5.0),
                "p95_db": _pctl(vals, 95.0),
                "delta_floor_db": float(0.5 * (_pctl(vals, 95.0) - _pctl(vals, 5.0))),
            }
        )
    out["groups"] = groups
    return out


def _eval_floor_curve_for_case(
    floor_model: FloorXPDModel,
    freq_hz: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
) -> np.ndarray:
    f = np.asarray(freq_hz, dtype=float)
    if len(f) == 0:
        return np.asarray([], dtype=float)
    if isinstance(floor_model, ConstantFloorXPD):
        return np.full((len(f),), float(floor_model.xpd_floor_db), dtype=float)
    if isinstance(floor_model, AngleSensitiveFloorXPD):
        mu = (
            float(floor_model.base_db)
            + float(floor_model.yaw_slope_db_per_deg) * abs(float(yaw_deg))
            + float(floor_model.pitch_slope_db_per_deg) * abs(float(pitch_deg))
        )
        return np.full((len(f),), mu, dtype=float)
    if isinstance(floor_model, FreqDependentFloorXPD):
        ref_f = np.asarray(floor_model.freq_hz, dtype=float)
        ref_x = np.asarray(floor_model.xpd_floor_db, dtype=float)
        if len(ref_f) == 0 or len(ref_x) == 0:
            return np.full((len(f),), np.nan, dtype=float)
        return np.interp(f, ref_f, ref_x, left=float(ref_x[0]), right=float(ref_x[-1])).astype(float)
    return np.asarray(
        [
            float(
                floor_model.sample_floor_xpd_db(
                    f_hz=float(ff),
                    yaw_deg=float(yaw_deg),
                    pitch_deg=float(pitch_deg),
                    rng=None,
                )
            )
            for ff in f
        ],
        dtype=float,
    )


def _build_floor_subbands(
    freq_hz: np.ndarray,
    floor_curve_db: np.ndarray,
    floor_uncert_db: np.ndarray,
    num_subbands: int = 4,
) -> list[dict[str, Any]]:
    f = np.asarray(freq_hz, dtype=float)
    x = np.asarray(floor_curve_db, dtype=float)
    u = np.asarray(floor_uncert_db, dtype=float)
    if len(f) == 0 or len(x) != len(f):
        return []
    if len(u) != len(f):
        u = np.full((len(f),), np.nan, dtype=float)
    out: list[dict[str, Any]] = []
    for idx, (s, e) in enumerate(make_subbands(len(f), max(1, int(num_subbands)))):
        out.append(
            {
                "index": int(idx),
                "start_idx": int(s),
                "end_idx": int(e),
                "f_lo_hz": float(f[s]),
                "f_hi_hz": float(f[e - 1]),
                "xpd_floor_db": float(np.nanmedian(x[s:e])),
                "xpd_floor_uncert_db": float(np.nanmedian(u[s:e])),
            }
        )
    return out


def _build_floor_reference_with_curve(
    bundles: list[StandardOutputBundle],
    floor_model: FloorXPDModel,
    freq_hz: np.ndarray,
    bin_keys: list[str] | None = None,
    num_subbands: int = 4,
) -> dict[str, Any]:
    out = _build_floor_reference(bundles, bin_keys=bin_keys)
    f = np.asarray(freq_hz, dtype=float)
    if len(f) == 0:
        return out
    rows = []
    for b in bundles:
        if str(b.scenario_id).upper() != "C0":
            continue
        u = b.conditions.to_dict()
        rows.append(
            {
                "yaw_deg": float(u.get("yaw_deg", 0.0)),
                "pitch_deg": float(u.get("pitch_deg", 0.0)),
            }
        )
    if not rows:
        return out
    curves = []
    for r in rows:
        curves.append(
            _eval_floor_curve_for_case(
                floor_model=floor_model,
                freq_hz=f,
                yaw_deg=float(r["yaw_deg"]),
                pitch_deg=float(r["pitch_deg"]),
            )
        )
    X = np.asarray(curves, dtype=float)
    floor_curve = np.nanmedian(X, axis=0)
    p_lo = np.nanpercentile(X, 5.0, axis=0)
    p_hi = np.nanpercentile(X, 95.0, axis=0)
    uncert = 0.5 * (p_hi - p_lo)
    out["frequency_hz"] = f.tolist()
    out["xpd_floor_db"] = np.asarray(floor_curve, dtype=float).tolist()
    out["xpd_floor_uncert_db"] = np.asarray(uncert, dtype=float).tolist()
    out["xpd_floor_p_lo_db"] = np.asarray(p_lo, dtype=float).tolist()
    out["xpd_floor_p_hi_db"] = np.asarray(p_hi, dtype=float).tolist()
    out["percentiles"] = [5.0, 95.0]
    out["subbands"] = _build_floor_subbands(
        freq_hz=f,
        floor_curve_db=floor_curve,
        floor_uncert_db=uncert,
        num_subbands=int(num_subbands),
    )
    return out


def _lookup_floor(reference: dict[str, Any], U: dict[str, Any]) -> tuple[float, float]:
    raw_base = reference.get("xpd_floor_db", np.nan)
    if isinstance(raw_base, list):
        arr = np.asarray(raw_base, dtype=float)
        arr = arr[np.isfinite(arr)]
        base = float(np.median(arr)) if len(arr) else float("nan")
    else:
        base = float(raw_base)
    raw_delta = reference.get("delta_floor_db", np.nan)
    if isinstance(raw_delta, list):
        arrd = np.asarray(raw_delta, dtype=float)
        arrd = arrd[np.isfinite(arrd)]
        delta = float(np.median(arrd)) if len(arrd) else float("nan")
    else:
        delta = float(raw_delta)
    if not np.isfinite(delta):
        u = np.asarray(reference.get("xpd_floor_uncert_db", []), dtype=float)
        u = u[np.isfinite(u)]
        if len(u):
            delta = float(np.median(u))
    groups = list(reference.get("groups", []))
    if not groups:
        return base, delta
    keys = list(reference.get("bin_keys", ["yaw_deg", "pitch_deg"]))
    if not keys:
        return base, delta
    best = None
    for g in groups:
        dist = 0.0
        valid = False
        for k in keys:
            if (k not in U) or (k not in g):
                continue
            try:
                uv = float(U.get(k, np.nan))
                gv = float(g.get(k, np.nan))
            except Exception:
                continue
            if np.isfinite(uv) and np.isfinite(gv):
                valid = True
                dist += abs(uv - gv)
        if not valid:
            continue
        score = (dist, -int(g.get("count", 0)))
        if best is None or score < best[0]:
            best = (score, g)
    if best is None:
        return base, delta
    gg = best[1]
    return float(gg.get("xpd_floor_db", base)), float(gg.get("delta_floor_db", delta))


def _claim_caution_flag(excess_db: float, uncert_db: float, mode: str, scale: float) -> bool:
    m = str(mode).lower().strip()
    if m == "off":
        return False
    if not (np.isfinite(excess_db) and np.isfinite(uncert_db)):
        return False
    thr = abs(float(uncert_db))
    if m == "scaled":
        thr = thr * max(float(scale), 0.0)
    return bool(abs(float(excess_db)) <= thr)


def _floor_curve_from_reference(reference: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq = np.asarray(reference.get("frequency_hz", []), dtype=float)
    floor = np.asarray(reference.get("xpd_floor_db", []), dtype=float)
    uncert = np.asarray(reference.get("xpd_floor_uncert_db", []), dtype=float)
    if freq.ndim != 1 or floor.ndim != 1 or len(freq) == 0 or len(freq) != len(floor):
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
    if len(uncert) != len(floor):
        uncert = np.full((len(floor),), np.nan, dtype=float)
    return freq, floor, uncert


def _floor_subbands_from_reference(reference: dict[str, Any]) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    sb = reference.get("subbands", [])
    if not isinstance(sb, list):
        return [], np.asarray([], dtype=float), np.asarray([], dtype=float)
    rows: list[dict[str, float]] = []
    f = []
    u = []
    for i, r in enumerate(sb):
        if not isinstance(r, dict):
            continue
        xf = float(r.get("xpd_floor_db", np.nan))
        xu = float(r.get("xpd_floor_uncert_db", np.nan))
        rows.append(
            {
                "index": float(r.get("index", i)),
                "f_lo_hz": float(r.get("f_lo_hz", np.nan)),
                "f_hi_hz": float(r.get("f_hi_hz", np.nan)),
                "xpd_floor_db": xf,
                "xpd_floor_uncert_db": xu,
            }
        )
        f.append(xf)
        u.append(xu)
    return rows, np.asarray(f, dtype=float), np.asarray(u, dtype=float)


def _case_label(scenario_id: str, params: dict[str, Any]) -> str:
    s = str(scenario_id).upper()
    p = dict(params)
    if s == "C0":
        return (
            f"d={float(p.get('distance_m', np.nan)):.2f}m"
            f"_yaw={float(p.get('yaw_deg', 0.0)):+.1f}"
            f"_pitch={float(p.get('pitch_deg', 0.0)):+.1f}"
            f"_rep={int(p.get('rep_id', 0))}"
        )
    if s == "A2":
        return (
            f"d={float(p.get('distance_m', np.nan)):.2f}m"
            f"_y={float(p.get('y_plane', np.nan)):.2f}"
            f"_rep={int(p.get('rep_id', 0))}"
        )
    if s == "A3":
        return (
            f"off={float(p.get('offset', np.nan)):.2f}"
            f"_rx=({float(p.get('rx_x', np.nan)):.2f},{float(p.get('rx_y', np.nan)):.2f})"
            f"_rep={int(p.get('rep_id', 0))}"
        )
    if s == "A4":
        return (
            f"mat={_sanitize_token(p.get('material', 'na'))}"
            f"_y={float(p.get('y_plane', np.nan)):.2f}"
            f"_d={float(p.get('distance_m', np.nan)):.2f}"
            f"_rep={int(p.get('rep_id', 0))}"
        )
    if s == "A5":
        return (
            f"rho={float(p.get('rho', np.nan)):.2f}"
            f"_rep={int(p.get('rep_id', 0))}"
            f"_outer={int(p.get('rep_outer', 0))}"
        )
    if s.startswith("B"):
        return f"rx=({float(p.get('rx_x', np.nan)):.2f},{float(p.get('rx_y', np.nan)):.2f})"
    return _sanitize_token(params)


def _plane_material_name(plane: Plane) -> str:
    try:
        mat = plane.material
        name = str(getattr(mat, "name", "") or "")
        if name:
            return name
        return str(getattr(mat, "kind", "unknown"))
    except Exception:
        return "unknown"


def _plane_vertices_xyz(plane: Plane) -> np.ndarray:
    p0 = np.asarray(plane.p0, dtype=float)
    u, v = plane.local_axes()
    hu = float(plane.half_extent_u) if plane.half_extent_u is not None else 0.0
    hv = float(plane.half_extent_v) if plane.half_extent_v is not None else 0.0
    if hu <= 0.0 or hv <= 0.0:
        return np.asarray([p0], dtype=float)
    return np.asarray(
        [
            p0 + hu * u + hv * v,
            p0 + hu * u - hv * v,
            p0 - hu * u - hv * v,
            p0 - hu * u + hv * v,
        ],
        dtype=float,
    )


def _plane_object_type(plane: Plane, scenario_id: str) -> str:
    pid = int(getattr(plane, "id", -1))
    mat_name = _plane_material_name(plane).lower()
    sid = str(scenario_id).upper()
    if "absorber" in mat_name:
        return "absorber"
    if pid >= 9500:
        return "furniture"
    if pid in {201, 301, 302}:
        return "obstacle"
    if sid.startswith("B") and pid in {101, 102, 103, 104}:
        return "wall"
    if sid.startswith("B") and pid in {105, 106}:
        return "floor_ceiling"
    if sid in {"A2", "A3", "A4", "A5"}:
        return "reflector"
    return "object"


def _serialize_scene_objects(scene: list[Plane], scenario_id: str) -> list[dict[str, Any]]:
    sid = str(scenario_id).upper()
    # For corner scenarios, infer the two reflector boundaries and suppress
    # stress objects that are geometrically behind the reflector walls.
    x_wall: float | None = None
    y_wall: float | None = None
    if sid in {"A3", "A5"}:
        for pl in scene:
            if _plane_object_type(pl, scenario_id) != "reflector":
                continue
            vv = _plane_vertices_xyz(pl)
            if vv.ndim != 2 or vv.shape[1] < 2 or len(vv) < 2:
                continue
            xs = vv[:, 0]
            ys = vv[:, 1]
            if float(np.nanstd(xs)) < 1e-8:
                x_wall = float(np.nanmedian(xs))
            if float(np.nanstd(ys)) < 1e-8:
                y_wall = float(np.nanmedian(ys))

    objects: list[dict[str, Any]] = []
    for idx, plane in enumerate(scene):
        verts = _plane_vertices_xyz(plane)
        obj_type = _plane_object_type(plane, scenario_id)
        if obj_type == "absorber":
            # In top-view plotting, represent vertical LOS blockers as a thin line-like strip
            # around the in-plane u-axis extent. This avoids exaggerated projected area that
            # can look like non-physical "refraction" in 2D.
            p0 = np.asarray(plane.p0, dtype=float)
            u, _ = plane.local_axes()
            hu = float(plane.half_extent_u) if plane.half_extent_u is not None else 0.05
            p_a = p0 + hu * u
            p_b = p0 - hu * u
            dxy = np.asarray([p_b[0] - p_a[0], p_b[1] - p_a[1]], dtype=float)
            nrm = float(np.hypot(dxy[0], dxy[1]))
            if nrm < 1e-9:
                dxy = np.asarray([1.0, 0.0], dtype=float)
                nrm = 1.0
            perp = np.asarray([-dxy[1], dxy[0]], dtype=float) / nrm
            t = 0.01
            poly_xy = [
                [float(p_a[0] + t * perp[0]), float(p_a[1] + t * perp[1])],
                [float(p_b[0] + t * perp[0]), float(p_b[1] + t * perp[1])],
                [float(p_b[0] - t * perp[0]), float(p_b[1] - t * perp[1])],
                [float(p_a[0] - t * perp[0]), float(p_a[1] - t * perp[1])],
            ]
        else:
            poly_xy = [[float(v[0]), float(v[1])] for v in verts]
        if obj_type == "furniture" and sid in {"A3", "A5"} and x_wall is not None and y_wall is not None and poly_xy:
            cx = float(np.mean([p[0] for p in poly_xy]))
            cy = float(np.mean([p[1] for p in poly_xy]))
            # Remove furniture/scatterers that sit behind the corner walls.
            if (cx >= float(x_wall) - 1e-6) or (cy >= float(y_wall) - 1e-6):
                continue
        objects.append(
            {
                "name": f"plane_{int(getattr(plane, 'id', idx))}",
                "type": obj_type,
                "material": _plane_material_name(plane),
                "poly_xy": poly_xy,
                "closed": bool(len(poly_xy) >= 3),
            }
        )
    return objects


def _estimate_path_power_lin(path_rec: dict[str, Any]) -> float:
    sg = np.asarray(path_rec.get("scalar_gain_f", []), dtype=complex)
    if len(sg) > 0:
        return float(np.mean(np.abs(sg) ** 2))
    af = np.asarray(path_rec.get("A_f", []), dtype=complex)
    if len(af) > 0:
        return float(np.mean(np.abs(af) ** 2))
    return 0.0


def _extract_rays_topk(path_records: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if int(k) <= 0:
        return []
    scored = []
    for i, p in enumerate(path_records):
        points = p.get("points", [])
        if not points:
            continue
        p_lin = _estimate_path_power_lin(p)
        scored.append((float(p_lin), int(i), p))
    scored.sort(key=lambda t: t[0], reverse=True)
    out = []
    for p_lin, i, p in scored[: int(k)]:
        seg_count = int(p.get("segment_count", 0))
        n_bounce = int(max(0, seg_count - 1))
        out.append(
            {
                "ray_id": int(i),
                "n_bounce": int(n_bounce),
                "P_lin": float(p_lin),
                "tau_s": float(p.get("tau_s", np.nan)),
                "surface_ids": list((p.get("meta", {}) or {}).get("surface_ids", [])),
                "vertices_xyz": [[float(x), float(y), float(z)] for x, y, z in np.asarray(p.get("points", []), dtype=float)],
            }
        )
    return out


def _scene_bounds(
    tx: np.ndarray,
    rx: np.ndarray,
    objects: list[dict[str, Any]],
    rays: list[dict[str, Any]],
) -> dict[str, float]:
    xs: list[float] = [float(tx[0]), float(rx[0])]
    ys: list[float] = [float(tx[1]), float(rx[1])]
    for obj in objects:
        for pt in obj.get("poly_xy", []):
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
    for r in rays:
        for p in r.get("vertices_xyz", []):
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
    if not xs or not ys:
        return {"xmin": -1.0, "xmax": 1.0, "ymin": -1.0, "ymax": 1.0}
    xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
    ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    pad_x = 0.05 * max(1.0, xmax - xmin)
    pad_y = 0.05 * max(1.0, ymax - ymin)
    return {
        "xmin": float(xmin - pad_x),
        "xmax": float(xmax + pad_x),
        "ymin": float(ymin - pad_y),
        "ymax": float(ymax + pad_y),
    }


def _build_room_scene(kind: str) -> list[Plane]:
    scene = B0_room_box.build_scene()
    if kind == "B2":
        scene.append(
            Plane(
                id=201,
                p0=np.array([5.0, 0.0, 1.5]),
                normal=np.array([-1.0, 0.0, 0.0]),
                material=Material.pec(),
                u_axis=np.array([0.0, 1.0, 0.0]),
                v_axis=np.array([0.0, 0.0, 1.0]),
                half_extent_u=1.8,
                half_extent_v=1.5,
            )
        )
    if kind == "B3":
        scene.extend(
            [
                Plane(
                    id=301,
                    p0=np.array([5.5, 1.5, 1.5]),
                    normal=np.array([-1.0, 0.0, 0.0]),
                    material=Material.pec(),
                    u_axis=np.array([0.0, 1.0, 0.0]),
                    v_axis=np.array([0.0, 0.0, 1.0]),
                    half_extent_u=1.5,
                    half_extent_v=1.5,
                ),
                Plane(
                    id=302,
                    p0=np.array([5.5, 1.5, 1.5]),
                    normal=np.array([0.0, -1.0, 0.0]),
                    material=Material.pec(),
                    u_axis=np.array([1.0, 0.0, 0.0]),
                    v_axis=np.array([0.0, 0.0, 1.0]),
                    half_extent_u=1.5,
                    half_extent_v=1.5,
                ),
            ]
        )
    return scene


def _build_scene_snapshot(
    scenario_id: str,
    params: dict[str, Any],
    *,
    basis: str,
    los_block_mode: str,
    a5_stress_mode: str,
    a5_scatterer_count: int,
) -> tuple[np.ndarray, np.ndarray, list[Plane]]:
    s = str(scenario_id).upper()
    tx, rx = default_antennas(basis=basis)
    scene: list[Plane] = []
    los_blocker = str(los_block_mode).lower().strip() == "occluder"

    if s == "C0":
        rx.position[:] = [float(params.get("distance_m", 1.0)), 0.0, 1.5]
        return np.asarray(tx.position, dtype=float), np.asarray(rx.position, dtype=float), scene

    if s == "A2":
        rx.position[:] = [float(params.get("distance_m", 6.0)), 0.0, 1.5]
        scene = A2_pec_plane.build_scene(y_plane=float(params.get("y_plane", 2.0)))
        if los_blocker:
            scene.append(make_los_blocker_plane(tx.position, rx.position, plane_id=9101))
        return np.asarray(tx.position, dtype=float), np.asarray(rx.position, dtype=float), scene

    if s == "A3":
        tx.position[:] = [0.0, 0.0, 1.5]
        rx.position[:] = [float(params.get("rx_x", 3.5)), float(params.get("rx_y", 4.5)), 1.5]
        scene = A3_corner_2bounce.build_scene(offset=float(params.get("offset", 3.5)))
        if los_blocker:
            scene.append(make_los_blocker_plane(tx.position, rx.position, plane_id=9201))
        return np.asarray(tx.position, dtype=float), np.asarray(rx.position, dtype=float), scene

    if s == "A4":
        rx.position[:] = [float(params.get("distance_m", 6.0)), 0.0, 1.5]
        scene = A4_dielectric_plane.build_scene(
            str(params.get("material", "wood")),
            y_plane=float(params.get("y_plane", 2.0)),
            y_late_offset=float(params.get("y_late_offset", 2.4)),
            include_late_panel=bool(params.get("include_late_panel", True)),
        )
        if los_blocker:
            scene.append(
                make_los_blocker_plane(
                    tx.position,
                    rx.position,
                    plane_id=9301,
                    half_extent_u=0.12,
                    half_extent_v=0.30,
                )
            )
        return np.asarray(tx.position, dtype=float), np.asarray(rx.position, dtype=float), scene

    if s == "A5":
        rx.position[:] = [float(params.get("rx_x", 2.5)), float(params.get("rx_y", 0.5)), 1.5]
        scene = A5_depol_stress.build_scene(offset=float(params.get("offset", 3.5)))
        mode = str(a5_stress_mode).lower().strip()
        if mode in {"geometry", "hybrid"} and int(a5_scatterer_count) > 0:
            append_fn = getattr(A5_depol_stress, "_append_stress_scatterers", None)
            if callable(append_fn):
                append_fn(
                    scene,
                    seed=int(params.get("seed", A5_depol_stress.BASE_SEED)),
                    count=int(a5_scatterer_count),
                    offset=float(params.get("offset", 3.5)),
                )
        if los_blocker:
            scene.append(
                make_los_blocker_plane(
                    tx.position,
                    rx.position,
                    plane_id=9401,
                    half_extent_u=0.12,
                    half_extent_v=0.30,
                )
            )
        return np.asarray(tx.position, dtype=float), np.asarray(rx.position, dtype=float), scene

    if s in {"B1", "B2", "B3"}:
        tx.position[:] = [2.0, 0.0, 1.5]
        rx.position[:] = [float(params.get("rx_x", 2.0)), float(params.get("rx_y", 0.0)), 1.5]
        scene = _build_room_scene(s)
        return np.asarray(tx.position, dtype=float), np.asarray(rx.position, dtype=float), scene

    return np.asarray(tx.position, dtype=float), np.asarray(rx.position, dtype=float), scene


def _export_scene_debug_for_case(
    case_rec: dict[str, Any],
    *,
    out_dir: Path,
    basis: str,
    los_block_mode: str,
    a5_stress_mode: str,
    a5_scatterer_count: int,
    debug_rays_k: int,
    scene_plane: str,
    git_hash: str,
    cmdline: str,
    seed: int,
) -> Path:
    scenario_id = str(case_rec.get("scenario_id", "NA"))
    case_id = str(case_rec.get("case_id", "NA"))
    params = dict(case_rec.get("params", {}))
    tx, rx, scene = _build_scene_snapshot(
        scenario_id,
        params,
        basis=basis,
        los_block_mode=los_block_mode,
        a5_stress_mode=a5_stress_mode,
        a5_scatterer_count=int(a5_scatterer_count),
    )
    rays_topk = _extract_rays_topk(list(case_rec.get("paths", [])), int(debug_rays_k))
    objects = _serialize_scene_objects(scene, scenario_id=scenario_id)
    bounds = _scene_bounds(tx=tx, rx=rx, objects=objects, rays=rays_topk)
    case_label = str(case_rec.get("case_label", _case_label(scenario_id, params)))
    scene_dict = {
        "scene_schema": "scene_debug_v1",
        "scenario_id": scenario_id,
        "case_id": case_id,
        "case_label": case_label,
        "coord_frame": {"units": "m", "plane": str(scene_plane), "z_up": True},
        "bounds": bounds,
        "tx": {"x": float(tx[0]), "y": float(tx[1]), "z": float(tx[2])},
        "rx": {"x": float(rx[0]), "y": float(rx[1]), "z": float(rx[2])},
        "objects": objects,
        "rays_topk": rays_topk,
        "meta": {
            "git_hash": str(git_hash),
            "cmd": str(cmdline),
            "seed": int(seed),
        },
    }
    fname = f"{_sanitize_token(scenario_id)}__{_sanitize_token(case_id)}__scene_debug.json"
    out_path = out_dir / fname
    out_path.write_text(json.dumps(scene_dict, indent=2), encoding="utf-8")
    return out_path


def _run_room_case(kind: str, rx_x: float, rx_y: float, f_hz: np.ndarray, basis: str = "circular") -> list[dict[str, Any]]:
    tx, rx = default_antennas(basis=basis)
    tx.position[:] = [2.0, 0.0, 1.5]
    rx.position[:] = [rx_x, rx_y, 1.5]
    scene = _build_room_scene(kind)
    paths = trace_paths(scene, tx, rx, f_hz, max_bounce=2, los_enabled=True)
    return paths_to_records(paths)


def _build_case_records(args: argparse.Namespace, f_hz: np.ndarray) -> list[dict[str, Any]]:
    s = str(args.scenario).upper()
    basis = str(args.basis)
    n_rep = max(1, int(args.n_rep))
    los_block_mode = str(args.los_block_mode).lower().strip()
    los_blocker = bool(los_block_mode == "occluder")
    out: list[dict[str, Any]] = []
    if s == "C0":
        dlist = _parse_float_list(args.dist_list, [1.0, 2.0, 3.0, 4.0, 5.0])
        yaw_list = _parse_float_list(args.yaw_list, [-10.0, 0.0, 10.0])
        pitch_list = _parse_float_list(args.pitch_list, [0.0])
        cid = 0
        for d in dlist:
            for yaw in yaw_list:
                for pit in pitch_list:
                    for rep_id in range(n_rep):
                        paths = C0_free_space.run_case({"distance_m": d, "yaw_deg": yaw, "pitch_deg": pit}, f_hz, basis=basis)
                        out.append(
                            {
                                "case_id": str(cid),
                                "scenario_id": "C0",
                                "link_id": f"C0_{cid}",
                                "params": {"distance_m": d, "yaw_deg": yaw, "pitch_deg": pit, "rep_id": int(rep_id)},
                                "paths": paths_to_records(paths),
                                "meta": {
                                    "d_m": d,
                                    "yaw_deg": yaw,
                                    "pitch_deg": pit,
                                    "rep_id": int(rep_id),
                                    "material_class": "free_space",
                                    "basis": basis,
                                },
                            }
                        )
                        cid += 1
        return out

    if s == "A2":
        dlist = _parse_float_list(args.dist_list, [4.0, 6.0, 8.0])
        ylist = _parse_float_list(args.a2_y_list, [1.5, 2.0, 2.5])
        cid = 0
        for d in dlist:
            for y in ylist:
                for rep_id in range(n_rep):
                    p = {"y_plane": float(y), "distance_m": float(d), "rep_id": int(rep_id)}
                    paths = A2_pec_plane.run_case(
                        p,
                        f_hz,
                        basis=basis,
                        los_blocker=bool(los_blocker),
                    )
                    out.append(
                        {
                            "case_id": str(cid),
                            "scenario_id": "A2",
                            "link_id": f"A2_{cid}",
                            "params": p,
                            "paths": paths_to_records(paths),
                            "meta": {
                                "d_m": float(d),
                                "material_class": "PEC",
                                "obstacle_flag": 1,
                                "rep_id": int(rep_id),
                                "los_block_method": "physical_occluder" if bool(los_blocker) else "synthetic_los_off",
                                "basis": basis,
                            },
                        }
                    )
                    cid += 1
        return out

    if s == "A3":
        base = A3_corner_2bounce.build_sweep_params()
        cid = 0
        for p0 in base:
            for rep_id in range(n_rep):
                p = dict(p0)
                p["rep_id"] = int(rep_id)
                paths = A3_corner_2bounce.run_case(
                    p,
                    f_hz,
                    basis=basis,
                    los_blocker=bool(los_blocker),
                )
                d = float(np.linalg.norm(np.asarray([p["rx_x"], p["rx_y"], 1.5]) - np.asarray([0.0, 0.0, 1.5])))
                out.append(
                    {
                        "case_id": str(cid),
                        "scenario_id": "A3",
                        "link_id": f"A3_{cid}",
                        "params": p,
                        "paths": paths_to_records(paths),
                        "meta": {
                            "d_m": d,
                            "material_class": "PEC",
                            "obstacle_flag": 1,
                            "rep_id": int(rep_id),
                            "los_block_method": "physical_occluder" if bool(los_blocker) else "synthetic_los_off",
                            "basis": basis,
                        },
                    }
                )
                cid += 1
        return out

    if s == "A4":
        mats = [m.strip() for m in str(args.material_list).split(",") if m.strip()] or list(DEFAULT_MATERIAL_SPECS.keys())
        yvals = _parse_float_list(args.a4_y_list, [1.5, 2.0, 2.5])
        dvals = _parse_float_list(args.dist_list, [6.0])
        include_late_panel = _parse_bool(args.a4_include_late_panel, default=True)
        late_offset_m = float(args.a4_late_offset_m)
        cid = 0
        for m in mats:
            for y in yvals:
                for d in dvals:
                    for rep_id in range(n_rep):
                        p = {
                            "material": m,
                            "y_plane": float(y),
                            "distance_m": float(d),
                            "rep_id": int(rep_id),
                            "include_late_panel": bool(include_late_panel),
                            "y_late_offset": late_offset_m,
                        }
                        paths = A4_dielectric_plane.run_case(
                            p,
                            f_hz,
                            basis=basis,
                            los_blocker=bool(los_blocker),
                            include_late_panel=bool(include_late_panel),
                            y_late_offset=late_offset_m,
                        )
                        out.append(
                            {
                                "case_id": str(cid),
                                "scenario_id": "A4",
                                "link_id": f"A4_{cid}",
                                "params": p,
                                "paths": paths_to_records(paths),
                                "meta": {
                                    "d_m": float(d),
                                    "material_class": m,
                                    "obstacle_flag": 1,
                                    "rep_id": int(rep_id),
                                    "los_block_method": "physical_occluder" if bool(los_blocker) else "synthetic_los_off",
                                    "basis": basis,
                                },
                            }
                        )
                        cid += 1
        return out

    if s == "A5":
        cid = 0
        params = A5_depol_stress.build_sweep_params()
        if int(args.a5_max_cases) > 0:
            params = params[: int(args.a5_max_cases)]
        stress_on = bool(args.stress_flag)
        stress_mode = str(args.a5_stress_mode) if stress_on else "none"
        for rep_outer in range(n_rep):
            for p0 in params:
                p = dict(p0)
                p["rep_outer"] = int(rep_outer)
                p["seed"] = int(p.get("seed", 0)) + int(rep_outer) * 10000
                paths = A5_depol_stress.run_case(
                    p,
                    f_hz,
                    basis=basis,
                    stress_mode=stress_mode,
                    scatterer_count=int(args.a5_scatterer_count) if stress_on else 0,
                    los_blocker=bool(los_blocker),
                )
                d = float(np.linalg.norm(np.asarray([p["rx_x"], p["rx_y"], 1.5]) - np.asarray([0.0, 0.0, 1.5])))
                out.append(
                    {
                        "case_id": str(cid),
                        "scenario_id": "A5",
                        "link_id": f"A5_{cid}",
                        "params": p,
                        "paths": paths_to_records(paths),
                        "meta": {
                            "d_m": d,
                            "material_class": "PEC",
                            "roughness_flag": int(stress_on),
                            "human_flag": int(stress_on),
                            "obstacle_flag": 1,
                            "stress_mode": str(stress_mode),
                            "los_block_method": "physical_occluder" if bool(los_blocker) else "synthetic_los_off",
                            "basis": basis,
                        },
                    }
                )
                cid += 1
        return out

    # B1/B2/B3 room-grid style
    kind = s
    x_vals = np.arange(float(args.grid_x_min), float(args.grid_x_max) + 1e-9, float(args.grid_step_m))
    y_vals = np.arange(float(args.grid_y_min), float(args.grid_y_max) + 1e-9, float(args.grid_step_m))
    cid = 0
    for x in x_vals:
        for y in y_vals:
            paths = _run_room_case(kind=kind, rx_x=float(x), rx_y=float(y), f_hz=f_hz, basis=basis)
            out.append(
                {
                    "case_id": str(cid),
                    "scenario_id": kind,
                    "link_id": f"{kind}_{cid}",
                    "params": {"rx_x": float(x), "rx_y": float(y)},
                    "paths": paths,
                    "meta": {
                        "d_m": float(np.linalg.norm(np.asarray([x, y, 1.5]) - np.asarray([2.0, 0.0, 1.5]))),
                        "material_class": "PEC",
                        "obstacle_flag": int(kind in {"B2", "B3"}),
                        "rx_x": float(x),
                        "rx_y": float(y),
                        "basis": basis,
                    },
                }
            )
            cid += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["C0", "A2", "A3", "A4", "A5", "B1", "B2", "B3"])
    parser.add_argument("--basis", type=str, default="circular", choices=["circular", "linear"])
    parser.add_argument("--out-h5", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs/standard")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nf", type=int, default=128)
    parser.add_argument("--f-center-hz", type=float, default=8e9)
    parser.add_argument("--delay-bin-ns", type=float, default=0.25)
    parser.add_argument("--Te-ns", type=float, default=3.0)
    parser.add_argument("--Tmax-ns", type=float, default=30.0)
    parser.add_argument("--xpr-model-config", type=str, default=None)
    parser.add_argument("--floor-model-config", type=str, default=None)
    parser.add_argument("--floor-reference-json", type=str, default=None)
    parser.add_argument("--floor-reference-out", type=str, default="")
    parser.add_argument("--claim-caution-mode", type=str, default="scaled", choices=["scaled", "half_width", "off"])
    parser.add_argument("--claim-caution-scale", type=float, default=1.0)
    parser.add_argument("--dist-list", type=str, default="")
    parser.add_argument("--yaw-list", type=str, default="-10,0,10")
    parser.add_argument("--pitch-list", type=str, default="0")
    parser.add_argument("--n-rep", type=int, default=5)
    parser.add_argument("--a2-y-list", type=str, default="1.5,2.0,2.5")
    parser.add_argument("--a4-y-list", type=str, default="1.5,2.0,2.5")
    parser.add_argument("--a4-include-late-panel", type=str, default="true")
    parser.add_argument("--a4-late-offset-m", type=float, default=2.4)
    parser.add_argument("--los-block-mode", type=str, default="occluder", choices=["synthetic", "occluder"])
    parser.add_argument("--a5-stress-mode", type=str, default="hybrid", choices=["none", "synthetic", "geometry", "hybrid"])
    parser.add_argument("--a5-scatterer-count", type=int, default=3)
    parser.add_argument("--a5-max-cases", type=int, default=0)
    parser.add_argument("--material-list", type=str, default="")
    parser.add_argument("--stress-flag", action="store_true")
    parser.add_argument("--strict-los-blocked", action="store_true")
    parser.add_argument("--max-links", type=int, default=0)
    parser.add_argument("--ds-reference", type=str, default="total", choices=["total", "co"])
    parser.add_argument("--el-proxy-mode", type=str, default="early_sum", choices=["early_sum", "dominant_early_ray"])
    parser.add_argument("--grid-x-min", type=float, default=2.0)
    parser.add_argument("--grid-x-max", type=float, default=8.0)
    parser.add_argument("--grid-y-min", type=float, default=-3.0)
    parser.add_argument("--grid-y-max", type=float, default=3.0)
    parser.add_argument("--grid-step-m", type=float, default=1.0)
    parser.add_argument("--room-target-los-n", type=int, default=0)
    parser.add_argument("--room-target-nlos-n", type=int, default=0)
    parser.add_argument("--export-scene-debug", type=str, default="true")
    parser.add_argument("--debug-rays-k", type=int, default=20)
    parser.add_argument("--scene-plane", type=str, default="xy")
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    commit, branch = _git_meta()
    run_id = str(args.run_id or f"{args.scenario}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    f_hz = uwb_frequency(nf=int(args.nf))
    xpr_cfg = _load_json(args.xpr_model_config)
    floor_cfg = _load_json(args.floor_model_config)
    floor_ref_in = _load_json(args.floor_reference_json)
    xpr_model = _make_xpr_model(str(args.scenario), xpr_cfg)
    floor_model = _make_floor_model(floor_cfg)

    case_records = _build_case_records(args, f_hz=f_hz)
    if int(args.max_links) > 0:
        case_records = case_records[: int(args.max_links)]

    scene_debug_files: list[str] = []
    scene_debug_dir = Path(args.out_dir) / "scene_debug"
    if _parse_bool(args.export_scene_debug, default=True):
        scene_debug_dir.mkdir(parents=True, exist_ok=True)
        cmdline = " ".join(shlex.quote(x) for x in sys.argv)
        for c in case_records:
            scene_path = _export_scene_debug_for_case(
                c,
                out_dir=scene_debug_dir,
                basis=str(args.basis),
                los_block_mode=str(args.los_block_mode),
                a5_stress_mode=str(args.a5_stress_mode) if bool(args.stress_flag) else "none",
                a5_scatterer_count=int(args.a5_scatterer_count) if bool(args.stress_flag) else 0,
                debug_rays_k=int(args.debug_rays_k),
                scene_plane=str(args.scene_plane),
                git_hash=commit,
                cmdline=cmdline,
                seed=int(args.seed),
            )
            scene_debug_files.append(str(scene_path))

    bundles: list[StandardOutputBundle] = []
    for c in case_records:
        scenario_id = str(c["scenario_id"])
        link_id = str(c["link_id"])
        case_id = str(c["case_id"])
        paths = list(c.get("paths", []))
        ray_rows = build_ray_table_from_rt(paths, matrix_source="A", include_material=True, include_angles=True)
        ray_rows = add_el_db(ray_rows, f_center_hz=float(args.f_center_hz), method="fspl")

        los_link = int(any(int(r.get("los_flag_ray", 0)) == 1 for r in ray_rows))
        if scenario_id in {"A2", "A3", "A4", "A5"} and bool(args.strict_los_blocked):
            if los_link == 1:
                raise SystemExit(f"LOS blocked check failed: scenario={scenario_id}, link_id={link_id}")

        dt = max(float(args.delay_bin_ns), 1e-6) * 1e-9
        delay_tau_s = np.arange(0.0, max(float(args.Tmax_ns), float(args.Te_ns)) * 1e-9 + dt, dt, dtype=float)
        synth_U = {
            "scenario_id": scenario_id,
            "basis": str(args.basis),
            "f_center_hz": float(args.f_center_hz),
            "yaw_deg": float(c.get("meta", {}).get("yaw_deg", 0.0)),
            "pitch_deg": float(c.get("meta", {}).get("pitch_deg", 0.0)),
            "roughness_flag": int(c.get("meta", {}).get("roughness_flag", 0)),
            "human_flag": int(c.get("meta", {}).get("human_flag", 0)),
            "material_class": str(c.get("meta", {}).get("material_class", "NA")),
            "EL_proxy_db": float(np.nanmedian(np.asarray([r.get("EL_db", np.nan) for r in ray_rows], dtype=float))),
        }
        pdp_obj = synthesize_dualcp_pdp(
            ray_rows,
            delay_tau_s,
            xpr_model=xpr_model,
            link_U=synth_U,
            rng=rng,
            include_xpd_tau=True,
            floor_model=floor_model if scenario_id == "C0" else None,
        )
        pdp = {
            "delay_tau_s": np.asarray(pdp_obj.delay_tau_s, dtype=float),
            "P_co": np.asarray(pdp_obj.P_co, dtype=float),
            "P_cross": np.asarray(pdp_obj.P_cross, dtype=float),
        }
        p_total = pdp["P_co"] + pdp["P_cross"]
        tau0 = estimate_tau0(
            pdp["delay_tau_s"],
            p_total,
            method="threshold",
            noise_tail_s=8e-9,
            margin_db=6.0,
        )
        early_mask, late_mask = make_early_late_masks(
            pdp["delay_tau_s"],
            tau0_s=float(tau0["tau0_s"]),
            Te_s=float(args.Te_ns) * 1e-9,
            Tmax_s=float(args.Tmax_ns) * 1e-9,
        )
        metrics = compute_link_metrics(
            pdp=pdp,
            delay_tau_s=pdp["delay_tau_s"],
            masks=(early_mask, late_mask),
            ds_reference=str(args.ds_reference),
            window_params={
                "tau0_s": float(tau0["tau0_s"]),
                "Te_s": float(args.Te_ns) * 1e-9,
                "Tmax_s": float(args.Tmax_ns) * 1e-9,
                "tau0_method": str(tau0["method"]),
                "noise_floor_def": "tail_median+margin",
            },
        )
        el_proxy_db = compute_el_proxy(
            ray_rows,
            pdp=pdp,
            mode=str(args.el_proxy_mode),
            early_mask=early_mask,
            L_ref_mode="los|minL",
            f_center_hz=float(args.f_center_hz),
        )
        link_meta = dict(c.get("meta", {}))
        link_meta["LOSflag"] = int(los_link)
        link_meta["scenario_id"] = scenario_id
        link_meta["case_id"] = case_id
        link_meta["delay_tau_s"] = pdp["delay_tau_s"].tolist()
        U = build_link_U_from_scenario(
            link_meta,
            ray_rows=ray_rows,
            pdp=pdp,
            masks=(early_mask, late_mask),
            el_proxy_db=float(el_proxy_db),
        )

        bundle = StandardOutputBundle(
            link_id=link_id,
            scenario_id=scenario_id,
            case_id=case_id,
            rays=RayTable(rows=ray_rows),
            pdp=pdp_obj,
            metrics=metrics,
            conditions=U,
            provenance={
                "git_commit": commit,
                "git_branch": branch,
                "seed": int(args.seed),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "scenario_id": scenario_id,
                "case_id": case_id,
                "link_params": c.get("params", {}),
                "cmdline": " ".join(shlex.quote(x) for x in sys.argv),
            },
        )
        bundles.append(bundle)

    floor_ref_use: dict[str, Any] | None = None
    if floor_ref_in:
        floor_ref_use = dict(floor_ref_in)
    elif str(args.scenario).upper() == "C0":
        floor_ref_use = _build_floor_reference_with_curve(
            bundles,
            floor_model=floor_model,
            freq_hz=f_hz,
            bin_keys=["yaw_deg", "pitch_deg"],
            num_subbands=4,
        )
        out_floor = Path(args.floor_reference_out) if str(args.floor_reference_out).strip() else (Path(args.out_dir) / "floor_reference.json")
        out_floor.parent.mkdir(parents=True, exist_ok=True)
        out_floor.write_text(json.dumps(floor_ref_use, indent=2), encoding="utf-8")

    if floor_ref_use is not None:
        f_curve_hz, f_curve_db, f_curve_unc = _floor_curve_from_reference(floor_ref_use)
        sb_rows, sb_floor, sb_unc = _floor_subbands_from_reference(floor_ref_use)
        for b in bundles:
            u_dict = b.conditions.to_dict()
            floor_db, delta = _lookup_floor(floor_ref_use, u_dict)
            if np.isfinite(floor_db):
                ex_e = float(b.metrics.XPD_early_db - floor_db)
                ex_l = float(b.metrics.XPD_late_db - floor_db)
                caution_e = _claim_caution_flag(ex_e, float(delta), mode=str(args.claim_caution_mode), scale=float(args.claim_caution_scale))
                caution_l = _claim_caution_flag(ex_l, float(delta), mode=str(args.claim_caution_mode), scale=float(args.claim_caution_scale))
                b.metrics.extras["xpd_floor_db"] = float(floor_db)
                b.metrics.extras["delta_floor_db"] = float(delta) if np.isfinite(delta) else np.nan
                b.metrics.extras["XPD_early_excess_db"] = ex_e
                b.metrics.extras["XPD_late_excess_db"] = ex_l
                b.metrics.extras["claim_caution_early"] = caution_e
                b.metrics.extras["claim_caution_late"] = caution_l
                b.metrics.extras["claim_caution_mode"] = str(args.claim_caution_mode)
                b.metrics.extras["claim_caution_scale"] = float(args.claim_caution_scale)

            if len(f_curve_db) > 0:
                # NOTE: This is scalar-link metric minus floor curve (not true per-frequency XPD excess).
                # We keep it for calibration diagnostics with explicit naming.
                minus_e_curve = np.asarray(b.metrics.XPD_early_db - f_curve_db, dtype=float)
                minus_l_curve = np.asarray(b.metrics.XPD_late_db - f_curve_db, dtype=float)
                c_e_curve = [
                    _claim_caution_flag(float(xx), float(uu), mode=str(args.claim_caution_mode), scale=float(args.claim_caution_scale))
                    for xx, uu in zip(minus_e_curve.tolist(), f_curve_unc.tolist())
                ]
                c_l_curve = [
                    _claim_caution_flag(float(xx), float(uu), mode=str(args.claim_caution_mode), scale=float(args.claim_caution_scale))
                    for xx, uu in zip(minus_l_curve.tolist(), f_curve_unc.tolist())
                ]
                b.metrics.extras["xpd_floor_freq_hz"] = np.asarray(f_curve_hz, dtype=float).tolist()
                b.metrics.extras["xpd_floor_curve_db"] = np.asarray(f_curve_db, dtype=float).tolist()
                b.metrics.extras["xpd_floor_uncert_curve_db"] = np.asarray(f_curve_unc, dtype=float).tolist()
                b.metrics.extras["XPD_early_minus_floor_curve_db"] = minus_e_curve.tolist()
                b.metrics.extras["XPD_late_minus_floor_curve_db"] = minus_l_curve.tolist()
                b.metrics.extras["claim_caution_early_curve"] = c_e_curve
                b.metrics.extras["claim_caution_late_curve"] = c_l_curve

            if len(sb_floor) > 0:
                minus_e_sb = np.asarray(b.metrics.XPD_early_db - sb_floor, dtype=float)
                minus_l_sb = np.asarray(b.metrics.XPD_late_db - sb_floor, dtype=float)
                c_e_sb = [
                    _claim_caution_flag(float(xx), float(uu), mode=str(args.claim_caution_mode), scale=float(args.claim_caution_scale))
                    for xx, uu in zip(minus_e_sb.tolist(), sb_unc.tolist())
                ]
                c_l_sb = [
                    _claim_caution_flag(float(xx), float(uu), mode=str(args.claim_caution_mode), scale=float(args.claim_caution_scale))
                    for xx, uu in zip(minus_l_sb.tolist(), sb_unc.tolist())
                ]
                b.metrics.extras["xpd_floor_subbands"] = sb_rows
                b.metrics.extras["xpd_floor_subband_db"] = sb_floor.tolist()
                b.metrics.extras["xpd_floor_uncert_subband_db"] = sb_unc.tolist()
                b.metrics.extras["XPD_early_minus_floor_subband_db"] = minus_e_sb.tolist()
                b.metrics.extras["XPD_late_minus_floor_subband_db"] = minus_l_sb.tolist()
                b.metrics.extras["claim_caution_early_subband"] = c_e_sb
                b.metrics.extras["claim_caution_late_subband"] = c_l_sb

    warnings: list[str] = []
    if str(args.scenario).upper().startswith("B"):
        los_n = int(sum(int(b.conditions.LOSflag) == 1 for b in bundles))
        nlos_n = int(sum(int(b.conditions.LOSflag) == 0 for b in bundles))
        t_los = int(max(0, args.room_target_los_n))
        t_nlos = int(max(0, args.room_target_nlos_n))
        if t_los > 0 and los_n < t_los:
            warnings.append(f"room_target_los_n unmet: {los_n} < {t_los}")
        if t_nlos > 0 and nlos_n < t_nlos:
            warnings.append(f"room_target_nlos_n unmet: {nlos_n} < {t_nlos}")

    run_meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "scenario_id": str(args.scenario),
        "basis": str(args.basis),
        "seed": int(args.seed),
        "n_rep": int(max(1, args.n_rep)),
        "nf": int(args.nf),
        "f_center_hz": float(args.f_center_hz),
        "Te_s": float(args.Te_ns) * 1e-9,
        "Tmax_s": float(args.Tmax_ns) * 1e-9,
        "delay_bin_s": float(args.delay_bin_ns) * 1e-9,
        "cmdline": " ".join(shlex.quote(x) for x in sys.argv),
        "git_commit": commit,
        "git_branch": branch,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "warnings": list(warnings),
    }
    save_run(run_meta, bundles, out_h5=args.out_h5, run_id=run_id)
    csv_out = export_csv(bundles, out_dir=args.out_dir)
    summary = {
        "run_id": run_id,
        "out_h5": str(args.out_h5),
        "out_dir": str(args.out_dir),
        "basis": str(args.basis),
        "n_links": len(bundles),
        "link_metrics_csv": csv_out["link_metrics_csv"],
        "rays_csv": csv_out["rays_csv"],
        "warnings": list(warnings),
        "floor_reference_json": str(args.floor_reference_json or (Path(args.out_dir) / "floor_reference.json"))
        if str(args.scenario).upper() == "C0" or args.floor_reference_json
        else "",
        "scene_debug_dir": str(scene_debug_dir) if _parse_bool(args.export_scene_debug, default=True) else "",
        "scene_debug_files": scene_debug_files,
    }
    out_summary = Path(args.out_dir) / "run_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(args.out_h5))
    print(csv_out["link_metrics_csv"])
    print(csv_out["rays_csv"])
    print(str(out_summary))


if __name__ == "__main__":
    main()

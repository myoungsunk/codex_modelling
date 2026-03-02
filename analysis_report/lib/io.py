"""I/O helpers for analysis reports over standard outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rt_io.standard_outputs_hdf5 import load_run


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    try:
        cfg = json.loads(txt)
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        pass
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(txt)
        if isinstance(cfg, dict):
            return cfg
    except Exception as exc:
        raise ValueError(
            "Config parse failed. Use JSON-compatible YAML, or install pyyaml. "
            f"path={p}"
        ) from exc
    raise ValueError(f"Config must be a mapping: {p}")


def _resolve_path(base_dir: Path, maybe_path: str | None) -> Path | None:
    if maybe_path is None:
        return None
    s = str(maybe_path).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    q = (base_dir / p)
    if q.exists():
        return q.resolve()
    return q.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _scenario_from_name(name: str) -> str:
    up = str(name).upper()
    for s in ["C0", "A2", "A3", "A4", "A5", "B1", "B2", "B3"]:
        if s in up:
            return s
    return "NA"


def discover_runs(input_paths: list[str], scenario_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in input_paths:
        root = Path(str(raw)).expanduser().resolve()
        candidates: list[Path] = []
        if root.is_file() and root.name == "run_summary.json":
            candidates = [root]
        elif root.is_dir() and (root / "run_summary.json").exists():
            candidates = [root / "run_summary.json"]
        elif root.is_dir():
            candidates = sorted(root.rglob("run_summary.json"))
        for c in candidates:
            key = str(c)
            if key in seen:
                continue
            seen.add(key)
            run_dir = c.parent
            summary = _read_json(c)
            run_id = str(summary.get("run_id", run_dir.name))
            scenario_id = str(summary.get("scenario_id", ""))
            if not scenario_id:
                scenario_id = _scenario_from_name(run_id)
            if scenario_map:
                for k, v in scenario_map.items():
                    if str(k) in run_id or str(k) in run_dir.name:
                        scenario_id = str(v)
                        break
            link_metrics_csv = _resolve_path(run_dir, str(summary.get("link_metrics_csv", "")))
            rays_csv = _resolve_path(run_dir, str(summary.get("rays_csv", "")))
            out_h5 = _resolve_path(run_dir, str(summary.get("out_h5", "")))
            scene_debug_dir = _resolve_path(run_dir, str(summary.get("scene_debug_dir", "")))
            runs.append(
                {
                    "run_dir": run_dir,
                    "run_summary_path": c,
                    "summary": summary,
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "link_metrics_csv": link_metrics_csv,
                    "rays_csv": rays_csv,
                    "out_h5": out_h5,
                    "scene_debug_dir": scene_debug_dir,
                }
            )
    runs.sort(key=lambda r: (str(r.get("scenario_id", "")), str(r.get("run_id", ""))))
    return runs


def _parse_jsonish(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s:
        return v
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            return v
    return v


def _to_float(v: Any) -> float:
    try:
        if v is None or v == "":
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


_LINK_ALIASES = {
    "xpd_early_db": "XPD_early_db",
    "xpd_late_db": "XPD_late_db",
    "l_pol_db": "L_pol_db",
    "rho_early_lin": "rho_early_lin",
    "rho_early_db": "rho_early_db",
    "delay_spread_rms_s": "delay_spread_rms_s",
    "ds_rms_s": "delay_spread_rms_s",
    "early_energy_fraction": "early_energy_fraction",
    "el_proxy_db": "EL_proxy_db",
    "losflag": "LOSflag",
    "material": "material_class",
}


_NUMERIC_KEYS = {
    "XPD_early_db",
    "XPD_late_db",
    "XPD_early_excess_db",
    "XPD_late_excess_db",
    "L_pol_db",
    "rho_early_lin",
    "rho_early_db",
    "delay_spread_rms_s",
    "early_energy_fraction",
    "d_m",
    "EL_proxy_db",
    "LOSflag",
    "roughness_flag",
    "human_flag",
    "obstacle_flag",
    "yaw_deg",
    "pitch_deg",
}


def _normalize_link_row(row: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    # alias lift
    for k, dst in _LINK_ALIASES.items():
        if (dst not in out) and (k in out):
            out[dst] = out[k]
    for k in list(out.keys()):
        out[k] = _parse_jsonish(out[k])
    for k in _NUMERIC_KEYS:
        if k in out:
            out[k] = _to_float(out.get(k))
    out.setdefault("scenario_id", run.get("scenario_id", "NA"))
    out.setdefault("run_id", run.get("run_id", run.get("run_dir", "")))
    out.setdefault("case_id", str(out.get("case_id", "")))
    out.setdefault("link_id", str(out.get("link_id", "")))
    out.setdefault("case_label", str(out.get("case_label", out.get("link_id", out.get("case_id", "")))))
    return out


def _normalize_ray_row(row: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for k in ["tau_s", "L_m", "P_lin", "EL_db", "incidence_deg", "n_bounce", "los_flag_ray", "ray_index"]:
        if k in out:
            out[k] = _to_float(out[k])
    out.setdefault("scenario_id", run.get("scenario_id", "NA"))
    out.setdefault("run_id", run.get("run_id", ""))
    return out


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _load_from_h5(run: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_h5 = run.get("out_h5")
    if out_h5 is None or not Path(out_h5).exists():
        return [], []
    run_id = str(run.get("run_id", ""))
    if not run_id:
        return [], []
    payload = load_run(out_h5, run_id=run_id)
    metrics_rows: list[dict[str, Any]] = []
    ray_rows: list[dict[str, Any]] = []
    for b in payload.get("bundles", []):
        m = dict(b.get("metrics", {}))
        u = dict(b.get("conditions", {}))
        m.update(u)
        m["scenario_id"] = str(b.get("scenario_id", run.get("scenario_id", "NA")))
        m["case_id"] = str(b.get("case_id", ""))
        m["link_id"] = str(b.get("link_id", ""))
        metrics_rows.append(_normalize_link_row(m, run=run))
        for rr in b.get("rays", {}).get("rows", []):
            rec = dict(rr)
            rec["scenario_id"] = m["scenario_id"]
            rec["case_id"] = m["case_id"]
            rec["link_id"] = m["link_id"]
            ray_rows.append(_normalize_ray_row(rec, run=run))
    return metrics_rows, ray_rows


def load_link_metrics(run: dict[str, Any]) -> list[dict[str, Any]]:
    p = run.get("link_metrics_csv")
    if p is not None and Path(p).exists():
        rows = _read_csv_rows(Path(p))
        return [_normalize_link_row(r, run=run) for r in rows]
    rows, _ = _load_from_h5(run)
    return rows


def load_rays(run: dict[str, Any]) -> list[dict[str, Any]]:
    p = run.get("rays_csv")
    if p is not None and Path(p).exists():
        rows = _read_csv_rows(Path(p))
        return [_normalize_ray_row(r, run=run) for r in rows]
    _, rows = _load_from_h5(run)
    return rows


def load_scene_debug_paths(run: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
    summary = dict(run.get("summary", {}))
    files = summary.get("scene_debug_files", [])
    if isinstance(files, list):
        for x in files:
            p = _resolve_path(Path(run["run_dir"]), str(x))
            if p is not None and p.exists():
                out.append(p)
    if out:
        return sorted(out)
    sd = run.get("scene_debug_dir")
    if sd is not None and Path(sd).exists():
        out = sorted(Path(sd).glob("*__scene_debug.json"))
    return out


def load_scene_debug_map(run: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for p in load_scene_debug_paths(run):
        try:
            obj = _read_json(p)
        except Exception:
            continue
        scenario_id = str(obj.get("scenario_id", run.get("scenario_id", "NA")))
        case_id = str(obj.get("case_id", ""))
        obj["_path"] = str(p)
        out[(scenario_id, case_id)] = obj
    return out


def load_pdp_npz(run: dict[str, Any], link_id: str) -> dict[str, np.ndarray] | None:
    run_dir = Path(run.get("run_dir", "."))
    p = run_dir / f"pdp_{link_id}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {
        "delay_tau_s": np.asarray(d.get("delay_tau_s", []), dtype=float),
        "P_co": np.asarray(d.get("P_co", []), dtype=float),
        "P_cross": np.asarray(d.get("P_cross", []), dtype=float),
        "XPD_tau_db": np.asarray(d.get("XPD_tau_db", []), dtype=float),
    }


def unique_scenarios(rows: list[dict[str, Any]]) -> list[str]:
    vals = sorted({str(r.get("scenario_id", "NA")) for r in rows})
    return vals


def rows_by_scenario(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        s = str(r.get("scenario_id", "NA"))
        out.setdefault(s, []).append(r)
    return out


def case_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("scenario_id", "NA")), str(row.get("case_id", ""))


def collect_all(config: dict[str, Any]) -> dict[str, Any]:
    scenario_map = dict(config.get("scenario_map", {})) if isinstance(config.get("scenario_map", {}), dict) else {}
    input_paths = [str(x) for x in config.get("input_runs", [])]
    runs = discover_runs(input_paths, scenario_map=scenario_map)
    link_rows: list[dict[str, Any]] = []
    ray_rows: list[dict[str, Any]] = []
    scene_map: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        lrows = load_link_metrics(run)
        rrows = load_rays(run)
        link_rows.extend(lrows)
        ray_rows.extend(rrows)
        sdm = load_scene_debug_map(run)
        for k, v in sdm.items():
            scene_map[k] = v
    return {
        "runs": runs,
        "link_rows": link_rows,
        "ray_rows": ray_rows,
        "scene_map": scene_map,
    }

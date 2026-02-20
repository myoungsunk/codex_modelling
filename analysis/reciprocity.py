"""Reciprocity sanity checks based on path invariants.

This module compares forward and swapped Tx/Rx traces using robust invariants:
- delay tau
- singular values of 2x2 path matrix over frequency

Matching is performed by reversed point sequence + bounce_count + surface pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rt_core.antenna import Antenna
from rt_core.geometry import normalize
from rt_core.tracer import PathResult, trace_paths


EPS = 1e-15


@dataclass
class ReciprocityMatch:
    path_index_forward: int
    path_index_reverse: int
    bounce_count: int
    surface_pattern: str
    tau_f_s: float
    tau_r_s: float
    delta_tau_s: float
    delta_sigma_max_db: float
    delta_sigma_median_db: float
    match_method: str


def _projected_hv(boresight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = normalize(np.asarray(boresight, dtype=float))
    href = np.array([0.0, 1.0, 0.0], dtype=float)
    h = href - float(np.dot(href, b)) * b
    if np.linalg.norm(h) < 1e-9:
        href2 = np.array([0.0, 0.0, 1.0], dtype=float)
        h = href2 - float(np.dot(href2, b)) * b
    h = normalize(h)
    v = normalize(np.cross(b, h))
    return h, v


def _swapped_antennas(tx: Antenna, rx: Antenna) -> tuple[Antenna, Antenna]:
    """Create reversed-link antennas (positions swapped, boresight reset to face each other)."""

    p_tx = np.asarray(tx.position, dtype=float)
    p_rx = np.asarray(rx.position, dtype=float)

    b_tx_rev = normalize(p_tx - p_rx)  # reverse transmitter at old RX points toward old TX
    h_tx_rev, v_tx_rev = _projected_hv(b_tx_rev)
    tx_rev = Antenna(
        position=p_rx.copy(),
        boresight=b_tx_rev,
        h_axis=h_tx_rev,
        v_axis=v_tx_rev,
        basis=rx.basis,
        convention=rx.convention,
        cross_pol_leakage_db=rx.cross_pol_leakage_db,
        axial_ratio_db=rx.axial_ratio_db,
        enable_coupling=rx.enable_coupling,
        global_up=np.asarray(rx.global_up, dtype=float),
    )

    b_rx_rev = normalize(p_rx - p_tx)  # reverse receiver at old TX points toward old RX
    h_rx_rev, v_rx_rev = _projected_hv(b_rx_rev)
    rx_rev = Antenna(
        position=p_tx.copy(),
        boresight=b_rx_rev,
        h_axis=h_rx_rev,
        v_axis=v_rx_rev,
        basis=tx.basis,
        convention=tx.convention,
        cross_pol_leakage_db=tx.cross_pol_leakage_db,
        axial_ratio_db=tx.axial_ratio_db,
        enable_coupling=tx.enable_coupling,
        global_up=np.asarray(tx.global_up, dtype=float),
    )
    return tx_rev, rx_rev


def _surface_pattern(surface_ids: list[int]) -> str:
    return "-".join(str(int(x)) for x in surface_ids) if surface_ids else "none"


def _path_key(path: PathResult, reverse_order: bool = False) -> tuple[int, str, tuple[tuple[float, float, float], ...]]:
    pts = [np.asarray(p, dtype=float) for p in path.points]
    sids = list(path.surface_ids)
    if reverse_order:
        pts = list(reversed(pts))
        sids = list(reversed(sids))
    interior = tuple(tuple(np.round(p, 6).tolist()) for p in pts[1:-1])
    return int(path.bounce_count), _surface_pattern(sids), interior


def _point_mismatch_forward_reverse(pf: PathResult, pr: PathResult) -> float:
    """Distance between forward interior points and reversed reverse interior points."""

    af = [np.asarray(p, dtype=float) for p in pf.points[1:-1]]
    br = [np.asarray(p, dtype=float) for p in reversed(pr.points[1:-1])]
    if len(af) != len(br):
        return float("inf")
    if len(af) == 0:
        return 0.0
    d = [float(np.linalg.norm(a - b)) for a, b in zip(af, br)]
    return float(np.mean(np.asarray(d, dtype=float)))


def _matrix_f(path: PathResult, matrix_source: str) -> np.ndarray:
    if str(matrix_source).upper() == "J":
        return np.asarray(path.J_f, dtype=np.complex128)
    return np.asarray(path.A_f, dtype=np.complex128)


def _sv_delta_db(a_f: np.ndarray, b_f: np.ndarray) -> tuple[float, float]:
    dvals = []
    n = min(len(a_f), len(b_f))
    for k in range(n):
        sa = np.linalg.svd(a_f[k], compute_uv=False)
        sb = np.linalg.svd(b_f[k], compute_uv=False)
        # compare sorted singular values (largest->smallest)
        m = min(len(sa), len(sb))
        for i in range(m):
            da = 20.0 * np.log10((float(sa[i]) + EPS) / (float(sb[i]) + EPS))
            dvals.append(abs(float(da)))
    if not dvals:
        return np.nan, np.nan
    arr = np.asarray(dvals, dtype=float)
    return float(np.max(arr)), float(np.median(arr))


def reciprocity_sanity(
    scene: list,
    tx: Antenna,
    rx: Antenna,
    f_hz: np.ndarray,
    max_bounce: int = 2,
    los_enabled: bool = True,
    use_fspl: bool = True,
    force_cp_swap_on_odd_reflection: bool = False,
    matrix_source: str = "J",
    tau_tol_s: float = 1e-12,
    sigma_tol_db: float = 1e-6,
) -> dict[str, Any]:
    """Run forward/reverse traces and compare path invariants."""

    fwd = trace_paths(
        scene,
        tx,
        rx,
        np.asarray(f_hz, dtype=float),
        max_bounce=max_bounce,
        los_enabled=los_enabled,
        use_fspl=use_fspl,
        force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
    )

    tx_rev, rx_rev = _swapped_antennas(tx, rx)
    rev = trace_paths(
        scene,
        tx_rev,
        rx_rev,
        np.asarray(f_hz, dtype=float),
        max_bounce=max_bounce,
        los_enabled=los_enabled,
        use_fspl=use_fspl,
        force_cp_swap_on_odd_reflection=force_cp_swap_on_odd_reflection,
    )

    rev_map: dict[tuple[int, str, tuple[tuple[float, float, float], ...]], list[tuple[int, PathResult]]] = {}
    for i_rev, p in enumerate(rev):
        rev_map.setdefault(_path_key(p, reverse_order=True), []).append((i_rev, p))
    rev_items: list[tuple[int, PathResult, str]] = []
    for i_rev, p in enumerate(rev):
        rev_items.append((i_rev, p, _surface_pattern(list(reversed(p.surface_ids)))))

    matches: list[ReciprocityMatch] = []
    unmatched_forward: list[dict[str, Any]] = []
    used_rev_ids: set[int] = set()

    for i_fwd, pf in enumerate(fwd):
        key = _path_key(pf, reverse_order=False)
        cand = rev_map.get(key, [])
        match_method = "exact_key"
        if not cand:
            # Fallback 1: same bounce + same reversed surface pattern.
            pat = _surface_pattern(pf.surface_ids)
            bcnt = int(pf.bounce_count)
            cand = [(i, pr) for i, pr, rp in rev_items if int(pr.bounce_count) == bcnt and rp == pat]
            match_method = "fallback_surface"
        if not cand:
            # Fallback 2: same bounce only (best tau + point mismatch).
            bcnt = int(pf.bounce_count)
            cand = [(i, pr) for i, pr, _ in rev_items if int(pr.bounce_count) == bcnt]
            match_method = "fallback_bounce"
        if not cand:
            unmatched_forward.append(
                {
                    "path_index_forward": int(i_fwd),
                    "bounce_count": int(pf.bounce_count),
                    "surface_pattern": _surface_pattern(pf.surface_ids),
                    "tau_s": float(pf.tau_s),
                }
            )
            continue
        # choose best tau match among unused candidates
        best = None
        best_idx = -1
        best_dt = np.inf
        best_pm = np.inf
        best_rev_idx = -1
        for i, (idx_rev, pr) in enumerate(cand):
            if id(pr) in used_rev_ids:
                continue
            dt = abs(float(pf.tau_s) - float(pr.tau_s))
            pm = _point_mismatch_forward_reverse(pf, pr)
            if (dt < best_dt) or (np.isclose(dt, best_dt) and pm < best_pm):
                best_dt = dt
                best_pm = pm
                best = pr
                best_idx = i
                best_rev_idx = int(idx_rev)
        if best is None:
            unmatched_forward.append(
                {
                    "path_index_forward": int(i_fwd),
                    "bounce_count": int(pf.bounce_count),
                    "surface_pattern": _surface_pattern(pf.surface_ids),
                    "tau_s": float(pf.tau_s),
                }
            )
            continue
        used_rev_ids.add(id(best))
        dmax, dmed = _sv_delta_db(_matrix_f(pf, matrix_source), _matrix_f(best, matrix_source))
        matches.append(
            ReciprocityMatch(
                path_index_forward=int(i_fwd),
                path_index_reverse=int(best_rev_idx),
                bounce_count=int(pf.bounce_count),
                surface_pattern=_surface_pattern(pf.surface_ids),
                tau_f_s=float(pf.tau_s),
                tau_r_s=float(best.tau_s),
                delta_tau_s=float(abs(float(pf.tau_s) - float(best.tau_s))),
                delta_sigma_max_db=float(dmax),
                delta_sigma_median_db=float(dmed),
                match_method=str(match_method),
            )
        )

    matched = len(matches)
    n_fwd = len(fwd)
    matched_ratio = float(matched / n_fwd) if n_fwd > 0 else 0.0
    dtau_max = float(max((m.delta_tau_s for m in matches), default=np.nan))
    dsig_max = float(max((m.delta_sigma_max_db for m in matches), default=np.nan))
    dsig_med = float(np.nanmedian([m.delta_sigma_median_db for m in matches])) if matches else float("nan")
    tau_mismatch_count = int(sum(1 for m in matches if np.isfinite(m.delta_tau_s) and m.delta_tau_s > float(tau_tol_s)))
    matrix_mismatch_count = int(
        sum(1 for m in matches if np.isfinite(m.delta_sigma_max_db) and m.delta_sigma_max_db > float(sigma_tol_db))
    )

    violations: list[dict[str, Any]] = []
    for u in unmatched_forward:
        violations.append(
            {
                "type": "unmatched_geometry",
                "path_index_forward": int(u.get("path_index_forward", -1)),
                "path_index_reverse": -1,
                "bounce_count": int(u.get("bounce_count", 0)),
                "surface_pattern": str(u.get("surface_pattern", "none")),
                "delta_tau_s": np.nan,
                "delta_sigma_max_db": np.nan,
            }
        )
    for m in matches:
        if np.isfinite(m.delta_tau_s) and m.delta_tau_s > float(tau_tol_s):
            violations.append(
                {
                    "type": "tau_mismatch",
                    "path_index_forward": int(m.path_index_forward),
                    "path_index_reverse": int(m.path_index_reverse),
                    "bounce_count": int(m.bounce_count),
                    "surface_pattern": str(m.surface_pattern),
                    "delta_tau_s": float(m.delta_tau_s),
                    "delta_sigma_max_db": float(m.delta_sigma_max_db),
                }
            )
        if np.isfinite(m.delta_sigma_max_db) and m.delta_sigma_max_db > float(sigma_tol_db):
            violations.append(
                {
                    "type": "matrix_invariant_mismatch",
                    "path_index_forward": int(m.path_index_forward),
                    "path_index_reverse": int(m.path_index_reverse),
                    "bounce_count": int(m.bounce_count),
                    "surface_pattern": str(m.surface_pattern),
                    "delta_tau_s": float(m.delta_tau_s),
                    "delta_sigma_max_db": float(m.delta_sigma_max_db),
                }
            )

    return {
        "n_forward": int(n_fwd),
        "n_reverse": int(len(rev)),
        "n_matched": int(matched),
        "matched_ratio": matched_ratio,
        "delta_tau_max_s": dtau_max,
        "delta_sigma_max_db": dsig_max,
        "delta_sigma_median_db": dsig_med,
        "unmatched_count": int(len(unmatched_forward)),
        "tau_mismatch_count": int(tau_mismatch_count),
        "matrix_mismatch_count": int(matrix_mismatch_count),
        "tau_tol_s": float(tau_tol_s),
        "sigma_tol_db": float(sigma_tol_db),
        "matrix_source": str(matrix_source),
        "matches": [
            {
                "path_index_forward": m.path_index_forward,
                "path_index_reverse": m.path_index_reverse,
                "bounce_count": m.bounce_count,
                "surface_pattern": m.surface_pattern,
                "tau_f_s": m.tau_f_s,
                "tau_r_s": m.tau_r_s,
                "delta_tau_s": m.delta_tau_s,
                "delta_sigma_max_db": m.delta_sigma_max_db,
                "delta_sigma_median_db": m.delta_sigma_median_db,
                "match_method": m.match_method,
            }
            for m in matches
        ],
        "unmatched_forward": unmatched_forward,
        "violations": violations,
    }

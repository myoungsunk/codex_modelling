"""Automated plotting suite P0~P21 with interpretation-safe labeling and selection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.ctf_cir import ctf_to_cir, pdp
from analysis.xpd_stats import gof_model_selection_db, make_subbands, model_quantiles_db, pathwise_xpd
from plots.plot_config import PlotConfig
from rt_core.polarization import fresnel_reflection
from scenarios.A4_dielectric_plane import MATERIALS


def _ensure_dir(out_dir: str | Path) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=180)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def _write_skip(out: Path, name: str, reason: str) -> None:
    (out / f"{name}.SKIPPED.txt").write_text(reason + "\n", encoding="utf-8")


def _matrix_f(path: dict[str, Any], matrix_source: str) -> np.ndarray:
    if matrix_source == "J" and "J_f" in path:
        return np.asarray(path["J_f"], dtype=np.complex128)
    return np.asarray(path["A_f"], dtype=np.complex128)


def _path_power(path: dict[str, Any], matrix_source: str) -> float:
    M = _matrix_f(path, matrix_source)
    return float(np.mean(np.abs(M) ** 2))


def _all_cases(data: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    out = []
    for sid, sc in data["scenarios"].items():
        for cid, case in sc["cases"].items():
            out.append((sid, cid, case))
    return out


def _scope_cases(data: dict[str, Any], config: PlotConfig) -> list[tuple[str, str, dict[str, Any]]]:
    if config.scenario_id is not None:
        if config.scenario_id not in data["scenarios"]:
            raise ValueError(f"scenario_id not found: {config.scenario_id}")
        sc = data["scenarios"][config.scenario_id]["cases"]
        if config.case_id is not None:
            if config.case_id not in sc:
                raise ValueError(f"case_id not found in scenario {config.scenario_id}: {config.case_id}")
            return [(config.scenario_id, config.case_id, sc[config.case_id])]
        return [(config.scenario_id, cid, case) for cid, case in sc.items()]

    # scenario not specified
    if config.case_id is not None:
        out = []
        for sid, cid, case in _all_cases(data):
            if cid == config.case_id:
                out.append((sid, cid, case))
        if out:
            return out
    return _all_cases(data)


def _select_case(data: dict[str, Any], config: PlotConfig) -> tuple[str, str, dict[str, Any]]:
    scope = _scope_cases(data, config)

    # Priority 1: explicit scenario/case already handled in _scope_cases.
    if len(scope) == 1 and config.scenario_id is not None and config.case_id is not None:
        return scope[0]

    # Priority 2: largest number of paths.
    max_paths = max((len(case["paths"]) for _, _, case in scope), default=0)
    cand = [(sid, cid, case) for sid, cid, case in scope if len(case["paths"]) == max_paths]

    # Priority 3: strongest-path case.
    def strongest(case: dict[str, Any]) -> float:
        if not case["paths"]:
            return -np.inf
        return max(_path_power(p, config.matrix_source) for p in case["paths"])

    best = max(cand, key=lambda x: strongest(x[2])) if cand else scope[0]
    return best


def _topk_paths(paths: list[dict[str, Any]], config: PlotConfig) -> list[dict[str, Any]]:
    if config.top_k_paths <= 0 or len(paths) <= config.top_k_paths:
        return list(paths)
    idx = np.argsort([-_path_power(p, config.matrix_source) for p in paths])
    return [paths[int(i)] for i in idx[: config.top_k_paths]]


def _scenario_exact_bounce(sid: str, config: PlotConfig, exact_bounce_map: dict[str, int]) -> int | None:
    if not config.apply_exact_bounce:
        return None
    return exact_bounce_map.get(sid)


def _exact_info_for_sid(sid: str, config: PlotConfig, exact_bounce_map: dict[str, int]) -> str:
    exact = _scenario_exact_bounce(sid, config, exact_bounce_map)
    return "None" if exact is None else str(exact)


def _plot_meta_line(data: dict[str, Any], config: PlotConfig, exact_info: str) -> str:
    meta = data.get("meta", {})
    basis = meta.get("basis", "NA")
    conv = meta.get("convention", config.convention)
    ant = meta.get("antenna_config", {})
    return (
        f"basis={basis}, conv={conv}, matrix={config.matrix_source}, exact_bounce={exact_info}, "
        f"coupling={ant.get('enable_coupling', 'NA')}, txLeak={ant.get('tx_cross_pol_leakage_db', 'NA')}dB, "
        f"rxLeak={ant.get('rx_cross_pol_leakage_db', 'NA')}dB"
    )


def _title(ax: plt.Axes, title: str, data: dict[str, Any], config: PlotConfig, exact_info: str) -> None:
    ax.set_title(title + "\n" + _plot_meta_line(data, config, exact_info), fontsize=10)


def _collect_samples(
    data: dict[str, Any],
    config: PlotConfig,
    exact_bounce_map: dict[str, int],
    scenario_ids: list[str] | None = None,
    subbands: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    allowed = set(scenario_ids) if scenario_ids is not None else None
    for sid, cid, case in _scope_cases(data, config):
        if allowed is not None and sid not in allowed:
            continue
        exact = _scenario_exact_bounce(sid, config, exact_bounce_map)
        vals = pathwise_xpd(
            case["paths"],
            subbands=subbands,
            exact_bounce=exact,
            matrix_source=config.matrix_source,
            power_floor=10.0 ** (config.power_floor_db / 10.0),
        )
        for v in vals:
            vv = dict(v)
            vv["scenario_id"] = sid
            vv["case_id"] = cid
            vv["material"] = str(case["params"].get("material", "NA"))
            out.append(vv)
    return out


def p0_geometry_ray_overlay(data: dict[str, Any], out: Path, config: PlotConfig, exact_bounce_map: dict[str, int]) -> None:
    sid, cid, case = _select_case(data, config)
    paths = _topk_paths(case["paths"], config)
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    for p in paths:
        pts = np.asarray(p.get("points", []), dtype=float)
        if len(pts) < 2:
            continue
        b = int(p["meta"]["bounce_count"])
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=cmap.get(b, "k"), alpha=0.8)
    _title(ax, f"P0 Geometry + Ray Overlay [{sid}/{cid}]", data, config, _exact_info_for_sid(sid, config, exact_bounce_map))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P0_geometry_overlay")


def p1_tau_power_scatter(data: dict[str, Any], out: Path, config: PlotConfig, exact_bounce_map: dict[str, int]) -> None:
    sid, cid, case = _select_case(data, config)
    paths = _topk_paths(case["paths"], config)
    fig, ax = plt.subplots(figsize=(7, 4))
    if paths:
        tau = np.array([p["tau_s"] for p in paths], dtype=float)
        pw = np.array([_path_power(p, config.matrix_source) for p in paths], dtype=float)
        b = np.array([p["meta"]["bounce_count"] for p in paths], dtype=int)
        sc = ax.scatter(tau * 1e9, 10 * np.log10(pw + 1e-18), c=b, cmap="viridis", s=24)
        fig.colorbar(sc, ax=ax, label="bounce_count")
    _title(ax, f"P1 Path Scatter: delay vs power [{sid}/{cid}]", data, config, _exact_info_for_sid(sid, config, exact_bounce_map))
    ax.set_xlabel("tau [ns]")
    ax.set_ylabel("power [dB]")
    if config.tau_plot_max_ns is not None:
        ax.set_xlim(0.0, float(config.tau_plot_max_ns))
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P1_tau_power")


def p2_hij_magnitude(data: dict[str, Any], out: Path, config: PlotConfig, exact_bounce_map: dict[str, int]) -> None:
    sid, cid, case = _select_case(data, config)
    freq = np.asarray(data["frequency"], dtype=float)
    paths = _topk_paths(case["paths"], config)

    H = np.zeros((len(freq), 2, 2), dtype=np.complex128)
    for p in paths:
        tau = float(p["tau_s"])
        A = _matrix_f(p, config.matrix_source)
        H += A * np.exp(-1j * 2.0 * np.pi * freq * tau)[:, None, None]

    fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    for i in range(2):
        for j in range(2):
            axs[i, j].plot(freq * 1e-9, 20 * np.log10(np.abs(H[:, i, j]) + 1e-12))
            axs[i, j].set_title(f"|H{i+1}{j+1}(f)|")
            axs[i, j].grid(True, alpha=0.3)
    axs[1, 0].set_xlabel("f [GHz]")
    axs[1, 1].set_xlabel("f [GHz]")
    fig.suptitle(_plot_meta_line(data, config, _exact_info_for_sid(sid, config, exact_bounce_map)) + f" | selected={sid}/{cid}", fontsize=9)
    _save(fig, out, "P2_Hij_magnitude")


def p3_pdp(data: dict[str, Any], out: Path, config: PlotConfig, exact_bounce_map: dict[str, int]) -> None:
    sid, cid, case = _select_case(data, config)
    freq = np.asarray(data["frequency"], dtype=float)
    paths = _topk_paths(case["paths"], config)

    H = np.zeros((len(freq), 2, 2), dtype=np.complex128)
    for p in paths:
        tau = float(p["tau_s"])
        A = _matrix_f(p, config.matrix_source)
        H += A * np.exp(-1j * 2.0 * np.pi * freq * tau)[:, None, None]

    h, tau = ctf_to_cir(H, freq)
    P = pdp(h)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tau * 1e9, 10 * np.log10(P["co"] + 1e-18), label="co")
    ax.plot(tau * 1e9, 10 * np.log10(P["cross"] + 1e-18), label="cross")
    ax.plot(tau * 1e9, 10 * np.log10(P["sum"] + 1e-18), label="sum")
    _title(ax, f"P3 PDP/CIR [{sid}/{cid}]", data, config, _exact_info_for_sid(sid, config, exact_bounce_map))
    ax.set_xlabel("tau [ns]")
    ax.set_ylabel("power [dB]")
    if config.tau_plot_max_ns is not None:
        ax.set_xlim(0.0, float(config.tau_plot_max_ns))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P3_PDP")


def p4_main_taps(data: dict[str, Any], out: Path, config: PlotConfig, exact_bounce_map: dict[str, int]) -> None:
    sid, cid, case = _select_case(data, config)
    freq = np.asarray(data["frequency"], dtype=float)
    paths = _topk_paths(case["paths"], config)

    H = np.zeros((len(freq), 2, 2), dtype=np.complex128)
    for p in paths:
        tau = float(p["tau_s"])
        A = _matrix_f(p, config.matrix_source)
        H += A * np.exp(-1j * 2.0 * np.pi * freq * tau)[:, None, None]

    h, tau = ctf_to_cir(H, freq, nfft=2048)
    p = np.sum(np.abs(h) ** 2, axis=(1, 2))
    idx = np.argsort(p)[-8:]
    idx.sort()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stem(tau[idx] * 1e9, 10 * np.log10(p[idx] + 1e-18))
    _title(ax, f"P4 Main taps zoom [{sid}/{cid}]", data, config, _exact_info_for_sid(sid, config, exact_bounce_map))
    ax.set_xlabel("tau [ns]")
    ax.set_ylabel("power [dB]")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P4_main_taps")


def _cp_metrics_enabled(data: dict[str, Any], out: Path, plot_name: str) -> bool:
    basis = str(data.get("meta", {}).get("basis", "linear"))
    if basis != "circular":
        _write_skip(out, plot_name, f"Skipped: CP metrics require circular basis. current basis={basis}")
        return False
    return True


def p5_cp_same_vs_opp(data: dict[str, Any], out: Path, config: PlotConfig, exact_bounce_map: dict[str, int]) -> None:
    if not _cp_metrics_enabled(data, out, "P5_cp_same_vs_opp"):
        return

    samples = _collect_samples(data, config, exact_bounce_map)
    if not samples:
        _write_skip(out, "P5_cp_same_vs_opp", "Skipped: no samples")
        return
    tau = np.array([s["tau_s"] for s in samples], dtype=float)
    same = np.array([s["co_power"] for s in samples], dtype=float)
    opp = np.array([s["cross_power"] for s in samples], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(tau * 1e9, 10 * np.log10(same + 1e-18), label="same-hand", s=14)
    ax.scatter(tau * 1e9, 10 * np.log10(opp + 1e-18), label="opposite-hand", s=14)
    _title(ax, "P5 CP same vs opposite", data, config, "per-scenario map" if config.apply_exact_bounce else "None")
    ax.set_xlabel("tau [ns]")
    ax.set_ylabel("power [dB]")
    if config.tau_plot_max_ns is not None:
        ax.set_xlim(0.0, float(config.tau_plot_max_ns))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P5_cp_same_vs_opp")


def p6_parity_xpd_box(data: dict[str, Any], out: Path, config: PlotConfig, exact_bounce_map: dict[str, int]) -> None:
    samples = _collect_samples(data, config, exact_bounce_map)
    odd = [s["xpd_db"] for s in samples if s["parity"] == "odd"]
    even = [s["xpd_db"] for s in samples if s["parity"] == "even"]
    if not odd and not even:
        _write_skip(out, "P6_parity_xpd", "Skipped: no parity samples")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([odd if odd else [np.nan], even if even else [np.nan]], tick_labels=["odd", "even"])
    _title(ax, "P6 XPD by parity", data, config, "per-scenario map" if config.apply_exact_bounce else "None")
    ax.set_ylabel("XPD [dB]")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P6_parity_xpd")


def p7_xpd_vs_bounce(data: dict[str, Any], out: Path, config: PlotConfig, exact_bounce_map: dict[str, int]) -> None:
    samples = []
    for sid, cid, case in _scope_cases(data, config):
        exact = _scenario_exact_bounce(sid, config, exact_bounce_map)
        vals = pathwise_xpd(
            case["paths"],
            exact_bounce=exact,
            matrix_source=config.matrix_source,
            power_floor=10.0 ** (config.power_floor_db / 10.0),
        )
        for i, s in enumerate(vals):
            angs = case["paths"][int(s["path_index"])]["meta"].get("incidence_angles", [])
            s2 = dict(s)
            s2["incidence_mean"] = float(np.nanmean(angs)) if angs else 0.0
            samples.append(s2)

    if not samples:
        _write_skip(out, "P7_xpd_vs_bounce", "Skipped: no samples")
        return

    x = np.array([s["bounce_count"] for s in samples], dtype=float)
    y = np.array([s["xpd_db"] for s in samples], dtype=float)
    c = np.array([s["incidence_mean"] for s in samples], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(x, y, c=c, cmap="plasma", s=16)
    fig.colorbar(sc, ax=ax, label="incidence angle [rad]")
    _title(ax, "P7 XPD vs bounce_count", data, config, "per-scenario map" if config.apply_exact_bounce else "None")
    ax.set_xlabel("bounce_count")
    ax.set_ylabel("XPD [dB]")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P7_xpd_vs_bounce")


def p8_xpd_vs_f_material(data: dict[str, Any], out: Path, config: PlotConfig, exact_bounce_map: dict[str, int]) -> None:
    if "A4" not in data["scenarios"]:
        _write_skip(out, "P8_xpd_vs_f_material", "Skipped: scenario A4 missing")
        return

    freq = np.asarray(data["frequency"], dtype=float)
    floor = 10.0 ** (config.power_floor_db / 10.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    for cid, case in data["scenarios"]["A4"]["cases"].items():
        if not case["paths"]:
            continue
        p = case["paths"][0]
        M = _matrix_f(p, config.matrix_source)
        co = np.abs(M[:, 0, 0]) ** 2 + np.abs(M[:, 1, 1]) ** 2
        cr = np.maximum(np.abs(M[:, 0, 1]) ** 2 + np.abs(M[:, 1, 0]) ** 2, floor)
        label = str(case["params"].get("material", cid))
        ax.plot(freq * 1e-9, 10 * np.log10((co + 1e-18) / (cr + 1e-18)), label=label, alpha=0.85)
    _title(ax, "P8 XPD(f) per material", data, config, _exact_info_for_sid("A4", config, exact_bounce_map))
    ax.set_xlabel("f [GHz]")
    ax.set_ylabel("XPD [dB]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P8_xpd_vs_f_material")


def p9_subband_mu_sigma(data: dict[str, Any], out: Path, config: PlotConfig, exact_bounce_map: dict[str, int]) -> None:
    if "A4" not in data["scenarios"]:
        _write_skip(out, "P9_subband_mu_sigma", "Skipped: scenario A4 missing")
        return

    freq = np.asarray(data["frequency"], dtype=float)
    bands = make_subbands(len(freq), 3)
    samples = []
    for _, _, case in [("A4", cid, c) for cid, c in data["scenarios"]["A4"]["cases"].items()]:
        samples.extend(
            pathwise_xpd(
                case["paths"],
                subbands=bands,
                exact_bounce=_scenario_exact_bounce("A4", config, exact_bounce_map),
                matrix_source=config.matrix_source,
                power_floor=10.0 ** (config.power_floor_db / 10.0),
            )
        )

    if not samples:
        _write_skip(out, "P9_subband_mu_sigma", "Skipped: no samples")
        return

    mu, sg = [], []
    for b in range(len(bands)):
        vals = np.array([s["xpd_db"] for s in samples if s.get("subband") == b], dtype=float)
        mu.append(np.nan if len(vals) == 0 else np.mean(vals))
        sg.append(np.nan if len(vals) == 0 else np.std(vals, ddof=1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(np.arange(len(bands)), mu, yerr=sg, fmt="o-")
    _title(ax, "P9 subband mu/sigma", data, config, _exact_info_for_sid("A4", config, exact_bounce_map))
    ax.set_xlabel("subband index")
    ax.set_ylabel("XPD [dB]")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P9_subband_mu_sigma")


def p10_parity_collapse(data: dict[str, Any], out: Path, config: PlotConfig, exact_bounce_map: dict[str, int]) -> None:
    base_sel = [x for x in ["A2", "A2R", "A3", "A3R"] if x in data["scenarios"]]
    stress_sel = ["A5"] if "A5" in data["scenarios"] else []
    base = _collect_samples(data, config, exact_bounce_map, scenario_ids=base_sel)
    stress = _collect_samples(data, config, exact_bounce_map, scenario_ids=stress_sel)
    if not base or not stress:
        _write_skip(out, "P10_parity_collapse", "Skipped: base/stress samples missing")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([s["xpd_db"] for s in base], bins=25, alpha=0.5, label="A2/A2R/A3/A3R")
    ax.hist([s["xpd_db"] for s in stress], bins=25, alpha=0.5, label="A5")
    _title(ax, "P10 parity separation collapse", data, config, "per-scenario map" if config.apply_exact_bounce else "None")
    ax.set_xlabel("XPD [dB]")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    _save(fig, out, "P10_parity_collapse")


def p11_var_vs_rho(data: dict[str, Any], out: Path, config: PlotConfig) -> None:
    if "A5" not in data["scenarios"]:
        _write_skip(out, "P11_xpd_var_vs_rho", "Skipped: scenario A5 missing")
        return

    rho_to_vals: dict[float, list[float]] = {}
    for _, case in data["scenarios"]["A5"]["cases"].items():
        rho = float(case["params"].get("rho", 0.0))
        vals = pathwise_xpd(
            case["paths"],
            matrix_source=config.matrix_source,
            power_floor=10.0 ** (config.power_floor_db / 10.0),
        )
        if not vals:
            continue
        bucket = rho_to_vals.setdefault(rho, [])
        bucket.extend(float(v["xpd_db"]) for v in vals)

    if not rho_to_vals:
        _write_skip(out, "P11_xpd_var_vs_rho", "Skipped: no rho samples")
        return

    rows: list[tuple[float, int, float | None]] = []
    for rho in sorted(rho_to_vals.keys()):
        arr = np.asarray(rho_to_vals[rho], dtype=float)
        n = int(len(arr))
        if n >= 2:
            rows.append((rho, n, float(np.var(arr, ddof=1))))
        else:
            rows.append((rho, n, None))

    fig, ax = plt.subplots(figsize=(7, 4))

    valid = [(rho, n, var) for rho, n, var in rows if var is not None]
    invalid = [(rho, n) for rho, n, var in rows if var is None]

    if valid:
        vx = np.asarray([x[0] for x in valid], dtype=float)
        vy = np.asarray([x[2] for x in valid], dtype=float)
        ax.plot(vx, vy, "o-", label="Var(XPD), n>=2")
        for rho, n, var in valid:
            ax.annotate(f"n={n}", (rho, float(var)), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    if invalid:
        ref_y = float(np.nanmax([v[2] for v in valid])) * 0.05 if valid else 1e-6
        ref_y = max(ref_y, 1e-6)
        ix = np.asarray([x[0] for x in invalid], dtype=float)
        iy = np.full_like(ix, ref_y, dtype=float)
        ax.scatter(ix, iy, marker="x", color="red", label="insufficient n (<2)")
        for (rho, n), y in zip(invalid, iy):
            ax.annotate(f"n={n} insufficient", (rho, float(y)), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color="red")

    _title(ax, "P11 XPD variance vs rho", data, config, "None")
    ax.set_xlabel("rho")
    ax.set_ylabel("Var(XPD)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P11_xpd_var_vs_rho")


def p12_delay_conditioned(data: dict[str, Any], out: Path, config: PlotConfig, exact_bounce_map: dict[str, int]) -> None:
    samples = _collect_samples(data, config, exact_bounce_map)
    if not samples:
        _write_skip(out, "P12_delay_conditioned", "Skipped: no samples")
        return
    tau = np.array([s["tau_s"] for s in samples], dtype=float)
    bins = np.quantile(tau, [0.0, 0.33, 0.66, 1.0])
    mu, sg = [], []
    for i in range(3):
        vals = np.array([s["xpd_db"] for s in samples if bins[i] <= s["tau_s"] <= bins[i + 1]], dtype=float)
        mu.append(float(np.mean(vals)) if len(vals) else np.nan)
        sg.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar([0, 1, 2], mu, yerr=sg, fmt="o-")
    _title(ax, "P12 delay-conditioned mu/sigma", data, config, "per-scenario map" if config.apply_exact_bounce else "None")
    ax.set_xlabel("delay bin")
    ax.set_ylabel("XPD [dB]")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P12_delay_conditioned")


def p13_k_factor(data: dict[str, Any], out: Path, config: PlotConfig) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for sid, sc in data["scenarios"].items():
        xs, ys = [], []
        for i, (_, case) in enumerate(sc["cases"].items()):
            paths = case["paths"]
            if not paths:
                continue
            pwr = np.array([_path_power(p, config.matrix_source) for p in paths], dtype=float)
            j = int(np.argmax(pwr))
            k = pwr[j] / (np.sum(pwr) - pwr[j] + 1e-18)
            xs.append(i)
            ys.append(10 * np.log10(k + 1e-18))
        if xs:
            ax.plot(xs, ys, "o-", label=sid)
    _title(ax, "P13 scenario K-factor trend", data, config, "None")
    ax.set_xlabel("case index")
    ax.set_ylabel("K [dB]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P13_k_factor")


def p14_tau_error_hist(data: dict[str, Any], out: Path, config: PlotConfig) -> None:
    c0 = 299_792_458.0
    errs_ps = []
    for _, _, case in _scope_cases(data, config):
        for p in case["paths"]:
            pts = np.asarray(p.get("points", []), dtype=float)
            if len(pts) < 2:
                continue
            d = float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))
            errs_ps.append((float(p["tau_s"]) - d / c0) * 1e12)
    if not errs_ps:
        _write_skip(out, "P14_tau_error_hist", "Skipped: points not available")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errs_ps, bins=30, alpha=0.8)
    _title(ax, "P14 tau error histogram", data, config, "None")
    ax.set_xlabel("tau_error [ps]")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P14_tau_error_hist")


def p15_incidence_distribution(data: dict[str, Any], out: Path, config: PlotConfig) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    any_vals = False
    for sid, _, case in _scope_cases(data, config):
        vals = []
        for p in case["paths"]:
            vals.extend(p["meta"].get("incidence_angles", []))
        if vals:
            any_vals = True
            ax.hist(np.rad2deg(np.asarray(vals, dtype=float)), bins=18, alpha=0.35, label=sid)
    if not any_vals:
        _write_skip(out, "P15_incidence_distribution", "Skipped: no incidence metadata")
        plt.close(fig)
        return
    _title(ax, "P15 incidence angle distribution", data, config, "None")
    ax.set_xlabel("incidence angle [deg]")
    ax.set_ylabel("count")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P15_incidence_distribution")


def p16_fresnel_curves(data: dict[str, Any], out: Path, config: PlotConfig) -> None:
    freq = np.asarray(data["frequency"], dtype=float)
    fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    theta = np.deg2rad(45.0)
    for name, mat in MATERIALS.items():
        gs, gp = fresnel_reflection(mat, theta_i=theta, f_hz=freq)
        axs[0].plot(freq * 1e-9, np.abs(gs), label=f"{name} |Gs|")
        axs[0].plot(freq * 1e-9, np.abs(gp), "--", label=f"{name} |Gp|")
        axs[1].plot(freq * 1e-9, np.unwrap(np.angle(gs)), label=f"{name} ∠Gs")
        axs[1].plot(freq * 1e-9, np.unwrap(np.angle(gp)), "--", label=f"{name} ∠Gp")
    axs[0].set_title("P16 Fresnel magnitude/phase vs frequency\n" + _plot_meta_line(data, config, "None"), fontsize=9)
    axs[0].set_ylabel("magnitude")
    axs[1].set_xlabel("f [GHz]")
    axs[1].set_ylabel("phase [rad]")
    axs[0].grid(True, alpha=0.3)
    axs[1].grid(True, alpha=0.3)
    axs[0].legend(fontsize=7, ncol=2)
    _save(fig, out, "P16_fresnel_curves")


def p17_cp_same_opp_vs_bounce(data: dict[str, Any], out: Path, config: PlotConfig, exact_bounce_map: dict[str, int]) -> None:
    if not _cp_metrics_enabled(data, out, "P17_cp_same_opp_vs_bounce"):
        return
    samples = _collect_samples(data, config, exact_bounce_map)
    if not samples:
        _write_skip(out, "P17_cp_same_opp_vs_bounce", "Skipped: no samples")
        return

    b = np.array([int(s["bounce_count"]) for s in samples], dtype=int)
    same = np.array([float(s["co_power"]) for s in samples], dtype=float)
    opp = np.array([float(s["cross_power"]) for s in samples], dtype=float)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(b, 10 * np.log10(same + 1e-18), s=12, label="same")
    axs[0].scatter(b + 0.05, 10 * np.log10(opp + 1e-18), s=12, label="opp")
    axs[0].set_xlabel("bounce_count")
    axs[0].set_ylabel("power [dB]")
    axs[0].legend(fontsize=8)
    axs[0].grid(True, alpha=0.3)

    uniq = sorted(set(b.tolist()))
    ratios = [10 * np.log10((np.mean(same[b == k]) + 1e-18) / (np.mean(opp[b == k]) + 1e-18)) for k in uniq]
    axs[1].bar(np.arange(len(uniq)), ratios)
    axs[1].set_xticks(np.arange(len(uniq)), [str(k) for k in uniq])
    axs[1].set_xlabel("bounce_count")
    axs[1].set_ylabel("10log10(Psame/Popp) [dB]")
    axs[1].grid(True, alpha=0.3)
    axs[0].set_title("P17 same/opp power vs bounce")
    axs[1].set_title("P17 same-vs-opp ratio by bounce")
    fig.suptitle(_plot_meta_line(data, config, "per-scenario map" if config.apply_exact_bounce else "None"), fontsize=9)
    _save(fig, out, "P17_cp_same_opp_vs_bounce")


def p17_a6_bounce_compare(data: dict[str, Any], out: Path, config: PlotConfig) -> None:
    if not _cp_metrics_enabled(data, out, "P17_A6_bounce_compare"):
        return
    if "A6" not in data["scenarios"]:
        _write_skip(out, "P17_A6_bounce_compare", "Skipped: scenario A6 missing")
        return

    # For A6 benchmark, default to propagation-only matrix when possible.
    bench_matrix_source = "J" if config.matrix_source == "A" else config.matrix_source
    floor = 10.0 ** (config.power_floor_db / 10.0)

    bounce1: list[float] = []
    bounce2: list[float] = []
    for _, case in data["scenarios"]["A6"]["cases"].items():
        for s in pathwise_xpd(case["paths"], matrix_source=bench_matrix_source, power_floor=floor):
            b = int(s["bounce_count"])
            xpd = float(s["xpd_db"])
            if b == 1:
                bounce1.append(xpd)
            elif b == 2:
                bounce2.append(xpd)

    if len(bounce1) == 0 or len(bounce2) == 0:
        _write_skip(out, "P17_A6_bounce_compare", "Skipped: need both bounce=1 and bounce=2 samples in A6")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([bounce1, bounce2], tick_labels=[f"bounce=1 (n={len(bounce1)})", f"bounce=2 (n={len(bounce2)})"])
    ax.set_title(
        "P17 A6 CP benchmark: same/opp ratio distribution\n"
        + _plot_meta_line(data, config, "A6 near-normal, exact_bounce=None")
        + f", benchmark_matrix={bench_matrix_source}"
    )
    ax.set_ylabel("XPD [dB] (10log10(Psame/Popp))")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P17_A6_bounce_compare")


def p18_singular_values_vs_delay(data: dict[str, Any], out: Path, config: PlotConfig) -> None:
    paths = []
    for _, _, case in _scope_cases(data, config):
        paths.extend(case["paths"])
    if not paths:
        _write_skip(out, "P18_singular_values_delay", "Skipped: no paths")
        return

    tau = np.array([float(p["tau_s"]) for p in paths], dtype=float) * 1e9
    s1, s2, cond = [], [], []
    for p in paths:
        M = np.mean(_matrix_f(p, config.matrix_source), axis=0)
        sv = np.linalg.svd(M, compute_uv=False)
        smax, smin = float(sv[0]), float(max(sv[-1], 1e-15))
        s1.append(smax)
        s2.append(smin)
        cond.append(smax / smin)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(tau, 20 * np.log10(np.asarray(s1) + 1e-18), s=12, label="σ1")
    axs[0].scatter(tau, 20 * np.log10(np.asarray(s2) + 1e-18), s=12, label="σ2")
    axs[0].set_xlabel("tau [ns]")
    axs[0].set_ylabel("magnitude [dB]")
    axs[0].legend(fontsize=8)
    axs[0].grid(True, alpha=0.3)

    axs[1].scatter(tau, 20 * np.log10(np.asarray(cond) + 1e-18), s=12)
    axs[1].set_xlabel("tau [ns]")
    axs[1].set_ylabel("cond [dB]")
    axs[1].grid(True, alpha=0.3)
    axs[0].set_title("P18 singular values vs delay")
    axs[1].set_title("P18 condition number vs delay")
    fig.suptitle(_plot_meta_line(data, config, "None"), fontsize=9)
    _save(fig, out, "P18_singular_values_delay")


def p19_reciprocity_sanity(data: dict[str, Any], out: Path, config: PlotConfig) -> None:
    info = data.get("meta", {}).get("reciprocity_sanity", None)
    if not info:
        _write_skip(out, "P19_reciprocity_sanity", "Skipped: reciprocity_sanity results missing in dataset meta")
        return

    entries = info.get("entries", [])
    if not entries:
        _write_skip(out, "P19_reciprocity_sanity", "Skipped: no reciprocity entries")
        return

    x_tau = []
    y_dsig = []
    c_bounce = []
    match_lines = []
    for e in entries:
        sid = str(e.get("scenario_id", "NA"))
        cid = str(e.get("case_id", "NA"))
        match_lines.append(f"{sid}/{cid}: {e.get('matched_ratio', np.nan):.2f}")
        for m in e.get("matches", []):
            x_tau.append(float(m.get("tau_f_s", 0.0)) * 1e9)
            y_dsig.append(float(m.get("delta_sigma_max_db", np.nan)))
            c_bounce.append(int(m.get("bounce_count", 0)))

    if not x_tau:
        _write_skip(out, "P19_reciprocity_sanity", "Skipped: no matched paths")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    sc = ax.scatter(np.asarray(x_tau), np.asarray(y_dsig), c=np.asarray(c_bounce), cmap="viridis", s=18)
    fig.colorbar(sc, ax=ax, label="bounce_count")
    dtau_max = float(info.get("delta_tau_max_s_global", np.nan))
    mr = float(info.get("matched_ratio_global", np.nan))
    dsmax = float(info.get("delta_sigma_max_db_global", np.nan))
    ax.set_title(
        "P19 Reciprocity sanity (invariant-based)\n"
        + _plot_meta_line(data, config, "None")
        + f", matched_ratio={mr:.3f}, dTau_max={dtau_max:.3e}s, dSigma_max={dsmax:.3e}dB"
    )
    ax.set_xlabel("tau [ns]")
    ax.set_ylabel("delta_sigma_max [dB]")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P19_reciprocity_sanity")


def p20_xpd_fit_gof(data: dict[str, Any], out: Path, config: PlotConfig, exact_bounce_map: dict[str, int]) -> None:
    samples = _collect_samples(data, config, exact_bounce_map)
    if not samples:
        _write_skip(out, "P20_xpd_fit_gof", "Skipped: no XPD samples")
        return

    min_n = 20
    bootstrap_B = 200
    parity_vals = {
        "odd": np.asarray([float(s["xpd_db"]) for s in samples if s.get("parity") == "odd"], dtype=float),
        "even": np.asarray([float(s["xpd_db"]) for s in samples if s.get("parity") == "even"], dtype=float),
    }

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    summary_lines: list[str] = []
    for idx, key in enumerate(["odd", "even"]):
        ax = axs[idx]
        vals = parity_vals[key]
        vals = vals[np.isfinite(vals)]
        floor_db = np.nan
        ant_cfg = data.get("meta", {}).get("antenna_config", {})
        if bool(ant_cfg):
            tx_l = float(ant_cfg.get("tx_cross_pol_leakage_db", 35.0))
            rx_l = float(ant_cfg.get("rx_cross_pol_leakage_db", 35.0))
            floor_db = float(20.0 * np.log10(1.0 / (10.0 ** (-tx_l / 20.0) + 10.0 ** (-rx_l / 20.0) + 1e-15)))
        gr = gof_model_selection_db(
            vals,
            min_n=min_n,
            bootstrap_B=bootstrap_B,
            seed=idx,
            floor_db=floor_db if np.isfinite(floor_db) else None,
            pinned_tol_db=0.5,
        )
        status = str(gr.get("status", "NA"))

        if status != "OK" or str(gr.get("best_model", "NA")) == "NA":
            ax.text(0.5, 0.5, f"{key}: {status}\nn={int(gr.get('n', 0))}", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("theoretical quantile")
            ax.set_ylabel("sample quantile [dB]")
            ax.grid(True, alpha=0.3)
            summary_lines.append(f"{key}: {status}(n={int(gr.get('n', 0))})")
            continue

        best_model = str(gr.get("best_model", "normal_db"))
        best = (gr.get("best_metrics", {}) or {})
        params = dict(best.get("params", {}))
        probs = (np.arange(1, len(vals) + 1, dtype=float) - 0.5) / max(len(vals), 1)
        x_theory = model_quantiles_db(best_model, params, probs)
        y_sample = np.sort(vals)
        ax.plot(x_theory, y_sample, "o", ms=3, alpha=0.8, label=f"{key} samples")
        if np.std(x_theory) > 0.0:
            slope, intercept = np.polyfit(x_theory, y_sample, 1)
            ax.plot(x_theory, slope * x_theory + intercept, "-", lw=1.2, label="fit")
        ax.set_xlabel("theoretical quantile")
        ax.set_ylabel("sample quantile [dB]")
        warn_tag = " [WARN]" if bool(best.get("warning", False) or gr.get("warning", False)) else ""
        ax.set_title(
            f"{key}{warn_tag}: model={best_model}, n={int(gr['n'])}, n_fit={int(gr.get('n_fit', 0))}\n"
            f"qq_r={float(best.get('qq_r', np.nan)):.3f}, ks_D={float(best.get('ks_D', np.nan)):.3f}, "
            f"ks_p_boot={float(best.get('ks_p_boot', np.nan)):.3f}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        summary_lines.append(
            f"{key}: model={best_model}, n={int(gr['n'])}, qq_r={float(best.get('qq_r', np.nan)):.3f}, "
            f"ks_p_boot={float(best.get('ks_p_boot', np.nan)):.3f}"
        )

    fig.suptitle(
        "P20 XPD fit GOF (model selection in dB, parity buckets)\n"
        + _plot_meta_line(data, config, "per-scenario map" if config.apply_exact_bounce else "None")
        + f", min_n={min_n}, bootstrap_B={bootstrap_B}",
        fontsize=9,
    )
    if summary_lines:
        fig.text(0.01, 0.01, " | ".join(summary_lines), fontsize=8)
    _save(fig, out, "P20_xpd_fit_gof")


def p21_tap_vs_path_consistency(data: dict[str, Any], out: Path, config: PlotConfig) -> None:
    info = data.get("meta", {}).get("tap_path_consistency", None)
    if not info:
        _write_skip(out, "P21_tap_vs_path_consistency", "Skipped: tap_path_consistency missing in dataset meta")
        return
    entries = info.get("entries", [])
    if not entries:
        _write_skip(out, "P21_tap_vs_path_consistency", "Skipped: empty tap_path_consistency entries")
        return

    x = np.asarray([float(e.get("xpd_path_strongest_db", np.nan)) for e in entries], dtype=float)
    y = np.asarray([float(e.get("xpd_tap_window_db", e.get("xpd_tap_peak_db", np.nan))) for e in entries], dtype=float)
    d = np.asarray([float(e.get("delta_xpd_db", np.nan)) for e in entries], dtype=float)
    w = np.asarray([bool(e.get("wrap_detected", False)) for e in entries], dtype=bool)
    n_paths = np.asarray([int(e.get("n_paths", 0)) for e in entries], dtype=float)
    overlap_count = np.asarray([int(e.get("overlap_count", 0)) for e in entries], dtype=float)
    reasons = [str(e.get("outlier_reason", "NONE")) for e in entries]
    sids = [str(e.get("scenario_id", "NA")) for e in entries]
    cids = [str(e.get("case_id", "NA")) for e in entries]

    good = np.isfinite(x) & np.isfinite(y)
    if not np.any(good):
        _write_skip(out, "P21_tap_vs_path_consistency", "Skipped: no finite XPD points")
        return

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    ax = axs[0]
    idx = np.where(good)[0]
    cval = np.where(w[good], 1.0, 0.0)
    sc = ax.scatter(x[good], y[good], c=cval, cmap="coolwarm", s=24, alpha=0.85)
    fig.colorbar(sc, ax=ax, ticks=[0, 1], label="wrap_detected (0/1)")

    mn = float(np.nanmin(np.concatenate([x[good], y[good]])))
    mx = float(np.nanmax(np.concatenate([x[good], y[good]])))
    pad = 0.05 * max(1.0, mx - mn)
    lo, hi = mn - pad, mx + pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="y=x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # Label outliers by scenario/case.
    outlier_idx = [i for i in idx if np.isfinite(d[i]) and float(d[i]) > 10.0]
    for i in outlier_idx[:20]:
        ax.annotate(f"{sids[i]}/{cids[i]}", (x[i], y[i]), textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.set_title("Path XPD vs Tap-window XPD")
    ax.set_xlabel("XPD_path_strongest [dB]")
    ax.set_ylabel("XPD_tap_window [dB]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ΔXPD vs n_paths
    ax2 = axs[1]
    g2 = np.isfinite(d)
    sc2 = ax2.scatter(n_paths[g2], d[g2], c=overlap_count[g2], cmap="viridis", s=24, alpha=0.85)
    fig.colorbar(sc2, ax=ax2, label="overlap_count")
    ax2.set_title("ΔXPD vs n_paths")
    ax2.set_xlabel("n_paths")
    ax2.set_ylabel("ΔXPD [dB]")
    ax2.grid(True, alpha=0.3)

    # ΔXPD vs overlap_count
    ax3 = axs[2]
    sc3 = ax3.scatter(overlap_count[g2], d[g2], c=np.where(w[g2], 1.0, 0.0), cmap="coolwarm", s=24, alpha=0.85)
    fig.colorbar(sc3, ax=ax3, label="wrap_detected (0/1)")
    ax3.set_title("ΔXPD vs overlap_count")
    ax3.set_xlabel("overlap_count")
    ax3.set_ylabel("ΔXPD [dB]")
    ax3.grid(True, alpha=0.3)
    # annotate a few largest outliers with reason
    if np.any(g2):
        big = np.argsort(-np.nan_to_num(d, nan=-np.inf))[:8]
        for i in big:
            if not np.isfinite(d[i]):
                continue
            ax3.annotate(f"{sids[i]}/{cids[i]}:{reasons[i]}", (overlap_count[i], d[i]), textcoords="offset points", xytext=(3, 3), fontsize=7)

    fig.suptitle(
        "P21 tap-wise vs path-wise consistency (windowed)\n"
        + _plot_meta_line(data, config, "None")
        + f", outliers(ΔXPD>10dB)={len(outlier_idx)}, wraps={int(np.sum(w))}, overlap_cases={int(np.sum(overlap_count > 1))}"
    )
    _save(fig, out, "P21_tap_vs_path_consistency")


def generate_all_plots(
    data: dict[str, Any],
    out_dir: str | Path,
    config: PlotConfig,
    exact_bounce_map: dict[str, int] | None = None,
) -> None:
    exact_map = exact_bounce_map or {}
    out = _ensure_dir(out_dir)
    p0_geometry_ray_overlay(data, out, config, exact_map)
    p1_tau_power_scatter(data, out, config, exact_map)
    p2_hij_magnitude(data, out, config, exact_map)
    p3_pdp(data, out, config, exact_map)
    p4_main_taps(data, out, config, exact_map)
    p5_cp_same_vs_opp(data, out, config, exact_map)
    p6_parity_xpd_box(data, out, config, exact_map)
    p7_xpd_vs_bounce(data, out, config, exact_map)
    p8_xpd_vs_f_material(data, out, config, exact_map)
    p9_subband_mu_sigma(data, out, config, exact_map)
    p10_parity_collapse(data, out, config, exact_map)
    p11_var_vs_rho(data, out, config)
    p12_delay_conditioned(data, out, config, exact_map)
    p13_k_factor(data, out, config)
    p14_tau_error_hist(data, out, config)
    p15_incidence_distribution(data, out, config)
    p16_fresnel_curves(data, out, config)
    p17_cp_same_opp_vs_bounce(data, out, config, exact_map)
    p17_a6_bounce_compare(data, out, config)
    p18_singular_values_vs_delay(data, out, config)
    p19_reciprocity_sanity(data, out, config)
    p20_xpd_fit_gof(data, out, config, exact_map)
    p21_tap_vs_path_consistency(data, out, config)


def _run_self_test() -> None:
    # minimal synthetic dataset
    nf = 8
    freq = np.linspace(6e9, 7e9, nf)
    A = np.ones((nf, 2, 2), dtype=np.complex128)
    data_lin = {
        "meta": {"basis": "linear", "convention": "IEEE-RHCP", "antenna_config": {"enable_coupling": True}},
        "frequency": freq,
        "scenarios": {
            "A2": {
                "cases": {
                    "0": {
                        "params": {},
                        "paths": [
                            {"tau_s": 1e-8, "A_f": A, "J_f": A, "meta": {"bounce_count": 0, "incidence_angles": [], "surface_ids": [], "segment_basis_uv": []}},
                            {"tau_s": 2e-8, "A_f": A, "J_f": A, "meta": {"bounce_count": 1, "incidence_angles": [0.5], "surface_ids": [1], "segment_basis_uv": []}},
                        ],
                    }
                }
            },
            "S2": {"cases": {"1": {"params": {}, "paths": [{"tau_s": 3e-8, "A_f": A, "J_f": A, "meta": {"bounce_count": 2, "incidence_angles": [], "surface_ids": [], "segment_basis_uv": []}}]}}},
        },
    }

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td:
        out = Path(td)
        cfg = PlotConfig(scenario_id="S2", case_id="1", matrix_source="A", apply_exact_bounce=True)

        # (a) CP plots skipped for linear basis.
        generate_all_plots(data_lin, out, cfg, exact_bounce_map={"A2": 1})
        assert not (out / "P5_cp_same_vs_opp.png").exists()
        assert (out / "P5_cp_same_vs_opp.SKIPPED.txt").exists()

        # (b) selection honors requested scenario/case.
        sid, cid, _ = _select_case(data_lin, cfg)
        assert sid == "S2" and cid == "1"

        # (c) exact_bounce filtering follows map.
        s = _collect_samples(data_lin, PlotConfig(matrix_source="A", apply_exact_bounce=True), {"A2": 1}, scenario_ids=["A2"])
        assert all(int(x["bounce_count"]) == 1 for x in s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _run_self_test()


if __name__ == "__main__":
    main()

"""Automated plotting suite P0~P13 using matplotlib only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from analysis.ctf_cir import ctf_to_cir, pdp, synthesize_ctf
from analysis.xpd_stats import conditional_fit, pathwise_xpd
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


def _all_paths(data: dict[str, Any], scenario_ids: list[str] | None = None) -> list[dict]:
    ids = scenario_ids if scenario_ids is not None else list(data["scenarios"].keys())
    out = []
    for sid in ids:
        for cid, case in data["scenarios"][sid]["cases"].items():
            for p in case["paths"]:
                cp = dict(p)
                cp["scenario"] = sid
                cp["case_id"] = cid
                cp["params"] = case["params"]
                out.append(cp)
    return out


def p0_geometry_ray_overlay(data: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    for sid in ["A2", "A2R", "A3", "A3R", "A5"]:
        if sid not in data["scenarios"]:
            continue
        first_case = next(iter(data["scenarios"][sid]["cases"].values()))
        for p in first_case["paths"]:
            pts = np.asarray(p.get("points", []), dtype=float)
            if len(pts) < 2:
                continue
            b = int(p["meta"]["bounce_count"])
            ax.plot(pts[:, 0], pts[:, 1], "-o", color=cmap.get(b, "k"), alpha=0.8)
    ax.set_title("P0 Geometry + Ray Overlay")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P0_geometry_overlay")


def p1_tau_power_scatter(data: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    paths = _all_paths(data)
    tau = np.array([p["tau_s"] for p in paths], dtype=float)
    pw = np.array([np.mean(np.abs(np.asarray(p["A_f"])) ** 2) for p in paths], dtype=float)
    b = np.array([p["meta"]["bounce_count"] for p in paths], dtype=int)
    sc = ax.scatter(tau * 1e9, 10 * np.log10(pw + 1e-18), c=b, cmap="viridis", s=16)
    fig.colorbar(sc, ax=ax, label="bounce_count")
    ax.set_title("P1 Path Scatter: delay vs power")
    ax.set_xlabel("tau [ns]")
    ax.set_ylabel("power [dB]")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P1_tau_power")


def p2_hij_magnitude(data: dict[str, Any], out: Path) -> None:
    freq = np.asarray(data["frequency"], dtype=float)
    case = next(iter(next(iter(data["scenarios"].values()))["cases"].values()))
    H = synthesize_ctf(case["paths"], freq)
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    for i in range(2):
        for j in range(2):
            axs[i, j].plot(freq * 1e-9, 20 * np.log10(np.abs(H[:, i, j]) + 1e-12))
            axs[i, j].set_title(f"|H{i+1}{j+1}(f)|")
            axs[i, j].grid(True, alpha=0.3)
    axs[1, 0].set_xlabel("f [GHz]")
    axs[1, 1].set_xlabel("f [GHz]")
    _save(fig, out, "P2_Hij_magnitude")


def p3_pdp(data: dict[str, Any], out: Path) -> None:
    freq = np.asarray(data["frequency"], dtype=float)
    case = next(iter(next(iter(data["scenarios"].values()))["cases"].values()))
    H = synthesize_ctf(case["paths"], freq)
    h, tau = ctf_to_cir(H, freq)
    P = pdp(h)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tau * 1e9, 10 * np.log10(P["co"] + 1e-18), label="co")
    ax.plot(tau * 1e9, 10 * np.log10(P["cross"] + 1e-18), label="cross")
    ax.plot(tau * 1e9, 10 * np.log10(P["sum"] + 1e-18), label="sum")
    ax.set_title("P3 PDP/CIR")
    ax.set_xlabel("tau [ns]")
    ax.set_ylabel("power [dB]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P3_PDP")


def p4_main_taps(data: dict[str, Any], out: Path) -> None:
    freq = np.asarray(data["frequency"], dtype=float)
    case = next(iter(next(iter(data["scenarios"].values()))["cases"].values()))
    H = synthesize_ctf(case["paths"], freq)
    h, tau = ctf_to_cir(H, freq, nfft=2048)
    p = np.sum(np.abs(h) ** 2, axis=(1, 2))
    idx = np.argsort(p)[-8:]
    idx.sort()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stem(tau[idx] * 1e9, 10 * np.log10(p[idx] + 1e-18))
    ax.set_title("P4 Main taps zoom")
    ax.set_xlabel("tau [ns]")
    ax.set_ylabel("power [dB]")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P4_main_taps")


def p5_cp_same_vs_opp(data: dict[str, Any], out: Path) -> None:
    sel = [x for x in ["A2", "A2R", "A3", "A3R", "A5"] if x in data["scenarios"]]
    paths = _all_paths(data, scenario_ids=sel)
    tau = np.array([p["tau_s"] for p in paths], dtype=float)
    same = np.array([np.mean(np.abs(np.asarray(p["A_f"])[:, 0, 0]) ** 2 + np.abs(np.asarray(p["A_f"])[:, 1, 1]) ** 2) for p in paths])
    opp = np.array([np.mean(np.abs(np.asarray(p["A_f"])[:, 0, 1]) ** 2 + np.abs(np.asarray(p["A_f"])[:, 1, 0]) ** 2) for p in paths])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(tau * 1e9, 10 * np.log10(same + 1e-18), label="same-hand", s=12)
    ax.scatter(tau * 1e9, 10 * np.log10(opp + 1e-18), label="opposite-hand", s=12)
    ax.set_title("P5 CP same vs opposite PDP")
    ax.set_xlabel("tau [ns]")
    ax.set_ylabel("power [dB]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P5_cp_same_vs_opp")


def p6_parity_xpd_box(data: dict[str, Any], out: Path) -> None:
    sel = [x for x in ["A2", "A2R", "A3", "A3R", "A5"] if x in data["scenarios"]]
    samples = pathwise_xpd(_all_paths(data, sel))
    odd = [s["xpd_db"] for s in samples if s["parity"] == "odd"]
    even = [s["xpd_db"] for s in samples if s["parity"] == "even"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([odd, even], labels=["odd", "even"])
    ax.set_title("P6 XPD by parity")
    ax.set_ylabel("XPD [dB]")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P6_parity_xpd")


def p7_xpd_vs_bounce(data: dict[str, Any], out: Path) -> None:
    paths = _all_paths(data)
    x = np.array([p["meta"]["bounce_count"] for p in paths], dtype=float)
    y = np.array([pathwise_xpd([p])[0]["xpd_db"] for p in paths], dtype=float)
    c = np.array(
        [
            float(np.nanmean(angles)) if len((angles := p["meta"].get("incidence_angles", []))) > 0 else 0.0
            for p in paths
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(x, y, c=c, cmap="plasma", s=16)
    fig.colorbar(sc, ax=ax, label="incidence angle [rad]")
    ax.set_title("P7 XPD vs bounce_count")
    ax.set_xlabel("bounce_count")
    ax.set_ylabel("XPD [dB]")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P7_xpd_vs_bounce")


def p8_xpd_vs_f_material(data: dict[str, Any], out: Path) -> None:
    freq = np.asarray(data["frequency"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    if "A4" in data["scenarios"]:
        for cid, case in data["scenarios"]["A4"]["cases"].items():
            H = synthesize_ctf(case["paths"], freq)
            co = np.abs(H[:, 0, 0]) ** 2 + np.abs(H[:, 1, 1]) ** 2
            cr = np.abs(H[:, 0, 1]) ** 2 + np.abs(H[:, 1, 0]) ** 2
            label = str(case["params"].get("material", cid))
            ax.plot(freq * 1e-9, 10 * np.log10((co + 1e-18) / (cr + 1e-18)), label=label, alpha=0.8)
    ax.set_title("P8 XPD(f) per material")
    ax.set_xlabel("f [GHz]")
    ax.set_ylabel("XPD [dB]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P8_xpd_vs_f_material")


def p9_subband_mu_sigma(data: dict[str, Any], out: Path) -> None:
    freq = np.asarray(data["frequency"], dtype=float)
    n = len(freq)
    bands = [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]
    samples = []
    for p in _all_paths(data, ["A4"] if "A4" in data["scenarios"] else None):
        for s in pathwise_xpd([p], subbands=bands):
            samples.append(s)
    mu, sg = [], []
    for b in range(len(bands)):
        vals = np.array([s["xpd_db"] for s in samples if s.get("subband") == b], dtype=float)
        mu.append(np.nan if len(vals) == 0 else np.mean(vals))
        sg.append(np.nan if len(vals) == 0 else np.std(vals))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(np.arange(len(bands)), mu, yerr=sg, fmt="o-")
    ax.set_title("P9 subband mu/sigma")
    ax.set_xlabel("subband index")
    ax.set_ylabel("XPD [dB]")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P9_subband_mu_sigma")


def p10_parity_collapse(data: dict[str, Any], out: Path) -> None:
    base_sel = [x for x in ["A2", "A2R", "A3", "A3R"] if x in data["scenarios"]]
    base = pathwise_xpd(_all_paths(data, base_sel) if base_sel else _all_paths(data))
    stress = pathwise_xpd(_all_paths(data, ["A5"]) if "A5" in data["scenarios"] else _all_paths(data))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([s["xpd_db"] for s in base], bins=25, alpha=0.5, label="A2/A3")
    ax.hist([s["xpd_db"] for s in stress], bins=25, alpha=0.5, label="A5")
    ax.set_title("P10 parity separation collapse")
    ax.set_xlabel("XPD [dB]")
    ax.set_ylabel("count")
    ax.legend()
    _save(fig, out, "P10_parity_collapse")


def p11_var_vs_rho(data: dict[str, Any], out: Path) -> None:
    xs, ys = [], []
    if "A5" in data["scenarios"]:
        for _, case in data["scenarios"]["A5"]["cases"].items():
            rho = float(case["params"].get("rho", 0.0))
            vals = np.array([s["xpd_db"] for s in pathwise_xpd(case["paths"])], dtype=float)
            if len(vals) > 0:
                xs.append(rho)
                ys.append(float(np.var(vals)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-")
    ax.set_title("P11 XPD variance vs rho")
    ax.set_xlabel("rho")
    ax.set_ylabel("Var(XPD)")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P11_xpd_var_vs_rho")


def p12_delay_conditioned(data: dict[str, Any], out: Path) -> None:
    samples = pathwise_xpd(_all_paths(data))
    if not samples:
        return
    tau = np.array([s["tau_s"] for s in samples], dtype=float)
    bins = np.quantile(tau, [0.0, 0.33, 0.66, 1.0])
    mu, sg = [], []
    for i in range(3):
        vals = np.array([s["xpd_db"] for s in samples if bins[i] <= s["tau_s"] <= bins[i + 1]], dtype=float)
        mu.append(float(np.mean(vals)) if len(vals) else np.nan)
        sg.append(float(np.std(vals)) if len(vals) else np.nan)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar([0, 1, 2], mu, yerr=sg, fmt="o-")
    ax.set_title("P12 delay-conditioned mu/sigma")
    ax.set_xlabel("delay bin")
    ax.set_ylabel("XPD [dB]")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P12_delay_conditioned")


def p13_k_factor(data: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for sid, sc in data["scenarios"].items():
        xs, ys = [], []
        for i, (_, case) in enumerate(sc["cases"].items()):
            paths = case["paths"]
            if not paths:
                continue
            pwr = np.array([np.mean(np.abs(np.asarray(p["A_f"])) ** 2) for p in paths], dtype=float)
            j = int(np.argmax(pwr))
            k = pwr[j] / (np.sum(pwr) - pwr[j] + 1e-18)
            xs.append(i)
            ys.append(10 * np.log10(k + 1e-18))
        if xs:
            ax.plot(xs, ys, "o-", label=sid)
    ax.set_title("P13 scenario K-factor trend")
    ax.set_xlabel("case index")
    ax.set_ylabel("K [dB]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P13_k_factor")


def p14_tau_error_hist(data: dict[str, Any], out: Path) -> None:
    c0 = 299_792_458.0
    errs_ps = []
    for p in _all_paths(data):
        pts = np.asarray(p.get("points", []), dtype=float)
        if len(pts) < 2:
            continue
        d = float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))
        tau_geo = d / c0
        errs_ps.append((float(p["tau_s"]) - tau_geo) * 1e12)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errs_ps, bins=30, alpha=0.8)
    ax.set_title("P14 tau error histogram")
    ax.set_xlabel("tau_error [ps]")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P14_tau_error_hist")


def p15_incidence_distribution(data: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for sid in data["scenarios"]:
        vals = []
        for case in data["scenarios"][sid]["cases"].values():
            for p in case["paths"]:
                vals.extend(p["meta"].get("incidence_angles", []))
        if vals:
            ax.hist(np.rad2deg(np.asarray(vals, dtype=float)), bins=20, alpha=0.4, label=sid)
    ax.set_title("P15 incidence angle distribution")
    ax.set_xlabel("incidence angle [deg]")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out, "P15_incidence_distribution")


def p16_fresnel_curves(data: dict[str, Any], out: Path) -> None:
    freq = np.asarray(data["frequency"], dtype=float)
    fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    theta = np.deg2rad(45.0)
    for name, mat in MATERIALS.items():
        gs, gp = fresnel_reflection(mat, theta_i=theta, f_hz=freq)
        axs[0].plot(freq * 1e-9, np.abs(gs), label=f"{name} |Gs|")
        axs[0].plot(freq * 1e-9, np.abs(gp), "--", label=f"{name} |Gp|")
        axs[1].plot(freq * 1e-9, np.unwrap(np.angle(gs)), label=f"{name} ∠Gs")
        axs[1].plot(freq * 1e-9, np.unwrap(np.angle(gp)), "--", label=f"{name} ∠Gp")
    axs[0].set_title("P16 Fresnel magnitude/phase vs frequency")
    axs[0].set_ylabel("magnitude")
    axs[1].set_xlabel("f [GHz]")
    axs[1].set_ylabel("phase [rad]")
    axs[0].grid(True, alpha=0.3)
    axs[1].grid(True, alpha=0.3)
    axs[0].legend(fontsize=7, ncol=2)
    _save(fig, out, "P16_fresnel_curves")


def p17_cp_same_opp_vs_bounce(data: dict[str, Any], out: Path) -> None:
    paths = _all_paths(data)
    b = np.array([int(p["meta"]["bounce_count"]) for p in paths], dtype=int)
    same = np.array([np.mean(np.abs(np.asarray(p["A_f"])[:, 0, 0]) ** 2 + np.abs(np.asarray(p["A_f"])[:, 1, 1]) ** 2) for p in paths], dtype=float)
    opp = np.array([np.mean(np.abs(np.asarray(p["A_f"])[:, 0, 1]) ** 2 + np.abs(np.asarray(p["A_f"])[:, 1, 0]) ** 2) for p in paths], dtype=float)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(b, 10 * np.log10(same + 1e-18), s=12, label="same")
    axs[0].scatter(b + 0.05, 10 * np.log10(opp + 1e-18), s=12, label="opp")
    axs[0].set_title("P17 same/opp power vs bounce")
    axs[0].set_xlabel("bounce_count")
    axs[0].set_ylabel("power [dB]")
    axs[0].legend(fontsize=8)
    axs[0].grid(True, alpha=0.3)

    uniq = sorted(set(b.tolist()))
    ratios = [10 * np.log10((np.mean(same[b == k]) + 1e-18) / (np.mean(opp[b == k]) + 1e-18)) for k in uniq]
    axs[1].bar(np.arange(len(uniq)), ratios)
    axs[1].set_xticks(np.arange(len(uniq)), [str(k) for k in uniq])
    axs[1].set_title("P17 same-vs-opp ratio by bounce")
    axs[1].set_xlabel("bounce_count")
    axs[1].set_ylabel("10log10(Psame/Popp) [dB]")
    axs[1].grid(True, alpha=0.3)
    _save(fig, out, "P17_cp_same_opp_vs_bounce")


def p18_singular_values_vs_delay(data: dict[str, Any], out: Path) -> None:
    paths = _all_paths(data)
    tau = np.array([float(p["tau_s"]) for p in paths], dtype=float) * 1e9
    s1, s2, cond = [], [], []
    for p in paths:
        M = np.mean(np.asarray(p["A_f"], dtype=np.complex128), axis=0)
        sv = np.linalg.svd(M, compute_uv=False)
        smax, smin = float(sv[0]), float(max(sv[-1], 1e-15))
        s1.append(smax)
        s2.append(smin)
        cond.append(smax / smin)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(tau, 20 * np.log10(np.asarray(s1) + 1e-18), s=12, label="σ1")
    axs[0].scatter(tau, 20 * np.log10(np.asarray(s2) + 1e-18), s=12, label="σ2")
    axs[0].set_title("P18 singular values vs delay")
    axs[0].set_xlabel("tau [ns]")
    axs[0].set_ylabel("magnitude [dB]")
    axs[0].legend(fontsize=8)
    axs[0].grid(True, alpha=0.3)

    axs[1].scatter(tau, 20 * np.log10(np.asarray(cond) + 1e-18), s=12)
    axs[1].set_title("P18 condition number vs delay")
    axs[1].set_xlabel("tau [ns]")
    axs[1].set_ylabel("cond [dB]")
    axs[1].grid(True, alpha=0.3)
    _save(fig, out, "P18_singular_values_delay")


def generate_all_plots(data: dict[str, Any], out_dir: str | Path) -> None:
    out = _ensure_dir(out_dir)
    p0_geometry_ray_overlay(data, out)
    p1_tau_power_scatter(data, out)
    p2_hij_magnitude(data, out)
    p3_pdp(data, out)
    p4_main_taps(data, out)
    p5_cp_same_vs_opp(data, out)
    p6_parity_xpd_box(data, out)
    p7_xpd_vs_bounce(data, out)
    p8_xpd_vs_f_material(data, out)
    p9_subband_mu_sigma(data, out)
    p10_parity_collapse(data, out)
    p11_var_vs_rho(data, out)
    p12_delay_conditioned(data, out)
    p13_k_factor(data, out)
    p14_tau_error_hist(data, out)
    p15_incidence_distribution(data, out)
    p16_fresnel_curves(data, out)
    p17_cp_same_opp_vs_bounce(data, out)
    p18_singular_values_vs_delay(data, out)

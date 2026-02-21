"""XPD/XPR estimation and conditional statistics.

This module provides power-based path/tap metrics that are stable for UWB,
including support for both embedded matrices (A_f) and propagation-only
matrices (J_f).
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats


EPS = 1e-15
MatrixSource = Literal["A", "J"]
BasisName = Literal["linear", "circular"]


def _matrix_from_path(path: dict[str, Any], matrix_source: MatrixSource = "A") -> NDArray[np.complex128]:
    if matrix_source == "J" and "J_f" in path:
        return np.asarray(path["J_f"], dtype=np.complex128)
    return np.asarray(path["A_f"], dtype=np.complex128)


def _resolve_basis(
    input_basis: BasisName | None,
    eval_basis: BasisName | None,
) -> tuple[BasisName | None, BasisName | None]:
    src = str(input_basis).lower() if input_basis is not None else None
    dst = str(eval_basis).lower() if eval_basis is not None else None
    src_b = src if src in {"linear", "circular"} else None
    dst_b = dst if dst in {"linear", "circular"} else src_b
    return src_b, dst_b


def _matrix_in_eval_basis(
    M_f: NDArray[np.complex128],
    input_basis: BasisName | None,
    eval_basis: BasisName | None,
    convention: str,
) -> NDArray[np.complex128]:
    src_b, dst_b = _resolve_basis(input_basis, eval_basis)
    if src_b is None or dst_b is None or src_b == dst_b:
        return M_f
    from analysis.ctf_cir import convert_basis

    return convert_basis(M_f, src=src_b, dst=dst_b, convention=convention)


def xpd_variable_definition(eval_basis: BasisName | None) -> str:
    if str(eval_basis).lower() == "circular":
        return "XPD_circular_same_vs_opposite_hand_power_ratio_dB"
    return "XPD_linear_copol_vs_crosspol_power_ratio_dB"


def _co_cross_power_spectra(M_f: NDArray[np.complex128]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    co_f = np.abs(M_f[:, 0, 0]) ** 2 + np.abs(M_f[:, 1, 1]) ** 2
    cross_f = np.abs(M_f[:, 0, 1]) ** 2 + np.abs(M_f[:, 1, 0]) ** 2
    return co_f.astype(float), cross_f.astype(float)


def make_subbands(nf: int, num_subbands: int) -> list[tuple[int, int]]:
    if num_subbands <= 0:
        raise ValueError("num_subbands must be positive")
    idx = np.linspace(0, nf, num_subbands + 1, dtype=int)
    out = []
    for i in range(num_subbands):
        s, e = int(idx[i]), int(idx[i + 1])
        if e > s:
            out.append((s, e))
    return out


def subband_centers_hz(f_hz: NDArray[np.float64], subbands: list[tuple[int, int]]) -> NDArray[np.float64]:
    f = np.asarray(f_hz, dtype=float)
    return np.asarray([float(np.mean(f[s:e])) for s, e in subbands], dtype=float)


def xpd_xpr_from_power(co_power: float, cross_power: float, power_floor: float = 1e-12) -> dict[str, float]:
    co = float(co_power)
    cross = max(float(cross_power), float(power_floor))
    xpd_linear = (co + EPS) / (cross + EPS)
    xpd_db = float(10.0 * np.log10(xpd_linear))
    return {
        "co_power": co,
        "cross_power": cross,
        "xpd_linear": float(xpd_linear),
        "xpr_linear": float(xpd_linear),
        "xpd_db": xpd_db,
        "xpr_db": xpd_db,
    }


def pathwise_xpd(
    paths: list[dict[str, Any]],
    subbands: list[tuple[int, int]] | None = None,
    exact_bounce: int | None = None,
    bounce_filter: set[int] | None = None,
    power_floor: float = 1e-12,
    matrix_source: MatrixSource = "A",
    input_basis: BasisName | None = None,
    eval_basis: BasisName | None = None,
    convention: str = "IEEE-RHCP",
) -> list[dict[str, Any]]:
    """Compute path-wise XPD/XPR using mean POWER over frequency.

    For each path and frequency span:
      co = mean(|M11|^2 + |M22|^2)
      cross = mean(|M12|^2 + |M21|^2)
      XPD = 10log10(co/cross)

    where M is A_f (embedded) or J_f (propagation-only).
    """

    out: list[dict[str, Any]] = []
    src_b, dst_b = _resolve_basis(input_basis, eval_basis)
    rv_name = xpd_variable_definition(dst_b)
    for i, p in enumerate(paths):
        M_f = _matrix_from_path(p, matrix_source=matrix_source)
        M_f = _matrix_in_eval_basis(M_f, src_b, dst_b, convention=convention)
        meta = p.get("meta", {})
        bcnt = int(meta.get("bounce_count", 0))

        if exact_bounce is not None and bcnt != exact_bounce:
            continue
        if bounce_filter is not None and bcnt not in bounce_filter:
            continue

        co_f, cross_f = _co_cross_power_spectra(M_f)
        base = {
            "path_index": i,
            "bounce_count": bcnt,
            "parity": "even" if bcnt % 2 == 0 else "odd",
            "tau_s": float(p.get("tau_s", 0.0)),
            "matrix_source": matrix_source,
            "input_basis": src_b,
            "eval_basis": dst_b,
            "convention": str(convention),
            "xpd_variable": rv_name,
        }

        if subbands is None:
            stats = xpd_xpr_from_power(float(np.mean(co_f)), float(np.mean(cross_f)), power_floor=power_floor)
            out.append({**base, **stats})
            continue

        for bidx, (s, e) in enumerate(subbands):
            stats = xpd_xpr_from_power(float(np.mean(co_f[s:e])), float(np.mean(cross_f[s:e])), power_floor=power_floor)
            out.append({**base, "subband": bidx, **stats})

    return out


def tapwise_xpd(
    h_tau: NDArray[np.complex128],
    tau_s: NDArray[np.float64],
    win_s: tuple[float, float] | None = None,
    power_floor: float = 1e-12,
    eval_basis: BasisName | None = None,
    convention: str = "IEEE-RHCP",
) -> dict[str, NDArray[np.float64]]:
    p = np.abs(h_tau) ** 2
    co = p[:, 0, 0] + p[:, 1, 1]
    cross = np.maximum(p[:, 0, 1] + p[:, 1, 0], power_floor)
    xpd = 10.0 * np.log10((co + EPS) / (cross + EPS))
    if win_s is None:
        m = np.ones_like(tau_s, dtype=bool)
    else:
        m = (tau_s >= win_s[0]) & (tau_s <= win_s[1])
    return {
        "tau_s": tau_s[m],
        "xpd_db": xpd[m],
        "co": co[m],
        "cross": cross[m],
        "eval_basis": np.asarray([str(eval_basis or "linear")] * int(np.sum(m))),
        "convention": np.asarray([str(convention)] * int(np.sum(m))),
    }


def tapwise_xpd_from_paths(
    paths: list[dict[str, Any]],
    f_hz: NDArray[np.float64],
    matrix_source: MatrixSource = "A",
    input_basis: BasisName | None = None,
    eval_basis: BasisName | None = None,
    convention: str = "IEEE-RHCP",
    nfft: int = 2048,
    window: str = "hann",
    win_s: tuple[float, float] | None = None,
    power_floor: float = 1e-12,
) -> dict[str, NDArray[np.float64]]:
    """Compute tap-wise XPD random variable from path list with explicit basis handling."""

    from analysis.ctf_cir import ctf_to_cir, synthesize_ctf_with_basis

    H_f = synthesize_ctf_with_basis(
        paths,
        f_hz=np.asarray(f_hz, dtype=float),
        matrix_source=matrix_source,
        input_basis=input_basis,
        eval_basis=eval_basis,
        convention=convention,
    )
    h_tau, tau = ctf_to_cir(H_f, np.asarray(f_hz, dtype=float), nfft=nfft, window=window)  # type: ignore[arg-type]
    out = tapwise_xpd(
        h_tau,
        tau,
        win_s=win_s,
        power_floor=power_floor,
        eval_basis=eval_basis,
        convention=convention,
    )
    out["matrix_source"] = np.asarray([str(matrix_source)] * len(out["tau_s"]))
    out["xpd_variable"] = np.asarray([xpd_variable_definition(eval_basis)] * len(out["tau_s"]))
    return out


def early_late_split(samples: list[dict[str, Any]], split_tau_s: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    early = [s for s in samples if float(s.get("tau_s", 0.0)) <= split_tau_s]
    late = [s for s in samples if float(s.get("tau_s", 0.0)) > split_tau_s]
    return early, late


def fit_normal_db(values: NDArray[np.float64]) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"mu": np.nan, "sigma": np.nan, "n": 0}
    mu = float(np.mean(v))
    sigma = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return {"mu": mu, "sigma": sigma, "n": int(len(v))}


def conditional_fit(samples: list[dict[str, Any]], keys: list[str], value_key: str = "xpd_db") -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {}
    for s in samples:
        k = "|".join(str(s.get(x, "NA")) for x in keys)
        buckets.setdefault(k, []).append(float(s[value_key]))
    return {k: fit_normal_db(np.asarray(v, dtype=float)) for k, v in buckets.items()}


def fit_linear_mu_frequency(
    subband_samples: list[dict[str, Any]],
    subband_centers: NDArray[np.float64],
    condition_keys: list[str],
    value_key: str = "xpd_db",
) -> dict[str, dict[str, Any]]:
    """Fit mu(f)=mu0 + mu1*(f-fc) using subband-wise means."""

    centers = np.asarray(subband_centers, dtype=float)
    fc = float(np.mean(centers)) if len(centers) else 0.0
    groups: dict[str, dict[int, list[float]]] = {}

    for s in subband_samples:
        if "subband" not in s:
            continue
        g = "|".join(str(s.get(k, "NA")) for k in condition_keys)
        b = int(s["subband"])
        groups.setdefault(g, {}).setdefault(b, []).append(float(s[value_key]))

    out: dict[str, dict[str, Any]] = {}
    for g, bmap in groups.items():
        xs, ys = [], []
        for b, vals in sorted(bmap.items()):
            if 0 <= b < len(centers) and len(vals) > 0:
                xs.append(float(centers[b]))
                ys.append(float(np.mean(vals)))
        if len(xs) >= 2:
            x = np.asarray(xs, dtype=float) - fc
            y = np.asarray(ys, dtype=float)
            mu1, mu0 = np.polyfit(x, y, 1)
        elif len(xs) == 1:
            mu0, mu1 = ys[0], 0.0
        else:
            mu0, mu1 = np.nan, np.nan
        out[g] = {
            "mu0_db": float(mu0),
            "mu1_db_per_hz": float(mu1),
            "fc_hz": fc,
            "num_points": len(xs),
            "subband_centers_hz": xs,
            "subband_mu_db": ys,
        }
    return out


def incidence_angle_bin_label(incidence_angles_rad: list[float], bins_deg: list[float]) -> str:
    if not incidence_angles_rad:
        return "NA"
    ang_deg = float(np.rad2deg(np.nanmean(np.asarray(incidence_angles_rad, dtype=float))))
    b = np.asarray(bins_deg, dtype=float)
    if len(b) < 2:
        return f"{ang_deg:.1f}deg"
    idx = int(np.digitize([ang_deg], b, right=False)[0]) - 1
    idx = int(np.clip(idx, 0, len(b) - 2))
    return f"[{b[idx]:.0f},{b[idx+1]:.0f})"


def save_stats_json(path: str | Path, stats_obj: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats_obj, indent=2), encoding="utf-8")
    return p


def gof_normal_db(
    values_db: list[float] | NDArray[np.float64],
    min_n: int = 20,
    bootstrap_B: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Goodness-of-fit for Normal model on dB-domain samples.

    Returns:
      n, mu, sigma, qq_r, qq_r2, ks_D, ks_p_raw, ks_p_boot, status, warning
    """

    v = np.asarray(values_db, dtype=float)
    v = v[np.isfinite(v)]
    n = int(len(v))
    if n < int(min_n):
        return {
            "n": n,
            "mu": np.nan,
            "sigma": np.nan,
            "qq_r": np.nan,
            "qq_r2": np.nan,
            "ks_D": np.nan,
            "ks_p_raw": np.nan,
            "ks_p_boot": np.nan,
            "status": "INSUFFICIENT",
            "warning": True,
        }

    mu = float(np.mean(v))
    sigma = float(np.std(v, ddof=1)) if n > 1 else 0.0
    if sigma <= 0.0:
        return {
            "n": n,
            "mu": mu,
            "sigma": sigma,
            "qq_r": 1.0,
            "qq_r2": 1.0,
            "ks_D": 0.0,
            "ks_p_raw": 1.0,
            "ks_p_boot": np.nan,
            "status": "DEGENERATE",
            "warning": True,
        }

    (_, _), (_, _, qq_r) = stats.probplot(v, dist="norm", fit=True)
    qq_r = float(qq_r)
    qq_r2 = float(qq_r**2)

    ks = stats.kstest(v, "norm", args=(mu, sigma))
    ks_D = float(ks.statistic)
    ks_p_raw = float(ks.pvalue)

    ks_p_boot = np.nan
    B = int(max(0, bootstrap_B))
    if B > 0:
        rng = np.random.default_rng(int(seed))
        d_boot = np.zeros(B, dtype=float)
        for b in range(B):
            x = rng.normal(loc=mu, scale=sigma, size=n)
            mb = float(np.mean(x))
            sb = float(np.std(x, ddof=1)) if n > 1 else 0.0
            if sb <= 0.0:
                d_boot[b] = 0.0
            else:
                d_boot[b] = float(stats.kstest(x, "norm", args=(mb, sb)).statistic)
        ks_p_boot = float(np.mean(d_boot >= ks_D))

    warning = bool((qq_r < 0.98) or (np.isfinite(ks_p_boot) and ks_p_boot < 0.05))
    return {
        "n": n,
        "mu": mu,
        "sigma": sigma,
        "qq_r": qq_r,
        "qq_r2": qq_r2,
        "ks_D": ks_D,
        "ks_p_raw": ks_p_raw,
        "ks_p_boot": ks_p_boot,
        "status": "OK",
        "warning": warning,
    }


def _safe_std(v: NDArray[np.float64]) -> float:
    if len(v) <= 1:
        return 0.0
    return float(np.std(v, ddof=1))


def _normal_loglik(v: NDArray[np.float64], mu: float, sigma: float) -> float:
    if sigma <= 0.0 or len(v) == 0:
        return -np.inf
    z = (v - mu) / sigma
    ll = -0.5 * z * z - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)
    return float(np.sum(ll))


def _ks_statistic_from_cdf(v: NDArray[np.float64], cdf_vals: NDArray[np.float64]) -> float:
    if len(v) == 0:
        return np.nan
    x = np.sort(v)
    f = np.asarray(cdf_vals, dtype=float)
    f = np.clip(f, 0.0, 1.0)
    ecdf_hi = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
    ecdf_lo = np.arange(0, len(x), dtype=float) / float(len(x))
    d_plus = np.max(ecdf_hi - f)
    d_minus = np.max(f - ecdf_lo)
    return float(max(d_plus, d_minus))


def _fit_normal_model(v: NDArray[np.float64]) -> dict[str, Any]:
    mu = float(np.mean(v)) if len(v) else np.nan
    sigma = _safe_std(v)
    if not np.isfinite(mu) or sigma <= 0.0:
        return {"status": "FAIL", "warning": "degenerate sigma", "params": {}, "k": 2, "loglik": -np.inf}
    ll = _normal_loglik(v, mu, sigma)
    return {"status": "OK", "params": {"mu": mu, "sigma": sigma}, "k": 2, "loglik": ll}


def _fit_gmm2_model(v: NDArray[np.float64], seed: int = 0, restarts: int = 4, max_iter: int = 200) -> dict[str, Any]:
    if len(v) < 4:
        return {"status": "FAIL", "warning": "insufficient samples for gmm2", "params": {}, "k": 5, "loglik": -np.inf}
    best: dict[str, Any] | None = None
    std0 = max(_safe_std(v), 1e-3)
    rng = np.random.default_rng(int(seed))
    for ridx in range(max(1, int(restarts))):
        if ridx == 0:
            mu1 = float(np.quantile(v, 0.25))
            mu2 = float(np.quantile(v, 0.75))
        else:
            pick = rng.choice(v, size=2, replace=False)
            mu1, mu2 = float(min(pick)), float(max(pick))
        s1 = std0
        s2 = std0
        w = 0.5
        prev_ll = -np.inf
        for _ in range(int(max_iter)):
            n1 = stats.norm.pdf(v, loc=mu1, scale=max(s1, 1e-3))
            n2 = stats.norm.pdf(v, loc=mu2, scale=max(s2, 1e-3))
            mix = w * n1 + (1.0 - w) * n2 + EPS
            r1 = (w * n1) / mix
            r2 = 1.0 - r1
            w = float(np.clip(np.mean(r1), 1e-4, 1.0 - 1e-4))
            sw1 = float(np.sum(r1))
            sw2 = float(np.sum(r2))
            mu1 = float(np.sum(r1 * v) / max(sw1, EPS))
            mu2 = float(np.sum(r2 * v) / max(sw2, EPS))
            s1 = float(np.sqrt(np.sum(r1 * (v - mu1) ** 2) / max(sw1, EPS)))
            s2 = float(np.sqrt(np.sum(r2 * (v - mu2) ** 2) / max(sw2, EPS)))
            s1 = max(s1, 1e-3)
            s2 = max(s2, 1e-3)
            ll = float(np.sum(np.log(mix)))
            if abs(ll - prev_ll) < 1e-8:
                break
            prev_ll = ll
        if mu1 > mu2:
            mu1, mu2 = mu2, mu1
            s1, s2 = s2, s1
            w = 1.0 - w
        cur = {
            "status": "OK",
            "params": {"weight": float(w), "mu1": float(mu1), "sigma1": float(s1), "mu2": float(mu2), "sigma2": float(s2)},
            "k": 5,
            "loglik": float(prev_ll),
        }
        if best is None or float(cur["loglik"]) > float(best["loglik"]):
            best = cur
    return best if best is not None else {"status": "FAIL", "warning": "gmm2 fit failed", "params": {}, "k": 5, "loglik": -np.inf}


def _fit_truncnorm_model(v: NDArray[np.float64]) -> dict[str, Any]:
    if len(v) < 4:
        return {
            "status": "FAIL",
            "warning": "insufficient samples for truncnorm",
            "params": {},
            "k": 4,
            "loglik": -np.inf,
        }
    lower = float(np.min(v))
    upper = float(np.max(v))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return {"status": "FAIL", "warning": "invalid bounds", "params": {}, "k": 4, "loglik": -np.inf}
    mu0 = float(np.mean(v))
    sig0 = max(_safe_std(v), 1e-3)

    def nll(theta: NDArray[np.float64]) -> float:
        mu = float(theta[0])
        sigma = float(np.exp(theta[1]))
        a = (lower - mu) / sigma
        b = (upper - mu) / sigma
        z = stats.norm.cdf(b) - stats.norm.cdf(a)
        if z <= 1e-12:
            return 1e12
        t = (v - mu) / sigma
        ll = -0.5 * t * t - np.log(sigma) - 0.5 * np.log(2.0 * np.pi) - np.log(z)
        out = float(-np.sum(ll))
        if not np.isfinite(out):
            return 1e12
        return out

    res = optimize.minimize(
        nll,
        x0=np.asarray([mu0, np.log(sig0)], dtype=float),
        method="L-BFGS-B",
        bounds=[(None, None), (-20.0, 20.0)],
        options={"maxiter": 400},
    )
    if not res.success:
        return {"status": "FAIL", "warning": f"truncnorm optimize failed: {res.message}", "params": {}, "k": 4, "loglik": -np.inf}
    mu = float(res.x[0])
    sigma = float(np.exp(res.x[1]))
    ll = float(-res.fun)
    return {
        "status": "OK",
        "params": {"mu": mu, "sigma": sigma, "lower": lower, "upper": upper},
        "k": 4,
        "loglik": ll,
    }


def _mode_center_db(v: NDArray[np.float64], bin_width_db: float = 0.25) -> tuple[float, float]:
    if len(v) == 0:
        return np.nan, 0.0
    width = max(float(bin_width_db), 1e-3)
    lo = float(np.min(v))
    hi = float(np.max(v))
    if hi <= lo:
        return lo, 1.0
    nbins = int(np.clip(np.ceil((hi - lo) / width), 8, 512))
    hist, edges = np.histogram(v, bins=nbins)
    j = int(np.argmax(hist))
    ctr = float(0.5 * (edges[j] + edges[j + 1]))
    frac = float(hist[j] / max(len(v), 1))
    return ctr, frac


def _fit_spike_slab_model(
    v: NDArray[np.float64],
    floor_db: float | None = None,
    seed: int = 0,
    max_iter: int = 200,
) -> dict[str, Any]:
    if len(v) < 4:
        return {"status": "FAIL", "warning": "insufficient samples for spike-slab", "params": {}, "k": 4, "loglik": -np.inf}
    if floor_db is not None and np.isfinite(floor_db):
        spike_loc = float(floor_db)
    else:
        spike_loc, _ = _mode_center_db(v, bin_width_db=0.25)
    std0 = max(_safe_std(v), 1e-3)
    sigma_spike = float(max(0.1, min(0.5, 0.1 * std0)))
    slab_mu = float(np.mean(v))
    slab_sigma = std0
    pi = 0.2
    _ = np.random.default_rng(int(seed))
    prev_ll = -np.inf
    for _iter in range(int(max_iter)):
        p_sp = stats.norm.pdf(v, loc=spike_loc, scale=sigma_spike)
        p_sl = stats.norm.pdf(v, loc=slab_mu, scale=max(slab_sigma, 1e-3))
        mix = pi * p_sp + (1.0 - pi) * p_sl + EPS
        r = (pi * p_sp) / mix
        pi = float(np.clip(np.mean(r), 1e-4, 1.0 - 1e-4))
        w_sl = np.maximum(1.0 - r, EPS)
        sw = float(np.sum(w_sl))
        slab_mu = float(np.sum(w_sl * v) / sw)
        slab_sigma = float(np.sqrt(np.sum(w_sl * (v - slab_mu) ** 2) / sw))
        slab_sigma = max(slab_sigma, 1e-3)
        ll = float(np.sum(np.log(mix)))
        if abs(ll - prev_ll) < 1e-8:
            break
        prev_ll = ll
    return {
        "status": "OK",
        "params": {
            "pi_spike": float(pi),
            "spike_loc": float(spike_loc),
            "spike_sigma": float(sigma_spike),
            "slab_mu": float(slab_mu),
            "slab_sigma": float(slab_sigma),
        },
        "k": 4,
        "loglik": float(prev_ll),
    }


def _model_cdf(model: str, x: NDArray[np.float64], params: dict[str, float]) -> NDArray[np.float64]:
    xv = np.asarray(x, dtype=float)
    if model == "normal_db":
        return stats.norm.cdf(xv, loc=float(params["mu"]), scale=max(float(params["sigma"]), 1e-9))
    if model == "gmm2_db":
        w = float(params["weight"])
        c1 = stats.norm.cdf(xv, loc=float(params["mu1"]), scale=max(float(params["sigma1"]), 1e-9))
        c2 = stats.norm.cdf(xv, loc=float(params["mu2"]), scale=max(float(params["sigma2"]), 1e-9))
        return np.clip(w * c1 + (1.0 - w) * c2, 0.0, 1.0)
    if model == "truncnorm_db":
        mu = float(params["mu"])
        sigma = max(float(params["sigma"]), 1e-9)
        lo = float(params["lower"])
        hi = float(params["upper"])
        a = (lo - mu) / sigma
        b = (hi - mu) / sigma
        z = max(float(stats.norm.cdf(b) - stats.norm.cdf(a)), 1e-12)
        t = (xv - mu) / sigma
        c = (stats.norm.cdf(t) - stats.norm.cdf(a)) / z
        c = np.where(xv <= lo, 0.0, c)
        c = np.where(xv >= hi, 1.0, c)
        return np.clip(c, 0.0, 1.0)
    if model == "spike_slab_db":
        p = float(params["pi_spike"])
        c_sp = stats.norm.cdf(xv, loc=float(params["spike_loc"]), scale=max(float(params["spike_sigma"]), 1e-9))
        c_sl = stats.norm.cdf(xv, loc=float(params["slab_mu"]), scale=max(float(params["slab_sigma"]), 1e-9))
        return np.clip(p * c_sp + (1.0 - p) * c_sl, 0.0, 1.0)
    raise ValueError(f"unknown model: {model}")


def _model_sample(model: str, n: int, params: dict[str, float], rng: np.random.Generator) -> NDArray[np.float64]:
    if model == "normal_db":
        return rng.normal(float(params["mu"]), max(float(params["sigma"]), 1e-9), size=n).astype(float)
    if model == "gmm2_db":
        w = float(params["weight"])
        m = rng.random(n) < w
        x = np.empty(n, dtype=float)
        x[m] = rng.normal(float(params["mu1"]), max(float(params["sigma1"]), 1e-9), size=int(np.sum(m)))
        x[~m] = rng.normal(float(params["mu2"]), max(float(params["sigma2"]), 1e-9), size=int(np.sum(~m)))
        return x
    if model == "truncnorm_db":
        mu = float(params["mu"])
        sigma = max(float(params["sigma"]), 1e-9)
        lo = float(params["lower"])
        hi = float(params["upper"])
        a = (lo - mu) / sigma
        b = (hi - mu) / sigma
        return stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n, random_state=rng).astype(float)
    if model == "spike_slab_db":
        p = float(params["pi_spike"])
        m = rng.random(n) < p
        x = np.empty(n, dtype=float)
        x[m] = rng.normal(float(params["spike_loc"]), max(float(params["spike_sigma"]), 1e-9), size=int(np.sum(m)))
        x[~m] = rng.normal(float(params["slab_mu"]), max(float(params["slab_sigma"]), 1e-9), size=int(np.sum(~m)))
        return x
    raise ValueError(f"unknown model: {model}")


def _numeric_ppf_from_cdf(
    model: str,
    q: NDArray[np.float64],
    params: dict[str, float],
    low: float,
    high: float,
) -> NDArray[np.float64]:
    out = np.empty_like(q, dtype=float)
    for i, qq in enumerate(q):
        lo = low
        hi = high
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            cm = float(_model_cdf(model, np.asarray([mid], dtype=float), params)[0])
            if cm < qq:
                lo = mid
            else:
                hi = mid
        out[i] = 0.5 * (lo + hi)
    return out


def model_quantiles_db(model: str, params: dict[str, float], probs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return model quantiles for QQ plots."""

    q = np.asarray(probs, dtype=float)
    q = np.clip(q, 1e-6, 1.0 - 1e-6)
    if model == "normal_db":
        return stats.norm.ppf(q, loc=float(params["mu"]), scale=max(float(params["sigma"]), 1e-9))
    if model == "gmm2_db":
        lo = float(min(params["mu1"] - 8.0 * params["sigma1"], params["mu2"] - 8.0 * params["sigma2"]))
        hi = float(max(params["mu1"] + 8.0 * params["sigma1"], params["mu2"] + 8.0 * params["sigma2"]))
        return _numeric_ppf_from_cdf(model, q, params, low=lo, high=hi)
    if model == "truncnorm_db":
        mu = float(params["mu"])
        sigma = max(float(params["sigma"]), 1e-9)
        lo = float(params["lower"])
        hi = float(params["upper"])
        a = (lo - mu) / sigma
        b = (hi - mu) / sigma
        return stats.truncnorm.ppf(q, a, b, loc=mu, scale=sigma)
    if model == "spike_slab_db":
        lo = float(min(params["spike_loc"] - 8.0 * params["spike_sigma"], params["slab_mu"] - 8.0 * params["slab_sigma"]))
        hi = float(max(params["spike_loc"] + 8.0 * params["spike_sigma"], params["slab_mu"] + 8.0 * params["slab_sigma"]))
        return _numeric_ppf_from_cdf(model, q, params, low=lo, high=hi)
    raise ValueError(f"unknown model: {model}")


def _bootstrap_ks_p(
    model: str,
    params: dict[str, float],
    n: int,
    d_obs: float,
    B: int,
    seed: int,
    floor_db: float | None = None,
) -> float:
    if B <= 0 or not np.isfinite(d_obs):
        return np.nan
    rng = np.random.default_rng(int(seed))
    d_boot = np.empty(B, dtype=float)
    for b in range(B):
        x = _model_sample(model, n, params, rng)
        fit = _fit_candidate_model(model, x, seed=seed + 1000 + b, floor_db=floor_db)
        if fit.get("status") != "OK":
            d_boot[b] = np.nan
            continue
        p = fit["params"]
        xs = np.sort(x)
        d_boot[b] = _ks_statistic_from_cdf(xs, _model_cdf(model, xs, p))
    good = np.isfinite(d_boot)
    if not np.any(good):
        return np.nan
    return float(np.mean(d_boot[good] >= d_obs))


def _fit_candidate_model(
    model: str,
    values_db: NDArray[np.float64],
    seed: int = 0,
    floor_db: float | None = None,
) -> dict[str, Any]:
    if model == "normal_db":
        return _fit_normal_model(values_db)
    if model == "gmm2_db":
        return _fit_gmm2_model(values_db, seed=seed)
    if model == "truncnorm_db":
        return _fit_truncnorm_model(values_db)
    if model == "spike_slab_db":
        return _fit_spike_slab_model(values_db, floor_db=floor_db, seed=seed)
    raise ValueError(f"unknown model: {model}")


def _evaluate_candidate_model(
    model: str,
    values_db: NDArray[np.float64],
    fit: dict[str, Any],
    n_total: int,
    bootstrap_B: int = 0,
    seed: int = 0,
    floor_db: float | None = None,
) -> dict[str, Any]:
    if fit.get("status") != "OK":
        return {
            "status": "FAIL",
            "warning": str(fit.get("warning", "fit failed")),
            "k": int(fit.get("k", 0)),
            "loglik": -np.inf,
            "aic": np.inf,
            "bic": np.inf,
            "qq_r": np.nan,
            "qq_r2": np.nan,
            "ks_D": np.nan,
            "ks_p_raw": np.nan,
            "ks_p_boot": np.nan,
            "params": {},
        }
    p = dict(fit["params"])
    ll = float(fit["loglik"])
    k = int(fit["k"])
    n = len(values_db)
    aic = float(2.0 * k - 2.0 * ll)
    bic = float(np.log(max(n, 1)) * k - 2.0 * ll)

    xs = np.sort(values_db)
    cdf_vals = _model_cdf(model, xs, p)
    ks_D = _ks_statistic_from_cdf(xs, cdf_vals)
    ks_p_raw = float(stats.kstwo.sf(ks_D, max(n, 1))) if np.isfinite(ks_D) else np.nan
    probs = (np.arange(1, n + 1, dtype=float) - 0.5) / float(n)
    try:
        q_theory = model_quantiles_db(model, p, probs)
        if np.std(q_theory) <= 0.0 or np.std(xs) <= 0.0:
            qq_r = 1.0
        else:
            qq_r = float(np.corrcoef(q_theory, xs)[0, 1])
    except Exception:
        qq_r = np.nan
    qq_r2 = float(qq_r**2) if np.isfinite(qq_r) else np.nan
    ks_p_boot = _bootstrap_ks_p(model, p, n, ks_D, int(max(0, bootstrap_B)), seed=seed, floor_db=floor_db)
    warn = bool((np.isfinite(qq_r) and qq_r < 0.98) or (np.isfinite(ks_p_boot) and ks_p_boot < 0.05))
    return {
        "status": "OK",
        "warning": warn,
        "k": k,
        "loglik": ll,
        "aic": aic,
        "bic": bic,
        "qq_r": qq_r,
        "qq_r2": qq_r2,
        "ks_D": ks_D,
        "ks_p_raw": ks_p_raw,
        "ks_p_boot": ks_p_boot,
        "params": p,
        "n": int(n_total),
        "n_fit": int(n),
    }


def floor_pinned_exclusion_mask(
    values_db: list[float] | NDArray[np.float64],
    floor_db: float | None = None,
    pinned_tol_db: float = 0.5,
    pinned_min_fraction: float = 0.15,
) -> dict[str, Any]:
    """Build exclusion mask for floor-limited / pinned samples."""

    v = np.asarray(values_db, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {
            "values": v,
            "mask_excluded": np.zeros(0, dtype=bool),
            "mask_floor": np.zeros(0, dtype=bool),
            "mask_pinned": np.zeros(0, dtype=bool),
            "floor_db": np.nan,
            "pinned_center_db": np.nan,
            "pinned_fraction": 0.0,
            "pinned_ratio": 0.0,
            "floor_ratio": 0.0,
            "point_mass_ratio": 0.0,
            "point_mass_kind": "none",
        }

    tol = max(float(pinned_tol_db), 1e-6)
    mask_floor = np.zeros(n, dtype=bool)
    if floor_db is not None and np.isfinite(floor_db):
        mask_floor = np.abs(v - float(floor_db)) <= tol

    mode_center, mode_frac = _mode_center_db(v, bin_width_db=min(0.5, tol))
    mask_pinned = np.zeros(n, dtype=bool)
    pinned_center = np.nan
    if np.isfinite(mode_center) and mode_frac >= float(pinned_min_fraction):
        mask_pinned = np.abs(v - mode_center) <= tol
        pinned_center = float(mode_center)
    mask_excluded = mask_floor | mask_pinned
    if np.all(mask_excluded):
        # Keep at least some samples for post-filter fit.
        mask_excluded = mask_floor.copy()
    floor_count = int(np.sum(mask_floor))
    pinned_count = int(np.sum(mask_pinned))
    excluded_count = int(np.sum(mask_excluded))
    floor_ratio = float(floor_count / max(n, 1))
    pinned_ratio = float(pinned_count / max(n, 1))
    point_mass_ratio = float(excluded_count / max(n, 1))
    point_mass_kind = "none"
    if floor_count > 0 and floor_count >= pinned_count:
        point_mass_kind = "floor_spike"
    elif pinned_count > 0:
        if np.isfinite(pinned_center) and pinned_center >= float(np.quantile(v, 0.9)):
            point_mass_kind = "right_censored_or_upper_spike"
        else:
            point_mass_kind = "mode_spike"
    return {
        "values": v,
        "mask_excluded": mask_excluded,
        "mask_floor": mask_floor,
        "mask_pinned": mask_pinned,
        "floor_db": float(floor_db) if floor_db is not None and np.isfinite(floor_db) else np.nan,
        "pinned_center_db": pinned_center,
        "pinned_fraction": float(mode_frac),
        "pinned_ratio": pinned_ratio,
        "floor_ratio": floor_ratio,
        "point_mass_ratio": point_mass_ratio,
        "point_mass_kind": point_mass_kind,
    }


def gof_model_selection_db(
    values_db: list[float] | NDArray[np.float64],
    min_n: int = 20,
    bootstrap_B: int = 200,
    seed: int = 0,
    floor_db: float | None = None,
    pinned_tol_db: float = 0.5,
) -> dict[str, Any]:
    """GOF with model selection for XPD[dB].

    Candidates:
      1) normal_db
      2) gmm2_db
      3) truncnorm_db
      4) spike_slab_db

    Includes pre/post single-normal diagnostics and exclusion of floor/pinned samples.
    """

    v_in = np.asarray(values_db, dtype=float)
    v_in = v_in[np.isfinite(v_in)]
    n_total = int(len(v_in))
    pre_normal = gof_normal_db(v_in, min_n=min_n, bootstrap_B=bootstrap_B, seed=seed)

    excl = floor_pinned_exclusion_mask(v_in, floor_db=floor_db, pinned_tol_db=pinned_tol_db)
    v = np.asarray(excl["values"], dtype=float)
    mask_excluded = np.asarray(excl["mask_excluded"], dtype=bool)
    v_fit = v[~mask_excluded]

    post_normal = gof_normal_db(v_fit, min_n=min_n, bootstrap_B=bootstrap_B, seed=seed + 97)
    n_fit = int(len(v_fit))
    n_excluded = int(np.sum(mask_excluded))
    floor_count = int(np.sum(np.asarray(excl["mask_floor"], dtype=bool)))
    pinned_count = int(np.sum(np.asarray(excl["mask_pinned"], dtype=bool)))
    floor_ratio = float(excl.get("floor_ratio", floor_count / max(n_total, 1)))
    pinned_ratio = float(excl.get("pinned_ratio", pinned_count / max(n_total, 1)))
    point_mass_ratio = float(excl.get("point_mass_ratio", n_excluded / max(n_total, 1)))
    point_mass_kind = str(excl.get("point_mass_kind", "none"))
    if n_fit < int(min_n):
        diag = "INSUFFICIENT_N"
        if point_mass_ratio >= 0.5:
            diag = "POINT_MASS_DOMINANT_INSUFFICIENT"
        return {
            "status": "INSUFFICIENT",
            "n": n_total,
            "n_total": n_total,
            "n_fit": n_fit,
            "n_excluded": n_excluded,
            "excluded_floor_count": floor_count,
            "excluded_pinned_count": pinned_count,
            "floor_ratio": floor_ratio,
            "pinned_ratio": pinned_ratio,
            "point_mass_ratio": point_mass_ratio,
            "point_mass_kind": point_mass_kind,
            "floor_db": excl["floor_db"],
            "pinned_center_db": excl["pinned_center_db"],
            "pinned_fraction": excl["pinned_fraction"],
            "pre_normal": pre_normal,
            "post_normal": post_normal,
            "best_model": "NA",
            "models": {},
            "warning": True,
            "single_normal_fail": True,
            "alternative_improved": False,
            "diagnostic_class": diag,
            "residual_borderline": False,
            "gof_continuous": {"ks_p_boot": np.nan, "qq_r": np.nan},
        }

    candidates = ["normal_db", "gmm2_db", "truncnorm_db", "spike_slab_db"]
    model_stats: dict[str, dict[str, Any]] = {}
    for i, model in enumerate(candidates):
        fit = _fit_candidate_model(model, v_fit, seed=seed + i * 13, floor_db=floor_db)
        # Full bootstrap only for normal here; non-normal gets raw first, then best model gets boot.
        B = int(bootstrap_B) if model == "normal_db" else 0
        model_stats[model] = _evaluate_candidate_model(
            model,
            v_fit,
            fit,
            n_total=n_total,
            bootstrap_B=B,
            seed=seed + i * 13,
            floor_db=floor_db,
        )

    ok_models = {k: m for k, m in model_stats.items() if m.get("status") == "OK" and np.isfinite(float(m.get("bic", np.inf)))}
    if ok_models:
        best_model = sorted(ok_models.items(), key=lambda kv: (float(kv[1]["bic"]), float(kv[1]["aic"])))[0][0]
    else:
        best_model = "normal_db"

    if model_stats.get(best_model, {}).get("status") == "OK" and not np.isfinite(float(model_stats[best_model].get("ks_p_boot", np.nan))):
        fit_best = _fit_candidate_model(best_model, v_fit, seed=seed + 700, floor_db=floor_db)
        b_best = int(bootstrap_B) if best_model == "normal_db" else int(min(max(int(bootstrap_B), 0), 60))
        model_stats[best_model] = _evaluate_candidate_model(
            best_model,
            v_fit,
            fit_best,
            n_total=n_total,
            bootstrap_B=b_best,
            seed=seed + 700,
            floor_db=floor_db,
        )

    best = model_stats.get(best_model, {})
    single_normal_fail = bool(pre_normal.get("status") == "OK" and pre_normal.get("warning", True))
    best_ok = bool(best.get("status") == "OK")
    best_qq = float(best.get("qq_r", np.nan))
    best_ks_boot = float(best.get("ks_p_boot", np.nan))
    pass_strict = bool(np.isfinite(best_qq) and np.isfinite(best_ks_boot) and best_qq >= 0.98 and best_ks_boot >= 0.05)
    best_pass = bool(best_ok and pass_strict)
    bic_norm = float(model_stats.get("normal_db", {}).get("bic", np.inf))
    bic_best = float(best.get("bic", np.inf))
    alternative_improved = bool(best_model != "normal_db" and np.isfinite(bic_norm) and np.isfinite(bic_best) and (bic_norm - bic_best > 10.0))
    residual_borderline = bool(best_ok and not pass_strict)
    if not best_ok:
        status = "FAIL_MODEL"
        warning = True
        diagnostic_class = "MODEL_FAILURE"
    elif pass_strict:
        status = "PASS_ALTERNATIVE" if (best_model != "normal_db" and single_normal_fail) else "PASS"
        warning = False
        diagnostic_class = "MODEL_PASS_ALTERNATIVE" if status == "PASS_ALTERNATIVE" else "MODEL_PASS"
    elif best_model != "normal_db" and alternative_improved:
        status = "PASS_ALTERNATIVE"
        warning = True
        diagnostic_class = "MODEL_PASS_ALTERNATIVE"
    else:
        status = "FAIL_MODEL"
        warning = True
        diagnostic_class = "MODEL_FAILURE"
    if point_mass_ratio >= 0.5 and status != "FAIL_MODEL":
        diagnostic_class = "POINT_MASS_DOMINANT"
    model_reason = "single-normal adequate"
    if single_normal_fail and alternative_improved and best_pass:
        model_reason = "single-normal fail; alternative model selected and passes GOF"
    elif single_normal_fail and best_model != "normal_db":
        model_reason = "single-normal fail; alternative selected but GOF still borderline"
    elif single_normal_fail:
        model_reason = "single-normal fail; no better candidate found"
    elif best_model != "normal_db":
        model_reason = "alternative selected by information criterion"

    return {
        "status": status,
        "warning": warning,
        "n": n_total,
        "n_total": n_total,
        "n_fit": n_fit,
        "n_excluded": n_excluded,
        "excluded_floor_count": floor_count,
        "excluded_pinned_count": pinned_count,
        "floor_ratio": floor_ratio,
        "pinned_ratio": pinned_ratio,
        "point_mass_ratio": point_mass_ratio,
        "point_mass_kind": point_mass_kind,
        "floor_db": excl["floor_db"],
        "pinned_center_db": excl["pinned_center_db"],
        "pinned_fraction": excl["pinned_fraction"],
        "pre_normal": pre_normal,
        "post_normal": post_normal,
        "best_model": best_model,
        "best_metrics": best,
        "models": model_stats,
        "single_normal_fail": single_normal_fail,
        "alternative_improved": alternative_improved,
        "residual_borderline": residual_borderline,
        "diagnostic_class": diagnostic_class,
        "gof_continuous": {
            "ks_p_boot": best_ks_boot,
            "qq_r": best_qq,
        },
        "model_reason": model_reason,
    }


def coupling_eps_from_params(cross_pol_leakage_db: float, axial_ratio_db: float, enable_coupling: bool = True) -> float:
    """Return coupling epsilon that matches rt_core.antenna.Antenna._coupling_matrix."""

    if not enable_coupling:
        return 0.0
    leak = 10.0 ** (-float(cross_pol_leakage_db) / 20.0)
    ar = 10.0 ** (float(axial_ratio_db) / 20.0)
    ar_leak = abs((ar - 1.0) / (ar + 1.0))
    return float(np.clip(leak + ar_leak, 0.0, 0.49))


def estimate_leakage_floor_db(
    tx_cross_pol_leakage_db: float,
    rx_cross_pol_leakage_db: float,
    tx_axial_ratio_db: float = 0.0,
    rx_axial_ratio_db: float = 0.0,
    tx_enable_coupling: bool = True,
    rx_enable_coupling: bool = True,
) -> dict[str, float]:
    """Predict XPD floor from Tx/Rx coupling parameters.

    floor_db ~= 20*log10(1/(eps_tx + eps_rx + 1e-15))
    """

    eps_tx = coupling_eps_from_params(
        cross_pol_leakage_db=tx_cross_pol_leakage_db,
        axial_ratio_db=tx_axial_ratio_db,
        enable_coupling=tx_enable_coupling,
    )
    eps_rx = coupling_eps_from_params(
        cross_pol_leakage_db=rx_cross_pol_leakage_db,
        axial_ratio_db=rx_axial_ratio_db,
        enable_coupling=rx_enable_coupling,
    )
    floor_db = float(20.0 * np.log10(1.0 / (eps_tx + eps_rx + 1e-15)))
    return {"eps_tx": eps_tx, "eps_rx": eps_rx, "xpd_floor_db": floor_db}


def estimate_leakage_floor_from_antenna_config(antenna_config: dict[str, Any]) -> dict[str, float]:
    """Convenience wrapper to estimate leakage floor from runner antenna_config dict."""

    en = bool(antenna_config.get("enable_coupling", True))
    return estimate_leakage_floor_db(
        tx_cross_pol_leakage_db=float(antenna_config.get("tx_cross_pol_leakage_db", 35.0)),
        rx_cross_pol_leakage_db=float(antenna_config.get("rx_cross_pol_leakage_db", 35.0)),
        tx_axial_ratio_db=float(antenna_config.get("tx_axial_ratio_db", 0.0)),
        rx_axial_ratio_db=float(antenna_config.get("rx_axial_ratio_db", 0.0)),
        tx_enable_coupling=en,
        rx_enable_coupling=en,
    )


def leakage_limited_summary(
    xpd_samples_db: list[float] | NDArray[np.float64],
    xpd_floor_db: float,
    median_tolerance_db: float = 1.0,
    sigma_threshold_db: float = 1.0,
) -> dict[str, float | bool]:
    """Return leakage-floor proximity and trigger flag for XPD samples."""

    v = np.asarray(xpd_samples_db, dtype=float)
    if len(v) == 0:
        return {
            "median_xpd_db": np.nan,
            "sigma_xpd_db": np.nan,
            "delta_floor_db": np.nan,
            "is_leakage_limited": False,
        }
    median_xpd_db = float(np.median(v))
    sigma_xpd_db = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    delta_floor_db = float(abs(median_xpd_db - float(xpd_floor_db)))
    is_limited = bool(delta_floor_db < float(median_tolerance_db) and sigma_xpd_db < float(sigma_threshold_db))
    return {
        "median_xpd_db": median_xpd_db,
        "sigma_xpd_db": sigma_xpd_db,
        "delta_floor_db": delta_floor_db,
        "is_leakage_limited": is_limited,
    }


def censoring_profile_by_bucket(
    samples: list[dict[str, Any]],
    key_fields: list[str],
    value_key: str = "xpd_db",
    floor_db: float | None = None,
    pinned_tol_db: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Estimate floor/pinned point-mass ratios per conditional bucket."""

    buckets: dict[str, list[float]] = {}
    for s in samples:
        key = "|".join(str(s.get(k, "NA")) for k in key_fields)
        buckets.setdefault(key, []).append(float(s.get(value_key, np.nan)))

    out: dict[str, dict[str, Any]] = {}
    for k, vals in buckets.items():
        e = floor_pinned_exclusion_mask(vals, floor_db=floor_db, pinned_tol_db=pinned_tol_db)
        v = np.asarray(e["values"], dtype=float)
        m_ex = np.asarray(e["mask_excluded"], dtype=bool)
        n_total = int(len(v))
        n_fit = int(np.sum(~m_ex))
        out[k] = {
            "n_total": n_total,
            "n_fit": n_fit,
            "n_excluded": int(np.sum(m_ex)),
            "floor_ratio": float(e.get("floor_ratio", 0.0)),
            "pinned_ratio": float(e.get("pinned_ratio", 0.0)),
            "point_mass_ratio": float(e.get("point_mass_ratio", 0.0)),
            "point_mass_kind": str(e.get("point_mass_kind", "none")),
            "floor_db": float(e.get("floor_db", np.nan)),
            "pinned_center_db": float(e.get("pinned_center_db", np.nan)),
            "pinned_fraction": float(e.get("pinned_fraction", 0.0)),
        }
    return out

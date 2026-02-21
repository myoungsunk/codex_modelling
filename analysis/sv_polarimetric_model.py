"""Stochastic SV-style polarimetric channel model generator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from analysis.xpd_stats import conditional_fit, floor_pinned_exclusion_mask, pathwise_xpd


def _matrix_from_path(path: dict[str, Any], matrix_source: str = "A") -> NDArray[np.complex128]:
    use_j = str(matrix_source).upper() == "J" and "J_f" in path
    return np.asarray(path["J_f"] if use_j else path["A_f"], dtype=np.complex128)


def ray_matrix_from_kappa(kappa: float, rng: np.random.Generator) -> NDArray[np.complex128]:
    """Generate a 2x2 ray matrix with XPR/XPD control via kappa."""

    kk = max(float(kappa), 1e-6)
    inv = 1.0 / np.sqrt(kk)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=4)
    h = np.array(
        [
            [np.exp(1j * phi[0]), inv * np.exp(1j * phi[1])],
            [inv * np.exp(1j * phi[2]), np.exp(1j * phi[3])],
        ],
        dtype=np.complex128,
    )
    return h


def _sample_kappa_linear(parity: str, parity_fit: dict[str, dict[str, float]], rng: np.random.Generator) -> float:
    p = parity_fit.get(parity) or parity_fit.get("NA") or {"mu": 10.0, "sigma": 3.0}
    mu = float(p.get("mu", 10.0))
    sigma = max(float(p.get("sigma", 0.0)), 0.0)
    xpd_db = float(rng.normal(mu, sigma)) if sigma > 0 else mu
    return float(10.0 ** (xpd_db / 10.0))


def _resolve_num_synth_paths(
    num_rays: int,
    num_paths_mode: str,
    delay_samples_s: NDArray[np.float64],
    rt_case_path_counts: NDArray[np.int64] | None,
    rng: np.random.Generator,
) -> int:
    mode = str(num_paths_mode).strip().lower()
    if mode == "match_rt_total":
        return max(int(len(delay_samples_s)), 1)
    if mode == "sample_case_hist":
        if rt_case_path_counts is None or len(rt_case_path_counts) == 0:
            return max(int(num_rays), 1)
        c = np.asarray(rt_case_path_counts, dtype=int)
        c = c[c > 0]
        if len(c) == 0:
            return max(int(num_rays), 1)
        picks = rng.choice(c, size=len(c), replace=True)
        return max(int(np.sum(picks)), 1)
    return max(int(num_rays), 1)


def _resolve_sampling_schedule(
    num_rays: int,
    num_paths_mode: str,
    delay_samples_s: NDArray[np.float64],
    rt_case_path_counts: NDArray[np.int64] | None,
    rt_path_scenarios: list[str] | NDArray[np.str_] | None,
    rt_path_cases: list[str] | NDArray[np.str_] | None,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """Return per-synthetic-path scenario/case targets for weighted sampling."""

    mode = str(num_paths_mode).strip().lower()
    if mode in {"match_rt_per_case", "match_rt_per_scenario", "match_rt_total"}:
        sids = [str(x) for x in (list(rt_path_scenarios) if rt_path_scenarios is not None else [])]
        cids = [str(x) for x in (list(rt_path_cases) if rt_path_cases is not None else [])]
        n_obs = int(min(len(delay_samples_s), len(sids), len(cids))) if len(sids) and len(cids) else int(len(delay_samples_s))
        if n_obs > 0:
            sids = (sids[:n_obs] if len(sids) >= n_obs else ["NA"] * n_obs)
            cids = (cids[:n_obs] if len(cids) >= n_obs else ["NA"] * n_obs)
            if mode == "match_rt_per_case":
                out = [(sids[i], cids[i]) for i in range(n_obs)]
                rng.shuffle(out)
                return out
            if mode == "match_rt_per_scenario":
                out = [(sids[i], "ALL") for i in range(n_obs)]
                rng.shuffle(out)
                return out
            if mode == "match_rt_total":
                out = [("ALL", "ALL")] * n_obs
                return out
    n_paths = _resolve_num_synth_paths(
        num_rays=num_rays,
        num_paths_mode=num_paths_mode,
        delay_samples_s=delay_samples_s,
        rt_case_path_counts=rt_case_path_counts,
        rng=rng,
    )
    return [("ALL", "ALL")] * int(max(n_paths, 1))


def _sample_index_for_target(
    target_sid: str,
    target_cid: str,
    idx_by_case: dict[tuple[str, str], NDArray[np.int64]],
    idx_by_scenario: dict[str, NDArray[np.int64]],
    n_total: int,
    rng: np.random.Generator,
) -> int:
    if target_cid != "ALL":
        key = (target_sid, target_cid)
        arr = idx_by_case.get(key)
        if arr is not None and len(arr) > 0:
            return int(arr[int(rng.integers(0, len(arr)))])
    if target_sid != "ALL":
        arr2 = idx_by_scenario.get(target_sid)
        if arr2 is not None and len(arr2) > 0:
            return int(arr2[int(rng.integers(0, len(arr2)))])
    return int(rng.integers(0, max(n_total, 1)))


def _subband_index_for_freq(k: int, subbands: list[tuple[int, int]]) -> int:
    for bidx, (s, e) in enumerate(subbands):
        if int(s) <= int(k) < int(e):
            return int(bidx)
    return int(max(len(subbands) - 1, 0))


def _sample_xpd_from_profile(
    rng: np.random.Generator,
    mu_db: float,
    sigma_db: float,
    lower_db: float,
    upper_db: float,
    censor: dict[str, Any] | None,
) -> float:
    """Sample XPD[dB] with optional point-mass component."""

    if censor is not None:
        pinned_ratio = float(max(0.0, min(1.0, censor.get("pinned_ratio", 0.0))))
        floor_ratio = float(max(0.0, min(1.0 - pinned_ratio, censor.get("floor_ratio", 0.0))))
        u = float(rng.random())
        pinned_center = float(censor.get("pinned_center_db", np.nan))
        floor_db = float(censor.get("floor_db", np.nan))
        if u < pinned_ratio and np.isfinite(pinned_center):
            return float(np.clip(pinned_center, lower_db, upper_db))
        if u < (pinned_ratio + floor_ratio) and np.isfinite(floor_db):
            return float(np.clip(floor_db, lower_db, upper_db))
    return _truncated_normal_scalar(
        rng,
        mu=float(mu_db),
        sigma=float(max(sigma_db, 1e-6)),
        lower=float(lower_db),
        upper=float(upper_db),
    )


def _ks2_with_cap(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    cap: int,
    seed: int,
) -> tuple[float, float, int]:
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, 0
    cc = int(max(2, cap))
    rng = np.random.default_rng(int(seed))
    xs = x if len(x) <= cc else np.asarray(rng.choice(x, size=cc, replace=False), dtype=float)
    ys = y if len(y) <= cc else np.asarray(rng.choice(y, size=cc, replace=False), dtype=float)
    ks = scipy_stats.ks_2samp(xs, ys, alternative="two-sided", method="auto")
    return float(ks.statistic), float(ks.pvalue), int(min(len(xs), len(ys)))


def _truncated_normal_scalar(
    rng: np.random.Generator,
    mu: float,
    sigma: float,
    lower: float,
    upper: float,
) -> float:
    lo = float(min(lower, upper))
    hi = float(max(lower, upper))
    if hi <= lo:
        return float(mu)
    sig = float(max(sigma, 1e-6))
    a = (lo - mu) / sig
    b = (hi - mu) / sig
    return float(scipy_stats.truncnorm.rvs(a, b, loc=mu, scale=sig, random_state=rng))


def offdiag_phases(
    paths: list[dict[str, Any]],
    matrix_source: str = "A",
    per_ray_sampling: bool = True,
    center_freq_index: int | None = None,
    common_phase_removed: bool = True,
) -> NDArray[np.float64]:
    vals: list[float] = []
    for p in paths:
        m = _matrix_from_path(p, matrix_source=matrix_source)
        if m.ndim != 3 or m.shape[1:] != (2, 2):
            continue
        mats = m
        if per_ray_sampling:
            idx = int(m.shape[0] // 2 if center_freq_index is None else np.clip(center_freq_index, 0, m.shape[0] - 1))
            mats = m[idx : idx + 1]
        for mm in mats:
            x = np.asarray(mm, dtype=np.complex128)
            if common_phase_removed:
                det = complex(np.linalg.det(x))
                if abs(det) > 1e-15:
                    phi0 = 0.5 * float(np.angle(det))
                else:
                    phi0 = float(np.angle(np.trace(x) + 1e-15))
                x = x * np.exp(-1j * phi0)
            vals.append(float(np.angle(x[0, 1])))
            vals.append(float(np.angle(x[1, 0])))
    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def kuiper_uniform_test(
    angles_rad: NDArray[np.float64] | list[float],
    bootstrap_B: int = 500,
    seed: int = 0,
    tie_jitter_eps: float = 1e-12,
) -> dict[str, float]:
    """Rotation-invariant circular-uniformity test via Kuiper statistic + bootstrap p-value."""

    a = np.asarray(angles_rad, dtype=float)
    a = a[np.isfinite(a)]
    n = int(len(a))
    if n < 2:
        return {"n": n, "V": np.nan, "p_boot": np.nan}

    rng = np.random.default_rng(int(seed))
    u = (a % (2.0 * np.pi)) / (2.0 * np.pi)
    if len(np.unique(u)) < len(u):
        eps = float(max(tie_jitter_eps, 0.0))
        u = np.mod(u + rng.uniform(-eps, eps, size=n), 1.0)
    u = np.sort(u)
    i = np.arange(1, n + 1, dtype=float)
    d_plus = float(np.max(i / n - u))
    d_minus = float(np.max(u - (i - 1.0) / n))
    V = float(d_plus + d_minus)

    B = int(max(0, bootstrap_B))
    if B <= 0:
        return {"n": n, "V": V, "p_boot": np.nan}

    v_boot = np.zeros(B, dtype=float)
    for b in range(B):
        ub = np.sort(rng.uniform(0.0, 1.0, size=n))
        dbp = float(np.max(i / n - ub))
        dbm = float(np.max(ub - (i - 1.0) / n))
        v_boot[b] = dbp + dbm
    p_boot = float(np.mean(v_boot >= V))
    return {"n": n, "V": V, "p_boot": p_boot}


def generate_synthetic_paths(
    f_hz: NDArray[np.float64],
    num_rays: int,
    delay_samples_s: NDArray[np.float64],
    power_samples: NDArray[np.float64],
    parity_probs: dict[str, float],
    parity_fit: dict[str, dict[str, float]],
    parity_slope_model: dict[str, dict[str, Any]] | None = None,
    parity_subband_fit: dict[str, dict[str, dict[str, float]]] | None = None,
    parity_censoring: dict[str, dict[str, Any]] | None = None,
    parity_subband_censoring: dict[str, dict[str, Any]] | None = None,
    subbands: list[tuple[int, int]] | None = None,
    kappa_freq_mode: str = "piecewise_constant",
    incidence_probs: dict[str, float] | None = None,
    incidence_probs_by_parity: dict[str, dict[str, float]] | None = None,
    parity_incidence_fit: dict[str, dict[str, float]] | None = None,
    parity_incidence_slope_model: dict[str, dict[str, Any]] | None = None,
    empirical_xpd_by_condition: dict[str, list[float]] | None = None,
    matrix_source: str = "A",
    xpd_freq_noise_sigma_db: float = 0.0,
    sample_slope: bool = False,
    slope_sigma_db_per_hz: float = 0.0,
    kappa_min: float = 1e-6,
    kappa_max: float = 1e12,
    num_paths_mode: str = "fixed",
    rt_case_path_counts: NDArray[np.int64] | None = None,
    rt_path_scenarios: list[str] | NDArray[np.str_] | None = None,
    rt_path_cases: list[str] | NDArray[np.str_] | None = None,
    return_diagnostics: bool = False,
    seed: int = 0,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate synthetic per-ray paths with 2x2 matrices and delays."""

    rng = np.random.default_rng(seed)
    freq = np.asarray(f_hz, dtype=float)
    n_f = len(freq)

    d = np.asarray(delay_samples_s, dtype=float)
    p = np.asarray(power_samples, dtype=float)
    if len(d) == 0:
        d = np.array([0.0], dtype=float)
    if len(p) == 0:
        p = np.array([1.0], dtype=float)
    p = np.maximum(p, 1e-15)
    p = p / np.sum(p)

    parity_labels = ["odd", "even"]
    probs = np.array([parity_probs.get("odd", 0.5), parity_probs.get("even", 0.5)], dtype=float)
    if probs.sum() <= 0:
        probs[:] = 0.5
    probs /= probs.sum()

    schedule = _resolve_sampling_schedule(
        num_rays=num_rays,
        num_paths_mode=num_paths_mode,
        delay_samples_s=d,
        rt_case_path_counts=rt_case_path_counts,
        rt_path_scenarios=rt_path_scenarios,
        rt_path_cases=rt_path_cases,
        rng=rng,
    )
    n_paths = int(max(len(schedule), 1))

    paths: list[dict[str, Any]] = []
    kappa_total = 0
    kappa_trunc_count = 0
    xpd_min_db = float(10.0 * np.log10(max(float(kappa_min), 1e-15)))
    xpd_max_db = float(10.0 * np.log10(max(float(kappa_max), 1e-15)))

    inc_labels: list[str] = []
    inc_probs_global = np.array([], dtype=float)
    if incidence_probs:
        inc_labels = [str(k) for k in incidence_probs.keys()]
        inc_probs_global = np.asarray([float(incidence_probs[k]) for k in inc_labels], dtype=float)
        if np.sum(inc_probs_global) > 0:
            inc_probs_global = inc_probs_global / np.sum(inc_probs_global)
        else:
            inc_labels, inc_probs_global = [], np.array([], dtype=float)

    npair = int(min(len(d), len(np.asarray(power_samples, dtype=float))))
    pwr = np.asarray(power_samples, dtype=float)
    if npair > 0:
        pair_delay = np.asarray(d[:npair], dtype=float)
        pair_power = np.asarray(pwr[:npair], dtype=float)
    else:
        pair_delay = np.asarray(d, dtype=float)
        pair_power = np.asarray(pwr, dtype=float)
    sids_arr = np.asarray(list(rt_path_scenarios) if rt_path_scenarios is not None else ["NA"] * len(pair_delay), dtype=object)
    cids_arr = np.asarray(list(rt_path_cases) if rt_path_cases is not None else ["NA"] * len(pair_delay), dtype=object)
    n_lab = int(min(len(pair_delay), len(sids_arr), len(cids_arr)))
    idx_by_case: dict[tuple[str, str], list[int]] = {}
    idx_by_scenario: dict[str, list[int]] = {}
    for i in range(n_lab):
        sid_i = str(sids_arr[i])
        cid_i = str(cids_arr[i])
        idx_by_case.setdefault((sid_i, cid_i), []).append(int(i))
        idx_by_scenario.setdefault(sid_i, []).append(int(i))
    idx_by_case_np = {k: np.asarray(v, dtype=np.int64) for k, v in idx_by_case.items()}
    idx_by_scenario_np = {k: np.asarray(v, dtype=np.int64) for k, v in idx_by_scenario.items()}

    src = str(matrix_source).upper()
    empirical_src: dict[str, NDArray[np.float64]] = {}
    empirical_pool: dict[str, NDArray[np.float64]] = {}
    empirical_idx: dict[str, int] = {}
    if empirical_xpd_by_condition:
        for k, vals in empirical_xpd_by_condition.items():
            vv = np.asarray(vals, dtype=float)
            vv = vv[np.isfinite(vv)]
            if len(vv) > 0:
                empirical_src[str(k)] = vv
                empirical_pool[str(k)] = rng.permutation(vv)
                empirical_idx[str(k)] = 0

    def _draw_empirical(key: str) -> float | None:
        kk = str(key)
        if kk not in empirical_pool:
            return None
        i = int(empirical_idx.get(kk, 0))
        arr = empirical_pool[kk]
        if i >= len(arr):
            arr = rng.permutation(empirical_src[kk])
            empirical_pool[kk] = arr
            i = 0
        empirical_idx[kk] = i + 1
        return float(arr[i])

    parity_schedule: list[str] | None = None
    if "odd" in empirical_src and "even" in empirical_src:
        n_odd_obs = int(len(empirical_src["odd"]))
        n_even_obs = int(len(empirical_src["even"]))
        den = max(n_odd_obs + n_even_obs, 1)
        n_odd = int(round(n_paths * n_odd_obs / den))
        n_odd = int(np.clip(n_odd, 0, n_paths))
        n_even = int(max(n_paths - n_odd, 0))
        parity_schedule = ["odd"] * n_odd + ["even"] * n_even
        rng.shuffle(parity_schedule)

    for _i in range(n_paths):
        sid_target, cid_target = schedule[_i] if _i < len(schedule) else ("ALL", "ALL")
        if npair > 0:
            idx = _sample_index_for_target(
                target_sid=str(sid_target),
                target_cid=str(cid_target),
                idx_by_case=idx_by_case_np,
                idx_by_scenario=idx_by_scenario_np,
                n_total=npair,
                rng=rng,
            )
            tau = float(pair_delay[idx])
            ray_power = float(max(pair_power[idx], 1e-15))
        else:
            idx_d = int(rng.integers(0, len(d)))
            idx_p = int(rng.choice(np.arange(len(p)), p=p))
            tau = float(d[idx_d])
            ray_power = float(max(power_samples[idx_p], 1e-15)) if len(power_samples) > idx_p else float(max(np.mean(power_samples), 1e-15))

        if parity_schedule is not None and _i < len(parity_schedule):
            parity = str(parity_schedule[_i])
        else:
            parity = str(rng.choice(parity_labels, p=probs))
        incidence_bin = "NA"
        if incidence_probs_by_parity and parity in incidence_probs_by_parity and incidence_probs_by_parity[parity]:
            bins_p = [str(k) for k in incidence_probs_by_parity[parity].keys()]
            pb = np.asarray([float(incidence_probs_by_parity[parity][k]) for k in bins_p], dtype=float)
            if np.sum(pb) > 0:
                pb /= np.sum(pb)
                incidence_bin = str(rng.choice(bins_p, p=pb))
        elif len(inc_labels) > 0 and len(inc_probs_global) > 0:
            incidence_bin = str(rng.choice(inc_labels, p=inc_probs_global))

        fit_key = f"{parity}|{incidence_bin}" if incidence_bin != "NA" else parity
        pfit = (
            (parity_incidence_fit or {}).get(fit_key)
            or parity_fit.get(parity)
            or parity_fit.get("NA")
            or {"mu": 10.0, "sigma": 3.0}
        )
        mu = float(pfit.get("mu", 10.0))
        sigma = max(float(pfit.get("sigma", 0.0)), 0.0)
        slope = 0.0
        fc = float(np.mean(freq)) if n_f > 0 else 0.0
        if parity_incidence_slope_model is not None and fit_key in parity_incidence_slope_model:
            slope = float(parity_incidence_slope_model[fit_key].get("mu1_db_per_hz", 0.0))
            fc = float(parity_incidence_slope_model[fit_key].get("fc_hz", fc))
        elif parity_slope_model is not None and parity in parity_slope_model:
            slope = float(parity_slope_model[parity].get("mu1_db_per_hz", 0.0))
            fc = float(parity_slope_model[parity].get("fc_hz", fc))
        if sample_slope:
            slope_sig = float(max(slope_sigma_db_per_hz, 0.0))
            if slope_sig > 0.0:
                slope = float(rng.normal(slope, slope_sig))
        if n_f > 0:
            df_max = float(np.max(np.abs(freq - fc)))
            if df_max > 0.0:
                slope_lim = float(0.95 * (xpd_max_db - xpd_min_db) / df_max)
                slope = float(np.clip(slope, -slope_lim, slope_lim))

        xpd_pick = _draw_empirical(fit_key)
        if xpd_pick is None:
            xpd_pick = _draw_empirical(parity)
        if xpd_pick is None:
            xpd_pick = _draw_empirical("ALL")
        par_cens = (parity_censoring or {}).get(parity) if parity_censoring is not None else None
        if xpd_pick is not None:
            xpd0_db = _sample_xpd_from_profile(
                rng=rng,
                mu_db=float(xpd_pick),
                sigma_db=1e-6,
                lower_db=xpd_min_db,
                upper_db=xpd_max_db,
                censor=par_cens,
            )
        else:
            xpd0_db = _sample_xpd_from_profile(
                rng=rng,
                mu_db=mu,
                sigma_db=max(sigma, 1e-6),
                lower_db=xpd_min_db,
                upper_db=xpd_max_db,
                censor=par_cens,
            )
        path_point_mode = "continuous"
        path_point_value = np.nan
        if par_cens is not None:
            p_pin = float(max(0.0, min(1.0, par_cens.get("pinned_ratio", 0.0))))
            p_floor = float(max(0.0, min(1.0 - p_pin, par_cens.get("floor_ratio", 0.0))))
            u_path = float(rng.random())
            if u_path < p_pin and np.isfinite(float(par_cens.get("pinned_center_db", np.nan))):
                path_point_mode = "pinned"
                path_point_value = float(np.clip(float(par_cens.get("pinned_center_db", np.nan)), xpd_min_db, xpd_max_db))
            elif u_path < (p_pin + p_floor) and np.isfinite(float(par_cens.get("floor_db", np.nan))):
                path_point_mode = "floor"
                path_point_value = float(np.clip(float(par_cens.get("floor_db", np.nan)), xpd_min_db, xpd_max_db))
        phi = rng.uniform(0.0, 2.0 * np.pi, size=4)
        M_f = np.zeros((n_f, 2, 2), dtype=np.complex128)
        mode_freq = str(kappa_freq_mode).strip().lower()
        subband_xpd_db: dict[int, float] = {}
        for k in range(n_f):
            if mode_freq == "piecewise_constant" and subbands:
                bidx = _subband_index_for_freq(k, subbands)
                if bidx not in subband_xpd_db:
                    if path_point_mode in {"pinned", "floor"} and np.isfinite(path_point_value):
                        subband_xpd_db[bidx] = float(path_point_value)
                    else:
                        sb_fit = ((parity_subband_fit or {}).get(parity, {}) or {}).get(str(bidx), {})
                        mu_b = float(sb_fit.get("mu", mu))
                        sg_b = max(float(sb_fit.get("sigma", sigma)), 1e-6)
                        if xpd_pick is not None:
                            mu_b = float(xpd_pick)
                            sg_b = 1e-6
                        sb_key = f"{parity}|{bidx}"
                        sb_cens = (parity_subband_censoring or {}).get(sb_key) if parity_subband_censoring is not None else None
                        if sb_cens is not None:
                            sb_cens = dict(sb_cens)
                            sb_cens["pinned_ratio"] = 0.0
                            sb_cens["floor_ratio"] = 0.0
                        subband_xpd_db[bidx] = _sample_xpd_from_profile(
                            rng=rng,
                            mu_db=mu_b,
                            sigma_db=sg_b,
                            lower_db=xpd_min_db,
                            upper_db=xpd_max_db,
                            censor=sb_cens,
                        )
                xpd_db_k = float(subband_xpd_db[bidx])
                if xpd_freq_noise_sigma_db > 0.0:
                    xpd_db_k += float(rng.normal(0.0, xpd_freq_noise_sigma_db))
            else:
                xpd_mean = xpd0_db + slope * (freq[k] - fc)
                if xpd_freq_noise_sigma_db > 0.0:
                    xpd_mean += float(rng.normal(0.0, xpd_freq_noise_sigma_db))
                xpd_db_k = _truncated_normal_scalar(
                    rng,
                    mu=xpd_mean,
                    sigma=max(float(xpd_freq_noise_sigma_db), 1e-6),
                    lower=xpd_min_db,
                    upper=xpd_max_db,
                )
            if xpd_db_k < xpd_min_db or xpd_db_k > xpd_max_db:
                kappa_trunc_count += 1
            xpd_db_k = float(np.clip(xpd_db_k, xpd_min_db, xpd_max_db))
            kappa_k = float(10.0 ** (xpd_db_k / 10.0))
            kappa_total += 1
            inv = 1.0 / np.sqrt(max(kappa_k, 1e-15))
            H = np.array(
                [
                    [np.exp(1j * phi[0]), inv * np.exp(1j * phi[1])],
                    [inv * np.exp(1j * phi[2]), np.exp(1j * phi[3])],
                ],
                dtype=np.complex128,
            )
            M_f[k] = H

        mean_power = float(np.mean(np.sum(np.abs(M_f) ** 2, axis=(1, 2))))
        scale = np.sqrt(ray_power / max(mean_power, 1e-15))
        M_f *= scale
        eye = np.eye(2, dtype=np.complex128)
        g_tx = np.repeat(eye[None, :, :], n_f, axis=0)
        g_rx = np.repeat(eye[None, :, :], n_f, axis=0)
        scalar = np.ones((n_f,), dtype=float)
        j_f = M_f.copy()
        a_f = M_f.copy()
        paths.append(
            {
                "tau_s": tau,
                "A_f": a_f,
                "J_f": j_f,
                "G_tx_f": g_tx,
                "G_rx_f": g_rx,
                "scalar_gain_f": scalar,
                "meta": {
                    "bounce_count": 1 if parity == "odd" else 2,
                    "parity": parity,
                    "incidence_angle_bin": incidence_bin,
                    "synthetic_matrix_source": src,
                    "synthetic_xpd0_db": float(xpd0_db),
                        "synthetic_slope_db_per_hz": float(slope),
                        "synthetic_sampling_scenario_id": str(sid_target),
                        "synthetic_sampling_case_id": str(cid_target),
                        "scenario_id": str(sid_target),
                        "case_id": str(cid_target),
                        "interactions": ["synthetic"],
                    "surface_ids": [],
                    "incidence_angles": [],
                    "AoD": [0.0, 0.0, 0.0],
                    "AoA": [0.0, 0.0, 0.0],
                    "segment_basis_uv": [],
                },
            }
        )
    diagnostics = {
        "matrix_source": src,
        "num_paths_mode": str(num_paths_mode),
        "kappa_freq_mode": str(kappa_freq_mode),
        "resolved_num_paths": int(n_paths),
        "xpd_freq_noise_sigma_db": float(xpd_freq_noise_sigma_db),
        "sample_slope": bool(sample_slope),
        "slope_sigma_db_per_hz": float(max(slope_sigma_db_per_hz, 0.0)),
        "kappa_min": float(kappa_min),
        "kappa_max": float(kappa_max),
        "kappa_clamp_count": 0,
        "kappa_total": int(kappa_total),
        "kappa_clamp_rate": 0.0,
        "kappa_truncation_count": int(kappa_trunc_count),
        "kappa_truncation_rate": float(kappa_trunc_count / max(kappa_total, 1)),
    }
    if return_diagnostics:
        return paths, diagnostics
    return paths


def summarize_rt_vs_synth(
    rt_paths: list[dict[str, Any]],
    synth_paths: list[dict[str, Any]],
    subbands: list[tuple[int, int]],
    rt_matrix_source: str = "A",
    synth_matrix_source: str = "A",
    phase_bootstrap_B: int = 500,
    ks_n_cap: int = 200,
    phase_test_basis: str = "unknown",
    common_phase_removed: bool = True,
    per_ray_phase_sampling: bool = True,
    floor_db: float | None = None,
    pinned_tol_db: float = 0.5,
    input_basis: str | None = None,
    eval_basis: str | None = None,
    convention: str = "IEEE-RHCP",
    seed: int = 0,
) -> dict[str, Any]:
    rt_samples = pathwise_xpd(
        rt_paths,
        matrix_source=rt_matrix_source,
        input_basis=input_basis,  # type: ignore[arg-type]
        eval_basis=eval_basis,  # type: ignore[arg-type]
        convention=convention,
    )
    sy_samples = pathwise_xpd(
        synth_paths,
        matrix_source=synth_matrix_source,
        input_basis=input_basis,  # type: ignore[arg-type]
        eval_basis=eval_basis,  # type: ignore[arg-type]
        convention=convention,
    )
    for s in rt_samples:
        i = int(s.get("path_index", -1))
        meta = rt_paths[i].get("meta", {}) if 0 <= i < len(rt_paths) else {}
        s["scenario_id"] = str(meta.get("scenario_id", "NA"))
    for s in sy_samples:
        i = int(s.get("path_index", -1))
        meta = synth_paths[i].get("meta", {}) if 0 <= i < len(synth_paths) else {}
        s["scenario_id"] = str(meta.get("scenario_id", "NA"))

    rt_par = conditional_fit(rt_samples, ["parity"])
    sy_par = conditional_fit(sy_samples, ["parity"])
    rt_x = np.asarray([float(s["xpd_db"]) for s in rt_samples], dtype=float)
    sy_x = np.asarray([float(s["xpd_db"]) for s in sy_samples], dtype=float)
    rt_ex = floor_pinned_exclusion_mask(rt_x, floor_db=floor_db, pinned_tol_db=pinned_tol_db)
    sy_ex = floor_pinned_exclusion_mask(sy_x, floor_db=floor_db, pinned_tol_db=pinned_tol_db)
    rt_xc = np.asarray(rt_ex["values"], dtype=float)[~np.asarray(rt_ex["mask_excluded"], dtype=bool)]
    sy_xc = np.asarray(sy_ex["values"], dtype=float)[~np.asarray(sy_ex["mask_excluded"], dtype=bool)]

    f3_mu_rt = float(np.mean(rt_xc)) if len(rt_xc) else np.nan
    f3_mu_sy = float(np.mean(sy_xc)) if len(sy_xc) else np.nan
    f3_delta = float(abs(f3_mu_rt - f3_mu_sy)) if np.isfinite(f3_mu_rt) and np.isfinite(f3_mu_sy) else np.nan
    f3_sigma_rt = float(np.std(rt_xc, ddof=1)) if len(rt_xc) > 1 else np.nan
    f3_sigma_sy = float(np.std(sy_xc, ddof=1)) if len(sy_xc) > 1 else np.nan
    f3_sigma_delta = float(abs(f3_sigma_rt - f3_sigma_sy)) if np.isfinite(f3_sigma_rt) and np.isfinite(f3_sigma_sy) else np.nan

    f3_ks2_d, f3_ks2_p, f3_ks2_n = _ks2_with_cap(rt_xc, sy_xc, cap=ks_n_cap, seed=seed + 77)
    f3_ks2_d_full, f3_ks2_p_full, _ = _ks2_with_cap(rt_x, sy_x, cap=ks_n_cap, seed=seed + 79)

    rt_by_scn: dict[str, list[float]] = {}
    sy_by_scn: dict[str, list[float]] = {}
    rt_by_scn_c: dict[str, list[float]] = {}
    sy_by_scn_c: dict[str, list[float]] = {}
    for s in rt_samples:
        sid = str(s.get("scenario_id", "NA"))
        rt_by_scn.setdefault(sid, []).append(float(s["xpd_db"]))
    for s in sy_samples:
        sid = str(s.get("scenario_id", "NA"))
        sy_by_scn.setdefault(sid, []).append(float(s["xpd_db"]))
    for sid, vals in rt_by_scn.items():
        e = floor_pinned_exclusion_mask(vals, floor_db=floor_db, pinned_tol_db=pinned_tol_db)
        vv = np.asarray(e["values"], dtype=float)
        mm = np.asarray(e["mask_excluded"], dtype=bool)
        rt_by_scn_c[sid] = [float(x) for x in vv[~mm]]
    for sid, vals in sy_by_scn.items():
        e = floor_pinned_exclusion_mask(vals, floor_db=floor_db, pinned_tol_db=pinned_tol_db)
        vv = np.asarray(e["values"], dtype=float)
        mm = np.asarray(e["mask_excluded"], dtype=bool)
        sy_by_scn_c[sid] = [float(x) for x in vv[~mm]]

    common_sids = sorted(set(rt_by_scn.keys()) & set(sy_by_scn.keys()))
    if common_sids:
        per_sid_cap = max(2, int(max(ks_n_cap, 2) // max(len(common_sids), 1)))
    else:
        per_sid_cap = 0
    rt_cat: list[float] = []
    sy_cat: list[float] = []
    rt_cat_c: list[float] = []
    sy_cat_c: list[float] = []
    for i_sid, sid in enumerate(common_sids):
        r = np.asarray(rt_by_scn[sid], dtype=float)
        s = np.asarray(sy_by_scn[sid], dtype=float)
        d, p_ks, n_used = _ks2_with_cap(r, s, cap=per_sid_cap, seed=seed + 200 + i_sid)
        if n_used > 0:
            rng_sid = np.random.default_rng(seed + 900 + i_sid)
            rr = r if len(r) <= per_sid_cap else np.asarray(rng_sid.choice(r, size=per_sid_cap, replace=False), dtype=float)
            ss = s if len(s) <= per_sid_cap else np.asarray(rng_sid.choice(s, size=per_sid_cap, replace=False), dtype=float)
            rt_cat.extend(rr.tolist())
            sy_cat.extend(ss.tolist())
        rc = np.asarray(rt_by_scn_c.get(sid, []), dtype=float)
        sc = np.asarray(sy_by_scn_c.get(sid, []), dtype=float)
        d_c, p_c, n_c = _ks2_with_cap(rc, sc, cap=per_sid_cap, seed=seed + 400 + i_sid)
        if n_c > 0:
            rng_sid_c = np.random.default_rng(seed + 1300 + i_sid)
            rrc = rc if len(rc) <= per_sid_cap else np.asarray(rng_sid_c.choice(rc, size=per_sid_cap, replace=False), dtype=float)
            ssc = sc if len(sc) <= per_sid_cap else np.asarray(rng_sid_c.choice(sc, size=per_sid_cap, replace=False), dtype=float)
            rt_cat_c.extend(rrc.tolist())
            sy_cat_c.extend(ssc.tolist())
    ks_strat_d, ks_strat_p, ks_strat_n = _ks2_with_cap(np.asarray(rt_cat, dtype=float), np.asarray(sy_cat, dtype=float), cap=10**9, seed=seed + 3000)
    ks_strat_d_c, ks_strat_p_c, ks_strat_n_c = _ks2_with_cap(np.asarray(rt_cat_c, dtype=float), np.asarray(sy_cat_c, dtype=float), cap=10**9, seed=seed + 3100)

    rt_odd = np.asarray([1.0 for s in rt_samples if str(s.get("parity", "NA")) == "odd"], dtype=float)
    sy_odd = np.asarray([1.0 for s in sy_samples if str(s.get("parity", "NA")) == "odd"], dtype=float)
    rt_odd_frac = float(len(rt_odd) / max(len(rt_samples), 1))
    sy_odd_frac = float(len(sy_odd) / max(len(sy_samples), 1))
    parity_frac_abs_diff = float(abs(rt_odd_frac - sy_odd_frac))
    primary_p = f3_ks2_p
    primary_d = f3_ks2_d
    primary_n = f3_ks2_n
    if not np.isfinite(primary_p):
        f3_ks2_status = "SKIPPED_INSUFFICIENT"
        f3_ks2_reason = "insufficient samples"
    elif primary_p >= 0.05 and np.isfinite(f3_delta) and np.isfinite(f3_sigma_delta) and f3_delta <= 3.0 and f3_sigma_delta <= 6.0:
        if np.isfinite(f3_ks2_p_full) and f3_ks2_p_full < 0.05:
            f3_ks2_status = "PASS_WITH_CENSORING"
            f3_ks2_reason = "continuous-only KS pass with censoring; full-mixture KS fails"
        else:
            f3_ks2_status = "PASS"
            f3_ks2_reason = "continuous-only KS and mu/sigma thresholds passed"
    elif parity_frac_abs_diff > 0.05:
        f3_ks2_status = "SKIPPED_CONDITION_MISMATCH"
        f3_ks2_reason = f"parity_fraction_diff={parity_frac_abs_diff:.3f}"
    else:
        f3_ks2_status = "FAIL"
        f3_ks2_reason = "continuous-only KS and/or mu/sigma thresholds failed"

    rt_sub = pathwise_xpd(
        rt_paths,
        subbands=subbands,
        matrix_source=rt_matrix_source,
        input_basis=input_basis,  # type: ignore[arg-type]
        eval_basis=eval_basis,  # type: ignore[arg-type]
        convention=convention,
    )
    sy_sub = pathwise_xpd(
        synth_paths,
        subbands=subbands,
        matrix_source=synth_matrix_source,
        input_basis=input_basis,  # type: ignore[arg-type]
        eval_basis=eval_basis,  # type: ignore[arg-type]
        convention=convention,
    )

    def _mu_sigma(samples: list[dict[str, Any]], nb: int, continuous_only: bool = False) -> tuple[list[float], list[float]]:
        mu: list[float] = []
        sg: list[float] = []
        for b in range(nb):
            vals = np.asarray([float(s["xpd_db"]) for s in samples if int(s.get("subband", -1)) == b], dtype=float)
            if continuous_only and len(vals) > 0:
                exb = floor_pinned_exclusion_mask(vals, floor_db=floor_db, pinned_tol_db=pinned_tol_db)
                vv = np.asarray(exb["values"], dtype=float)
                mm = np.asarray(exb["mask_excluded"], dtype=bool)
                vals = vv[~mm]
            if len(vals) == 0:
                mu.append(np.nan)
                sg.append(np.nan)
            elif len(vals) == 1:
                mu.append(float(vals[0]))
                sg.append(0.0)
            else:
                mu.append(float(np.mean(vals)))
                sg.append(float(np.std(vals, ddof=1)))
        return mu, sg

    nb = len(subbands)
    mu_rt, sg_rt = _mu_sigma(rt_sub, nb, continuous_only=False)
    mu_sy, sg_sy = _mu_sigma(sy_sub, nb, continuous_only=False)
    mu_rt_c, sg_rt_c = _mu_sigma(rt_sub, nb, continuous_only=True)
    mu_sy_c, sg_sy_c = _mu_sigma(sy_sub, nb, continuous_only=True)
    a_rt = np.asarray(mu_rt, dtype=float)
    a_sy = np.asarray(mu_sy, dtype=float)
    valid = np.isfinite(a_rt) & np.isfinite(a_sy)
    mu_rmse_full = float(np.sqrt(np.mean((a_rt[valid] - a_sy[valid]) ** 2))) if np.any(valid) else np.nan
    s_rt = np.asarray(sg_rt, dtype=float)
    s_sy = np.asarray(sg_sy, dtype=float)
    valid_s = np.isfinite(s_rt) & np.isfinite(s_sy)
    sigma_rmse_full = float(np.sqrt(np.mean((s_rt[valid_s] - s_sy[valid_s]) ** 2))) if np.any(valid_s) else np.nan
    a_rt_c = np.asarray(mu_rt_c, dtype=float)
    a_sy_c = np.asarray(mu_sy_c, dtype=float)
    valid_c = np.isfinite(a_rt_c) & np.isfinite(a_sy_c)
    mu_rmse = float(np.sqrt(np.mean((a_rt_c[valid_c] - a_sy_c[valid_c]) ** 2))) if np.any(valid_c) else np.nan
    s_rt_c = np.asarray(sg_rt_c, dtype=float)
    s_sy_c = np.asarray(sg_sy_c, dtype=float)
    valid_sc = np.isfinite(s_rt_c) & np.isfinite(s_sy_c)
    sigma_rmse = float(np.sqrt(np.mean((s_rt_c[valid_sc] - s_sy_c[valid_sc]) ** 2))) if np.any(valid_sc) else np.nan
    span_rt = float(np.nanmax(a_rt) - np.nanmin(a_rt)) if np.any(np.isfinite(a_rt)) else np.nan
    span_sy = float(np.nanmax(a_sy) - np.nanmin(a_sy)) if np.any(np.isfinite(a_sy)) else np.nan
    span_rt_c = float(np.nanmax(a_rt_c) - np.nanmin(a_rt_c)) if np.any(np.isfinite(a_rt_c)) else np.nan
    span_sy_c = float(np.nanmax(a_sy_c) - np.nanmin(a_sy_c)) if np.any(np.isfinite(a_sy_c)) else np.nan

    ph_rt = offdiag_phases(
        rt_paths,
        matrix_source=rt_matrix_source,
        per_ray_sampling=per_ray_phase_sampling,
        common_phase_removed=common_phase_removed,
    )
    ph_sy = offdiag_phases(
        synth_paths,
        matrix_source=synth_matrix_source,
        per_ray_sampling=per_ray_phase_sampling,
        common_phase_removed=common_phase_removed,
    )
    ku_rt = kuiper_uniform_test(ph_rt, bootstrap_B=phase_bootstrap_B, seed=seed)
    ku_sy = kuiper_uniform_test(ph_sy, bootstrap_B=phase_bootstrap_B, seed=seed + 1)
    if int(ku_sy.get("n", 0)) < 20:
        synth_phase_status = "INSUFFICIENT"
    elif np.isfinite(float(ku_sy.get("p_boot", np.nan))) and float(ku_sy["p_boot"]) >= 0.05:
        synth_phase_status = "PASS"
    else:
        synth_phase_status = "FAIL"

    f5_status = "FAIL"
    if np.isfinite(mu_rmse) and mu_rmse <= 3.0:
        f5_status = "PASS_WITH_CENSORING" if (np.isfinite(mu_rmse_full) and mu_rmse_full > 3.0) else "PASS"

    return {
        "rt_parity_xpd": rt_par,
        "synthetic_parity_xpd": sy_par,
        "f3_xpd_mu_rt_db": f3_mu_rt,
        "f3_xpd_mu_synth_db": f3_mu_sy,
        "f3_xpd_mu_delta_abs_db": f3_delta,
        "f3_xpd_sigma_rt_db": f3_sigma_rt,
        "f3_xpd_sigma_synth_db": f3_sigma_sy,
        "f3_xpd_sigma_delta_abs_db": f3_sigma_delta,
        "f3_xpd_ks2_D": primary_d,
        "f3_xpd_ks2_p": primary_p,
        "f3_xpd_ks2_n_used": primary_n,
        "f3_xpd_ks2_D_global_cont": f3_ks2_d,
        "f3_xpd_ks2_p_global_cont": f3_ks2_p,
        "f3_xpd_ks2_D_stratified_cont": ks_strat_d_c,
        "f3_xpd_ks2_p_stratified_cont": ks_strat_p_c,
        "f3_xpd_ks2_n_stratified_cont": ks_strat_n_c,
        "f3_xpd_ks2_D_global_full": f3_ks2_d_full,
        "f3_xpd_ks2_p_global_full": f3_ks2_p_full,
        "f3_xpd_ks2_D_stratified_full": ks_strat_d,
        "f3_xpd_ks2_p_stratified_full": ks_strat_p,
        "f3_xpd_ks2_n_stratified_full": ks_strat_n,
        "f3_continuous_only_primary": True,
        "f3_continuous_rt_n": int(len(rt_xc)),
        "f3_continuous_synth_n": int(len(sy_xc)),
        "f3_pinned_ratio_rt": float(rt_ex.get("pinned_ratio", 0.0)),
        "f3_pinned_ratio_synth": float(sy_ex.get("pinned_ratio", 0.0)),
        "f3_floor_ratio_rt": float(rt_ex.get("floor_ratio", 0.0)),
        "f3_floor_ratio_synth": float(sy_ex.get("floor_ratio", 0.0)),
        "f3_parity_frac_abs_diff": parity_frac_abs_diff,
        "f3_xpd_ks2_status": f3_ks2_status,
        "f3_xpd_ks2_reason": f3_ks2_reason,
        "f3_gate_pass": bool(f3_ks2_status in {"PASS", "PASS_WITH_CENSORING"}),
        "rt_num_paths": len(rt_paths),
        "synthetic_num_paths": len(synth_paths),
        "rt_subband_count": len(rt_sub),
        "synthetic_subband_count": len(sy_sub),
        "subband_mu_rt_db": mu_rt,
        "subband_mu_synth_db": mu_sy,
        "subband_sigma_rt_db": sg_rt,
        "subband_sigma_synth_db": sg_sy,
        "subband_mu_rt_cont_db": mu_rt_c,
        "subband_mu_synth_cont_db": mu_sy_c,
        "subband_sigma_rt_cont_db": sg_rt_c,
        "subband_sigma_synth_cont_db": sg_sy_c,
        "subband_mu_span_rt": span_rt,
        "subband_mu_span_synth": span_sy,
        "subband_mu_span_rt_cont": span_rt_c,
        "subband_mu_span_synth_cont": span_sy_c,
        "subband_mu_rmse": mu_rmse,
        "subband_sigma_rmse": sigma_rmse,
        "subband_mu_rmse_full": mu_rmse_full,
        "subband_sigma_rmse_full": sigma_rmse_full,
        "f5_subband_mu_rmse_db": mu_rmse,
        "f5_subband_sigma_rmse_db": sigma_rmse,
        "f5_subband_mu_rmse_full_db": mu_rmse_full,
        "f5_subband_sigma_rmse_full_db": sigma_rmse_full,
        "f5_status": f5_status,
        "phase_test_basis": phase_test_basis,
        "common_phase_removed": bool(common_phase_removed),
        "per_ray_sampling": bool(per_ray_phase_sampling),
        "phase_uniformity_V_rt": float(ku_rt["V"]) if np.isfinite(ku_rt["V"]) else np.nan,
        "phase_uniformity_p_rt": float(ku_rt["p_boot"]) if np.isfinite(ku_rt["p_boot"]) else np.nan,
        "phase_uniformity_V_synth": float(ku_sy["V"]) if np.isfinite(ku_sy["V"]) else np.nan,
        "phase_uniformity_p_synth": float(ku_sy["p_boot"]) if np.isfinite(ku_sy["p_boot"]) else np.nan,
        "phase_uniformity_p": float(ku_sy["p_boot"]) if np.isfinite(ku_sy["p_boot"]) else np.nan,
        "phase_uniformity_rt_status": "INFO_DETERMINISTIC",
        "phase_uniformity_synth_status": synth_phase_status,
    }

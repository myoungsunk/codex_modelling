"""Conditional proxy distribution fitting for dual-CP metrics."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

import numpy as np
from scipy import stats


def _to_records(df_or_list: Any) -> list[dict[str, Any]]:
    if isinstance(df_or_list, list):
        return [dict(x) for x in df_or_list if isinstance(x, dict)]
    if isinstance(df_or_list, dict) and isinstance(df_or_list.get("rows"), list):
        return [dict(x) for x in df_or_list["rows"] if isinstance(x, dict)]
    try:
        import pandas as pd  # type: ignore

        if isinstance(df_or_list, pd.DataFrame):
            return df_or_list.to_dict(orient="records")
    except Exception:
        pass
    raise ValueError("Unsupported input type for fit_proxy_model")


def _float_or_nan(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def _bucket_key(row: dict[str, Any], u_keys: list[str]) -> str:
    return "|".join(str(row.get(k, "NA")) for k in u_keys)


def _weighted_lstsq(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    ww = np.sqrt(np.clip(np.asarray(w, dtype=float), 1e-12, np.inf))
    Xw = X * ww[:, None]
    yw = y * ww
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return np.asarray(beta, dtype=float)


def _fit_regression(
    rows: list[dict[str, Any]],
    z_key: str,
    numeric_u_keys: list[str],
    robust: bool = False,
    huber_delta: float = 1.5,
    max_iter: int = 30,
) -> dict[str, Any]:
    if not numeric_u_keys:
        return {"enabled": False}
    X = []
    y = []
    for r in rows:
        z = _float_or_nan(r.get(z_key, np.nan))
        if not np.isfinite(z):
            continue
        vals = []
        ok = True
        for k in numeric_u_keys:
            v = _float_or_nan(r.get(k, np.nan))
            if not np.isfinite(v):
                ok = False
                break
            vals.append(v)
        if not ok:
            continue
        X.append([1.0, *vals])
        y.append(z)
    if len(y) < max(4, len(numeric_u_keys) + 1):
        return {"enabled": False}
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
    beta = np.asarray(beta, dtype=float)
    if bool(robust):
        delta = float(max(huber_delta, 1e-6))
        for _ in range(int(max_iter)):
            pred_i = Xa @ beta
            resid_i = ya - pred_i
            med = float(np.median(resid_i))
            mad = float(np.median(np.abs(resid_i - med)))
            scale = max(1.4826 * mad, 1e-6)
            u = np.abs(resid_i) / (delta * scale)
            w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
            beta_new = _weighted_lstsq(Xa, ya, w)
            if np.linalg.norm(beta_new - beta) <= 1e-8 * (1.0 + np.linalg.norm(beta)):
                beta = beta_new
                break
            beta = beta_new
    pred = Xa @ beta
    resid = ya - pred
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    return {
        "enabled": True,
        "robust": bool(robust),
        "keys": list(numeric_u_keys),
        "intercept": float(beta[0]),
        "coef": [float(x) for x in beta[1:]],
        "sigma_resid": sigma,
        "n_fit": int(len(ya)),
    }


def predict_distribution(model: dict[str, Any], U: dict[str, Any]) -> tuple[float, float, str]:
    """Predict Normal(mu,sigma) parameters for one condition U."""

    u_keys = list(model.get("u_keys", []))
    bkey = _bucket_key(U, u_keys)
    buckets = model.get("bucket_stats", {}) or {}
    min_bucket_n = int(model.get("bucket_min_n", 1))
    b = buckets.get(bkey)
    if isinstance(b, dict):
        n_b = int(b.get("n", 0))
        mu = _float_or_nan(b.get("mu", np.nan))
        sigma = max(_float_or_nan(b.get("sigma", np.nan)), 1e-6)
        if n_b >= min_bucket_n and np.isfinite(mu) and np.isfinite(sigma):
            return float(mu), float(sigma), "bucket"

    reg = model.get("regression", {}) or {}
    if bool(reg.get("enabled", False)):
        vals = []
        ok = True
        for k in reg.get("keys", []):
            v = _float_or_nan(U.get(k, np.nan))
            if not np.isfinite(v):
                ok = False
                break
            vals.append(v)
        if ok:
            mu = float(reg.get("intercept", 0.0) + np.dot(np.asarray(reg.get("coef", []), dtype=float), np.asarray(vals, dtype=float)))
            sigma = float(max(_float_or_nan(reg.get("sigma_resid", 1.0)), 1e-6))
            if np.isfinite(mu) and np.isfinite(sigma):
                return mu, sigma, "regression"

    g = model.get("global", {}) or {}
    mu = _float_or_nan(g.get("mu", np.nan))
    sigma = max(_float_or_nan(g.get("sigma", np.nan)), 1e-6)
    if not np.isfinite(mu):
        mu = 0.0
    if not np.isfinite(sigma):
        sigma = 1.0
    return float(mu), float(sigma), "global"


def fit_proxy_model(
    df_or_list: Any,
    z_key: str,
    u_keys: list[str],
    method: str = "binned+regression",
    seed: int = 0,
    dist_family: str = "normal",
    robust_regression: bool = False,
    min_bucket_n: int = 3,
) -> dict[str, Any]:
    rows = _to_records(df_or_list)
    vals = np.asarray([_float_or_nan(r.get(z_key, np.nan)) for r in rows], dtype=float)
    mask = np.isfinite(vals)
    rows_z = [rows[i] for i in range(len(rows)) if bool(mask[i])]
    z = vals[mask]
    if len(z) == 0:
        raise ValueError(f"No finite samples for z_key={z_key}")

    buckets: dict[str, list[float]] = {}
    bucket_u: dict[str, dict[str, Any]] = {}
    for r, zv in zip(rows_z, z):
        k = _bucket_key(r, u_keys)
        buckets.setdefault(k, []).append(float(zv))
        bucket_u.setdefault(k, {uk: r.get(uk, "NA") for uk in u_keys})
    bucket_stats: dict[str, dict[str, Any]] = {}
    for k, arr in buckets.items():
        v = np.asarray(arr, dtype=float)
        bucket_stats[k] = {
            "u": bucket_u[k],
            "n": int(len(v)),
            "mu": float(np.mean(v)),
            "sigma": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        }

    numeric_u_keys = []
    for uk in u_keys:
        x = np.asarray([_float_or_nan(r.get(uk, np.nan)) for r in rows_z], dtype=float)
        if np.all(np.isfinite(x)):
            numeric_u_keys.append(str(uk))
    reg = (
        _fit_regression(
            rows_z,
            z_key=z_key,
            numeric_u_keys=numeric_u_keys,
            robust=bool(robust_regression),
        )
        if "regression" in str(method).lower()
        else {"enabled": False}
    )

    model = {
        "schema_version": "dualcp_proxy_model_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": str(method),
        "dist_family": str(dist_family),
        "z_key": str(z_key),
        "u_keys": list(u_keys),
        "bucket_min_n": int(max(1, min_bucket_n)),
        "global": {
            "n": int(len(z)),
            "mu": float(np.mean(z)),
            "sigma": float(np.std(z, ddof=1)) if len(z) > 1 else 0.0,
        },
        "bucket_stats": bucket_stats,
        "regression": reg,
    }

    mu_hat = []
    sigma_hat = []
    for r in rows_z:
        mu_i, sig_i, _ = predict_distribution(model, r)
        mu_hat.append(float(mu_i))
        sigma_hat.append(float(max(sig_i, 1e-6)))
    mu_hat_a = np.asarray(mu_hat, dtype=float)
    sigma_hat_a = np.asarray(sigma_hat, dtype=float)
    resid_std = (z - mu_hat_a) / sigma_hat_a
    dist = str(dist_family).lower().strip()
    if dist == "student_t":
        try:
            df, loc, scl = stats.t.fit(resid_std)
            ks = stats.kstest(resid_std, lambda xx: stats.t.cdf(xx, df, loc=loc, scale=scl))
            resid_dist = {"family": "student_t", "df": float(df), "loc": float(loc), "scale": float(max(scl, 1e-9))}
            sim_sampler = lambda size: float(max(scl, 1e-9)) * rng.standard_t(float(df), size=size) + float(loc)
        except Exception:
            ks = stats.kstest(resid_std, "norm")
            resid_dist = {"family": "normal"}
            sim_sampler = lambda size: rng.standard_normal(size)
    elif dist == "laplace":
        loc, scl = stats.laplace.fit(resid_std)
        ks = stats.kstest(resid_std, lambda xx: stats.laplace.cdf(xx, loc=loc, scale=scl))
        resid_dist = {"family": "laplace", "loc": float(loc), "scale": float(max(scl, 1e-9))}
        sim_sampler = lambda size: rng.laplace(loc=float(loc), scale=float(max(scl, 1e-9)), size=size)
    else:
        ks = stats.kstest(resid_std, "norm")
        resid_dist = {"family": "normal"}
        sim_sampler = lambda size: rng.standard_normal(size)
    n = len(resid_std)
    if n >= 2:
        q = stats.norm.ppf((np.arange(1, n + 1, dtype=float) - 0.5) / float(n))
        qq_r = float(np.corrcoef(np.sort(resid_std), q)[0, 1])
    else:
        qq_r = float("nan")
    rng = np.random.default_rng(int(seed))
    wd = float(stats.wasserstein_distance(resid_std, np.asarray(sim_sampler(n), dtype=float)))
    model["residual_dist"] = resid_dist
    model["gof"] = {
        "n": int(n),
        "ks_D": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "qq_r": float(qq_r),
        "wasserstein": wd,
    }
    return model


def save_proxy_model(path: str, model: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, default=str)

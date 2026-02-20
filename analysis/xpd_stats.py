"""XPD estimation and conditional statistics.

Example:
    >>> import numpy as np
    >>> from analysis.xpd_stats import xpd_db_from_matrix
    >>> A = np.array([[1, 0],[0,1]], dtype=np.complex128)
    >>> xpd_db_from_matrix(A) > 40
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats


EPS = 1e-15


def xpd_db_from_matrix(A: NDArray[np.complex128], power_floor: float = 1e-12) -> float:
    co = float(np.abs(A[0, 0]) ** 2 + np.abs(A[1, 1]) ** 2)
    cross = max(float(np.abs(A[0, 1]) ** 2 + np.abs(A[1, 0]) ** 2), float(power_floor))
    return float(10.0 * np.log10((co + EPS) / (cross + EPS)))


def pathwise_xpd(
    paths: list[dict],
    subbands: list[tuple[int, int]] | None = None,
    exact_bounce: int | None = None,
    bounce_filter: set[int] | None = None,
    power_floor: float = 1e-12,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, p in enumerate(paths):
        A_f = np.asarray(p["A_f"], dtype=np.complex128)
        meta = p["meta"]
        bcnt = int(meta["bounce_count"])
        if exact_bounce is not None and bcnt != exact_bounce:
            continue
        if bounce_filter is not None and bcnt not in bounce_filter:
            continue
        if subbands is None:
            A = np.mean(A_f, axis=0)
            xpd = xpd_db_from_matrix(A, power_floor=power_floor)
            out.append(
                {
                    "path_index": i,
                    "xpd_db": xpd,
                    "bounce_count": bcnt,
                    "parity": "even" if bcnt % 2 == 0 else "odd",
                    "tau_s": float(p["tau_s"]),
                }
            )
        else:
            for bidx, (s, e) in enumerate(subbands):
                A = np.mean(A_f[s:e], axis=0)
                out.append(
                    {
                        "path_index": i,
                        "subband": bidx,
                        "xpd_db": xpd_db_from_matrix(A, power_floor=power_floor),
                        "bounce_count": bcnt,
                        "parity": "even" if bcnt % 2 == 0 else "odd",
                        "tau_s": float(p["tau_s"]),
                    }
                )
    return out


def tapwise_xpd(
    h_tau: NDArray[np.complex128],
    tau_s: NDArray[np.float64],
    win_s: tuple[float, float] | None = None,
    power_floor: float = 1e-12,
) -> dict[str, NDArray[np.float64]]:
    p = np.abs(h_tau) ** 2
    co = p[:, 0, 0] + p[:, 1, 1]
    cross = np.maximum(p[:, 0, 1] + p[:, 1, 0], power_floor)
    xpd = 10.0 * np.log10((co + EPS) / (cross + EPS))
    if win_s is None:
        m = np.ones_like(tau_s, dtype=bool)
    else:
        m = (tau_s >= win_s[0]) & (tau_s <= win_s[1])
    return {"tau_s": tau_s[m], "xpd_db": xpd[m], "co": co[m], "cross": cross[m]}


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


def conditional_fit(samples: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {}
    for s in samples:
        k = "|".join(str(s.get(x, "NA")) for x in keys)
        buckets.setdefault(k, []).append(float(s["xpd_db"]))
    return {k: fit_normal_db(np.asarray(v, dtype=float)) for k, v in buckets.items()}


def save_stats_json(path: str | Path, stats_obj: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats_obj, indent=2), encoding="utf-8")
    return p

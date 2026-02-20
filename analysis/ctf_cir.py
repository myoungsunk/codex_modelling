"""Channel synthesis and CIR/PDP analysis.

Example:
    >>> import numpy as np
    >>> from analysis.ctf_cir import synthesize_ctf
    >>> H = synthesize_ctf([], np.linspace(6e9,7e9,4))
    >>> H.shape
    (4, 2, 2)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import hann, kaiser


WindowName = Literal["hann", "kaiser", "none"]


def linear_to_circular_matrix(convention: str = "IEEE-RHCP") -> NDArray[np.complex128]:
    if convention.upper().startswith("IEEE"):
        return np.array(
            [[1 / np.sqrt(2), 1 / np.sqrt(2)], [-1j / np.sqrt(2), 1j / np.sqrt(2)]],
            dtype=np.complex128,
        )
    return np.array(
        [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1j / np.sqrt(2), -1j / np.sqrt(2)]],
        dtype=np.complex128,
    )


def convert_basis(H_f: NDArray[np.complex128], src: str, dst: str, convention: str = "IEEE-RHCP") -> NDArray[np.complex128]:
    if src == dst:
        return H_f
    U = linear_to_circular_matrix(convention)
    if src == "linear" and dst == "circular":
        return np.einsum("ab,kbc,cd->kad", U.conj().T, H_f, U)
    if src == "circular" and dst == "linear":
        return np.einsum("ab,kbc,cd->kad", U, H_f, U.conj().T)
    raise ValueError(f"unsupported basis conversion: {src} -> {dst}")


def synthesize_ctf(paths: list[dict], f_hz: NDArray[np.float64]) -> NDArray[np.complex128]:
    """H(f)=sum_l A_l(f)exp(-j2pi f tau_l)."""

    freq = np.asarray(f_hz, dtype=float)
    h_f = np.zeros((len(freq), 2, 2), dtype=np.complex128)
    for p in paths:
        tau = float(p["tau_s"])
        a = np.asarray(p["A_f"], dtype=np.complex128)
        phase = np.exp(-1j * 2.0 * np.pi * freq * tau)[:, None, None]
        h_f += a * phase
    return h_f


def ctf_to_cir(
    H_f: NDArray[np.complex128],
    f_hz: NDArray[np.float64],
    nfft: int | None = None,
    window: WindowName = "hann",
    kaiser_beta: float = 8.0,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    freq = np.asarray(f_hz, dtype=float)
    n = len(freq)
    m = int(nfft or n)
    if m < n:
        raise ValueError("nfft must be >= len(f_hz)")

    if window == "hann":
        w = hann(n, sym=False)
    elif window == "kaiser":
        w = kaiser(n, beta=kaiser_beta, sym=False)
    else:
        w = np.ones(n)

    hw = H_f * w[:, None, None]
    if m > n:
        pad = np.zeros((m - n, 2, 2), dtype=np.complex128)
        hw = np.concatenate([hw, pad], axis=0)

    h_tau = np.fft.ifft(hw, axis=0)
    df = float(freq[1] - freq[0]) if n > 1 else 1.0
    tau = np.arange(m, dtype=float) / (m * df)
    return h_tau, tau


def pdp(h_tau: NDArray[np.complex128]) -> dict[str, NDArray[np.float64]]:
    p = np.abs(h_tau) ** 2
    co = p[:, 0, 0] + p[:, 1, 1]
    cross = p[:, 0, 1] + p[:, 1, 0]
    return {"pdp_ij": p, "co": co, "cross": cross, "sum": co + cross}


def cache_case_result(path: str | Path, scenario_id: str, case_id: str, H_f: NDArray[np.complex128], h_tau: NDArray[np.complex128], tau: NDArray[np.float64]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "a") as f:
        grp = f.require_group(f"cache/{scenario_id}/{case_id}")
        for name in list(grp.keys()):
            del grp[name]
        grp.create_dataset("H_f", data=H_f)
        grp.create_dataset("h_tau", data=h_tau)
        grp.create_dataset("tau_s", data=tau)

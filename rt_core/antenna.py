"""Antenna port models and projections to/from wave bases.

Example:
    >>> import numpy as np
    >>> from rt_core.antenna import Antenna
    >>> ant = Antenna(position=np.zeros(3), boresight=np.array([1,0,0]), h_axis=np.array([0,1,0]), v_axis=np.array([0,0,1]), basis="linear")
    >>> W = ant.wave_basis(np.array([1.0, 0.0, 0.0]))
    >>> W.shape
    (3, 2)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from rt_core.geometry import normalize
from rt_core.polarization import transverse_basis

Vec3 = NDArray[np.float64]
CMat = NDArray[np.complex128]


@dataclass(frozen=True)
class Antenna:
    position: Vec3
    boresight: Vec3
    h_axis: Vec3
    v_axis: Vec3
    basis: str = "linear"  # linear or circular
    convention: str = "IEEE-RHCP"
    cross_pol_leakage_db: float = 35.0
    axial_ratio_db: float = 0.0
    enable_coupling: bool = True
    global_up: Vec3 = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))

    def __post_init__(self) -> None:
        p = np.asarray(self.position, dtype=float)
        b = normalize(np.asarray(self.boresight, dtype=float))
        h = np.asarray(self.h_axis, dtype=float)
        h = h - float(np.dot(h, b)) * b
        h = normalize(h)
        v = np.asarray(self.v_axis, dtype=float)
        v = v - float(np.dot(v, b)) * b - float(np.dot(v, h)) * h
        v = normalize(v)
        object.__setattr__(self, "position", p)
        object.__setattr__(self, "boresight", b)
        object.__setattr__(self, "h_axis", h)
        object.__setattr__(self, "v_axis", v)
        gup = np.asarray(self.global_up, dtype=float)
        if np.linalg.norm(gup) < 1e-9:
            gup = np.array([0.0, 0.0, 1.0], dtype=float)
        object.__setattr__(self, "global_up", normalize(gup))

    def wave_basis(self, direction: Vec3) -> CMat:
        # Use a global reference vector to avoid locking wave basis to antenna H axis.
        u, v = transverse_basis(direction, self.global_up)
        return np.column_stack([u, v]).astype(np.complex128)

    def port_basis_vectors(self, direction: Vec3) -> CMat:
        """Return polarization vectors (columns) for two ports."""

        k = normalize(np.asarray(direction, dtype=float))
        eh = self.h_axis - float(np.dot(self.h_axis, k)) * k
        if np.linalg.norm(eh) < 1e-9:
            eh = self.v_axis - float(np.dot(self.v_axis, k)) * k
        eh = normalize(eh)
        ev = self.v_axis - float(np.dot(self.v_axis, k)) * k - float(np.dot(self.v_axis, eh)) * eh
        if np.linalg.norm(ev) < 1e-9:
            ev = np.cross(k, eh)
        ev = normalize(ev)

        if self.basis == "linear":
            return np.column_stack([eh, ev]).astype(np.complex128)

        if self.convention.upper().startswith("IEEE"):
            r = (eh - 1j * ev) / np.sqrt(2.0)
            l = (eh + 1j * ev) / np.sqrt(2.0)
        else:
            r = (eh + 1j * ev) / np.sqrt(2.0)
            l = (eh - 1j * ev) / np.sqrt(2.0)
        return np.column_stack([r, l]).astype(np.complex128)

    def tx_emit_matrix(self, direction: Vec3, wave_basis: CMat | None = None) -> NDArray[np.complex128]:
        wb = self.wave_basis(direction) if wave_basis is None else wave_basis
        pb = self.port_basis_vectors(direction)
        return (wb.conj().T @ pb).astype(np.complex128)

    def rx_receive_matrix(self, direction: Vec3, wave_basis: CMat | None = None) -> NDArray[np.complex128]:
        wb = self.wave_basis(direction) if wave_basis is None else wave_basis
        # Receive circular handedness is defined for the incoming wave, i.e., opposite look direction.
        pb = self.port_basis_vectors(-np.asarray(direction, dtype=float))
        return (pb.conj().T @ wb).astype(np.complex128)

    def _coupling_matrix(self, n_f: int) -> NDArray[np.complex128]:
        if not self.enable_coupling:
            eye = np.eye(2, dtype=np.complex128)
            return np.repeat(eye[None, :, :], n_f, axis=0)
        leak = 10.0 ** (-self.cross_pol_leakage_db / 20.0)
        ar = 10.0 ** (self.axial_ratio_db / 20.0)
        ar_leak = abs((ar - 1.0) / (ar + 1.0))
        eps = float(np.clip(leak + ar_leak, 0.0, 0.49))
        base = np.array([[1.0, eps], [eps, 1.0]], dtype=np.complex128)
        base /= np.sqrt(1.0 + eps**2)
        return np.repeat(base[None, :, :], n_f, axis=0)

    def tx_port_to_wave(self, direction: Vec3, f_hz: NDArray[np.float64], wave_basis: CMat | None = None) -> NDArray[np.complex128]:
        proj = self.tx_emit_matrix(direction, wave_basis=wave_basis)
        cpl = self._coupling_matrix(len(f_hz))
        return np.einsum("ab,kbc->kac", proj, cpl).astype(np.complex128)

    def rx_wave_to_port(self, direction: Vec3, f_hz: NDArray[np.float64], wave_basis: CMat | None = None) -> NDArray[np.complex128]:
        proj = self.rx_receive_matrix(direction, wave_basis=wave_basis)
        cpl = self._coupling_matrix(len(f_hz))
        return np.einsum("kab,bc->kac", cpl.conj().transpose(0, 2, 1), proj).astype(np.complex128)

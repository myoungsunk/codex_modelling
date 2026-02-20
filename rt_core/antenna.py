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

from dataclasses import dataclass

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

    def wave_basis(self, direction: Vec3) -> CMat:
        u, v = transverse_basis(direction, self.h_axis)
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
        pb = self.port_basis_vectors(direction)
        return (pb.conj().T @ wb).astype(np.complex128)

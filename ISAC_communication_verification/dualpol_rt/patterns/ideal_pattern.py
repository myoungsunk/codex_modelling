from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dualpol_rt.geometry.visibility import normalize


def spherical_angles(direction_local: np.ndarray) -> tuple[float, float]:
    k = normalize(np.asarray(direction_local, dtype=float))
    theta = float(np.arccos(np.clip(k[2], -1.0, 1.0)))
    phi = float(np.arctan2(k[1], k[0]))
    if phi < 0.0:
        phi += 2.0 * np.pi
    return theta, phi


def spherical_basis(direction_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta, phi = spherical_angles(direction_local)
    if np.sin(theta) < 1e-9:
        e_theta = np.array([1.0, 0.0, 0.0], dtype=float)
        e_phi = np.array([0.0, 1.0, 0.0], dtype=float)
        return e_theta, e_phi
    e_theta = np.array(
        [
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta),
        ],
        dtype=float,
    )
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=float)
    return normalize(e_theta), normalize(e_phi)


@dataclass(frozen=True)
class IdealPattern:
    basis: str = "linear"
    circular_order: str = "LR"
    convention: str = "IEEE-RHCP"
    x_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    y_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=float))
    z_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))

    def __post_init__(self) -> None:
        x = normalize(np.asarray(self.x_axis, dtype=float))
        z = normalize(np.asarray(self.z_axis, dtype=float))
        y = np.asarray(self.y_axis, dtype=float)
        y = y - float(np.dot(y, z)) * z - float(np.dot(y, x)) * x
        y = normalize(y)
        x = normalize(np.cross(y, z))
        rot = np.column_stack([x, y, z]).astype(float)
        rot.setflags(write=False)
        object.__setattr__(self, "x_axis", x)
        object.__setattr__(self, "y_axis", y)
        object.__setattr__(self, "z_axis", z)
        object.__setattr__(self, "_rotation_local_to_world", rot)

    @property
    def port_count(self) -> int:
        return 2

    def world_to_local(self, vec_world: np.ndarray) -> np.ndarray:
        return (self._rotation_local_to_world.T @ np.asarray(vec_world, dtype=float)).astype(float)

    def local_to_world(self, vec_local: np.ndarray) -> np.ndarray:
        return (self._rotation_local_to_world.astype(np.complex128) @ np.asarray(vec_local, dtype=np.complex128)).astype(np.complex128)

    def _projected_hv(self, direction_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = normalize(np.asarray(direction_local, dtype=float))
        h = np.array([1.0, 0.0, 0.0], dtype=float)
        v = np.array([0.0, 1.0, 0.0], dtype=float)
        h = h - float(np.dot(h, k)) * k
        if np.linalg.norm(h) < 1e-9:
            h = v - float(np.dot(v, k)) * k
        h = normalize(h)
        v = v - float(np.dot(v, k)) * k - float(np.dot(v, h)) * h
        if np.linalg.norm(v) < 1e-9:
            v = np.cross(k, h)
        v = normalize(v)
        return h, v

    def _linear_or_circular_vector(self, port_id: int, direction_local: np.ndarray) -> np.ndarray:
        h, v = self._projected_hv(direction_local)
        if self.basis.lower() == "linear":
            return h if int(port_id) == 0 else v
        if self.circular_order.upper() == "LR":
            left = (h + 1j * v) / np.sqrt(2.0)
            right = (h - 1j * v) / np.sqrt(2.0)
        else:
            right = (h - 1j * v) / np.sqrt(2.0)
            left = (h + 1j * v) / np.sqrt(2.0)
        return left if int(port_id) == 0 else right

    def port_field(self, port_id: int, f_hz: float | np.ndarray, k_local: np.ndarray) -> np.ndarray:
        freqs = np.atleast_1d(np.asarray(f_hz, dtype=float))
        e_theta, e_phi = spherical_basis(k_local)
        field_local = self._linear_or_circular_vector(int(port_id), k_local)
        comp_theta = np.vdot(e_theta, field_local)
        comp_phi = np.vdot(e_phi, field_local)
        return np.column_stack(
            [
                np.full(freqs.shape, comp_theta, dtype=np.complex128),
                np.full(freqs.shape, comp_phi, dtype=np.complex128),
            ]
        )

    def port_field_hv(self, port_id: int, f_hz: float | np.ndarray, k_local: np.ndarray) -> np.ndarray:
        field_theta_phi = self.port_field(port_id, f_hz, k_local)
        return np.column_stack([field_theta_phi[:, 1], field_theta_phi[:, 0]]).astype(np.complex128)

    def port_vector_local(self, port_id: int, f_hz: float | np.ndarray, k_local: np.ndarray) -> np.ndarray:
        freqs = np.atleast_1d(np.asarray(f_hz, dtype=float))
        e_theta_local, e_phi_local = spherical_basis(k_local)
        field = self.port_field(port_id, freqs, k_local)
        out = np.zeros((len(freqs), 3), dtype=np.complex128)
        for idx in range(len(freqs)):
            field_local = field[idx, 0] * e_theta_local + field[idx, 1] * e_phi_local
            out[idx] = field_local
        return out

    def port_vector_world(self, port_id: int, f_hz: float | np.ndarray, k_world: np.ndarray) -> np.ndarray:
        freqs = np.atleast_1d(np.asarray(f_hz, dtype=float))
        k_local = self.world_to_local(k_world)
        field_local = self.port_vector_local(port_id, freqs, k_local)
        out = np.zeros((len(freqs), 3), dtype=np.complex128)
        for idx in range(len(freqs)):
            out[idx] = self.local_to_world(field_local[idx])
        return out

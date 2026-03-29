from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from dualpol_rt.geometry.visibility import normalize
from dualpol_rt.patterns.ideal_pattern import spherical_basis


@dataclass(frozen=True)
class AntennaPose:
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    x_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    y_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=float))
    z_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=float)
        x = normalize(np.asarray(self.x_axis, dtype=float))
        z = normalize(np.asarray(self.z_axis, dtype=float))
        y = np.asarray(self.y_axis, dtype=float)
        y = y - float(np.dot(y, z)) * z - float(np.dot(y, x)) * x
        y = normalize(y)
        x = normalize(np.cross(y, z))
        rotation = np.column_stack([x, y, z]).astype(float)
        rotation.setflags(write=False)
        origin.setflags(write=False)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "x_axis", x)
        object.__setattr__(self, "y_axis", y)
        object.__setattr__(self, "z_axis", z)
        object.__setattr__(self, "_rotation_local_to_world", rotation)

    def world_to_local(self, vec_world: np.ndarray) -> np.ndarray:
        return (self._rotation_local_to_world.T @ np.asarray(vec_world, dtype=float)).astype(float)

    def local_to_world(self, vec_local: np.ndarray) -> np.ndarray:
        return (self._rotation_local_to_world.astype(np.complex128) @ np.asarray(vec_local, dtype=np.complex128)).astype(np.complex128)

    def rotated_local(
        self,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
    ) -> "AntennaPose":
        """Rotate the pose around its local axes.

        The local boresight is +z. Yaw is applied around local +y, pitch around
        local +x, and roll around local +z using right-handed positive angles.
        """

        yaw = np.deg2rad(float(yaw_deg))
        pitch = np.deg2rad(float(pitch_deg))
        roll = np.deg2rad(float(roll_deg))

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        rot_yaw = np.array(
            [
                [cy, 0.0, sy],
                [0.0, 1.0, 0.0],
                [-sy, 0.0, cy],
            ],
            dtype=float,
        )
        rot_pitch = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cp, -sp],
                [0.0, sp, cp],
            ],
            dtype=float,
        )
        rot_roll = np.array(
            [
                [cr, -sr, 0.0],
                [sr, cr, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        rotation = self._rotation_local_to_world @ rot_roll @ rot_pitch @ rot_yaw
        return AntennaPose(origin=self.origin, x_axis=rotation[:, 0], y_axis=rotation[:, 1], z_axis=rotation[:, 2])

    @classmethod
    def facing_from_points(cls, origin: np.ndarray, target: np.ndarray, up_hint: np.ndarray | None = None) -> "AntennaPose":
        origin_arr = np.asarray(origin, dtype=float)
        target_arr = np.asarray(target, dtype=float)
        z_axis = normalize(target_arr - origin_arr)
        up = np.asarray([0.0, 0.0, 1.0] if up_hint is None else up_hint, dtype=float)
        x_axis = up - float(np.dot(up, z_axis)) * z_axis
        if np.linalg.norm(x_axis) < 1e-9:
            alt = np.array([1.0, 0.0, 0.0], dtype=float) if abs(z_axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0], dtype=float)
            x_axis = alt - float(np.dot(alt, z_axis)) * z_axis
        x_axis = normalize(x_axis)
        y_axis = normalize(np.cross(z_axis, x_axis))
        return cls(origin=origin_arr, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)


@dataclass(frozen=True)
class AntennaFamily:
    name: str
    pattern: object
    pose: AntennaPose
    port_labels: tuple[str, str]
    basis_labels: tuple[str, str]
    normalization_mode: str
    sparams: np.ndarray | None = None

    @property
    def port_count(self) -> int:
        return int(self.pattern.port_count)

    def with_pose(self, pose: AntennaPose) -> "AntennaFamily":
        return replace(self, pose=pose)

    def normalized_copy(self, mode: str) -> "AntennaFamily":
        pattern = self.pattern.normalized_copy(mode) if hasattr(self.pattern, "normalized_copy") else self.pattern
        return replace(self, pattern=pattern, normalization_mode=str(mode))

    def port_field(self, port_id: int, f_hz: float | np.ndarray, k_local: np.ndarray) -> np.ndarray:
        return self.pattern.port_field(port_id, f_hz, k_local)

    def port_field_hv(self, port_id: int, f_hz: float | np.ndarray, k_local: np.ndarray) -> np.ndarray:
        return self.pattern.port_field_hv(port_id, f_hz, k_local)

    def port_vector_world(self, port_id: int, f_hz: float | np.ndarray, k_world: np.ndarray) -> np.ndarray:
        freqs = np.atleast_1d(np.asarray(f_hz, dtype=float))
        k_local = self.pose.world_to_local(k_world)
        if hasattr(self.pattern, "port_vector_local"):
            local_vectors = self.pattern.port_vector_local(port_id, freqs, k_local)
        else:
            e_theta_local, e_phi_local = spherical_basis(k_local)
            field = self.pattern.port_field(port_id, freqs, k_local)
            local_vectors = np.zeros((len(freqs), 3), dtype=np.complex128)
            for idx in range(len(freqs)):
                local_vectors[idx] = field[idx, 0] * e_theta_local + field[idx, 1] * e_phi_local
        out = np.zeros((len(freqs), 3), dtype=np.complex128)
        for idx in range(len(freqs)):
            out[idx] = self.pose.local_to_world(local_vectors[idx])
        return out

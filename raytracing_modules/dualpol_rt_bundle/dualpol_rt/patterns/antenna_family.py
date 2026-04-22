from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from dualpol_rt.geometry.visibility import normalize
from dualpol_rt.patterns.ffd_pattern import FFDPattern
from dualpol_rt.patterns.ideal_pattern import IdealPattern
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
    antenna_mode: str = "generic"
    basis: str = "linear"
    convention: str = "IEEE-RHCP"
    cross_pol_leakage_db: float = float("inf")
    axial_ratio_db: float = 0.0
    enable_coupling: bool = True
    tx_pattern_cos_exp: float = 0.0
    rx_pattern_cos_exp: float = 0.0

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


def _pose_from_boresight(pos: np.ndarray | None = None, boresight: np.ndarray | None = None) -> AntennaPose:
    origin = np.zeros(3, dtype=float) if pos is None else np.asarray(pos, dtype=float)
    if boresight is None:
        return AntennaPose(origin=origin)
    direction = normalize(np.asarray(boresight, dtype=float))
    return AntennaPose.facing_from_points(origin, origin + direction)


def _as_cp_pattern(
    pattern_data: FFDPattern | tuple[str | Path, str | Path] | dict[str, str | Path],
    *,
    handedness: str,
    normalization_mode: str,
) -> FFDPattern:
    if isinstance(pattern_data, FFDPattern):
        return pattern_data.normalized_copy(normalization_mode) if pattern_data.normalization_mode != normalization_mode else pattern_data
    handed = str(handedness).lower()
    if isinstance(pattern_data, tuple):
        if len(pattern_data) != 2:
            raise ValueError("realistic CP antenna expects exactly two pattern files")
        port_order = ("RHCP", "LHCP") if handed == "right" else ("LHCP", "RHCP")
        mapping = {port_order[0]: pattern_data[0], port_order[1]: pattern_data[1]}
        return FFDPattern.from_port_files(mapping, port_order=port_order, normalization_mode=normalization_mode, basis="circular")
    if isinstance(pattern_data, dict):
        port_order = tuple(str(name) for name in pattern_data.keys())
        if len(port_order) != 2:
            raise ValueError("realistic CP antenna expects exactly two named CP ports")
        return FFDPattern.from_port_files(pattern_data, port_order=port_order, normalization_mode=normalization_mode, basis="circular")
    raise TypeError("pattern_data must be an FFDPattern, a 2-file tuple, or a 2-port mapping")


def _cp_decomposition(field_xyz: np.ndarray) -> tuple[complex, complex]:
    ex = complex(field_xyz[0])
    ey = complex(field_xyz[1])
    rhcp = (ex - 1j * ey) / np.sqrt(2.0)
    lhcp = (ex + 1j * ey) / np.sqrt(2.0)
    return rhcp, lhcp


def _fit_cosine_exponent(pattern: FFDPattern, port_name: str) -> float:
    field = np.asarray(pattern.fields_by_port[port_name], dtype=np.complex128)
    freq_idx = len(pattern.freq_grid_hz) // 2
    power_theta = np.mean(np.abs(field[freq_idx, ..., 0]) ** 2 + np.abs(field[freq_idx, ..., 1]) ** 2, axis=1)
    if power_theta.size <= 1:
        return 0.0
    peak = float(np.max(power_theta))
    if peak <= 0.0:
        return 0.0
    cos_theta = np.cos(np.asarray(pattern.theta_grid_rad, dtype=float))
    mask = (cos_theta > 1e-6) & (power_theta > 0.0)
    if int(np.count_nonzero(mask)) < 2:
        return 0.0
    x = np.log(np.maximum(cos_theta[mask], 1e-12))
    y = np.log(np.maximum(power_theta[mask] / peak, 1e-12))
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return 0.0
    return float(max(np.dot(x, y) / denom, 0.0))


def _cp_pattern_quality(pattern: FFDPattern, handedness: str) -> tuple[float, float, float]:
    port_name = pattern.port_names[0]
    freq_hz = float(pattern.freq_grid_hz[len(pattern.freq_grid_hz) // 2])
    field = pattern.port_vector_local(0, np.array([freq_hz], dtype=float), np.array([0.0, 0.0, 1.0], dtype=float))[0]
    rhcp, lhcp = _cp_decomposition(field)
    handed = str(handedness).lower()
    co = rhcp if handed == "right" else lhcp
    cross = lhcp if handed == "right" else rhcp
    co_mag = float(abs(co))
    cross_mag = float(abs(cross))
    xpd_db = float(max(20.0 * np.log10((co_mag + 1e-15) / (cross_mag + 1e-15)), 0.0))
    ar_linear = (co_mag + cross_mag) / max(co_mag - cross_mag, 1e-15)
    ar_db = float(20.0 * np.log10(max(ar_linear, 1.0)))
    fitted_exp = _fit_cosine_exponent(pattern, port_name)
    return ar_db, xpd_db, fitted_exp


def make_ideal_cp_antenna(
    handedness: str = "right",
    pos: np.ndarray | None = None,
    boresight: np.ndarray | None = None,
    *,
    name: str | None = None,
    normalization_mode: str = "raw",
) -> AntennaFamily:
    handed = str(handedness).lower()
    if handed not in {"right", "left"}:
        raise ValueError("handedness must be 'right' or 'left'")
    circular_order = "RL" if handed == "right" else "LR"
    labels = ("RHCP", "LHCP") if handed == "right" else ("LHCP", "RHCP")
    return AntennaFamily(
        name=str(name or f"ideal_cp_{handed}"),
        pattern=IdealPattern(basis="circular", circular_order=circular_order),
        pose=_pose_from_boresight(pos=pos, boresight=boresight),
        port_labels=labels,
        basis_labels=labels,
        normalization_mode=str(normalization_mode),
        antenna_mode="ideal_cp",
        basis="circular",
        convention="IEEE-RHCP",
        cross_pol_leakage_db=60.0,
        axial_ratio_db=0.0,
        enable_coupling=False,
        tx_pattern_cos_exp=0.0,
        rx_pattern_cos_exp=0.0,
    )


def make_realistic_patch_cp_antenna(
    pattern_data: FFDPattern | tuple[str | Path, str | Path] | dict[str, str | Path],
    handedness: str = "right",
    pos: np.ndarray | None = None,
    boresight: np.ndarray | None = None,
    *,
    name: str | None = None,
    normalization_mode: str = "raw",
) -> AntennaFamily:
    handed = str(handedness).lower()
    if handed not in {"right", "left"}:
        raise ValueError("handedness must be 'right' or 'left'")
    pattern = _as_cp_pattern(pattern_data, handedness=handed, normalization_mode=normalization_mode)
    axial_ratio_db, cross_pol_leakage_db, fitted_exp = _cp_pattern_quality(pattern, handed)
    labels = tuple(str(label) for label in pattern.port_names)
    return AntennaFamily(
        name=str(name or f"realistic_patch_cp_{handed}"),
        pattern=pattern,
        pose=_pose_from_boresight(pos=pos, boresight=boresight),
        port_labels=labels,
        basis_labels=labels,
        normalization_mode=str(normalization_mode),
        antenna_mode="realistic_patch_cp",
        basis="circular",
        convention="IEEE-RHCP",
        cross_pol_leakage_db=float(cross_pol_leakage_db),
        axial_ratio_db=float(axial_ratio_db),
        enable_coupling=True,
        tx_pattern_cos_exp=float(fitted_exp),
        rx_pattern_cos_exp=float(fitted_exp),
    )

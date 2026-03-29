from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from dualpol_rt.patterns.ideal_pattern import IdealPattern


def _clean_lines(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in text if line.strip() and not line.strip().startswith(("#", "!", "//"))]


def _parse_ffd_file(path: str | Path, freq_hz: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lines = _clean_lines(path)
    if len(lines) < 3:
        raise ValueError(f"FFD file is too short: {path}")
    theta_meta = [float(token) for token in lines[0].split()]
    phi_meta = [float(token) for token in lines[1].split()]
    if len(theta_meta) < 3 or len(phi_meta) < 3:
        raise ValueError(f"FFD header requires theta and phi grids: {path}")
    theta_start, theta_stop, theta_count = theta_meta[:3]
    phi_start, phi_stop, phi_count = phi_meta[:3]
    theta = np.linspace(np.deg2rad(theta_start), np.deg2rad(theta_stop), int(theta_count), dtype=float)
    phi = np.linspace(np.deg2rad(phi_start), np.deg2rad(phi_stop), int(phi_count), dtype=float)
    expected = int(theta_count) * int(phi_count)
    third = lines[2]

    def _assign_block(block_lines: list[str], out: np.ndarray, f_idx: int) -> None:
        if len(block_lines) < expected:
            raise ValueError(f"FFD data line count mismatch in {path}: expected {expected}, got {len(block_lines)}")
        for flat_idx in range(expected):
            tokens = [float(token) for token in block_lines[flat_idx].split()]
            if len(tokens) == 4:
                re_theta, im_theta, re_phi, im_phi = tokens
            elif len(tokens) >= 6:
                re_theta, im_theta, re_phi, im_phi = tokens[-4:]
            else:
                raise ValueError(f"Unsupported FFD row format in {path}: {block_lines[flat_idx]}")
            t_idx = flat_idx % len(theta)
            p_idx = flat_idx // len(theta)
            out[f_idx, t_idx, p_idx, 0] = re_theta + 1j * im_theta
            out[f_idx, t_idx, p_idx, 1] = re_phi + 1j * im_phi

    if third.lower().startswith("frequencies"):
        n_freq = int(third.split()[-1])
        freq_grid = np.zeros(n_freq, dtype=float)
        field = np.zeros((n_freq, len(theta), len(phi), 2), dtype=np.complex128)
        cursor = 3
        for f_idx in range(n_freq):
            if cursor >= len(lines):
                raise ValueError(f"Unexpected end of FFD file in {path}")
            freq_line = lines[cursor]
            cursor += 1
            if not freq_line.lower().startswith("frequency"):
                raise ValueError(f"Expected 'Frequency <Hz>' line in {path}, got: {freq_line}")
            freq_grid[f_idx] = float(freq_line.split()[1])
            block = lines[cursor : cursor + expected]
            cursor += expected
            _assign_block(block, field, f_idx)
        return freq_grid, theta, phi, field

    freq_val = float(freq_hz) if freq_hz is not None else 0.0
    field = np.zeros((1, len(theta), len(phi), 2), dtype=np.complex128)
    _assign_block(lines[2:], field, 0)
    return np.array([freq_val], dtype=float), theta, phi, field


def _parse_array_map(path: str | Path) -> list[tuple[str, str]]:
    lines = _clean_lines(path)
    entries: list[tuple[str, str]] = []
    for line in lines:
        tokens = line.split()
        if len(tokens) < 2:
            continue
        entries.append((tokens[0], tokens[-1]))
    if not entries:
        raise ValueError(f"No FFD entries found in array map: {path}")
    return entries


@dataclass(frozen=True)
class FFDPattern(IdealPattern):
    port_names: tuple[str, str] = ("port0", "port1")
    freq_grid_hz: np.ndarray | None = None
    theta_grid_rad: np.ndarray | None = None
    phi_grid_rad: np.ndarray | None = None
    fields_by_port: dict[str, np.ndarray] | None = field(default=None)
    normalization_mode: str = "raw"
    family_scale_f: np.ndarray | None = None

    @classmethod
    def from_port_files(
        cls,
        port_files: dict[str, str | Path],
        port_order: tuple[str, str] | None = None,
        freq_hz: float | None = None,
        normalization_mode: str = "raw",
        **kwargs,
    ) -> "FFDPattern":
        names = port_order or tuple(port_files.keys())
        if len(names) != 2:
            raise ValueError("FFDPattern expects exactly two ports for the current dual-port baseline")
        freq_grid = None
        theta = None
        phi = None
        fields: dict[str, np.ndarray] = {}
        for name in names:
            f_grid, t_grid, p_grid, field = _parse_ffd_file(port_files[name], freq_hz=freq_hz)
            if freq_grid is None:
                freq_grid = f_grid
                theta = t_grid
                phi = p_grid
            else:
                if not np.allclose(freq_grid, f_grid) or not np.allclose(theta, t_grid) or not np.allclose(phi, p_grid):
                    raise ValueError("All FFD ports must share the same frequency/theta/phi grids")
            fields[str(name)] = field
        return cls(
            port_names=tuple(str(name) for name in names),
            freq_grid_hz=freq_grid,
            theta_grid_rad=theta,
            phi_grid_rad=phi,
            fields_by_port=fields,
            normalization_mode=normalization_mode,
            **kwargs,
        )

    @classmethod
    def from_array_file(
        cls,
        array_file: str | Path,
        base_dir: str | Path | None = None,
        port_order: tuple[str, str] | None = None,
        freq_hz: float | None = None,
        normalization_mode: str = "raw",
        **kwargs,
    ) -> "FFDPattern":
        base = Path(base_dir) if base_dir is not None else Path(array_file).parent
        entries = _parse_array_map(array_file)
        mapping = {name: str(base / filename) for name, filename in entries}
        names = port_order or tuple(name for name, _filename in entries[:2])
        return cls.from_port_files(mapping, port_order=names, freq_hz=freq_hz, normalization_mode=normalization_mode, **kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.freq_grid_hz is None or self.theta_grid_rad is None or self.phi_grid_rad is None or self.fields_by_port is None:
            raise ValueError("FFDPattern requires frequency/theta/phi grids and per-port fields")
        freq = np.asarray(self.freq_grid_hz, dtype=float)
        theta = np.asarray(self.theta_grid_rad, dtype=float)
        phi = np.asarray(self.phi_grid_rad, dtype=float)
        if np.any(np.diff(freq) < 0.0):
            raise ValueError("freq_grid_hz must be sorted ascending")
        if len(freq) == 0:
            raise ValueError("freq_grid_hz must be non-empty")
        for port_name in self.port_names:
            if port_name not in self.fields_by_port:
                raise ValueError(f"Missing port field data for {port_name}")
            arr = np.asarray(self.fields_by_port[port_name], dtype=np.complex128)
            if arr.shape != (len(freq), len(theta), len(phi), 2):
                raise ValueError(f"Unexpected field shape for {port_name}: {arr.shape}")
            self.fields_by_port[port_name] = arr
        if self.family_scale_f is None:
            family_scale = np.ones(len(freq), dtype=float)
            if str(self.normalization_mode).lower() == "family_radiated_power":
                family_scale = self._family_scale_from_raw_fields()
        else:
            family_scale = np.asarray(self.family_scale_f, dtype=float)
            if family_scale.shape != (len(freq),):
                raise ValueError("family_scale_f must have shape [Nf]")
        freq.setflags(write=False)
        theta.setflags(write=False)
        phi.setflags(write=False)
        family_scale.setflags(write=False)
        object.__setattr__(self, "freq_grid_hz", freq)
        object.__setattr__(self, "theta_grid_rad", theta)
        object.__setattr__(self, "phi_grid_rad", phi)
        object.__setattr__(self, "family_scale_f", family_scale)

    def _family_scale_from_raw_fields(self) -> np.ndarray:
        energies = self._raw_radiation_energy_by_freq()
        return (1.0 / np.sqrt(np.maximum(energies, 1e-30))).astype(float)

    def _raw_radiation_energy_by_freq(self) -> np.ndarray:
        theta = np.asarray(self.theta_grid_rad, dtype=float)
        weights = np.sin(theta)[:, None]
        denom = float(np.sum(np.broadcast_to(weights, (len(theta), len(self.phi_grid_rad)))))
        energies = np.zeros(len(self.freq_grid_hz), dtype=float)
        for port_name in self.port_names:
            field = np.asarray(self.fields_by_port[port_name], dtype=np.complex128)
            mag2 = np.abs(field[..., 0]) ** 2 + np.abs(field[..., 1]) ** 2
            energies += np.sum(mag2 * weights[None, :, :], axis=(1, 2)) / max(denom, 1e-30)
        energies /= float(len(self.port_names))
        return energies

    def radiation_energy_by_freq(self) -> np.ndarray:
        energies = self._raw_radiation_energy_by_freq()
        return (energies * (np.asarray(self.family_scale_f, dtype=float) ** 2)).astype(float)

    def normalized_copy(self, mode: str = "raw") -> "FFDPattern":
        mode_norm = str(mode).lower()
        if mode_norm not in {"raw", "family_radiated_power"}:
            raise ValueError(f"unsupported normalization mode: {mode}")
        scale = np.ones(len(self.freq_grid_hz), dtype=float)
        if mode_norm == "family_radiated_power":
            scale = self._family_scale_from_raw_fields()
        return replace(self, normalization_mode=mode_norm, family_scale_f=scale)

    def _normalize_phi(self, phi: float) -> float:
        phi_vals = np.asarray(self.phi_grid_rad, dtype=float)
        phi_q = float(phi)
        while phi_q < phi_vals[0]:
            phi_q += 2.0 * np.pi
        while phi_q > phi_vals[-1]:
            phi_q -= 2.0 * np.pi
        return float(np.clip(phi_q, phi_vals[0], phi_vals[-1]))

    def _interpolate_theta_phi(self, grid_tp: np.ndarray, theta: float, phi: float) -> np.ndarray:
        theta_vals = np.asarray(self.theta_grid_rad, dtype=float)
        theta_q = float(np.clip(theta, theta_vals[0], theta_vals[-1]))
        phi_vals = np.asarray(self.phi_grid_rad, dtype=float)
        phi_q = self._normalize_phi(float(phi))

        if len(theta_vals) == 1 and len(phi_vals) == 1:
            return np.asarray(grid_tp[0, 0], dtype=np.complex128)
        if len(theta_vals) == 1:
            pi = int(np.searchsorted(phi_vals, phi_q, side="right") - 1)
            pi = int(np.clip(pi, 0, len(phi_vals) - 2))
            p0 = phi_vals[pi]
            p1 = phi_vals[pi + 1]
            wp = 0.0 if abs(p1 - p0) < 1e-12 else (phi_q - p0) / (p1 - p0)
            return (1.0 - wp) * grid_tp[0, pi] + wp * grid_tp[0, pi + 1]
        if len(phi_vals) == 1:
            ti = int(np.searchsorted(theta_vals, theta_q, side="right") - 1)
            ti = int(np.clip(ti, 0, len(theta_vals) - 2))
            t0 = theta_vals[ti]
            t1 = theta_vals[ti + 1]
            wt = 0.0 if abs(t1 - t0) < 1e-12 else (theta_q - t0) / (t1 - t0)
            return (1.0 - wt) * grid_tp[ti, 0] + wt * grid_tp[ti + 1, 0]
        if np.isclose(theta_q, theta_vals[0], atol=1e-12) or np.isclose(theta_q, theta_vals[-1], atol=1e-12):
            pole_idx = 0 if np.isclose(theta_q, theta_vals[0], atol=1e-12) else -1
            return np.mean(np.asarray(grid_tp[pole_idx], dtype=np.complex128), axis=0)

        ti = int(np.searchsorted(theta_vals, theta_q, side="right") - 1)
        pi = int(np.searchsorted(phi_vals, phi_q, side="right") - 1)
        ti = int(np.clip(ti, 0, len(theta_vals) - 2))
        pi = int(np.clip(pi, 0, len(phi_vals) - 2))
        t0 = theta_vals[ti]
        t1 = theta_vals[ti + 1]
        p0 = phi_vals[pi]
        p1 = phi_vals[pi + 1]
        wt = 0.0 if abs(t1 - t0) < 1e-12 else (theta_q - t0) / (t1 - t0)
        wp = 0.0 if abs(p1 - p0) < 1e-12 else (phi_q - p0) / (p1 - p0)
        g00 = grid_tp[ti, pi]
        g01 = grid_tp[ti, pi + 1]
        g10 = grid_tp[ti + 1, pi]
        g11 = grid_tp[ti + 1, pi + 1]
        return (1.0 - wt) * (1.0 - wp) * g00 + (1.0 - wt) * wp * g01 + wt * (1.0 - wp) * g10 + wt * wp * g11

    def _interpolate_freq(self, samples_f: np.ndarray, query_freq_hz: np.ndarray) -> np.ndarray:
        freq_grid = np.asarray(self.freq_grid_hz, dtype=float)
        q = np.atleast_1d(np.asarray(query_freq_hz, dtype=float))
        if len(freq_grid) == 1:
            if not np.isfinite(freq_grid[0]) or freq_grid[0] <= 0.0:
                return np.repeat(np.asarray(samples_f[0], dtype=np.complex128)[None, :], len(q), axis=0)
            if not np.allclose(q, freq_grid[0], atol=1e-9, rtol=0.0):
                raise ValueError(f"query frequency {q} is outside available FFD range [{freq_grid[0]}, {freq_grid[0]}]")
            return np.repeat(np.asarray(samples_f[0], dtype=np.complex128)[None, :], len(q), axis=0)
        if np.any(q < freq_grid[0]) or np.any(q > freq_grid[-1]):
            raise ValueError(f"query frequency {q} is outside available FFD range [{freq_grid[0]}, {freq_grid[-1]}]")
        out = np.zeros((len(q), samples_f.shape[-1]), dtype=np.complex128)
        for comp_idx in range(samples_f.shape[-1]):
            out[:, comp_idx] = np.interp(q, freq_grid, np.real(samples_f[:, comp_idx])) + 1j * np.interp(q, freq_grid, np.imag(samples_f[:, comp_idx]))
        return out

    def port_field(self, port_id: int, f_hz: float | np.ndarray, k_local: np.ndarray) -> np.ndarray:
        freqs = np.atleast_1d(np.asarray(f_hz, dtype=float))
        theta, phi = self._direction_angles_local(k_local)
        port_name = self.port_names[int(port_id)]
        raw_field = np.asarray(self.fields_by_port[port_name], dtype=np.complex128)
        angle_samples = np.zeros((len(self.freq_grid_hz), 2), dtype=np.complex128)
        for f_idx in range(len(self.freq_grid_hz)):
            angle_samples[f_idx] = self._interpolate_theta_phi(raw_field[f_idx], theta, phi)
        scaled_samples = angle_samples * np.asarray(self.family_scale_f, dtype=float)[:, None]
        return self._interpolate_freq(scaled_samples, freqs)

    def _direction_angles_local(self, k_local: np.ndarray) -> tuple[float, float]:
        k = np.asarray(k_local, dtype=float)
        kn = np.linalg.norm(k)
        if kn == 0.0:
            raise ValueError("direction vector must be non-zero")
        theta = float(np.arccos(np.clip(k[2] / kn, -1.0, 1.0)))
        phi = float(np.arctan2(k[1], k[0]))
        if phi < 0.0:
            phi += 2.0 * np.pi
        return theta, phi

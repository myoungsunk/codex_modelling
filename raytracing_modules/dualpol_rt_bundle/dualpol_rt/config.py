from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_MATERIAL_XPOL_COUPLING_SWEEP_DB = (20.0, 25.0, 30.0, 35.0, 40.0)


@dataclass(frozen=True)
class DepolConfig:
    enabled: bool = False


@dataclass(frozen=True)
class IncidenceSweepConfig:
    standard_band_deg: tuple[float, float] = (0.0, 75.0)
    grazing_band_deg: tuple[float, float] = (75.0, 90.0)

    def regime_for_angle(self, incidence_angle_deg: float) -> str:
        angle = float(incidence_angle_deg)
        if self.standard_band_deg[0] <= angle <= self.standard_band_deg[1]:
            return "standard"
        if self.grazing_band_deg[0] <= angle <= self.grazing_band_deg[1]:
            return "grazing"
        return "out_of_band"


@dataclass(frozen=True)
class ExperimentConfig:
    fc: float = 6.5e9
    bw: float = 500e6
    n_freq: int = 101
    room_size: tuple[float, float, float] = (10.0, 6.0, 3.0)
    tx_pos: np.ndarray = field(default_factory=lambda: np.array([1.5, 3.0, 2.2], dtype=float))
    rx_plane_z: float = 1.2
    grid_step: float = 0.2
    max_reflections: int = 2
    material_preset: str = "uniform_concrete"
    snr_db_list: tuple[float, ...] = (0.0, 10.0, 20.0)
    debug_rx_pos: np.ndarray = field(default_factory=lambda: np.array([5.0, 3.0, 1.2], dtype=float))
    material_xpol_coupling_db: float | None = None
    material_xpol_coupling_sweep_db: tuple[float, ...] = DEFAULT_MATERIAL_XPOL_COUPLING_SWEEP_DB
    incidence_sweep: IncidenceSweepConfig = field(default_factory=IncidenceSweepConfig)
    depol: DepolConfig = field(default_factory=DepolConfig)

    def __post_init__(self) -> None:
        tx = np.asarray(self.tx_pos, dtype=float)
        debug_rx = np.asarray(self.debug_rx_pos, dtype=float)
        tx.setflags(write=False)
        debug_rx.setflags(write=False)
        object.__setattr__(self, "tx_pos", tx)
        object.__setattr__(self, "debug_rx_pos", debug_rx)
        object.__setattr__(self, "material_xpol_coupling_sweep_db", tuple(float(value) for value in self.material_xpol_coupling_sweep_db))
        if self.n_freq <= 0:
            raise ValueError("n_freq must be positive")
        if self.max_reflections not in {0, 1, 2}:
            raise ValueError("Phase 1 baseline supports max_reflections in {0,1,2}")
        if self.grid_step <= 0.0:
            raise ValueError("grid_step must be positive")
        if self.material_xpol_coupling_db is not None and float(self.material_xpol_coupling_db) < 0.0:
            raise ValueError("material_xpol_coupling_db must be non-negative")
        if not self.material_xpol_coupling_sweep_db:
            raise ValueError("material_xpol_coupling_sweep_db must be non-empty")
        if any(value < 0.0 for value in self.material_xpol_coupling_sweep_db):
            raise ValueError("material_xpol_coupling_sweep_db must contain non-negative values")
        if self.incidence_sweep.standard_band_deg != (0.0, 75.0):
            raise ValueError("standard incidence sweep must remain [0 deg, 75 deg]")
        if self.incidence_sweep.grazing_band_deg != (75.0, 90.0):
            raise ValueError("grazing incidence sweep must remain [75 deg, 90 deg]")
        if self.depol.enabled:
            raise ValueError("Depolarization is intentionally disabled in this baseline; keep DepolConfig(enabled=False)")

    @property
    def freqs_hz(self) -> np.ndarray:
        if self.n_freq == 1:
            return np.array([self.fc], dtype=float)
        return np.linspace(self.fc - self.bw / 2.0, self.fc + self.bw / 2.0, self.n_freq, dtype=float)

    @property
    def rx_x(self) -> np.ndarray:
        return np.arange(0.5, self.room_size[0] - 0.5 + 1e-12, self.grid_step, dtype=float)

    @property
    def rx_y(self) -> np.ndarray:
        return np.arange(0.5, self.room_size[1] - 0.5 + 1e-12, self.grid_step, dtype=float)

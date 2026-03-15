"""Floor-XPD models for C0 hardware/alignment leakage proxy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FloorXPDModel:
    def sample_floor_xpd_db(
        self,
        f_hz: float | np.ndarray | None,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> float:
        raise NotImplementedError


@dataclass
class ConstantFloorXPD(FloorXPDModel):
    xpd_floor_db: float = 25.0
    sigma_db: float = 0.0

    def sample_floor_xpd_db(
        self,
        f_hz: float | np.ndarray | None,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> float:
        _ = f_hz, yaw_deg, pitch_deg
        if self.sigma_db > 0.0:
            rr = rng if rng is not None else np.random.default_rng()
            return float(rr.normal(float(self.xpd_floor_db), float(self.sigma_db)))
        return float(self.xpd_floor_db)


@dataclass
class FreqDependentFloorXPD(FloorXPDModel):
    freq_hz: np.ndarray
    xpd_floor_db: np.ndarray
    sigma_db: float = 0.0

    def sample_floor_xpd_db(
        self,
        f_hz: float | np.ndarray | None,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> float:
        _ = yaw_deg, pitch_deg
        if f_hz is None:
            base = float(np.median(np.asarray(self.xpd_floor_db, dtype=float)))
        else:
            f_in = np.asarray(f_hz, dtype=float)
            f0 = float(np.median(f_in))
            base = float(
                np.interp(
                    f0,
                    np.asarray(self.freq_hz, dtype=float),
                    np.asarray(self.xpd_floor_db, dtype=float),
                    left=float(np.asarray(self.xpd_floor_db, dtype=float)[0]),
                    right=float(np.asarray(self.xpd_floor_db, dtype=float)[-1]),
                )
            )
        if self.sigma_db > 0.0:
            rr = rng if rng is not None else np.random.default_rng()
            return float(rr.normal(base, float(self.sigma_db)))
        return base


@dataclass
class AngleSensitiveFloorXPD(FloorXPDModel):
    base_db: float = 25.0
    yaw_slope_db_per_deg: float = -0.05
    pitch_slope_db_per_deg: float = -0.02
    sigma_db: float = 0.0

    def sample_floor_xpd_db(
        self,
        f_hz: float | np.ndarray | None,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> float:
        _ = f_hz
        mu = (
            float(self.base_db)
            + float(self.yaw_slope_db_per_deg) * abs(float(yaw_deg))
            + float(self.pitch_slope_db_per_deg) * abs(float(pitch_deg))
        )
        if self.sigma_db > 0.0:
            rr = rng if rng is not None else np.random.default_rng()
            return float(rr.normal(mu, float(self.sigma_db)))
        return float(mu)

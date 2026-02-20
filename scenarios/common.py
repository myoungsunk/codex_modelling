"""Shared scenario helpers for building antennas and frequency sweeps."""

from __future__ import annotations

import numpy as np

from rt_core.antenna import Antenna


def default_antennas(
    basis: str = "linear",
    convention: str = "IEEE-RHCP",
    tx_cross_pol_leakage_db: float = 35.0,
    rx_cross_pol_leakage_db: float = 35.0,
    tx_axial_ratio_db: float = 0.0,
    rx_axial_ratio_db: float = 0.0,
    enable_coupling: bool = True,
) -> tuple[Antenna, Antenna]:
    tx = Antenna(
        position=np.array([0.0, 0.0, 1.5]),
        boresight=np.array([1.0, 0.0, 0.0]),
        h_axis=np.array([0.0, 1.0, 0.0]),
        v_axis=np.array([0.0, 0.0, 1.0]),
        basis=basis,
        convention=convention,
        cross_pol_leakage_db=tx_cross_pol_leakage_db,
        axial_ratio_db=tx_axial_ratio_db,
        enable_coupling=enable_coupling,
    )
    rx = Antenna(
        position=np.array([6.0, 0.0, 1.5]),
        boresight=np.array([-1.0, 0.0, 0.0]),
        h_axis=np.array([0.0, 1.0, 0.0]),
        v_axis=np.array([0.0, 0.0, 1.0]),
        basis=basis,
        convention=convention,
        cross_pol_leakage_db=rx_cross_pol_leakage_db,
        axial_ratio_db=rx_axial_ratio_db,
        enable_coupling=enable_coupling,
    )
    return tx, rx


def uwb_frequency(nf: int = 256, f_min: float = 6e9, f_max: float = 10e9) -> np.ndarray:
    return np.linspace(f_min, f_max, nf, dtype=float)


def paths_to_records(paths: list) -> list[dict]:
    rec = []
    for p in paths:
        rec.append(
            {
                "tau_s": p.tau_s,
                "A_f": p.A_f,
                "J_f": p.J_f,
                "scalar_gain_f": p.scalar_gain_f,
                "G_tx_f": p.G_tx_f,
                "G_rx_f": p.G_rx_f,
                "meta": p.meta_dict(),
                "points": [x.tolist() for x in p.points],
            }
        )
    return rec

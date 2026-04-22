"""Shared scenario helpers for building antennas and frequency sweeps."""

from __future__ import annotations

import numpy as np

from rt_core.antenna import Antenna
from rt_core.geometry import Material, Plane, normalize


def default_antennas(
    basis: str = "linear",
    convention: str = "IEEE-RHCP",
    tx_cross_pol_leakage_db: float = 35.0,
    rx_cross_pol_leakage_db: float = 35.0,
    tx_axial_ratio_db: float = 0.0,
    rx_axial_ratio_db: float = 0.0,
    enable_coupling: bool = True,
    tx_coupling_ref_freq_hz: float = 8.0e9,
    rx_coupling_ref_freq_hz: float = 8.0e9,
    tx_cross_pol_leakage_db_slope_per_ghz: float = 0.0,
    rx_cross_pol_leakage_db_slope_per_ghz: float = 0.0,
    tx_axial_ratio_db_slope_per_ghz: float = 0.0,
    rx_axial_ratio_db_slope_per_ghz: float = 0.0,
    tx_cross_coupling_phase_deg: float = 0.0,
    rx_cross_coupling_phase_deg: float = 0.0,
    tx_cross_coupling_phase_slope_deg_per_ghz: float = 0.0,
    rx_cross_coupling_phase_slope_deg_per_ghz: float = 0.0,
    tx_peak_gain_dbi: float = 0.0,
    rx_peak_gain_dbi: float = 0.0,
    tx_pattern_cos_exp: float = 0.0,
    rx_pattern_cos_exp: float = 0.0,
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
        coupling_ref_freq_hz=tx_coupling_ref_freq_hz,
        cross_pol_leakage_db_slope_per_ghz=tx_cross_pol_leakage_db_slope_per_ghz,
        axial_ratio_db_slope_per_ghz=tx_axial_ratio_db_slope_per_ghz,
        cross_coupling_phase_deg=tx_cross_coupling_phase_deg,
        cross_coupling_phase_slope_deg_per_ghz=tx_cross_coupling_phase_slope_deg_per_ghz,
        tx_peak_gain_dbi=tx_peak_gain_dbi,
        tx_pattern_cos_exp=tx_pattern_cos_exp,
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
        coupling_ref_freq_hz=rx_coupling_ref_freq_hz,
        cross_pol_leakage_db_slope_per_ghz=rx_cross_pol_leakage_db_slope_per_ghz,
        axial_ratio_db_slope_per_ghz=rx_axial_ratio_db_slope_per_ghz,
        cross_coupling_phase_deg=rx_cross_coupling_phase_deg,
        cross_coupling_phase_slope_deg_per_ghz=rx_cross_coupling_phase_slope_deg_per_ghz,
        rx_peak_gain_dbi=rx_peak_gain_dbi,
        rx_pattern_cos_exp=rx_pattern_cos_exp,
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
                "path_length_m": float(p.path_length_m),
                "segment_count": int(p.segment_count),
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


def make_los_blocker_plane(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    plane_id: int,
    half_extent_u: float = 0.35,
    half_extent_v: float = 0.55,
    material: Material | None = None,
) -> Plane:
    """Build a finite los-blocker plate centered on the Tx-Rx midpoint.

    The plate normal is aligned to the Tx->Rx direction so the direct segment is
    physically occluded when LOS tracing is enabled.
    """

    tx = np.asarray(tx_pos, dtype=float)
    rx = np.asarray(rx_pos, dtype=float)
    n = normalize(rx - tx)
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = normalize(np.cross(ref, n))
    v = normalize(np.cross(n, u))
    p0 = 0.5 * (tx + rx)
    mat = material if material is not None else Material.dielectric(eps_r=1.15, tan_delta=1.0, name="absorber_proxy")
    return Plane(
        id=int(plane_id),
        p0=np.asarray(p0, dtype=float),
        normal=np.asarray(n, dtype=float),
        material=mat,
        u_axis=u,
        v_axis=v,
        half_extent_u=float(half_extent_u),
        half_extent_v=float(half_extent_v),
    )

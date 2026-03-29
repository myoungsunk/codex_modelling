from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dualpol_rt.channel.builder import build_channel, convert_basis
from dualpol_rt.config import ExperimentConfig
from dualpol_rt.geometry.image_method import enumerate_paths
from dualpol_rt.metrics.comparison import ChannelDeltaSummary, ChannelSummary, compare_channels, summarize_channel
from dualpol_rt.metrics.xpr import condition_numbers, xpr_db
from dualpol_rt.patterns.antenna_family import AntennaFamily, AntennaPose
from dualpol_rt.patterns.ffd_pattern import FFDPattern
from dualpol_rt.scene.shoebox import build_shoebox


@dataclass(frozen=True)
class SystemMaps:
    rate_maps: dict[float, np.ndarray]
    capacity_maps: dict[float, np.ndarray]
    gain_normalized_rate_maps: dict[float, np.ndarray]
    xpr_db_map: np.ndarray
    condition_number_map: np.ndarray


@dataclass(frozen=True)
class CompareModeMaps:
    lp: SystemMaps
    cp: SystemMaps
    delta_rate_maps: dict[float, np.ndarray]
    delta_capacity_maps: dict[float, np.ndarray]
    delta_gain_normalized_rate_maps: dict[float, np.ndarray]
    delta_xpr_db_map: np.ndarray
    delta_condition_number_map: np.ndarray


@dataclass(frozen=True)
class RealisticCompareResult:
    rx_x: np.ndarray
    rx_y: np.ndarray
    raw: CompareModeMaps
    normalized: CompareModeMaps


def _identity_pose(origin: np.ndarray | None = None) -> AntennaPose:
    return AntennaPose(origin=np.zeros(3, dtype=float) if origin is None else np.asarray(origin, dtype=float))


def build_phase2_patterns(
    tx_lp_files: tuple[str | Path, str | Path],
    rx_lp_files: tuple[str | Path, str | Path],
    tx_cp_files: tuple[str | Path, str | Path],
    rx_cp_files: tuple[str | Path, str | Path],
    normalization_mode: str = "raw",
    tx_pose: AntennaPose | None = None,
    rx_pose: AntennaPose | None = None,
) -> dict[str, AntennaFamily]:
    """Build the four Phase 2 antenna families from individual HFSS FFD files."""
    tx_pose_use = _identity_pose() if tx_pose is None else tx_pose
    rx_pose_use = _identity_pose() if rx_pose is None else rx_pose
    tx_lp_pattern = FFDPattern.from_port_files(
        {"lp_p0": tx_lp_files[0], "lp_p1": tx_lp_files[1]},
        port_order=("lp_p0", "lp_p1"),
        normalization_mode=normalization_mode,
        sweep_order="theta_inner",
    )
    rx_lp_pattern = FFDPattern.from_port_files(
        {"lp_p0": rx_lp_files[0], "lp_p1": rx_lp_files[1]},
        port_order=("lp_p0", "lp_p1"),
        normalization_mode=normalization_mode,
        sweep_order="theta_inner",
    )
    tx_cp_pattern = FFDPattern.from_port_files(
        {"cp_p0": tx_cp_files[0], "cp_p1": tx_cp_files[1]},
        port_order=("cp_p0", "cp_p1"),
        normalization_mode=normalization_mode,
        sweep_order="theta_inner",
    )
    rx_cp_pattern = FFDPattern.from_port_files(
        {"cp_p0": rx_cp_files[0], "cp_p1": rx_cp_files[1]},
        port_order=("cp_p0", "cp_p1"),
        normalization_mode=normalization_mode,
        sweep_order="theta_inner",
    )
    return {
        "tx_lp": AntennaFamily("dual_linear_slant_tx", tx_lp_pattern, tx_pose_use, ("+45", "-45"), ("+45", "-45"), normalization_mode),
        "rx_lp": AntennaFamily("dual_linear_slant_rx", rx_lp_pattern, rx_pose_use, ("+45", "-45"), ("+45", "-45"), normalization_mode),
        "tx_cp": AntennaFamily("dual_cp_tx", tx_cp_pattern, tx_pose_use, ("RHCP", "LHCP"), ("RHCP", "LHCP"), normalization_mode),
        "rx_cp": AntennaFamily("dual_cp_rx", rx_cp_pattern, rx_pose_use, ("RHCP", "LHCP"), ("RHCP", "LHCP"), normalization_mode),
    }


def build_phase2_patterns_from_standard_files(
    ffd_dir: str | Path,
    normalization_mode: str = "raw",
    tx_pose: AntennaPose | None = None,
    rx_pose: AntennaPose | None = None,
) -> dict[str, AntennaFamily]:
    """Build LP(+45/-45) and CP(RHCP/LHCP) families from the standard four-file bundle."""
    root = Path(ffd_dir)
    lp_files = (root / "LP_+45_new_6G7G_11pts.ffd", root / "LP_-45_new_6G7G_11pts.ffd")
    cp_files = (root / "RHCP_new_6G7G_11pts.ffd", root / "LHCP_new_6G7G_11pts.ffd")
    missing = [str(path) for path in (*lp_files, *cp_files) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing standard HFSS FFD files: {missing}")
    return build_phase2_patterns(
        tx_lp_files=lp_files,
        rx_lp_files=lp_files,
        tx_cp_files=cp_files,
        rx_cp_files=cp_files,
        normalization_mode=normalization_mode,
        tx_pose=tx_pose,
        rx_pose=rx_pose,
    )


def _facing_families(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    tx_family: AntennaFamily,
    rx_family: AntennaFamily,
) -> tuple[AntennaFamily, AntennaFamily]:
    tx_pose = AntennaPose.facing_from_points(tx_pos, rx_pos)
    rx_pose = AntennaPose.facing_from_points(rx_pos, tx_pos)
    return tx_family.with_pose(tx_pose), rx_family.with_pose(rx_pose)


def _empty_system_maps(shape: tuple[int, int], snr_db_list: tuple[float, ...]) -> SystemMaps:
    return SystemMaps(
        rate_maps={float(snr): np.full(shape, np.nan, dtype=float) for snr in snr_db_list},
        capacity_maps={float(snr): np.full(shape, np.nan, dtype=float) for snr in snr_db_list},
        gain_normalized_rate_maps={float(snr): np.full(shape, np.nan, dtype=float) for snr in snr_db_list},
        xpr_db_map=np.full(shape, np.nan, dtype=float),
        condition_number_map=np.full(shape, np.nan, dtype=float),
    )


def _empty_compare_mode_maps(shape: tuple[int, int], snr_db_list: tuple[float, ...]) -> CompareModeMaps:
    return CompareModeMaps(
        lp=_empty_system_maps(shape, snr_db_list),
        cp=_empty_system_maps(shape, snr_db_list),
        delta_rate_maps={float(snr): np.full(shape, np.nan, dtype=float) for snr in snr_db_list},
        delta_capacity_maps={float(snr): np.full(shape, np.nan, dtype=float) for snr in snr_db_list},
        delta_gain_normalized_rate_maps={float(snr): np.full(shape, np.nan, dtype=float) for snr in snr_db_list},
        delta_xpr_db_map=np.full(shape, np.nan, dtype=float),
        delta_condition_number_map=np.full(shape, np.nan, dtype=float),
    )


def _write_summary(maps: SystemMaps, summary: ChannelSummary, idx: tuple[int, int]) -> None:
    iy, ix = idx
    for snr, value in summary.equal_power_rate.items():
        maps.rate_maps[float(snr)][iy, ix] = float(value)
    for snr, value in summary.waterfilled_capacity.items():
        maps.capacity_maps[float(snr)][iy, ix] = float(value)
    for snr, value in summary.gain_normalized_equal_power_rate.items():
        maps.gain_normalized_rate_maps[float(snr)][iy, ix] = float(value)
    maps.xpr_db_map[iy, ix] = float(summary.xpr_db)
    maps.condition_number_map[iy, ix] = float(summary.mean_condition_number)


def _write_delta(maps: CompareModeMaps, delta: ChannelDeltaSummary, idx: tuple[int, int]) -> None:
    iy, ix = idx
    for snr, value in delta.delta_equal_power_rate.items():
        maps.delta_rate_maps[float(snr)][iy, ix] = float(value)
    for snr, value in delta.delta_waterfilled_capacity.items():
        maps.delta_capacity_maps[float(snr)][iy, ix] = float(value)
    for snr, value in delta.delta_gain_normalized_equal_power_rate.items():
        maps.delta_gain_normalized_rate_maps[float(snr)][iy, ix] = float(value)
    maps.delta_xpr_db_map[iy, ix] = float(delta.delta_xpr_db)
    maps.delta_condition_number_map[iy, ix] = float(delta.delta_condition_number)


def _build_mode_channels(
    paths,
    freqs_hz: np.ndarray,
    tx_lp_family: AntennaFamily,
    rx_lp_family: AntennaFamily,
    tx_cp_family: AntennaFamily,
    rx_cp_family: AntennaFamily,
    normalization_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    tx_lp = tx_lp_family.normalized_copy(normalization_mode)
    rx_lp = rx_lp_family.normalized_copy(normalization_mode)
    tx_cp = tx_cp_family.normalized_copy(normalization_mode)
    rx_cp = rx_cp_family.normalized_copy(normalization_mode)
    H_lp = build_channel(paths, tx_pattern=tx_lp, rx_pattern=rx_lp, freqs_hz=freqs_hz, eval_basis="linear")
    H_cp = build_channel(paths, tx_pattern=tx_cp, rx_pattern=rx_cp, freqs_hz=freqs_hz, eval_basis="linear")
    return H_lp, H_cp


def evaluate_realistic_debug_point(
    cfg: ExperimentConfig,
    rx_pos: np.ndarray,
    tx_lp_family: AntennaFamily,
    rx_lp_family: AntennaFamily,
    tx_cp_family: AntennaFamily,
    rx_cp_family: AntennaFamily,
) -> dict[str, object]:
    scene = build_shoebox(room_size=cfg.room_size, material_preset=cfg.material_preset)
    rx = np.asarray(rx_pos, dtype=float)
    tx_lp_link, rx_lp_link = _facing_families(cfg.tx_pos, rx, tx_lp_family, rx_lp_family)
    tx_cp_link, rx_cp_link = _facing_families(cfg.tx_pos, rx, tx_cp_family, rx_cp_family)
    paths = enumerate_paths(scene, cfg.tx_pos, rx, cfg.max_reflections)

    H_lp_raw, H_cp_raw = _build_mode_channels(paths, cfg.freqs_hz, tx_lp_link, rx_lp_link, tx_cp_link, rx_cp_link, normalization_mode="raw")
    H_lp_norm, H_cp_norm = _build_mode_channels(
        paths,
        cfg.freqs_hz,
        tx_lp_link,
        rx_lp_link,
        tx_cp_link,
        rx_cp_link,
        normalization_mode="family_radiated_power",
    )
    same_family_basis_invariance = {
        "raw": float(np.max(np.abs(condition_numbers(H_lp_raw) - condition_numbers(convert_basis(H_lp_raw, src="linear", dst="circular"))))),
        "normalized": float(np.max(np.abs(condition_numbers(H_lp_norm) - condition_numbers(convert_basis(H_lp_norm, src="linear", dst="circular"))))),
    }

    return {
        "rx_pos": rx,
        "path_count": len(paths),
        "paths": paths,
        "H_lp_raw": H_lp_raw,
        "H_cp_raw": H_cp_raw,
        "H_lp_normalized": H_lp_norm,
        "H_cp_normalized": H_cp_norm,
        "lp_raw_summary": summarize_channel(H_lp_raw, cfg.snr_db_list),
        "cp_raw_summary": summarize_channel(H_cp_raw, cfg.snr_db_list),
        "raw_delta": compare_channels(H_lp_raw, H_cp_raw, cfg.snr_db_list),
        "lp_normalized_summary": summarize_channel(H_lp_norm, cfg.snr_db_list, gain_normalized_mode="channel_equal_power"),
        "cp_normalized_summary": summarize_channel(H_cp_norm, cfg.snr_db_list, gain_normalized_mode="channel_equal_power"),
        "normalized_delta": compare_channels(H_lp_norm, H_cp_norm, cfg.snr_db_list, gain_normalized_mode="channel_equal_power"),
        "raw_mean_channel_norms": {
            "lp": float(np.mean(np.linalg.norm(H_lp_raw, axis=(1, 2)))),
            "cp": float(np.mean(np.linalg.norm(H_cp_raw, axis=(1, 2)))),
        },
        "normalized_mean_channel_norms": {
            "lp": float(np.mean(np.linalg.norm(H_lp_norm, axis=(1, 2)))),
            "cp": float(np.mean(np.linalg.norm(H_cp_norm, axis=(1, 2)))),
        },
        "same_family_basis_invariance_lp": same_family_basis_invariance,
        "same_family_basis_invariance_cp": same_family_basis_invariance,
        "same_family_basis_invariance_note": "Representation invariance is evaluated on the LP-family channel only; CP-family channels already live in circular port space.",
    }


def evaluate_realistic_rx_grid(
    cfg: ExperimentConfig,
    tx_lp_family: AntennaFamily,
    rx_lp_family: AntennaFamily,
    tx_cp_family: AntennaFamily,
    rx_cp_family: AntennaFamily,
) -> RealisticCompareResult:
    scene = build_shoebox(room_size=cfg.room_size, material_preset=cfg.material_preset)
    shape = (len(cfg.rx_y), len(cfg.rx_x))
    raw_maps = _empty_compare_mode_maps(shape, cfg.snr_db_list)
    norm_maps = _empty_compare_mode_maps(shape, cfg.snr_db_list)

    for iy, y in enumerate(cfg.rx_y):
        for ix, x in enumerate(cfg.rx_x):
            rx_pos = np.array([x, y, cfg.rx_plane_z], dtype=float)
            paths = enumerate_paths(scene, cfg.tx_pos, rx_pos, cfg.max_reflections)
            tx_lp_link, rx_lp_link = _facing_families(cfg.tx_pos, rx_pos, tx_lp_family, rx_lp_family)
            tx_cp_link, rx_cp_link = _facing_families(cfg.tx_pos, rx_pos, tx_cp_family, rx_cp_family)
            H_lp_raw, H_cp_raw = _build_mode_channels(paths, cfg.freqs_hz, tx_lp_link, rx_lp_link, tx_cp_link, rx_cp_link, normalization_mode="raw")
            H_lp_norm, H_cp_norm = _build_mode_channels(
                paths,
                cfg.freqs_hz,
                tx_lp_link,
                rx_lp_link,
                tx_cp_link,
                rx_cp_link,
                normalization_mode="family_radiated_power",
            )
            lp_raw_summary = summarize_channel(H_lp_raw, cfg.snr_db_list)
            cp_raw_summary = summarize_channel(H_cp_raw, cfg.snr_db_list)
            raw_delta = compare_channels(H_lp_raw, H_cp_raw, cfg.snr_db_list)
            lp_norm_summary = summarize_channel(H_lp_norm, cfg.snr_db_list, gain_normalized_mode="channel_equal_power")
            cp_norm_summary = summarize_channel(H_cp_norm, cfg.snr_db_list, gain_normalized_mode="channel_equal_power")
            norm_delta = compare_channels(H_lp_norm, H_cp_norm, cfg.snr_db_list, gain_normalized_mode="channel_equal_power")
            _write_summary(raw_maps.lp, lp_raw_summary, (iy, ix))
            _write_summary(raw_maps.cp, cp_raw_summary, (iy, ix))
            _write_delta(raw_maps, raw_delta, (iy, ix))
            _write_summary(norm_maps.lp, lp_norm_summary, (iy, ix))
            _write_summary(norm_maps.cp, cp_norm_summary, (iy, ix))
            _write_delta(norm_maps, norm_delta, (iy, ix))

    return RealisticCompareResult(rx_x=cfg.rx_x, rx_y=cfg.rx_y, raw=raw_maps, normalized=norm_maps)

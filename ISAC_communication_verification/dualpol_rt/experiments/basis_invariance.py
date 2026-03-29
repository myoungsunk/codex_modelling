from __future__ import annotations

from dataclasses import asdict

import numpy as np

from dualpol_rt.channel.builder import build_channel, convert_basis
from dualpol_rt.channel.path_record import GridResult
from dualpol_rt.config import ExperimentConfig
from dualpol_rt.geometry.image_method import enumerate_paths
from dualpol_rt.metrics.achievable_rate import equal_power_rate, waterfilled_capacity
from dualpol_rt.metrics.xpr import condition_numbers, xpr_db
from dualpol_rt.scene.shoebox import build_shoebox


def evaluate_debug_point(
    cfg: ExperimentConfig,
    rx_pos: np.ndarray,
    tx_pattern,
    rx_pattern,
) -> dict[str, object]:
    scene = build_shoebox(room_size=cfg.room_size, material_preset=cfg.material_preset)
    paths = enumerate_paths(scene, cfg.tx_pos, np.asarray(rx_pos, dtype=float), cfg.max_reflections)
    H_lin = build_channel(paths, tx_pattern=tx_pattern, rx_pattern=rx_pattern, freqs_hz=cfg.freqs_hz, eval_basis="linear")
    H_circ = convert_basis(H_lin, src="linear", dst="circular")
    rates_lin = {float(snr): equal_power_rate(H_lin, snr) for snr in cfg.snr_db_list}
    rates_circ = {float(snr): equal_power_rate(H_circ, snr) for snr in cfg.snr_db_list}
    caps_lin = {float(snr): waterfilled_capacity(H_lin, snr) for snr in cfg.snr_db_list}
    caps_circ = {float(snr): waterfilled_capacity(H_circ, snr) for snr in cfg.snr_db_list}
    return {
        "config": asdict(cfg),
        "rx_pos": np.asarray(rx_pos, dtype=float),
        "path_count": len(paths),
        "H_linear": H_lin,
        "H_circular": H_circ,
        "rates_linear": rates_lin,
        "rates_circular": rates_circ,
        "capacity_linear": caps_lin,
        "capacity_circular": caps_circ,
        "xpr_db": xpr_db(H_lin),
        "condition_number": float(np.mean(condition_numbers(H_lin))),
    }


def evaluate_rx_grid(
    cfg: ExperimentConfig,
    tx_pattern,
    rx_pattern,
) -> GridResult:
    scene = build_shoebox(room_size=cfg.room_size, material_preset=cfg.material_preset)
    shape = (len(cfg.rx_y), len(cfg.rx_x))
    linear_rate_maps = {float(snr): np.full(shape, np.nan, dtype=float) for snr in cfg.snr_db_list}
    circular_rate_maps = {float(snr): np.full(shape, np.nan, dtype=float) for snr in cfg.snr_db_list}
    linear_capacity_maps = {float(snr): np.full(shape, np.nan, dtype=float) for snr in cfg.snr_db_list}
    circular_capacity_maps = {float(snr): np.full(shape, np.nan, dtype=float) for snr in cfg.snr_db_list}
    delta_rate_maps = {float(snr): np.full(shape, np.nan, dtype=float) for snr in cfg.snr_db_list}
    xpr_map = np.full(shape, np.nan, dtype=float)
    condition_map = np.full(shape, np.nan, dtype=float)

    for iy, y in enumerate(cfg.rx_y):
        for ix, x in enumerate(cfg.rx_x):
            rx_pos = np.array([x, y, cfg.rx_plane_z], dtype=float)
            paths = enumerate_paths(scene, cfg.tx_pos, rx_pos, cfg.max_reflections)
            H_lin = build_channel(paths, tx_pattern=tx_pattern, rx_pattern=rx_pattern, freqs_hz=cfg.freqs_hz, eval_basis="linear")
            H_circ = convert_basis(H_lin, src="linear", dst="circular")
            xpr_map[iy, ix] = xpr_db(H_lin)
            condition_map[iy, ix] = float(np.mean(condition_numbers(H_lin)))
            for snr in cfg.snr_db_list:
                snr_key = float(snr)
                rate_lin = equal_power_rate(H_lin, snr)
                rate_circ = equal_power_rate(H_circ, snr)
                cap_lin = waterfilled_capacity(H_lin, snr)
                cap_circ = waterfilled_capacity(H_circ, snr)
                linear_rate_maps[snr_key][iy, ix] = rate_lin
                circular_rate_maps[snr_key][iy, ix] = rate_circ
                linear_capacity_maps[snr_key][iy, ix] = cap_lin
                circular_capacity_maps[snr_key][iy, ix] = cap_circ
                delta_rate_maps[snr_key][iy, ix] = rate_circ - rate_lin

    return GridResult(
        rx_x=cfg.rx_x,
        rx_y=cfg.rx_y,
        linear_rate_maps=linear_rate_maps,
        circular_rate_maps=circular_rate_maps,
        linear_capacity_maps=linear_capacity_maps,
        circular_capacity_maps=circular_capacity_maps,
        delta_rate_maps=delta_rate_maps,
        xpr_db_map=xpr_map,
        condition_number_map=condition_map,
    )

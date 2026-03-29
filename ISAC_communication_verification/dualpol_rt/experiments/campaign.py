from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dualpol_rt.channel.builder import build_channel
from dualpol_rt.config import ExperimentConfig
from dualpol_rt.em.jones import attach_em_response
from dualpol_rt.em.materials import material_named
from dualpol_rt.experiments.realistic_compare import CompareModeMaps, RealisticCompareResult, SystemMaps
from dualpol_rt.geometry.image_method import enumerate_paths
from dualpol_rt.metrics.comparison import ChannelDeltaSummary, ChannelSummary, compare_channels, summarize_channel
from dualpol_rt.metrics.richness import PathRichnessMetrics, compute_mrs_scores, compute_path_richness, label_mrs_scores
from dualpol_rt.patterns import AntennaFamily, AntennaPose
from dualpol_rt.scene.scenarios import ScenarioSpec, build_scenario_layout, canonical_scenarios, get_scenario_spec, scenario_grid
from dualpol_rt.scene.surfaces import Scene, Surface


@dataclass(frozen=True)
class PoseProtocol:
    name: str
    yaw_range_deg: tuple[float, float] = (0.0, 0.0)
    pitch_range_deg: tuple[float, float] = (0.0, 0.0)
    roll_range_deg: tuple[float, float] = (0.0, 0.0)
    blocker: bool = False

    def sample_angles(self, rng: np.random.Generator) -> tuple[float, float, float]:
        yaw = float(rng.uniform(self.yaw_range_deg[0], self.yaw_range_deg[1]))
        pitch = float(rng.uniform(self.pitch_range_deg[0], self.pitch_range_deg[1]))
        roll = float(rng.uniform(self.roll_range_deg[0], self.roll_range_deg[1]))
        return yaw, pitch, roll


@dataclass(frozen=True)
class SceneRichnessSummary:
    scenario_name: str
    proxy: bool
    point_count: int
    median_mrs_score: float
    min_mrs_score: float
    max_mrs_score: float
    median_n_eff: float
    median_tau_rms_ns: float
    median_mean_aoa_spread_deg: float
    median_dpr_db: float
    median_pmi_env: float
    bucket_counts: dict[str, int]


@dataclass(frozen=True)
class CampaignSelection:
    mrs_low_threshold: float
    mrs_high_threshold: float
    representatives: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class CampaignResult:
    scene_summaries: tuple[SceneRichnessSummary, ...]
    selection: CampaignSelection
    screening_rows: tuple[dict[str, Any], ...]
    pose_rows: tuple[dict[str, Any], ...]
    full_compare_results: dict[str, RealisticCompareResult]
    headline_summary: dict[str, Any]
    artifact_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class _PointRecord:
    scenario_name: str
    proxy: bool
    rx_pos: np.ndarray
    richness: PathRichnessMetrics
    lp_rate_raw: dict[float, float]
    cp_rate_raw: dict[float, float]
    delta_rate_raw: dict[float, float]
    lp_capacity_raw: dict[float, float]
    cp_capacity_raw: dict[float, float]
    delta_capacity_raw: dict[float, float]
    lp_rate_norm: dict[float, float]
    cp_rate_norm: dict[float, float]
    delta_rate_norm: dict[float, float]
    lp_capacity_norm: dict[float, float]
    cp_capacity_norm: dict[float, float]
    delta_capacity_norm: dict[float, float]
    mrs_raw: float = np.nan
    mrs_score: float = np.nan
    mrs_label: str = ""
    selected_role: str = ""


def standard_pose_protocols() -> dict[str, PoseProtocol]:
    return {
        "P0": PoseProtocol(name="P0"),
        "P1": PoseProtocol(name="P1", yaw_range_deg=(-180.0, 180.0)),
        "P2": PoseProtocol(name="P2", yaw_range_deg=(-180.0, 180.0), roll_range_deg=(-90.0, 90.0)),
        "P3": PoseProtocol(name="P3", yaw_range_deg=(-180.0, 180.0), roll_range_deg=(-90.0, 90.0), blocker=True),
    }


def _make_cfg(spec: ScenarioSpec, fc: float, bw: float, n_freq: int, grid_step: float, snr_db_list: tuple[float, ...], max_reflections: int) -> ExperimentConfig:
    return ExperimentConfig(
        fc=fc,
        bw=bw,
        n_freq=n_freq,
        room_size=spec.room_size,
        tx_pos=spec.tx_pos,
        rx_plane_z=spec.rx_plane_z,
        grid_step=grid_step,
        max_reflections=max_reflections,
        material_preset=spec.material_preset,
        snr_db_list=snr_db_list,
        debug_rx_pos=np.array([spec.room_size[0] * 0.5, spec.room_size[1] * 0.5, spec.rx_plane_z], dtype=float),
    )


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
    paths: list,
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


def _families_for_pose(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    tx_lp_family: AntennaFamily,
    rx_lp_family: AntennaFamily,
    tx_cp_family: AntennaFamily,
    rx_cp_family: AntennaFamily,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> tuple[AntennaFamily, AntennaFamily, AntennaFamily, AntennaFamily]:
    tx_pose = AntennaPose.facing_from_points(tx_pos, rx_pos)
    rx_pose = AntennaPose.facing_from_points(rx_pos, tx_pos).rotated_local(yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg)
    return (
        tx_lp_family.with_pose(tx_pose),
        rx_lp_family.with_pose(rx_pose),
        tx_cp_family.with_pose(tx_pose),
        rx_cp_family.with_pose(rx_pose),
    )


def _human_blocker_scene(scene: Scene, tx_pos: np.ndarray, rx_pos: np.ndarray) -> Scene:
    center = 0.5 * (np.asarray(tx_pos, dtype=float) + np.asarray(rx_pos, dtype=float))
    center[2] = 0.85
    hx, hy, hz = 0.25, 0.15, 0.85
    x0, x1 = center[0] - hx, center[0] + hx
    y0, y1 = center[1] - hy, center[1] + hy
    z0, z1 = center[2] - hz, center[2] + hz
    start_id = max(surface.surface_id for surface in scene.surfaces) + 1
    material = material_named("human_blocker_proxy")
    blocker_surfaces = (
        Surface(start_id + 0, "blocker_x0", point=[x0, 0.5 * (y0 + y1), 0.5 * (z0 + z1)], normal=[-1.0, 0.0, 0.0], u_axis=[0.0, 1.0, 0.0], v_axis=[0.0, 0.0, 1.0], half_u=0.5 * (y1 - y0), half_v=0.5 * (z1 - z0), material=material),
        Surface(start_id + 1, "blocker_x1", point=[x1, 0.5 * (y0 + y1), 0.5 * (z0 + z1)], normal=[1.0, 0.0, 0.0], u_axis=[0.0, 1.0, 0.0], v_axis=[0.0, 0.0, 1.0], half_u=0.5 * (y1 - y0), half_v=0.5 * (z1 - z0), material=material),
        Surface(start_id + 2, "blocker_y0", point=[0.5 * (x0 + x1), y0, 0.5 * (z0 + z1)], normal=[0.0, -1.0, 0.0], u_axis=[1.0, 0.0, 0.0], v_axis=[0.0, 0.0, 1.0], half_u=0.5 * (x1 - x0), half_v=0.5 * (z1 - z0), material=material),
        Surface(start_id + 3, "blocker_y1", point=[0.5 * (x0 + x1), y1, 0.5 * (z0 + z1)], normal=[0.0, 1.0, 0.0], u_axis=[1.0, 0.0, 0.0], v_axis=[0.0, 0.0, 1.0], half_u=0.5 * (x1 - x0), half_v=0.5 * (z1 - z0), material=material),
        Surface(start_id + 4, "blocker_z0", point=[0.5 * (x0 + x1), 0.5 * (y0 + y1), z0], normal=[0.0, 0.0, -1.0], u_axis=[1.0, 0.0, 0.0], v_axis=[0.0, 1.0, 0.0], half_u=0.5 * (x1 - x0), half_v=0.5 * (y1 - y0), material=material),
        Surface(start_id + 5, "blocker_z1", point=[0.5 * (x0 + x1), 0.5 * (y0 + y1), z1], normal=[0.0, 0.0, 1.0], u_axis=[1.0, 0.0, 0.0], v_axis=[0.0, 1.0, 0.0], half_u=0.5 * (x1 - x0), half_v=0.5 * (y1 - y0), material=material),
    )
    return Scene(surfaces=tuple(scene.surfaces) + blocker_surfaces)


def _evaluate_point(
    scene: Scene,
    cfg: ExperimentConfig,
    rx_pos: np.ndarray,
    tx_lp_family: AntennaFamily,
    rx_lp_family: AntennaFamily,
    tx_cp_family: AntennaFamily,
    rx_cp_family: AntennaFamily,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    blocker: bool = False,
) -> tuple[PathRichnessMetrics, ChannelSummary, ChannelSummary, ChannelDeltaSummary, ChannelSummary, ChannelSummary, ChannelDeltaSummary]:
    use_scene = _human_blocker_scene(scene, cfg.tx_pos, rx_pos) if blocker else scene
    paths = enumerate_paths(use_scene, cfg.tx_pos, rx_pos, cfg.max_reflections)
    enriched_paths = [attach_em_response(path, freqs_hz=cfg.freqs_hz) if path.jones_f is None or path.scalar_factor_f is None else path for path in paths]
    tx_lp_pose, rx_lp_pose, tx_cp_pose, rx_cp_pose = _families_for_pose(
        cfg.tx_pos,
        rx_pos,
        tx_lp_family,
        rx_lp_family,
        tx_cp_family,
        rx_cp_family,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
    )
    H_lp_raw, H_cp_raw = _build_mode_channels(enriched_paths, cfg.freqs_hz, tx_lp_pose, rx_lp_pose, tx_cp_pose, rx_cp_pose, normalization_mode="raw")
    H_lp_norm, H_cp_norm = _build_mode_channels(enriched_paths, cfg.freqs_hz, tx_lp_pose, rx_lp_pose, tx_cp_pose, rx_cp_pose, normalization_mode="family_radiated_power")
    lp_raw_summary = summarize_channel(H_lp_raw, cfg.snr_db_list)
    cp_raw_summary = summarize_channel(H_cp_raw, cfg.snr_db_list)
    raw_delta = compare_channels(H_lp_raw, H_cp_raw, cfg.snr_db_list)
    lp_norm_summary = summarize_channel(H_lp_norm, cfg.snr_db_list)
    cp_norm_summary = summarize_channel(H_cp_norm, cfg.snr_db_list)
    norm_delta = compare_channels(H_lp_norm, H_cp_norm, cfg.snr_db_list)
    richness = compute_path_richness(enriched_paths, freqs_hz=cfg.freqs_hz)
    return richness, lp_raw_summary, cp_raw_summary, raw_delta, lp_norm_summary, cp_norm_summary, norm_delta


def _make_point_record(
    scenario_name: str,
    proxy: bool,
    rx_pos: np.ndarray,
    richness: PathRichnessMetrics,
    lp_raw_summary: ChannelSummary,
    cp_raw_summary: ChannelSummary,
    raw_delta: ChannelDeltaSummary,
    lp_norm_summary: ChannelSummary,
    cp_norm_summary: ChannelSummary,
    norm_delta: ChannelDeltaSummary,
) -> _PointRecord:
    return _PointRecord(
        scenario_name=scenario_name,
        proxy=proxy,
        rx_pos=np.asarray(rx_pos, dtype=float),
        richness=richness,
        lp_rate_raw={float(snr): float(value) for snr, value in lp_raw_summary.equal_power_rate.items()},
        cp_rate_raw={float(snr): float(value) for snr, value in cp_raw_summary.equal_power_rate.items()},
        delta_rate_raw={float(snr): float(value) for snr, value in raw_delta.delta_equal_power_rate.items()},
        lp_capacity_raw={float(snr): float(value) for snr, value in lp_raw_summary.waterfilled_capacity.items()},
        cp_capacity_raw={float(snr): float(value) for snr, value in cp_raw_summary.waterfilled_capacity.items()},
        delta_capacity_raw={float(snr): float(value) for snr, value in raw_delta.delta_waterfilled_capacity.items()},
        lp_rate_norm={float(snr): float(value) for snr, value in lp_norm_summary.equal_power_rate.items()},
        cp_rate_norm={float(snr): float(value) for snr, value in cp_norm_summary.equal_power_rate.items()},
        delta_rate_norm={float(snr): float(value) for snr, value in norm_delta.delta_equal_power_rate.items()},
        lp_capacity_norm={float(snr): float(value) for snr, value in lp_norm_summary.waterfilled_capacity.items()},
        cp_capacity_norm={float(snr): float(value) for snr, value in cp_norm_summary.waterfilled_capacity.items()},
        delta_capacity_norm={float(snr): float(value) for snr, value in norm_delta.delta_waterfilled_capacity.items()},
    )


def _flatten_point_record(record: _PointRecord, snr_db_list: tuple[float, ...]) -> dict[str, Any]:
    row = {
        "scenario_name": record.scenario_name,
        "proxy": record.proxy,
        "rx_x": float(record.rx_pos[0]),
        "rx_y": float(record.rx_pos[1]),
        "rx_z": float(record.rx_pos[2]),
        "path_count": int(record.richness.path_count),
        "n_eff": float(record.richness.n_eff),
        "tau_rms_ns": float(record.richness.tau_rms_s * 1e9),
        "mean_aoa_spread_deg": float(record.richness.mean_aoa_spread_deg),
        "aoa_azimuth_spread_deg": float(record.richness.aoa_azimuth_spread_deg),
        "aoa_elevation_spread_deg": float(record.richness.aoa_elevation_spread_deg),
        "aod_azimuth_spread_deg": float(record.richness.aod_azimuth_spread_deg),
        "aod_elevation_spread_deg": float(record.richness.aod_elevation_spread_deg),
        "dpr_db": float(record.richness.dpr_db),
        "pmi_env": float(record.richness.pmi_env),
        "mrs_raw": float(record.mrs_raw),
        "mrs_score": float(record.mrs_score),
        "mrs_label": record.mrs_label,
        "selected_role": record.selected_role,
    }
    for snr in snr_db_list:
        key = int(round(float(snr)))
        row[f"lp_rate_raw_snr_{key}"] = float(record.lp_rate_raw[float(snr)])
        row[f"cp_rate_raw_snr_{key}"] = float(record.cp_rate_raw[float(snr)])
        row[f"delta_rate_raw_snr_{key}"] = float(record.delta_rate_raw[float(snr)])
        row[f"lp_capacity_raw_snr_{key}"] = float(record.lp_capacity_raw[float(snr)])
        row[f"cp_capacity_raw_snr_{key}"] = float(record.cp_capacity_raw[float(snr)])
        row[f"delta_capacity_raw_snr_{key}"] = float(record.delta_capacity_raw[float(snr)])
        row[f"lp_rate_norm_snr_{key}"] = float(record.lp_rate_norm[float(snr)])
        row[f"cp_rate_norm_snr_{key}"] = float(record.cp_rate_norm[float(snr)])
        row[f"delta_rate_norm_snr_{key}"] = float(record.delta_rate_norm[float(snr)])
        row[f"lp_capacity_norm_snr_{key}"] = float(record.lp_capacity_norm[float(snr)])
        row[f"cp_capacity_norm_snr_{key}"] = float(record.cp_capacity_norm[float(snr)])
        row[f"delta_capacity_norm_snr_{key}"] = float(record.delta_capacity_norm[float(snr)])
    return row


def _summarize_scene_records(records: list[_PointRecord]) -> SceneRichnessSummary:
    if not records:
        raise ValueError("cannot summarize empty scene record list")
    scores = np.asarray([record.mrs_score for record in records], dtype=float)
    bucket_counts = {
        "mild": sum(1 for record in records if record.mrs_label == "mild"),
        "moderate": sum(1 for record in records if record.mrs_label == "moderate"),
        "rich": sum(1 for record in records if record.mrs_label == "rich"),
    }
    return SceneRichnessSummary(
        scenario_name=records[0].scenario_name,
        proxy=records[0].proxy,
        point_count=len(records),
        median_mrs_score=float(np.median(scores)),
        min_mrs_score=float(np.min(scores)),
        max_mrs_score=float(np.max(scores)),
        median_n_eff=float(np.median([record.richness.n_eff for record in records])),
        median_tau_rms_ns=float(np.median([record.richness.tau_rms_s for record in records]) * 1e9),
        median_mean_aoa_spread_deg=float(np.median([record.richness.mean_aoa_spread_deg for record in records])),
        median_dpr_db=float(np.median([record.richness.dpr_db for record in records])),
        median_pmi_env=float(np.median([record.richness.pmi_env for record in records])),
        bucket_counts=bucket_counts,
    )


def _select_representatives(records: list[_PointRecord]) -> CampaignSelection:
    non_proxy_records = [record for record in records if not record.proxy]
    scores = np.asarray([record.mrs_score for record in non_proxy_records], dtype=float)
    _, low, high = label_mrs_scores(scores)
    representatives: dict[str, dict[str, Any]] = {}
    used_scenarios: set[str] = set()
    scene_groups: dict[str, list[_PointRecord]] = {}
    for record in non_proxy_records:
        scene_groups.setdefault(record.scenario_name, []).append(record)
    for bucket in ("mild", "moderate", "rich"):
        bucket_records = [record for record in non_proxy_records if record.mrs_label == bucket]
        if not bucket_records:
            continue
        target = float(np.median([record.mrs_score for record in bucket_records]))
        candidate_scores: list[tuple[float, int, str]] = []
        for scenario_name, scene_records in scene_groups.items():
            if scenario_name in used_scenarios and len(scene_groups) >= 3:
                continue
            scene_bucket = [record for record in scene_records if record.mrs_label == bucket]
            if not scene_bucket:
                continue
            scene_median = float(np.median([record.mrs_score for record in scene_bucket]))
            candidate_scores.append((abs(scene_median - target), -len(scene_bucket), scenario_name))
        if not candidate_scores:
            continue
        candidate_scores.sort()
        chosen_name = candidate_scores[0][2]
        chosen_records = [record for record in scene_groups[chosen_name] if record.mrs_label == bucket]
        chosen_target = float(np.median([record.mrs_score for record in chosen_records]))
        representative = min(chosen_records, key=lambda record: (abs(record.mrs_score - target), abs(record.mrs_score - chosen_target)))
        representative.selected_role = bucket
        representatives[bucket] = {
            "scenario_name": chosen_name,
            "rx_pos": representative.rx_pos.copy(),
            "mrs_score": float(representative.mrs_score),
            "bucket": bucket,
        }
        used_scenarios.add(chosen_name)
    return CampaignSelection(mrs_low_threshold=low, mrs_high_threshold=high, representatives=representatives)


def _evaluate_screening(
    scenario_specs: tuple[ScenarioSpec, ...],
    tx_lp_family: AntennaFamily,
    rx_lp_family: AntennaFamily,
    tx_cp_family: AntennaFamily,
    rx_cp_family: AntennaFamily,
    fc: float,
    bw: float,
    screening_n_freq: int,
    screening_grid_step: float,
    snr_db_list: tuple[float, ...],
    max_reflections: int,
) -> tuple[list[_PointRecord], dict[str, dict[str, Any]]]:
    records: list[_PointRecord] = []
    grids: dict[str, dict[str, Any]] = {}
    for spec in scenario_specs:
        layout = build_scenario_layout(spec)
        rx_x, rx_y, valid_mask = scenario_grid(spec, screening_grid_step)
        cfg = _make_cfg(spec, fc, bw, screening_n_freq, screening_grid_step, snr_db_list, max_reflections)
        for iy, y in enumerate(rx_y):
            for ix, x in enumerate(rx_x):
                if not valid_mask[iy, ix]:
                    continue
                rx_pos = np.array([x, y, spec.rx_plane_z], dtype=float)
                richness, lp_raw_summary, cp_raw_summary, raw_delta, lp_norm_summary, cp_norm_summary, norm_delta = _evaluate_point(
                    layout.scene,
                    cfg,
                    rx_pos,
                    tx_lp_family,
                    rx_lp_family,
                    tx_cp_family,
                    rx_cp_family,
                )
                records.append(
                    _make_point_record(
                        scenario_name=spec.name,
                        proxy=spec.proxy,
                        rx_pos=rx_pos,
                        richness=richness,
                        lp_raw_summary=lp_raw_summary,
                        cp_raw_summary=cp_raw_summary,
                        raw_delta=raw_delta,
                        lp_norm_summary=lp_norm_summary,
                        cp_norm_summary=cp_norm_summary,
                        norm_delta=norm_delta,
                    )
                )
        grids[spec.name] = {"rx_x": rx_x, "rx_y": rx_y, "valid_mask": valid_mask}
    non_proxy_metrics = [record.richness for record in records if not record.proxy]
    reference_metrics = non_proxy_metrics if non_proxy_metrics else [record.richness for record in records]
    raw_scores, scaled_scores = compute_mrs_scores([record.richness for record in records], reference_metrics_list=reference_metrics)
    non_proxy_scores = np.asarray([score for record, score in zip(records, scaled_scores) if not record.proxy], dtype=float)
    labels, _low, _high = label_mrs_scores(non_proxy_scores if non_proxy_scores.size else scaled_scores)
    score_iter = iter(labels)
    for record, raw_score, scaled_score in zip(records, raw_scores, scaled_scores):
        label = next(score_iter) if not record.proxy else ("mild" if float(scaled_score) <= _low else "rich" if float(scaled_score) >= _high else "moderate")
        record.mrs_raw = float(raw_score)
        record.mrs_score = float(scaled_score)
        record.mrs_label = label
    return records, grids


def _evaluate_scene_grid(
    spec: ScenarioSpec,
    tx_lp_family: AntennaFamily,
    rx_lp_family: AntennaFamily,
    tx_cp_family: AntennaFamily,
    rx_cp_family: AntennaFamily,
    fc: float,
    bw: float,
    n_freq: int,
    grid_step: float,
    snr_db_list: tuple[float, ...],
    max_reflections: int,
) -> tuple[RealisticCompareResult, np.ndarray, list[_PointRecord]]:
    layout = build_scenario_layout(spec)
    rx_x, rx_y, valid_mask = scenario_grid(spec, grid_step)
    cfg = _make_cfg(spec, fc, bw, n_freq, grid_step, snr_db_list, max_reflections)
    shape = (len(rx_y), len(rx_x))
    raw_maps = _empty_compare_mode_maps(shape, snr_db_list)
    norm_maps = _empty_compare_mode_maps(shape, snr_db_list)
    point_records: list[_PointRecord] = []
    for iy, y in enumerate(rx_y):
        for ix, x in enumerate(rx_x):
            if not valid_mask[iy, ix]:
                continue
            rx_pos = np.array([x, y, spec.rx_plane_z], dtype=float)
            richness, lp_raw_summary, cp_raw_summary, raw_delta, lp_norm_summary, cp_norm_summary, norm_delta = _evaluate_point(
                layout.scene,
                cfg,
                rx_pos,
                tx_lp_family,
                rx_lp_family,
                tx_cp_family,
                rx_cp_family,
            )
            _write_summary(raw_maps.lp, lp_raw_summary, (iy, ix))
            _write_summary(raw_maps.cp, cp_raw_summary, (iy, ix))
            _write_delta(raw_maps, raw_delta, (iy, ix))
            _write_summary(norm_maps.lp, lp_norm_summary, (iy, ix))
            _write_summary(norm_maps.cp, cp_norm_summary, (iy, ix))
            _write_delta(norm_maps, norm_delta, (iy, ix))
            point_records.append(
                _make_point_record(
                    scenario_name=spec.name,
                    proxy=spec.proxy,
                    rx_pos=rx_pos,
                    richness=richness,
                    lp_raw_summary=lp_raw_summary,
                    cp_raw_summary=cp_raw_summary,
                    raw_delta=raw_delta,
                    lp_norm_summary=lp_norm_summary,
                    cp_norm_summary=cp_norm_summary,
                    norm_delta=norm_delta,
                )
            )
    return RealisticCompareResult(rx_x=rx_x, rx_y=rx_y, raw=raw_maps, normalized=norm_maps), valid_mask, point_records


def _aggregate_rate_rows(
    scenario_name: str,
    proxy: bool,
    protocol_name: str,
    point_records: list[_PointRecord],
    snr_db_list: tuple[float, ...],
    r0_by_snr: dict[float, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snr in snr_db_list:
        lp_rates = np.asarray([record.lp_rate_norm[float(snr)] for record in point_records], dtype=float)
        cp_rates = np.asarray([record.cp_rate_norm[float(snr)] for record in point_records], dtype=float)
        delta = cp_rates - lp_rates
        epsilon = 0.01 * float(np.mean(lp_rates)) if lp_rates.size else 0.0
        r0 = float(r0_by_snr[float(snr)])
        lp_p05 = float(np.percentile(lp_rates, 5.0)) if lp_rates.size else 0.0
        cp_p05 = float(np.percentile(cp_rates, 5.0)) if cp_rates.size else 0.0
        rows.append(
            {
                "scenario_name": scenario_name,
                "proxy": proxy,
                "protocol": protocol_name,
                "snr_db": float(snr),
                "samples": int(lp_rates.size),
                "lp_rate_mean": float(np.mean(lp_rates)) if lp_rates.size else 0.0,
                "cp_rate_mean": float(np.mean(cp_rates)) if cp_rates.size else 0.0,
                "delta_rate_mean": float(np.mean(delta)) if delta.size else 0.0,
                "epsilon": float(epsilon),
                "noninferiority_prob_0": float(np.mean(delta >= 0.0)) if delta.size else 0.0,
                "noninferiority_prob_epsilon": float(np.mean(delta >= -epsilon)) if delta.size else 0.0,
                "lp_rate_p05": lp_p05,
                "cp_rate_p05": cp_p05,
                "delta_rate_p05": float(cp_p05 - lp_p05),
                "delta_rate_distribution_p05": float(np.percentile(delta, 5.0)) if delta.size else 0.0,
                "r0": r0,
                "outage_lp": float(np.mean(lp_rates < r0)) if lp_rates.size else 0.0,
                "outage_cp": float(np.mean(cp_rates < r0)) if cp_rates.size else 0.0,
                "delta_outage": float(np.mean(cp_rates < r0) - np.mean(lp_rates < r0)) if lp_rates.size else 0.0,
            }
        )
    return rows


def _select_subset_points(rx_x: np.ndarray, rx_y: np.ndarray, valid_mask: np.ndarray, target_shape: tuple[int, int]) -> list[np.ndarray]:
    if not np.any(valid_mask):
        return []
    y_bins = np.array_split(np.arange(len(rx_y)), min(target_shape[1], len(rx_y)))
    x_bins = np.array_split(np.arange(len(rx_x)), min(target_shape[0], len(rx_x)))
    points: list[np.ndarray] = []
    seen: set[tuple[float, float]] = set()
    for y_idx_group in y_bins:
        for x_idx_group in x_bins:
            valid_indices = [(iy, ix) for iy in y_idx_group for ix in x_idx_group if valid_mask[iy, ix]]
            if not valid_indices:
                continue
            cy = float(np.mean(y_idx_group))
            cx = float(np.mean(x_idx_group))
            iy, ix = min(valid_indices, key=lambda pair: (pair[0] - cy) ** 2 + (pair[1] - cx) ** 2)
            key = (float(rx_x[ix]), float(rx_y[iy]))
            if key in seen:
                continue
            seen.add(key)
            points.append(np.array([rx_x[ix], rx_y[iy]], dtype=float))
    return points


def _evaluate_pose_samples(
    spec: ScenarioSpec,
    rx_points_xy: list[np.ndarray],
    protocols: tuple[PoseProtocol, ...],
    tx_lp_family: AntennaFamily,
    rx_lp_family: AntennaFamily,
    tx_cp_family: AntennaFamily,
    rx_cp_family: AntennaFamily,
    fc: float,
    bw: float,
    n_freq: int,
    snr_db_list: tuple[float, ...],
    max_reflections: int,
    pose_seed_count: int,
    base_seed: int,
) -> list[_PointRecord]:
    cfg = _make_cfg(spec, fc, bw, n_freq, 1.0, snr_db_list, max_reflections)
    layout = build_scenario_layout(spec)
    sample_records: list[_PointRecord] = []
    for protocol_index, protocol in enumerate(protocols):
        for point_index, point_xy in enumerate(rx_points_xy):
            rx_pos = np.array([float(point_xy[0]), float(point_xy[1]), spec.rx_plane_z], dtype=float)
            rng = np.random.default_rng(base_seed + 100000 * protocol_index + 1000 * point_index)
            for _sample_idx in range(pose_seed_count):
                yaw_deg, pitch_deg, roll_deg = protocol.sample_angles(rng)
                richness, lp_raw_summary, cp_raw_summary, raw_delta, lp_norm_summary, cp_norm_summary, norm_delta = _evaluate_point(
                    layout.scene,
                    cfg,
                    rx_pos,
                    tx_lp_family,
                    rx_lp_family,
                    tx_cp_family,
                    rx_cp_family,
                    yaw_deg=yaw_deg,
                    pitch_deg=pitch_deg,
                    roll_deg=roll_deg,
                    blocker=protocol.blocker,
                )
                record = _make_point_record(
                    scenario_name=spec.name,
                    proxy=spec.proxy,
                    rx_pos=rx_pos,
                    richness=richness,
                    lp_raw_summary=lp_raw_summary,
                    cp_raw_summary=cp_raw_summary,
                    raw_delta=raw_delta,
                    lp_norm_summary=lp_norm_summary,
                    cp_norm_summary=cp_norm_summary,
                    norm_delta=norm_delta,
                )
                record.selected_role = protocol.name
                sample_records.append(record)
    return sample_records


def _scene_reference_point(records: list[_PointRecord]) -> np.ndarray:
    median_score = float(np.median([record.mrs_score for record in records]))
    representative = min(records, key=lambda record: abs(record.mrs_score - median_score))
    return representative.rx_pos.copy()


def _sanitize_key(text: str) -> str:
    return str(text).replace("-", "_").replace(" ", "_")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_ready(row.get(key, "")) for key in fieldnames})


def _write_maps_npz(
    path: Path,
    screening_grids: dict[str, dict[str, Any]],
    screening_rows: list[dict[str, Any]],
    full_results: dict[str, tuple[RealisticCompareResult, np.ndarray]],
    snr_db_list: tuple[float, ...],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    screening_by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in screening_rows:
        screening_by_scene.setdefault(str(row["scenario_name"]), []).append(row)
    for scenario_name, meta in screening_grids.items():
        rx_x = np.asarray(meta["rx_x"], dtype=float)
        rx_y = np.asarray(meta["rx_y"], dtype=float)
        valid_mask = np.asarray(meta["valid_mask"], dtype=bool)
        mrs_map = np.full(valid_mask.shape, np.nan, dtype=float)
        delta_map = np.full(valid_mask.shape, np.nan, dtype=float)
        row_lookup = {(float(row["rx_x"]), float(row["rx_y"])): row for row in screening_by_scene.get(scenario_name, [])}
        for iy, y in enumerate(rx_y):
            for ix, x in enumerate(rx_x):
                row = row_lookup.get((float(x), float(y)))
                if row is None:
                    continue
                mrs_map[iy, ix] = float(row["mrs_score"])
                delta_map[iy, ix] = float(row.get("delta_rate_norm_snr_10", np.nan))
        key = _sanitize_key(scenario_name)
        arrays[f"screening_{key}_rx_x"] = rx_x
        arrays[f"screening_{key}_rx_y"] = rx_y
        arrays[f"screening_{key}_valid_mask"] = valid_mask.astype(np.uint8)
        arrays[f"screening_{key}_mrs_score"] = mrs_map
        arrays[f"screening_{key}_delta_rate_norm_snr_10"] = delta_map
    for scenario_name, (result, valid_mask) in full_results.items():
        key = _sanitize_key(scenario_name)
        arrays[f"full_{key}_rx_x"] = np.asarray(result.rx_x, dtype=float)
        arrays[f"full_{key}_rx_y"] = np.asarray(result.rx_y, dtype=float)
        arrays[f"full_{key}_valid_mask"] = np.asarray(valid_mask, dtype=np.uint8)
        for snr in snr_db_list:
            snr_key = int(round(float(snr)))
            arrays[f"full_{key}_norm_lp_rate_snr_{snr_key}"] = np.asarray(result.normalized.lp.rate_maps[float(snr)], dtype=float)
            arrays[f"full_{key}_norm_cp_rate_snr_{snr_key}"] = np.asarray(result.normalized.cp.rate_maps[float(snr)], dtype=float)
            arrays[f"full_{key}_norm_delta_rate_snr_{snr_key}"] = np.asarray(result.normalized.delta_rate_maps[float(snr)], dtype=float)
            arrays[f"full_{key}_raw_delta_rate_snr_{snr_key}"] = np.asarray(result.raw.delta_rate_maps[float(snr)], dtype=float)
            arrays[f"full_{key}_norm_lp_capacity_snr_{snr_key}"] = np.asarray(result.normalized.lp.capacity_maps[float(snr)], dtype=float)
            arrays[f"full_{key}_norm_cp_capacity_snr_{snr_key}"] = np.asarray(result.normalized.cp.capacity_maps[float(snr)], dtype=float)
            arrays[f"full_{key}_norm_delta_capacity_snr_{snr_key}"] = np.asarray(result.normalized.delta_capacity_maps[float(snr)], dtype=float)
            arrays[f"full_{key}_raw_delta_capacity_snr_{snr_key}"] = np.asarray(result.raw.delta_capacity_maps[float(snr)], dtype=float)
    np.savez(path, **arrays)


def run_indoor_campaign(
    tx_lp_family: AntennaFamily,
    rx_lp_family: AntennaFamily,
    tx_cp_family: AntennaFamily,
    rx_cp_family: AntennaFamily,
    scenario_names: tuple[str, ...] | None = None,
    output_dir: str | Path | None = None,
    fc: float = 6.5e9,
    bw: float = 500e6,
    screening_n_freq: int = 21,
    screening_grid_step: float = 0.5,
    full_n_freq: int = 101,
    full_grid_step: float = 0.2,
    pose_seed_count: int = 64,
    pose_grid_shape: tuple[int, int] = (10, 10),
    snr_db_list: tuple[float, ...] = (0.0, 10.0, 20.0),
    max_reflections: int = 2,
    base_seed: int = 20260326,
) -> CampaignResult:
    requested_specs = tuple(get_scenario_spec(name) for name in scenario_names) if scenario_names is not None else canonical_scenarios(include_proxy=True)
    screening_records, screening_grids = _evaluate_screening(
        requested_specs,
        tx_lp_family,
        rx_lp_family,
        tx_cp_family,
        rx_cp_family,
        fc,
        bw,
        screening_n_freq,
        screening_grid_step,
        snr_db_list,
        max_reflections,
    )
    selection = _select_representatives(screening_records)
    scene_summaries = tuple(
        _summarize_scene_records([record for record in screening_records if record.scenario_name == spec.name])
        for spec in requested_specs
    )

    selected_scene_names = []
    for bucket in ("mild", "moderate", "rich"):
        if bucket in selection.representatives:
            selected_scene_names.append(str(selection.representatives[bucket]["scenario_name"]))
    selected_scene_names = list(dict.fromkeys(selected_scene_names))

    full_results: dict[str, RealisticCompareResult] = {}
    full_masks: dict[str, np.ndarray] = {}
    pose_rows: list[dict[str, Any]] = []
    headline_summary: dict[str, Any] = {"selected_scenes": {}, "proxy_scenes": {}}

    for scene_index, scene_name in enumerate(selected_scene_names):
        spec = get_scenario_spec(scene_name)
        full_result, valid_mask, point_records = _evaluate_scene_grid(
            spec,
            tx_lp_family,
            rx_lp_family,
            tx_cp_family,
            rx_cp_family,
            fc,
            bw,
            full_n_freq,
            full_grid_step,
            snr_db_list,
            max_reflections,
        )
        full_results[scene_name] = full_result
        full_masks[scene_name] = valid_mask
        r0_by_snr = {
            float(snr): float(np.percentile([record.lp_rate_norm[float(snr)] for record in point_records], 5.0))
            for snr in snr_db_list
        }
        pose_rows.extend(_aggregate_rate_rows(scene_name, spec.proxy, "P0", point_records, snr_db_list, r0_by_snr))

        subset_points = _select_subset_points(full_result.rx_x, full_result.rx_y, valid_mask, pose_grid_shape)
        pose_protocols = tuple(standard_pose_protocols()[name] for name in ("P1", "P2"))
        pose_samples = _evaluate_pose_samples(
            spec,
            subset_points,
            pose_protocols,
            tx_lp_family,
            rx_lp_family,
            tx_cp_family,
            rx_cp_family,
            fc,
            bw,
            full_n_freq,
            snr_db_list,
            max_reflections,
            pose_seed_count,
            base_seed + 10000 * scene_index,
        )
        for protocol_name in ("P1", "P2"):
            protocol_records = [record for record in pose_samples if record.selected_role == protocol_name]
            pose_rows.extend(_aggregate_rate_rows(scene_name, spec.proxy, protocol_name, protocol_records, snr_db_list, r0_by_snr))
        headline_summary["selected_scenes"][scene_name] = {
            "proxy": spec.proxy,
            "r0_by_snr": {str(int(round(float(snr)))): float(value) for snr, value in r0_by_snr.items()},
            "protocol_rows": [row for row in pose_rows if row["scenario_name"] == scene_name and not row["proxy"]],
        }

    screening_by_scene = {spec.name: [record for record in screening_records if record.scenario_name == spec.name] for spec in requested_specs}
    proxy_specs = [spec for spec in requested_specs if spec.proxy]
    for proxy_index, spec in enumerate(proxy_specs):
        proxy_records = screening_by_scene[spec.name]
        if not proxy_records:
            continue
        reference_point = _scene_reference_point(proxy_records)
        r0_by_snr = {
            float(snr): float(np.percentile([record.lp_rate_norm[float(snr)] for record in proxy_records], 5.0))
            for snr in snr_db_list
        }
        if spec.name == "l_corridor_proxy":
            proxy_protocols = tuple(standard_pose_protocols()[name] for name in ("P1", "P2"))
        else:
            proxy_protocols = tuple(standard_pose_protocols()[name] for name in ("P2", "P3"))
        proxy_samples = _evaluate_pose_samples(
            spec,
            [reference_point[:2]],
            proxy_protocols,
            tx_lp_family,
            rx_lp_family,
            tx_cp_family,
            rx_cp_family,
            fc,
            bw,
            full_n_freq,
            snr_db_list,
            max_reflections,
            pose_seed_count,
            base_seed + 50000 + 10000 * proxy_index,
        )
        proxy_rows = []
        for protocol in proxy_protocols:
            protocol_records = [record for record in proxy_samples if record.selected_role == protocol.name]
            proxy_rows.extend(_aggregate_rate_rows(spec.name, True, protocol.name, protocol_records, snr_db_list, r0_by_snr))
        pose_rows.extend(proxy_rows)
        headline_summary["proxy_scenes"][spec.name] = {
            "proxy": True,
            "reference_point": reference_point.tolist(),
            "protocol_rows": proxy_rows,
        }

    screening_rows = [_flatten_point_record(record, snr_db_list) for record in screening_records]
    artifact_paths: dict[str, str] = {}
    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        maps_path = outdir / "maps.npz"
        summary_path = outdir / "summary.json"
        scene_csv_path = outdir / "scene_metrics.csv"
        pose_csv_path = outdir / "pose_metrics.csv"
        _write_maps_npz(
            maps_path,
            screening_grids=screening_grids,
            screening_rows=screening_rows,
            full_results={name: (result, full_masks[name]) for name, result in full_results.items()},
            snr_db_list=snr_db_list,
        )
        _write_csv(scene_csv_path, screening_rows)
        _write_csv(pose_csv_path, pose_rows)
        summary_payload = {
            "scenarios": [_json_ready(summary.__dict__) for summary in scene_summaries],
            "selection": {
                "mrs_low_threshold": float(selection.mrs_low_threshold),
                "mrs_high_threshold": float(selection.mrs_high_threshold),
                "representatives": _json_ready(selection.representatives),
            },
            "headline_summary": _json_ready(headline_summary),
            "notes": {
                "headline_excludes_proxy": True,
                "normalized_results_are_primary": True,
                "raw_results_are_secondary": True,
            },
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        artifact_paths = {
            "maps": str(maps_path),
            "summary": str(summary_path),
            "scene_metrics": str(scene_csv_path),
            "pose_metrics": str(pose_csv_path),
        }

    return CampaignResult(
        scene_summaries=scene_summaries,
        selection=selection,
        screening_rows=tuple(screening_rows),
        pose_rows=tuple(pose_rows),
        full_compare_results=full_results,
        headline_summary=headline_summary,
        artifact_paths=artifact_paths,
    )


__all__ = [
    "PoseProtocol",
    "SceneRichnessSummary",
    "CampaignSelection",
    "CampaignResult",
    "run_indoor_campaign",
    "standard_pose_protocols",
]

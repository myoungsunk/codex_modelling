"""Standalone deterministic dual-polarized ray-tracing baseline."""

from dualpol_rt.config import DepolConfig, ExperimentConfig, IncidenceSweepConfig
from dualpol_rt.channel.builder import build_channel, convert_basis
from dualpol_rt.channel.path_record import GridResult, PathRecord
from dualpol_rt.experiments.basis_invariance import evaluate_debug_point, evaluate_rx_grid
from dualpol_rt.experiments.campaign import CampaignResult, CampaignSelection, PoseProtocol, SceneRichnessSummary, run_indoor_campaign, standard_pose_protocols
from dualpol_rt.experiments.realistic_compare import (
    CompareModeMaps,
    RealisticCompareResult,
    SystemMaps,
    build_phase2_patterns,
    build_phase2_patterns_from_standard_files,
    evaluate_realistic_debug_point,
    evaluate_realistic_rx_grid,
)
from dualpol_rt.feature_extractor import (
    CIRTrace,
    compute_a_fp_variants,
    compute_cir_baseline_features,
    compute_gamma_cp_variants,
    extract_all_features,
    extract_first_path,
    ifft_to_cir,
)
from dualpol_rt.geometry.image_method import enumerate_paths
from dualpol_rt.metrics.achievable_rate import (
    equal_power_rate,
    gain_normalized_equal_power_rate,
    waterfilled_capacity,
)
from dualpol_rt.metrics.link_budget import (
    best_port_rate,
    best_port_snr_db,
    fixed_combiner_rate,
    gain_to_snr_db,
    per_port_rx_gain,
    port_imbalance_db,
    standard_fixed_rates,
    total_rx_gain,
    total_snr_db,
)
from dualpol_rt.metrics.comparison import ChannelDeltaSummary, ChannelSummary, compare_channels, summarize_channel
from dualpol_rt.metrics.richness import PathRichnessMetrics, compute_mrs_scores, compute_path_richness, label_mrs_scores
from dualpol_rt.metrics.xpr import condition_numbers, singular_values, xpr_db
from dualpol_rt.patterns import (
    AntennaFamily,
    AntennaPose,
    FFDPattern,
    IdealPattern,
    make_ideal_cp_antenna,
    make_realistic_patch_cp_antenna,
)
from dualpol_rt.scene import ScenarioSpec, build_scenario, canonical_scenarios, get_scenario_spec, scenario_grid
from dualpol_rt.scene.shoebox import build_shoebox

__all__ = [
    "ExperimentConfig",
    "DepolConfig",
    "IncidenceSweepConfig",
    "PathRecord",
    "GridResult",
    "AntennaPose",
    "AntennaFamily",
    "IdealPattern",
    "FFDPattern",
    "make_ideal_cp_antenna",
    "make_realistic_patch_cp_antenna",
    "build_shoebox",
    "enumerate_paths",
    "build_channel",
    "convert_basis",
    "equal_power_rate",
    "waterfilled_capacity",
    "gain_normalized_equal_power_rate",
    "total_rx_gain",
    "per_port_rx_gain",
    "port_imbalance_db",
    "gain_to_snr_db",
    "total_snr_db",
    "best_port_snr_db",
    "best_port_rate",
    "fixed_combiner_rate",
    "standard_fixed_rates",
    "xpr_db",
    "singular_values",
    "condition_numbers",
    "PathRichnessMetrics",
    "compute_path_richness",
    "compute_mrs_scores",
    "label_mrs_scores",
    "ChannelSummary",
    "ChannelDeltaSummary",
    "summarize_channel",
    "compare_channels",
    "ScenarioSpec",
    "evaluate_debug_point",
    "evaluate_rx_grid",
    "SystemMaps",
    "CompareModeMaps",
    "RealisticCompareResult",
    "build_phase2_patterns",
    "build_phase2_patterns_from_standard_files",
    "evaluate_realistic_debug_point",
    "evaluate_realistic_rx_grid",
    "CIRTrace",
    "ifft_to_cir",
    "extract_first_path",
    "compute_gamma_cp_variants",
    "compute_a_fp_variants",
    "compute_cir_baseline_features",
    "extract_all_features",
    "build_scenario",
    "canonical_scenarios",
    "get_scenario_spec",
    "scenario_grid",
    "PoseProtocol",
    "SceneRichnessSummary",
    "CampaignSelection",
    "CampaignResult",
    "run_indoor_campaign",
    "standard_pose_protocols",
]

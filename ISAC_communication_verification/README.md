# Dual-Pol Deterministic RT

Deterministic `2x2` dual-polarized ray-tracing baseline for shoebox indoor scenes, with a Phase 2 realistic antenna path that injects actual HFSS far-field patterns without changing the propagation core.

The repository now also includes an indoor richness and robustness campaign for realistic axis-aligned office, corridor, lecture-room, factory-lite, and proxy blocker scenes. It screens scenes by multipath richness, selects mild/moderate/rich representatives, runs normalized CP-vs-slant comparisons, and writes paper-facing artifacts.

## Scope
- Exact image-method path enumeration for LOS, 1-bounce, and 2-bounce shoebox paths
- Internal propagation solver fixed to the linear Jones basis `[H; V]`
- Circular results produced only by explicit basis conversion for diagnostics
- Phase 2 realistic comparison uses actual HFSS FFD antenna families:
  - dual-linear slant: `LP_+45_new_6G7G_11pts.ffd`, `LP_-45_new_6G7G_11pts.ffd`
  - dual-CP: `RHCP_new_6G7G_11pts.ffd`, `LHCP_new_6G7G_11pts.ffd`
- Metrics: equal-power achievable rate, water-filled capacity, gain-normalized achievable rate, XPR, singular values, condition number
- Indoor campaign:
  - canonical `open_office`, `mixed_office`, `lecture_room`, `straight_corridor`, `factory_lite`
  - exploratory `l_corridor_proxy`, `lobby_blocker_proxy`
  - multipath metrics `N_eff`, `tau_rms`, AoA/AoD spreads, `DPR`, `PMI_env`, and composite `MRS`
  - pose protocols `P0`, `P1`, `P2`, and blocker-only `P3`
  - artifacts `maps.npz`, `summary.json`, `scene_metrics.csv`, `pose_metrics.csv`
- MATLAB validation scripts are manual oracles only. Python does not call MATLAB automatically.

## Conventions
- Linear basis order: `[H; V]`
- Time convention: `exp(-j*w*t)`
- Circular basis order inside the solver: `[RHCP; LHCP]` with `circular_order="RL"`
- CP port order is standardized as `RHCP=port 0`, `LHCP=port 1` across the API and experiments
- HFSS Theta-Phi loading rule: `[H; V] = [E_phi; E_theta]`
- FFD boresight is interpreted as local `+z`
- Default Phase 2 pose is facing boresight: Tx local `+z` points to Rx, Rx local `+z` points to Tx
- When `theta=0`, the physical boresight is the `+z` pole and `phi` is only a diagnostic label

## Phase 2 FFD Input Format

Phase 2 expects one multi-frequency `.ffd` file per port. The canonical input is:

```text
<port>.ffd
  line 1: theta_start theta_stop theta_count
  line 2: phi_start phi_stop phi_count
  line 3: Frequencies N
  line 4: Frequency <Hz>
  then Ntheta*Nphi rows of: Re(Etheta) Im(Etheta) Re(Ephi) Im(Ephi)
  repeated for each frequency block
```

The current benchmark families are:

```text
LP family
  LP_+45_new_6G7G_11pts.ffd
  LP_-45_new_6G7G_11pts.ffd

CP family
  RHCP_new_6G7G_11pts.ffd
  LHCP_new_6G7G_11pts.ffd
```

These files cover `6.0-7.0 GHz` with `11` stored frequency points. The default experiment remains `fc=6.5 GHz`, `bw=500 MHz`, `n_freq=101`.

## Normalization Modes
- `raw`: use the HFSS-exported far field as-is
- `family_radiated_power`: scale each antenna family at each stored frequency so the family-average radiated field energy over angle is 1

Use both modes for Phase 2 interpretation:
- `raw` shows end-to-end antenna-family performance with only total transmit power held fixed
- `family_radiated_power` factors out much of the gross gain difference and highlights pattern/polarization/angular-coupling structure

## Quick Start

Phase 1 basis-invariance sanity check:

```python
from dualpol_rt import ExperimentConfig, IdealPattern, evaluate_rx_grid

cfg = ExperimentConfig()
pattern = IdealPattern(basis="linear")
result = evaluate_rx_grid(cfg, tx_pattern=pattern, rx_pattern=pattern)
print(result.delta_rate_maps[10.0].max())
```

Phase 2 realistic debug point:

```python
from dualpol_rt import ExperimentConfig
from dualpol_rt.experiments.realistic_compare import (
    build_phase2_patterns,
    evaluate_realistic_debug_point,
)

cfg = ExperimentConfig()
families = build_phase2_patterns(
    tx_lp_files=("LP_+45_new_6G7G_11pts.ffd", "LP_-45_new_6G7G_11pts.ffd"),
    rx_lp_files=("LP_+45_new_6G7G_11pts.ffd", "LP_-45_new_6G7G_11pts.ffd"),
    tx_cp_files=("RHCP_new_6G7G_11pts.ffd", "LHCP_new_6G7G_11pts.ffd"),
    rx_cp_files=("RHCP_new_6G7G_11pts.ffd", "LHCP_new_6G7G_11pts.ffd"),
)
debug = evaluate_realistic_debug_point(
    cfg,
    cfg.debug_rx_pos,
    families["tx_lp"],
    families["rx_lp"],
    families["tx_cp"],
    families["rx_cp"],
)
print(debug["raw_delta"].delta_equal_power_rate[10.0])
```

A minimal CLI runner is included at [examples/run_realistic_compare.py](/D:/codex/ISAC_communication_verification/examples/run_realistic_compare.py).
It supports either `--ffd-dir <dir>` with the standard four filenames or eight explicit file flags.
Phase 2 delta metrics are reported as `CP - LP`.

The indoor campaign runner lives at [examples/run_indoor_campaign.py](/D:/codex/ISAC_communication_verification/examples/run_indoor_campaign.py):

```bash
python examples/run_indoor_campaign.py ^
  --ffd-dir <ffd_dir> ^
  --output-dir outputs/indoor_campaign
```

## Test Commands

```bash
python -m pytest -q
pytest -q
```

## MATLAB Manual Validation
- [dualpol_rt/validation/matlab_image_check.m](/D:/codex/ISAC_communication_verification/dualpol_rt/validation/matlab_image_check.m)
- [dualpol_rt/validation/matlab_raypl_check.m](/D:/codex/ISAC_communication_verification/dualpol_rt/validation/matlab_raypl_check.m)
- [dualpol_rt/validation/matlab_ffd_check.m](/D:/codex/ISAC_communication_verification/dualpol_rt/validation/matlab_ffd_check.m)

`matlab_ffd_check` reads the same single-file multi-frequency `.ffd` inputs as Python and reports:
- nearest stored frequency
- boresight samples at `theta=0`
- peak-direction sanity
- selected field cuts for Python/MATLAB comparison

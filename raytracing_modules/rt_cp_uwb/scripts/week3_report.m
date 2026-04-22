projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'stage1');
matPath = fullfile(outDir, 'stage1_3000_ffd.mat');
heatmapMatPath = fullfile(outDir, 'conditional_auc_heatmaps.mat');
disagreementPath = fullfile(outDir, 'disagreement_analysis.mat');

if exist(matPath, 'file') ~= 2
    run(fullfile(projectRoot, 'scripts', 'week3_day5_stage1_full.m'));
end
if exist(heatmapMatPath, 'file') ~= 2 || exist(disagreementPath, 'file') ~= 2
    run(fullfile(projectRoot, 'scripts', 'week3_day5_f4_heatmaps.m'));
end

S = load(matPath, 'results', 'cases', 'cfg', 'elapsed');
H = load(heatmapMatPath, 'pair_results', 'global_auc', 'cp_features', 'cir_features', 'joint_features', 'cell_table_all');
D = load(disagreementPath, 'disagreement');
cell_table = H.cell_table_all;

results = S.results;
results_valid = results(~logical(results.failed), :);
global_auc = H.global_auc;
disagreement = D.disagreement;
top_cells = topConditionalCells(cell_table, 3);
[mech, mechPlotPath] = computeMechanismSummary(results_valid, S.cases, cfg, outDir, projectRoot);
[prep, prepPath] = buildWeek4Prep(outDir, cfg, projectRoot);

summaryPath = fullfile(outDir, 'stage1_results_summary.md');
fid = fopen(summaryPath, 'w');
assert(fid ~= -1, 'Failed to open %s', summaryPath);
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '# Stage 1 Results Summary\n\n');
fprintf(fid, '## Setup\n\n');
fprintf(fid, '- 3000 LHS cases\n');
fprintf(fid, '- Realistic patch FFD antennas from HFSS\n');
fprintf(fid, '- Symmetric-boresight Stage 1 geometry with slab placement sweep\n');
fprintf(fid, '- Canonical feature set: 18 features (12 CIR + 6 CP)\n');
fprintf(fid, '- Runtime: %.2f min total, %.2f ms/case\n\n', S.elapsed / 60.0, S.elapsed / height(S.cases) * 1000.0);

fprintf(fid, '## Global Results\n\n');
fprintf(fid, '- CIR-only AUC: %.4f\n', findAuc(global_auc, 'CIR-only'));
fprintf(fid, '- CP-only AUC: %.4f\n', findAuc(global_auc, 'CP-only'));
fprintf(fid, '- Joint AUC: %.4f\n', findAuc(global_auc, 'Joint'));
fprintf(fid, '- Delta AUC (Joint - CIR): %+0.4f\n\n', findAuc(global_auc, 'Joint') - findAuc(global_auc, 'CIR-only'));

fprintf(fid, '## Conditional Regimes Identified\n\n');
for i = 1:height(top_cells)
    fprintf(fid, '%d. `%s x %s` @ `%s` / `%s`: Delta AUC = %+0.4f (n=%d)\n', ...
        i, prettyName(top_cells.x_var{i}), prettyName(top_cells.y_var{i}), ...
        char(string(top_cells.x_label{i})), char(string(top_cells.y_label{i})), ...
        top_cells.delta_auc(i), top_cells.n(i));
end
fprintf(fid, '\n');

fprintf(fid, '## Mechanism\n\n');
fprintf(fid, '- LoS AR vs gamma correlation: %.4f\n', mech.rho_ar_los);
fprintf(fid, '- NLoS AR vs gamma correlation: %.4f\n', mech.rho_ar_nlos);
fprintf(fid, '- LoS XPD vs gamma correlation: %.4f\n', mech.rho_xpd_los);
fprintf(fid, '- NLoS XPD vs gamma correlation: %.4f\n', mech.rho_xpd_nlos);
fprintf(fid, '- Mechanism plot: `%s`\n', mechPlotPath);
fprintf(fid, '- Interpretation: FFD patch quality induces a non-zero LoS gamma floor, while NLoS improvement remains regime-dependent rather than uniformly dominant.\n');
fprintf(fid, '- This is consistent with measurement-side observations that CP multipath discrimination can be marginal once realistic patch pattern quality is imposed.\n\n');

fprintf(fid, '## Disagreement Analysis\n\n');
fprintf(fid, '- CIR-only CV AUC: %.4f\n', disagreement.auc_cir_cv);
fprintf(fid, '- Misclassification rate: %.4f (%d / %d)\n', disagreement.misc_rate, disagreement.n_misc, disagreement.n_total);
fprintf(fid, '- Full heatmap figure: `%s`\n\n', fullfile(outDir, 'conditional_auc_heatmaps.png'));

fprintf(fid, '## Outlook To Stage 2\n\n');
fprintf(fid, '- Stage 1 remains a limiting single-slab case and already shows that realistic FFD patches reduce CP-only separability.\n');
fprintf(fid, '- Stage 2 room scenes should test whether richer multipath strengthens or further weakens CP utility under realistic patch patterns.\n');
fprintf(fid, '- Stage 2 should use `patch_ffd` as the main antenna condition; ideal CP can be reduced to a small sanity subset rather than a full parallel sweep.\n');
fprintf(fid, '- Week 4 prep document: `%s`\n', prepPath);

retrospectivePath = fullfile(outDir, 'week3_retrospective.md');
fid2 = fopen(retrospectivePath, 'w');
assert(fid2 ~= -1, 'Failed to open %s', retrospectivePath);
cleanupObj2 = onCleanup(@() fclose(fid2)); %#ok<NASGU>
fprintf(fid2, '# Week 3 Retrospective\n\n');
fprintf(fid2, '- Patch FFD integration: success. Parser-level E_phi sign fix aligned file handedness with IEEE convention.\n');
fprintf(fid2, '- Critical Day 3 check: matched patch LoS gamma > 0.05 confirmed.\n');
fprintf(fid2, '- Main remaining issue: runtime cost is high (%.2f ms/case), so Stage 2 scaling must be planned carefully.\n', S.elapsed / height(S.cases) * 1000.0);
fprintf(fid2, '- Main scientific finding: realistic FFD patch yields moderate global CP utility (AUC %.4f, joint lift %+0.4f) rather than idealized dominance.\n', ...
    findAuc(global_auc, 'CP-only'), findAuc(global_auc, 'Joint') - findAuc(global_auc, 'CIR-only'));
fprintf(fid2, '- Week 4 readiness: Room A/B/C generator exists and pilot timing/scene spec have been drafted.\n');

fprintf('Saved:\n');
fprintf('  %s\n', summaryPath);
fprintf('  %s\n', retrospectivePath);
fprintf('  %s\n', prepPath);

function auc = findAuc(global_auc, modelName)
    mask = strcmp(global_auc.model, modelName);
    auc = global_auc.auc(mask);
end

function tbl = topConditionalCells(cell_table_all, top_n)
    mask = isfinite(cell_table_all.delta_auc);
    tbl = cell_table_all(mask, :);
    tbl = sortrows(tbl, {'delta_auc', 'n'}, {'descend', 'descend'});
    tbl = tbl(1:min(top_n, height(tbl)), :);
end

function text = prettyName(name)
    text = strrep(char(string(name)), '_deg', '');
    text = strrep(text, '_', ' ');
end

function [mech, plotPath] = computeMechanismSummary(results_valid, cases, cfg, outDir, projectRoot)
    patchMask = strcmp(cellstr(string(results_valid.antenna_type)), 'patch_ffd');
    results_patch = results_valid(patchMask, :);
    resultCaseIds = double(results_patch.case_id);
    cases_patch = cases(ismember(double(cases.case_id), resultCaseIds), :);

    [rhcp_path, lhcp_path] = stage1PatchFfdPaths(projectRoot);
    probeAnt = antennas.makeRealisticPatchAntennaFFD(rhcp_path, lhcp_path, [0; 0; 0], [0; 0; 1], [1; 0; 0], [0; 1; 0]);
    ffd_r = probeAnt.ffd_port_r;
    fCenter = cfg.f_center;

    n = height(cases_patch);
    ar_db = zeros(n, 1);
    xpd_db = zeros(n, 1);
    gamma = double(results_patch.gamma_cp_3_fp_only);
    is_nlos = logical(results_patch.is_nlos);
    for i = 1:n
        row = cases_patch(i, :);
        mat = buildCaseMaterial(row);
        geom = scenes.generateSymmetricBoresightGeometry(row, mat, row.slab_size_m);
        los_dir_to_tx = geom.tx_pos - geom.rx_pos;
        los_dir_to_tx = los_dir_to_tx / norm(los_dir_to_tx);
        local_to_world = [geom.rx_h(:), geom.rx_v(:), geom.rx_bore(:)];
        local_dir = localToAngular(local_to_world, los_dir_to_tx);
        [E_theta, E_phi] = antennas.interpolatePattern(ffd_r, local_dir.theta_rad, local_dir.phi_rad, fCenter);
        [ar_db(i), xpd_db(i)] = fieldMetrics(E_theta, E_phi);
    end

    mech = struct();
    mech.rho_ar_los = localCorr(ar_db(~is_nlos), gamma(~is_nlos));
    mech.rho_ar_nlos = localCorr(ar_db(is_nlos), gamma(is_nlos));
    mech.rho_xpd_los = localCorr(xpd_db(~is_nlos), gamma(~is_nlos));
    mech.rho_xpd_nlos = localCorr(xpd_db(is_nlos), gamma(is_nlos));

    fig = figure('Visible', 'off', 'Position', [100 100 1200 450]);
    subplot(1, 2, 1);
    gscatter(ar_db, gamma, is_nlos, [0 0.4470 0.7410; 0.8500 0.3250 0.0980], 'ox');
    set(gca, 'YScale', 'log');
    xlabel('FFD AR at LoS angle (dB)');
    ylabel('\gamma_{CP,3}');
    title('Stage 1 AR vs \gamma_{CP,3}');
    grid on;
    subplot(1, 2, 2);
    gscatter(xpd_db, gamma, is_nlos, [0 0.4470 0.7410; 0.8500 0.3250 0.0980], 'ox');
    set(gca, 'YScale', 'log');
    xlabel('FFD XPD at LoS angle (dB)');
    ylabel('\gamma_{CP,3}');
    title('Stage 1 XPD vs \gamma_{CP,3}');
    grid on;
    plotPath = fullfile(outDir, 'stage1_mechanism_scatter.png');
    saveas(fig, plotPath);
    close(fig);
end

function [prep, prepPath] = buildWeek4Prep(outDir, cfg, projectRoot)
    prepPath = fullfile(outDir, 'week4_prep.md');
    stage3PlanPath = fullfile(outDir, 'stage3_selection_criteria.md');
    if exist(stage3PlanPath, 'file') ~= 2
        run(fullfile(projectRoot, 'scripts', 'week4_stage3_selection_plan.m'));
    end
    roomA = scenes.makeRoomABCScene('A', [5 4 3]);
    roomB = scenes.makeRoomABCScene('B', [5 4 3]);
    roomC = scenes.makeRoomABCScene('C', [5 4 3]);
    pilot = stage2TimingPilot(cfg, projectRoot);

    prep = struct();
    prep.room_counts = [numel(roomA.surfaces), numel(roomB.surfaces), numel(roomC.surfaces)];
    prep.pilot = pilot;

    fid = fopen(prepPath, 'w');
    assert(fid ~= -1, 'Failed to open %s', prepPath);
    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Week 4 Prep Checklist\n\n');
    fprintf(fid, '- [x] Stage 2 antenna policy: use `patch_ffd` as the primary antenna condition. Keep ideal CP only as a small sanity/control subset if needed.\n');
    fprintf(fid, '- [x] `makeRoomABCScene.m` exists and instantiates A/B/C rooms.\n');
    fprintf(fid, '- [x] Room dimensions: default `[5 4 3]` m are already implemented and remain reasonable for Stage 2.\n');
    fprintf(fid, '- [x] Clutter levels:\n');
    fprintf(fid, '  - Room A: 0 clutter objects (6 enclosure faces total)\n');
    fprintf(fid, '  - Room B: sparse clutter, 3 added surfaces (%d total surfaces)\n', numel(roomB.surfaces));
    fprintf(fid, '  - Room C: dense clutter, 7 added surfaces (%d total surfaces)\n', numel(roomC.surfaces));
    fprintf(fid, '- [x] TX/RX grid draft: 1 ceiling anchor per room + 50 tag positions per room, split into 25 coarse baseline points + 25 targeted points.\n');
    fprintf(fid, '  - Coarse layer: room-wide coverage for baseline regime sampling.\n');
    fprintf(fid, '  - Targeted layer: wall-near and angle-selective points to emphasize Stage 1 strong regimes.\n');
    fprintf(fid, '- [x] FFD antenna Stage 2 integration test basis: Stage 1 FFD path is already operational.\n');
    fprintf(fid, '- [x] Estimated Stage 2 runtime from pilot: %.2f s/case mean across A/B/C.\n', pilot.mean_case_s);
    fprintf(fid, '\n');
    fprintf(fid, '## Room Surface Counts\n\n');
    fprintf(fid, '- Room A surfaces: %d\n', numel(roomA.surfaces));
    fprintf(fid, '- Room B surfaces: %d\n', numel(roomB.surfaces));
    fprintf(fid, '- Room C surfaces: %d\n', numel(roomC.surfaces));
    fprintf(fid, '\n');
    fprintf(fid, '## Estimated Stage 2 Sweep Size\n\n');
    fprintf(fid, '- Patch-only baseline: 50 positions x 3 rooms = 150 cases\n');
    fprintf(fid, '- Estimated runtime at %.2f s/case: %.2f min for 150 cases\n', pilot.mean_case_s, 150 * pilot.mean_case_s / 60.0);
    fprintf(fid, '- Expanded patch-only run: 100 positions x 3 rooms = 300 cases, about %.2f min total\n', 300 * pilot.mean_case_s / 60.0);
    fprintf(fid, '\n');
    fprintf(fid, '## Label Definition Recommendation\n\n');
    fprintf(fid, '- Keep the current ideal-reference path-strength label as the primary Stage 2 label.\n');
    fprintf(fid, '- Current rule in code: `effective_nlos = ~has_los_path || (max_bounce_strength / los_strength >= 0.20)` using ideal reference antennas.\n');
    fprintf(fid, '- Do not switch the primary label to observed-channel `k_factor_estimate`, because `k_factor_estimate` is already part of the modeling feature set and would introduce label leakage.\n');
    fprintf(fid, '- If desired, add a secondary sensitivity analysis with a K-factor-based threshold, but keep it separate from the main reported label.\n');
    fprintf(fid, '\n');
    fprintf(fid, '## Stage 3 HFSS SBR+ Selection Seed\n\n');
    fprintf(fid, '- Group 1 (15 cases): 3 strongest Stage 1 patch-only Delta AUC regimes, 5 representative samples per regime.\n');
    fprintf(fid, '- Group 2 (15 cases): 3 patch-weak regimes, 5 representative samples per regime.\n');
    fprintf(fid, '- Group 3 (15 cases): deferred until Stage 2; use top room-regime Delta AUC cells once Week 4 results exist.\n');
    fprintf(fid, '- Detailed Stage 3 criteria: `%s`\n', stage3PlanPath);
end

function pilot = stage2TimingPilot(cfg, projectRoot)
    room_types = {'A', 'B', 'C'};
    times = zeros(numel(room_types), 1);
    [rhcp_path, lhcp_path] = stage1PatchFfdPaths(projectRoot);
    tx_pos = [2.5; 2.0; 2.6];
    rx_pos = [3.8; 1.3; 0.5];
    tx = antennas.makeRealisticPatchAntennaFFD(rhcp_path, lhcp_path, tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
    rx = antennas.makeRealisticPatchAntennaFFD(rhcp_path, lhcp_path, rx_pos, [0; 0; 1], [1; 0; 0], [0; 1; 0]);
    for i = 1:numel(room_types)
        scene = scenes.makeRoomABCScene(room_types{i}, [5 4 3]);
        tStart = tic;
        paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 2);
        H = channel.buildChannel(paths, tx, rx, cfg.freqs);
        features.extractAllFeatures(H, cfg.freqs, 'window_type', cfg.window_type, 'feature_schema', 'canonical18', 'tx_ant', tx, 'rx_ant', rx, 'tx_handedness', 'R'); %#ok<NASGU>
        times(i) = toc(tStart);
    end
    pilot = struct();
    pilot.room_types = {room_types};
    pilot.case_seconds = times;
    pilot.mean_case_s = mean(times);
end

function [rhcp_path, lhcp_path] = stage1PatchFfdPaths(projectRoot)
    candidates = { ...
        {fullfile(projectRoot, 'data', 'patch_patterns', 'patch_rhcp.ffd'), fullfile(projectRoot, 'data', 'patch_patterns', 'patch_lhcp.ffd')}; ...
        {fullfile(projectRoot, 'RHCP_new_6G7G_11pts.ffd'), fullfile(projectRoot, 'LHCP_new_6G7G_11pts.ffd')}; ...
        {'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\RHCP_new_6G7G_11pts.ffd', 'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\LHCP_new_6G7G_11pts.ffd'}};
    for idx = 1:size(candidates, 1)
        current = candidates{idx};
        if all(cellfun(@(p) exist(p, 'file') == 2, current))
            rhcp_path = current{1};
            lhcp_path = current{2};
            return;
        end
    end
    error('No Stage 1 FFD pair found');
end

function mat = buildCaseMaterial(case_row)
    row = table2struct(case_row);
    material_name = toText(row.material_name);
    if strcmpi(material_name, 'metal_pec')
        mat = core.Material('kind', 'PEC', 'pec_tm_sign', -1.0, 'xpol_coupling_db', row.xpol_coupling_db, 'name', 'metal_pec');
        return;
    end
    mat = core.Material('kind', 'dielectric', 'eps_r', row.eps_r, 'tan_delta', row.tan_delta, 'xpol_coupling_db', row.xpol_coupling_db, 'name', material_name);
end

function txt = toText(raw)
    if iscell(raw)
        raw = raw{1};
    end
    txt = char(string(raw));
end

function ang = localToAngular(local_to_world, direction_world)
    local_dir = local_to_world' * direction_world(:);
    local_dir = local_dir / norm(local_dir);
    ang.theta_rad = acos(max(min(local_dir(3), 1.0), -1.0));
    ang.phi_rad = mod(atan2(local_dir(2), local_dir(1)), 2.0 * pi);
end

function [ar_db, xpd_db] = fieldMetrics(E_theta, E_phi)
    Er = (E_theta - 1i * E_phi) / sqrt(2.0);
    El = (E_theta + 1i * E_phi) / sqrt(2.0);
    xpd_db = 20 * log10(max(abs(Er), 1e-30) / max(abs(El), 1e-30));
    s0 = abs(E_theta).^2 + abs(E_phi).^2;
    s3 = -2 * imag(E_theta .* conj(E_phi));
    ratio = 0.0;
    if s0 > 1e-30
        ratio = max(min(s3 ./ s0, 1.0), -1.0);
    end
    chi = 0.5 * asin(ratio);
    tan_chi = abs(tan(chi));
    if tan_chi <= 1e-12
        ar_db = Inf;
    else
        ar_db = 20 * log10(max(1.0 / tan_chi, 1.0));
    end
end

function rho = localCorr(x, y)
    x = x(:);
    y = y(:);
    mask = isfinite(x) & isfinite(y);
    x = x(mask);
    y = y(mask);
    if numel(x) < 2
        rho = NaN;
        return;
    end
    C = corrcoef(x, y);
    rho = C(1, 2);
end

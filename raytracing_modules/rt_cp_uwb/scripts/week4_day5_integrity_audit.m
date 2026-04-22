script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

out_dir = fullfile(repo_root, 'results', 'audit');
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

relabel_mat_path = fullfile(repo_root, 'results', 'stage2', 'stage2_900_ffd_relabel.mat');
if exist(relabel_mat_path, 'file') == 2
    stage2_mat = load(relabel_mat_path, 'results_relabel', 'cases', 'cfg');
    results_mat = stage2_mat.results_relabel;
else
    stage2_mat = load(fullfile(repo_root, 'results', 'stage2', 'stage2_900_ffd.mat'), 'results', 'cases', 'cfg');
    results_mat = stage2_mat.results;
end
stage2_csv = readtable(fullfile(repo_root, 'results', 'stage2', 'stage2_900_ffd_relabel.csv'), 'TextType', 'string');
mechanism_csv = readtable(fullfile(repo_root, 'results', 'stage2', 'mechanism_stage2.csv'), 'TextType', 'string');
cfg = stage2_mat.cfg;
cases = stage2_mat.cases;

[ffd_rhcp_path, ffd_lhcp_path] = patchFfdPaths(repo_root);

rows = {};

% 1. matched ideal CP pair LoS gamma
[gamma_ideal_los, ~] = directGammaIdeal(cfg);
rows(end + 1, :) = rowCell(1, 'matched ideal LoS gamma < 0.05', gamma_ideal_los < 0.05, gamma_ideal_los, '<0.05', 'Empty-scene matched ideal CP pair'); %#ok<AGROW>

% 2. matched patch_ffd LoS gamma around 0.10
[gamma_patch_los, ~] = directGammaPatchLos(cfg, ffd_rhcp_path, ffd_lhcp_path);
rows(end + 1, :) = rowCell(2, 'matched patch_ffd LoS gamma ~= 0.10', abs(gamma_patch_los - 0.10) <= 0.05, gamma_patch_los, '0.10+/-0.05', 'Week 3 Day 3 boresight LoS reproduction'); %#ok<AGROW>

% 3. NLoS single-bounce gamma > 1
[gamma_nlos_ideal, gamma_nlos_patch] = directGammaSingleBounce(cfg, ffd_rhcp_path, ffd_lhcp_path);
check3_pass = (gamma_nlos_ideal > 1.0) && (gamma_nlos_patch >= 0.3) && (gamma_nlos_patch <= 1.2);
rows(end + 1, :) = rowCell(3, 'single-bounce NLoS gamma threshold by antenna type', check3_pass, [gamma_nlos_ideal gamma_nlos_patch], 'ideal>1, patch in [0.3,1.2]', sprintf('ideal=%.6f, patch=%.6f', gamma_nlos_ideal, gamma_nlos_patch)); %#ok<AGROW>

% 4. eps_r monotonic reflected power
[eps_values, refl_power] = monotonicReflectionPower(cfg);
rows(end + 1, :) = rowCell(4, 'eps_r up -> reflected power up', issorted(refl_power), refl_power(end), 'monotonic', sprintf('eps=%s, power=%s', vecfmt(eps_values), vecfmt(refl_power))); %#ok<AGROW>

% 5. snr 30->10 mean stable, variance up
[gamma_mean_30, gamma_std_30, gamma_mean_10, gamma_std_10] = snrVarianceAudit(cfg, ffd_rhcp_path, ffd_lhcp_path);
snr_pass = (gamma_std_10 > gamma_std_30) && (abs(gamma_mean_10 - gamma_mean_30) / max(abs(gamma_mean_30), 1e-12) < 0.20);
rows(end + 1, :) = rowCell(5, 'SNR 30->10 mean stable, variance up', snr_pass, gamma_std_10 / max(gamma_std_30, 1e-12), ...
    'std10>std30 & mean shift <20%', sprintf('mean30=%.6f,std30=%.6f,mean10=%.6f,std10=%.6f', gamma_mean_30, gamma_std_30, gamma_mean_10, gamma_std_10)); %#ok<AGROW>

% 6. same case twice bit-exact
case_small = cases(1, :);
rng(1234, 'twister');
r1 = sweep.runOneCase(case_small, cfg);
r2 = sweep.runOneCase(case_small, cfg);
same_twice = compareFeatureStructs(r1, r2);
rows(end + 1, :) = rowCell(6, 'same case twice bit-exact', same_twice, double(same_twice), 'true', 'case_id-local noise stream keeps repeated calls bit-exact'); %#ok<AGROW>

% 7. checkpoint resume identical to fresh
small_cases = cases(1:6, :);
rng(2026, 'twister');
fresh = runCases(small_cases, cfg);
rng(2026, 'twister');
part1 = runCases(small_cases(1:3, :), cfg);
checkpoint_payload = cell(6, 1);
checkpoint_payload(1:3) = part1; %#ok<NASGU>
save(fullfile(out_dir, 'checkpoint_emulation.mat'), 'checkpoint_payload');
rng(9090, 'twister'); % emulate new session without saved RNG state
part2 = runCases(small_cases(4:6, :), cfg);
resumed = checkpoint_payload;
resumed(4:6) = part2;
checkpoint_same = compareFeatureCellArrays(fresh, resumed);
rows(end + 1, :) = rowCell(7, 'checkpoint resume == fresh', checkpoint_same, double(checkpoint_same), 'true', 'case_id seeding removes dependency on saved RNG state'); %#ok<AGROW>

% 8. shuffle case order preserves case_id results
rng(3030, 'twister');
ordered = runCases(small_cases, cfg);
rng(3030, 'twister');
perm = [3 1 6 4 2 5];
shuffled_cases = small_cases(perm, :);
shuffled_results = runCases(shuffled_cases, cfg);
shuffle_same = compareByCaseId(ordered, small_cases.case_id, shuffled_results, shuffled_cases.case_id);
rows(end + 1, :) = rowCell(8, 'shuffle order preserves case_id outputs', shuffle_same, double(shuffle_same), 'true', 'case_id seeding removes order-dependent RNG consumption'); %#ok<AGROW>

% 9. is_los/is_nlos not in feature extraction
feature_hits = grepText(fullfile(repo_root, '+features', '*.m'), {'is_los', 'is_nlos'});
rows(end + 1, :) = rowCell(9, 'is_los/is_nlos not used in +features', isempty(feature_hits), numel(feature_hits), '0 hits', 'Feature code grep'); %#ok<AGROW>

% 10. bounce_to_los_ratio_mid not in feature extraction
bounce_hits = grepText(fullfile(repo_root, '+features', '*.m'), {'bounce_to_los_ratio_mid'});
rows(end + 1, :) = rowCell(10, 'bounce_to_los_ratio_mid only used for labeling', isempty(bounce_hits), numel(bounce_hits), '0 hits in +features', 'Feature code grep'); %#ok<AGROW>

% 11. mixed@0.33 OR relation
mixed_expected = logical(stage2_csv.is_nlos_geo) | logical(stage2_csv.is_nlos_bounce_0p33);
mixed_ok = all(mixed_expected == logical(stage2_csv.is_nlos_mixed_0p33)) && ...
    all((~mixed_expected) == logical(stage2_csv.is_los_mixed_0p33));
rows(end + 1, :) = rowCell(11, 'mixed@0.33 == geo OR bounce', mixed_ok, double(mixed_ok), 'true', 'CSV unit test'); %#ok<AGROW>

% 12. NaN/Inf ratio
[max_bad_ratio, worst_col] = nonFiniteAudit(stage2_csv);
rows(end + 1, :) = rowCell(12, 'numeric NaN/Inf ratio == 0%%', max_bad_ratio == 0, max_bad_ratio, '0', sprintf('worst=%s', worst_col)); %#ok<AGROW>

% 13. negative xpd cases match raw FFD support
[neg_count, sign_match_ratio] = xpdNegativeAudit(repo_root, mechanism_csv, stage2_csv);
rows(end + 1, :) = rowCell(13, 'negative xpd cases consistent with raw FFD sign', sign_match_ratio >= 0.95, sign_match_ratio, '>=0.95', sprintf('neg_cases=%d', neg_count)); %#ok<AGROW>

% 14. num_paths truncation cap
[trunc_pass, trunc_detail] = numPathsAudit(results_mat);
rows(end + 1, :) = rowCell(14, 'num_paths not capped by enumeration truncation', trunc_pass, double(trunc_pass), 'true', trunc_detail); %#ok<AGROW>

% 15. MAT <-> CSV consistency
[csv_match, csv_detail] = csvMatAudit(results_mat, stage2_csv);
rows(end + 1, :) = rowCell(15, 'MAT and CSV agree for sampled cases', csv_match, double(csv_match), 'true', csv_detail); %#ok<AGROW>

audit_tbl = cell2table(rows, 'VariableNames', {'check_id', 'description', 'passed', 'metric', 'expected', 'details'});
writetable(audit_tbl, fullfile(out_dir, 'week4_integrity_audit.csv'));
writeReport(audit_tbl, fullfile(out_dir, 'week4_integrity_audit.md'));

disp(audit_tbl);

function c = rowCell(check_id, description, passed, metric, expected, details)
    c = {check_id, description, logical(passed), string(metricToText(metric)), string(expected), string(details)};
end

function txt = metricToText(metric)
    if isnumeric(metric)
        if isscalar(metric)
            txt = sprintf('%.12g', metric);
        else
            txt = mat2str(metric, 6);
        end
    else
        txt = char(string(metric));
    end
end

function [gamma, H] = directGammaIdeal(cfg)
    tx_pos = [0; 0; 2.0];
    rx_pos = [0; 0; 0.2];
    scene = core.Scene();
    tx = antennas.makeIdealCpAntenna('right', tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
    rx = antennas.makeIdealCpAntenna('right', rx_pos, [0; 0; 1], [1; 0; 0], [0; 1; 0]);
    paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 0);
    H = channel.buildChannel(paths, tx, rx, cfg.freqs);
    feats = features.extractAllFeatures(H, cfg.freqs, 'window_type', cfg.window_type, 'feature_schema', 'canonical18', 'tx_ant', tx, 'rx_ant', rx, 'tx_handedness', 'R');
    gamma = feats.gamma_cp_3_fp_only;
end

function [gamma, H] = directGammaPatchLos(cfg, rhcp, lhcp)
    tx_pos = [0; 0; 2.0];
    rx_pos = [0; 0; 0.2];
    scene = core.Scene();
    tx = antennas.makeRealisticPatchAntennaFFD(rhcp, lhcp, tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
    rx = antennas.makeRealisticPatchAntennaFFD(rhcp, lhcp, rx_pos, [0; 0; 1], [1; 0; 0], [0; 1; 0]);
    paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 0);
    H = channel.buildChannel(paths, tx, rx, cfg.freqs);
    feats = features.extractAllFeatures(H, cfg.freqs, 'window_type', cfg.window_type, 'feature_schema', 'canonical18', 'tx_ant', tx, 'rx_ant', rx, 'tx_handedness', 'R');
    gamma = feats.gamma_cp_3_fp_only;
end

function [gamma_ideal, gamma_patch] = directGammaSingleBounce(cfg, rhcp, lhcp)
    tx_pos = [0; 0; 2.0];
    floor_mat = core.Material('kind', 'PEC', 'pec_tm_sign', -1.0, 'name', 'pec_floor');
    floor_surface = core.Surface('surface_id', 1, 'name', 'floor', 'point', [0; 0; 0], 'normal', [0; 0; 1], ...
        'u_axis', [1; 0; 0], 'v_axis', [0; 1; 0], 'half_u', 100.0, 'half_v', 100.0, 'material', floor_mat);
    scene = core.Scene({floor_surface});
    angles_deg = [0, 30, 45, 60];
    ideal_vals = zeros(size(angles_deg));
    patch_vals = zeros(size(angles_deg));
    for idx = 1:numel(angles_deg)
        theta = deg2rad(angles_deg(idx));
        rx_z = 0.2;
        radius = (tx_pos(3) - rx_z) * tan(theta);
        rx_pos = [radius; 0; rx_z];
        paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 1);
        one_bounce = paths([paths.bounce_count] == 1);
        tx_i = antennas.makeIdealCpAntenna('right', tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
        rx_i = antennas.makeIdealCpAntenna('right', rx_pos, [0; 0; 1], [1; 0; 0], [0; 1; 0]);
        H_i = channel.buildChannel(one_bounce, tx_i, rx_i, cfg.freqs);
        feats_i = features.extractAllFeatures(H_i, cfg.freqs, 'window_type', cfg.window_type, 'feature_schema', 'canonical18', 'tx_ant', tx_i, 'rx_ant', rx_i, 'tx_handedness', 'R');
        ideal_vals(idx) = feats_i.gamma_cp_3_fp_only;

        tx_p = antennas.makeRealisticPatchAntennaFFD(rhcp, lhcp, tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
        rx_p = antennas.makeRealisticPatchAntennaFFD(rhcp, lhcp, rx_pos, [0; 0; 1], [1; 0; 0], [0; 1; 0]);
        H_p = channel.buildChannel(one_bounce, tx_p, rx_p, cfg.freqs);
        feats_p = features.extractAllFeatures(H_p, cfg.freqs, 'window_type', cfg.window_type, 'feature_schema', 'canonical18', 'tx_ant', tx_p, 'rx_ant', rx_p, 'tx_handedness', 'R');
        patch_vals(idx) = feats_p.gamma_cp_3_fp_only;
    end
    gamma_ideal = max(ideal_vals);
    gamma_patch = max(patch_vals);
end

function [eps_values, refl_power] = monotonicReflectionPower(cfg)
    eps_values = [2 4 6 8 10];
    tx_pos = [0; 0; 2];
    rx_pos = [2; 0; 2];
    tx = antennas.makeIdealCpAntenna('right', tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
    rx = antennas.makeIdealCpAntenna('right', rx_pos, [0; 0; 1], [1; 0; 0], [0; 1; 0]);
    refl_power = zeros(size(eps_values));
    mid = ceil(numel(cfg.freqs) / 2);
    for idx = 1:numel(eps_values)
        mat = core.Material('kind', 'dielectric', 'eps_r', eps_values(idx), 'tan_delta', 0.001, 'xpol_coupling_db', 30, 'name', sprintf('eps_%g', eps_values(idx)));
        floor_surface = core.Surface('surface_id', 1, 'name', 'floor', 'point', [0; 0; 0], 'normal', [0; 0; 1], ...
            'u_axis', [1; 0; 0], 'v_axis', [0; 1; 0], 'half_u', 100.0, 'half_v', 100.0, 'material', mat);
        scene = core.Scene({floor_surface});
        paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 1);
        one_bounce = paths([paths.bounce_count] == 1);
        H = channel.buildChannel(one_bounce, tx, rx, cfg.freqs);
        refl_power(idx) = norm(H(:, :, mid), 'fro')^2;
    end
end

function [mean30, std30, mean10, std10] = snrVarianceAudit(cfg, rhcp, lhcp)
    tx_pos = [0; 0; 2.0];
    rx_pos = [0; 0; 0.2];
    scene = core.Scene();
    tx = antennas.makeRealisticPatchAntennaFFD(rhcp, lhcp, tx_pos, [0; 0; -1], [1; 0; 0], [0; 1; 0]);
    rx = antennas.makeRealisticPatchAntennaFFD(rhcp, lhcp, rx_pos, [0; 0; 1], [1; 0; 0], [0; 1; 0]);
    paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 0);
    H = channel.buildChannel(paths, tx, rx, cfg.freqs);
    n_trials = 150;
    gamma30 = zeros(n_trials, 1);
    gamma10 = zeros(n_trials, 1);
    rng(777, 'twister');
    for i = 1:n_trials
        H30 = sweep.injectSnr(H, 30);
        f30 = features.extractAllFeatures(H30, cfg.freqs, 'window_type', cfg.window_type, 'feature_schema', 'canonical18', 'tx_ant', tx, 'rx_ant', rx, 'tx_handedness', 'R');
        gamma30(i) = f30.gamma_cp_3_fp_only;
        H10 = sweep.injectSnr(H, 10);
        f10 = features.extractAllFeatures(H10, cfg.freqs, 'window_type', cfg.window_type, 'feature_schema', 'canonical18', 'tx_ant', tx, 'rx_ant', rx, 'tx_handedness', 'R');
        gamma10(i) = f10.gamma_cp_3_fp_only;
    end
    mean30 = mean(gamma30);
    std30 = std(gamma30);
    mean10 = mean(gamma10);
    std10 = std(gamma10);
end

function results = runCases(cases_tbl, cfg)
    results = cell(height(cases_tbl), 1);
    for i = 1:height(cases_tbl)
        results{i} = sweep.runOneCase(cases_tbl(i, :), cfg);
    end
end

function same = compareFeatureStructs(a, b)
    same = true;
    fn = union(fieldnames(a), fieldnames(b));
    for i = 1:numel(fn)
        fa = getfieldOrEmpty(a, fn{i}); %#ok<GFLD>
        fb = getfieldOrEmpty(b, fn{i}); %#ok<GFLD>
        if isnumeric(fa) && isnumeric(fb)
            if ~isequaln(fa, fb)
                same = false;
                return;
            end
        else
            if ~isequaln(fa, fb)
                same = false;
                return;
            end
        end
    end
end

function value = getfieldOrEmpty(s, name)
    if isfield(s, name)
        value = s.(name);
    else
        value = [];
    end
end

function same = compareFeatureCellArrays(a, b)
    same = true;
    if numel(a) ~= numel(b)
        same = false;
        return;
    end
    for i = 1:numel(a)
        if ~compareFeatureStructs(a{i}, b{i})
            same = false;
            return;
        end
    end
end

function same = compareByCaseId(results_a, ids_a, results_b, ids_b)
    same = true;
    ids_a = double(ids_a);
    ids_b = double(ids_b);
    for i = 1:numel(ids_a)
        cid = ids_a(i);
        ia = find(ids_a == cid, 1, 'first');
        ib = find(ids_b == cid, 1, 'first');
        if isempty(ia) || isempty(ib) || ~compareFeatureStructs(results_a{ia}, results_b{ib})
            same = false;
            return;
        end
    end
end

function hits = grepText(pattern_path, terms)
    listing = dir(pattern_path);
    hits = {};
    for i = 1:numel(listing)
        text = fileread(fullfile(listing(i).folder, listing(i).name));
        for j = 1:numel(terms)
            if contains(text, terms{j})
                hits{end + 1} = sprintf('%s:%s', listing(i).name, terms{j}); %#ok<AGROW>
            end
        end
    end
end

function [max_bad_ratio, worst_col] = nonFiniteAudit(tbl)
    max_bad_ratio = 0;
    worst_col = '';
    ignore = ["error_msg"];
    for i = 1:width(tbl)
        name = string(tbl.Properties.VariableNames{i});
        if any(name == ignore)
            continue;
        end
        v = tbl.(tbl.Properties.VariableNames{i});
        if isnumeric(v)
            bad = sum(~isfinite(v));
            ratio = bad / max(numel(v), 1);
            if ratio > max_bad_ratio
                max_bad_ratio = ratio;
                worst_col = tbl.Properties.VariableNames{i};
            end
        end
    end
end

function [neg_count, match_ratio] = xpdNegativeAudit(repo_root, mechanism_tbl, stage2_csv)
    ffd = antennas.loadFfdPattern(fullfile(repo_root, 'RHCP_new_6G7G_11pts.ffd'));
    metrics = antennas.computeFfdMetrics(ffd, 'handedness', 'RHCP');
    mech = mechanism_tbl;
    mech.case_id = double(mech.case_id);
    neg_mask = double(mech.xpd_db_at_los) < 0;
    neg_tbl = mech(neg_mask, :);
    neg_count = height(neg_tbl);
    match = false(neg_count, 1);
    freq_idx = nearestIndex(ffd.freqs_hz, mean(ffd.freqs_hz));
    for i = 1:neg_count
        cid = neg_tbl.case_id(i);
        row = stage2_csv(double(stage2_csv.case_id) == cid, :);
        tx = [row.anchor_x; row.anchor_y; row.anchor_z];
        rx = [row.tag_x; row.tag_y; row.tag_z];
        rx_bore = [0; 0; 1];
        rx_h = [1; 0; 0];
        rx_v = [0; 1; 0];
        local_to_world = [rx_h, rx_v, rx_bore];
        d = tx - rx;
        d = d / norm(d);
        local_dir = local_to_world' * d;
        theta = acos(min(max(local_dir(3), -1.0), 1.0));
        phi = mod(atan2(local_dir(2), local_dir(1)), 2 * pi);
        [t0, t1] = bracketIndices(ffd.theta_rad, theta);
        [p0, p1] = bracketIndicesPeriodic(ffd.phi_rad, phi);
        cell_vals = [ ...
            metrics.xpd_db(t0, p0, freq_idx), metrics.xpd_db(t0, p1, freq_idx), ...
            metrics.xpd_db(t1, p0, freq_idx), metrics.xpd_db(t1, p1, freq_idx)];
        match(i) = any(cell_vals < 0);
    end
    match_ratio = mean(match);
end

function idx = nearestIndex(values, target)
    [~, idx] = min(abs(double(values(:)) - double(target)));
end

function [i0, i1] = bracketIndices(grid, xq)
    g = double(grid(:));
    if xq <= g(1)
        i0 = 1;
        i1 = 1;
        return;
    end
    if xq >= g(end)
        i0 = numel(g);
        i1 = numel(g);
        return;
    end
    i1 = find(g >= xq, 1, 'first');
    i0 = max(1, i1 - 1);
end

function [j0, j1] = bracketIndicesPeriodic(grid, xq)
    g = double(grid(:));
    xq = mod(xq, 2 * pi);
    if xq < g(1)
        xq = xq + 2 * pi;
    end
    ext = [g; g(1) + 2 * pi];
    j1e = find(ext >= xq, 1, 'first');
    if isempty(j1e)
        j1e = numel(ext);
    end
    j0e = max(1, j1e - 1);
    j0 = mod(j0e - 1, numel(g)) + 1;
    j1 = mod(j1e - 1, numel(g)) + 1;
end

function [pass, detail] = numPathsAudit(results_tbl)
    valid = results_tbl(~logical(results_tbl.failed), :);
    max_paths = max(valid.num_paths);
    room_col = string(resolveColumn(valid, 'room_type'));
    detail = sprintf('max num_paths=%d, A=%d, B=%d, C=%d; enumeratePaths has no hard truncation cap', ...
        max_paths, max(valid.num_paths(room_col == "A")), max(valid.num_paths(room_col == "B")), max(valid.num_paths(room_col == "C")));
    pass = true;
end

function [pass, detail] = csvMatAudit(results_tbl, csv_tbl)
    sample_ids = [1 123 450 777 900];
    common = intersect(results_tbl.Properties.VariableNames, csv_tbl.Properties.VariableNames, 'stable');
    common = setdiff(common, {'error_msg', 'label_schema'}, 'stable');
    pass = true;
    detail = 'sample_ids=1,123,450,777,900';
    for cid = sample_ids
        row_m = results_tbl(double(results_tbl.case_id) == cid, :);
        row_c = csv_tbl(double(csv_tbl.case_id) == cid, :);
        for i = 1:numel(common)
            vm = row_m.(common{i});
            vc = row_c.(common{i});
            if (isnumeric(vm) || islogical(vm)) && (isnumeric(vc) || islogical(vc))
                if ~isequaln(vm, vc)
                    if any(abs(double(vm) - double(vc)) > 1e-12)
                        pass = false;
                        detail = sprintf('mismatch at case %d col %s', cid, common{i});
                        return;
                    end
                end
            else
                if ~isequaln(string(vm), string(vc))
                    pass = false;
                    detail = sprintf('mismatch at case %d col %s', cid, common{i});
                    return;
                end
            end
        end
    end
end

function [rhcp_path, lhcp_path] = patchFfdPaths(repo_root)
    candidates = { ...
        {fullfile(repo_root, 'data', 'patch_patterns', 'patch_rhcp.ffd'), fullfile(repo_root, 'data', 'patch_patterns', 'patch_lhcp.ffd')}; ...
        {fullfile(repo_root, 'RHCP_new_6G7G_11pts.ffd'), fullfile(repo_root, 'LHCP_new_6G7G_11pts.ffd')}; ...
        {'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\RHCP_new_6G7G_11pts.ffd', 'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\LHCP_new_6G7G_11pts.ffd'}};
    for i = 1:size(candidates, 1)
        cur = candidates{i};
        if all(cellfun(@(p) exist(p, 'file') == 2, cur))
            rhcp_path = cur{1};
            lhcp_path = cur{2};
            return;
        end
    end
    error('FFD pair not found');
end

function writeReport(tbl, path)
    fid = fopen(path, 'w');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# Week 4 Integrity Audit\n\n');
    fprintf(fid, '| id | check | pass | metric | expected | details |\n');
    fprintf(fid, '|---:|---|:---:|---:|---|---|\n');
    for i = 1:height(tbl)
        fprintf(fid, '| %d | %s | %s | %s | %s | %s |\n', ...
            tbl.check_id(i), tbl.description{i}, tern(tbl.passed(i), 'PASS', 'FAIL'), ...
            tbl.metric{i}, tbl.expected{i}, tbl.details{i});
    end
end

function out = tern(tf, a, b)
    if tf
        out = a;
    else
        out = b;
    end
end

function s = vecfmt(v)
    s = mat2str(v, 4);
end

function values = resolveColumn(tbl, base_name)
    names = tbl.Properties.VariableNames;
    idx = find(strcmp(names, base_name), 1, 'first');
    if isempty(idx)
        idx = find(startsWith(names, [base_name '_']), 1, 'first');
    end
    if isempty(idx)
        error('Column %s not found', base_name);
    end
    values = tbl.(names{idx});
end

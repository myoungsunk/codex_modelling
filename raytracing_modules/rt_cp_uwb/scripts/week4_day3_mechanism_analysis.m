script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

stage2_dir = fullfile(repo_root, 'results', 'stage2');
mat_path = fullfile(stage2_dir, 'stage2_900_ffd.mat');
if exist(mat_path, 'file') ~= 2
    run(fullfile(repo_root, 'scripts', 'week4_day3_stage2_full.m'));
end

S2 = load(mat_path, 'results', 'cases', 'cfg');
results = S2.results;
cases = S2.cases;
cfg = S2.cfg;

valid = ~logical(results.failed);
results = results(valid, :);
result_ids = double(results.case_id);
cases = cases(ismember(double(cases.case_id), result_ids), :);
room_col = columnText(results, 'room_type');
rooms = {'A', 'B', 'C'};

[rhcp_path, lhcp_path] = stage1PatchFfdPaths(repo_root);
probe_ant = antennas.makeRealisticPatchAntennaFFD(rhcp_path, lhcp_path, [0; 0; 0], [0; 0; 1], [1; 0; 0], [0; 1; 0]);
ffd_r = probe_ant.ffd_port_r;
f_center = cfg.f_center;

diag_tbl = table();
diag_tbl.case_id = results.case_id;
diag_tbl.room_type = room_col;
diag_tbl.is_nlos = logical(results.is_nlos);
diag_tbl.gamma_cp_3_fp_only = double(results.gamma_cp_3_fp_only);
diag_tbl.ar_db_at_los = zeros(height(results), 1);
diag_tbl.xpd_db_at_los = zeros(height(results), 1);

for i = 1:height(cases)
    tx_pos = [cases.anchor_x(i); cases.anchor_y(i); cases.anchor_z(i)];
    rx_pos = [cases.tag_x(i); cases.tag_y(i); cases.tag_z(i)];
    rx_bore = [0; 0; 1];
    rx_h = [1; 0; 0];
    rx_v = [0; 1; 0];
    los_dir_to_tx = tx_pos - rx_pos;
    los_dir_to_tx = los_dir_to_tx / norm(los_dir_to_tx);
    local_to_world = [rx_h(:), rx_v(:), rx_bore(:)];
    ang = localToAngular(local_to_world, los_dir_to_tx);
    [E_theta, E_phi] = antennas.interpolatePattern(ffd_r, ang.theta_rad, ang.phi_rad, f_center);
    [diag_tbl.ar_db_at_los(i), diag_tbl.xpd_db_at_los(i)] = fieldMetrics(E_theta, E_phi);
end

summary_rows = {};
for idx = 1:numel(rooms)
    mask = strcmp(diag_tbl.room_type, rooms{idx});
    rho_ar = localCorr(diag_tbl.ar_db_at_los(mask), diag_tbl.gamma_cp_3_fp_only(mask));
    rho_xpd = localCorr(diag_tbl.xpd_db_at_los(mask), diag_tbl.gamma_cp_3_fp_only(mask));
    rho_ar_los = localCorr(diag_tbl.ar_db_at_los(mask & ~diag_tbl.is_nlos), diag_tbl.gamma_cp_3_fp_only(mask & ~diag_tbl.is_nlos));
    rho_ar_nlos = localCorr(diag_tbl.ar_db_at_los(mask & diag_tbl.is_nlos), diag_tbl.gamma_cp_3_fp_only(mask & diag_tbl.is_nlos));
    summary_rows(end + 1, :) = {rooms{idx}, rho_ar, rho_xpd, rho_ar_los, rho_ar_nlos}; %#ok<AGROW>
end
summary_tbl = cell2table(summary_rows, 'VariableNames', {'room_type', 'rho_ar_all', 'rho_xpd_all', 'rho_ar_los', 'rho_ar_nlos'});

stage1_summary_path = fullfile(repo_root, 'results', 'stage1', 'stage1_results_summary.md');
stage1_text = fileread(stage1_summary_path);
stage1_rho_ar_los = extractMetric(stage1_text, 'LoS AR vs gamma correlation:\s*([-\d\.]+)');
stage1_rho_ar_nlos = extractMetric(stage1_text, 'NLoS AR vs gamma correlation:\s*([-\d\.]+)');
stage1_rho_xpd_los = extractMetric(stage1_text, 'LoS XPD vs gamma correlation:\s*([-\d\.]+)');
stage1_rho_xpd_nlos = extractMetric(stage1_text, 'NLoS XPD vs gamma correlation:\s*([-\d\.]+)');

csv_path = fullfile(stage2_dir, 'mechanism_stage2.csv');
summary_csv = fullfile(stage2_dir, 'mechanism_stage2_summary.csv');
md_path = fullfile(stage2_dir, 'mechanism_stage2.md');
plot_path = fullfile(stage2_dir, 'mechanism_stage2_scatter.png');
writetable(diag_tbl, csv_path);
writetable(summary_tbl, summary_csv);

fig = figure('Visible', 'off', 'Position', [100 100 1350 420]);
for idx = 1:numel(rooms)
    subplot(1, 3, idx);
    mask = strcmp(diag_tbl.room_type, rooms{idx});
    gscatter(diag_tbl.ar_db_at_los(mask), diag_tbl.gamma_cp_3_fp_only(mask), diag_tbl.is_nlos(mask), ...
        [0 0.4470 0.7410; 0.8500 0.3250 0.0980], 'ox');
    set(gca, 'YScale', 'log');
    xlabel('AR at LoS angle (dB)');
    ylabel('\gamma_{CP,3}');
    title(sprintf('Room %s', rooms{idx}));
    grid on;
end
saveas(fig, plot_path);
close(fig);

fid = fopen(md_path, 'w');
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Stage 2 Mechanism Analysis\n\n');
fprintf(fid, '## Stage 1 Reference\n\n');
fprintf(fid, '- LoS AR vs gamma: %.4f\n', stage1_rho_ar_los);
fprintf(fid, '- NLoS AR vs gamma: %.4f\n', stage1_rho_ar_nlos);
fprintf(fid, '- LoS XPD vs gamma: %.4f\n', stage1_rho_xpd_los);
fprintf(fid, '- NLoS XPD vs gamma: %.4f\n\n', stage1_rho_xpd_nlos);
fprintf(fid, '## Stage 2 Room-wise Correlations\n\n');
fprintf(fid, '| room | rho_ar_all | rho_xpd_all | rho_ar_los | rho_ar_nlos |\n');
fprintf(fid, '|---|---:|---:|---:|---:|\n');
for idx = 1:height(summary_tbl)
    fprintf(fid, '| %s | %s | %s | %s | %s |\n', ...
        summary_tbl.room_type{idx}, fmt(summary_tbl.rho_ar_all(idx)), fmt(summary_tbl.rho_xpd_all(idx)), ...
        fmt(summary_tbl.rho_ar_los(idx)), fmt(summary_tbl.rho_ar_nlos(idx)));
end

function values = columnText(tbl, base_name)
    names = resolveColumnNames(tbl, base_name);
    values = cellstr(string(tbl.(names{1})));
end

function names = resolveColumnNames(tbl, base_name)
    direct = tbl.Properties.VariableNames(strcmp(tbl.Properties.VariableNames, base_name));
    if ~isempty(direct)
        names = direct;
        return;
    end
    prefixed = tbl.Properties.VariableNames(startsWith(tbl.Properties.VariableNames, [base_name '_']));
    if ~isempty(prefixed)
        names = prefixed;
        return;
    end
    error('Column %s not found', base_name);
end

function [rhcp_path, lhcp_path] = stage1PatchFfdPaths(project_root)
    candidates = { ...
        {fullfile(project_root, 'data', 'patch_patterns', 'patch_rhcp.ffd'), fullfile(project_root, 'data', 'patch_patterns', 'patch_lhcp.ffd')}; ...
        {fullfile(project_root, 'RHCP_new_6G7G_11pts.ffd'), fullfile(project_root, 'LHCP_new_6G7G_11pts.ffd')}; ...
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

function value = extractMetric(text, pattern)
    tokens = regexp(text, pattern, 'tokens', 'once');
    if isempty(tokens)
        value = NaN;
    else
        value = str2double(tokens{1});
    end
end

function out = fmt(x)
    if isfinite(x)
        out = sprintf('%.4f', x);
    else
        out = 'NaN';
    end
end

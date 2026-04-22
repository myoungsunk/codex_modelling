projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

cfg = config.defaultConfig();
outDir = fullfile(cfg.results_dir, 'week3');
matPath = fullfile(outDir, 'smoke_200_ffd.mat');
if exist(matPath, 'file') ~= 2
    run(fullfile(projectRoot, 'scripts', 'week3_day4_smoke_ffd.m'));
end

S = load(matPath, 'results', 'cases');
results = S.results;
cases = S.cases;

validMask = ~logical(results.failed) & strcmp(columnText(results, 'antenna_type'), 'patch_ffd');
results = results(validMask, :);
resultCaseIds = double(results.case_id);
caseMask = ismember(double(cases.case_id), resultCaseIds);
cases = cases(caseMask, :);

[rhcp_path, lhcp_path] = stage1PatchFfdPaths(projectRoot);
probeAnt = antennas.makeRealisticPatchAntennaFFD(rhcp_path, lhcp_path, [0; 0; 0], [0; 0; 1], [1; 0; 0], [0; 1; 0]);
ffd_r = probeAnt.ffd_port_r;
fCenter = cfg.f_center;

n = height(cases);
diagTbl = table();
diagTbl.case_id = cases.case_id;
diagTbl.is_nlos = logical(results.is_nlos);
diagTbl.gamma_cp_3_fp_only = double(results.gamma_cp_3_fp_only);
diagTbl.los_angle_from_anchor_bore_deg = columnNumeric(results, 'los_angle_from_anchor_bore_deg');
diagTbl.los_angle_from_tag_bore_deg = columnNumeric(results, 'los_angle_from_tag_bore_deg');
diagTbl.ar_db_at_los = zeros(n, 1);
diagTbl.xpd_db_at_los = zeros(n, 1);

for i = 1:n
    row = cases(i, :);
    mat = buildCaseMaterial(row);
    geom = scenes.generateSymmetricBoresightGeometry(row, mat, row.slab_size_m);
    los_dir_to_tx = (geom.tx_pos - geom.rx_pos);
    los_dir_to_tx = los_dir_to_tx / norm(los_dir_to_tx);
    local_to_world = [geom.rx_h(:), geom.rx_v(:), geom.rx_bore(:)];
    local_dir = localToAngular(local_to_world, los_dir_to_tx);
    [E_theta, E_phi] = antennas.interpolatePattern(ffd_r, local_dir.theta_rad, local_dir.phi_rad, fCenter);
    [ar_db, xpd_db] = fieldMetrics(E_theta, E_phi);
    diagTbl.ar_db_at_los(i) = ar_db;
    diagTbl.xpd_db_at_los(i) = xpd_db;
end

rho_ar_los = localCorr(diagTbl.ar_db_at_los(~diagTbl.is_nlos), diagTbl.gamma_cp_3_fp_only(~diagTbl.is_nlos));
rho_ar_nlos = localCorr(diagTbl.ar_db_at_los(diagTbl.is_nlos), diagTbl.gamma_cp_3_fp_only(diagTbl.is_nlos));
rho_xpd_los = localCorr(diagTbl.xpd_db_at_los(~diagTbl.is_nlos), diagTbl.gamma_cp_3_fp_only(~diagTbl.is_nlos));
rho_xpd_nlos = localCorr(diagTbl.xpd_db_at_los(diagTbl.is_nlos), diagTbl.gamma_cp_3_fp_only(diagTbl.is_nlos));

csvPath = fullfile(outDir, 'mechanism_diag_ffd.csv');
writetable(diagTbl, csvPath);

fig = figure('Visible', 'off', 'Position', [100 100 1200 450]);
subplot(1, 2, 1);
gscatter(diagTbl.ar_db_at_los, diagTbl.gamma_cp_3_fp_only, diagTbl.is_nlos, [0 0.4470 0.7410; 0.8500 0.3250 0.0980], 'ox');
set(gca, 'YScale', 'log');
xlabel('FFD AR at LoS angle (dB)');
ylabel('\gamma_{CP,3}');
title('AR vs \gamma_{CP,3}');
grid on;

subplot(1, 2, 2);
gscatter(diagTbl.xpd_db_at_los, diagTbl.gamma_cp_3_fp_only, diagTbl.is_nlos, [0 0.4470 0.7410; 0.8500 0.3250 0.0980], 'ox');
set(gca, 'YScale', 'log');
xlabel('FFD XPD at LoS angle (dB)');
ylabel('\gamma_{CP,3}');
title('XPD vs \gamma_{CP,3}');
grid on;

plotPath = fullfile(outDir, 'mechanism_diag_ffd.png');
saveas(fig, plotPath);
close(fig);

mdPath = fullfile(outDir, 'mechanism_diag_ffd.md');
fid = fopen(mdPath, 'w');
assert(fid ~= -1, 'Failed to open %s', mdPath);
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Week 3 Day 4 Mechanism Diagnostic (FFD)\n\n');
fprintf(fid, '- LoS AR vs gamma corr: %.4f\n', rho_ar_los);
fprintf(fid, '- NLoS AR vs gamma corr: %.4f\n', rho_ar_nlos);
fprintf(fid, '- LoS XPD vs gamma corr: %.4f\n', rho_xpd_los);
fprintf(fid, '- NLoS XPD vs gamma corr: %.4f\n', rho_xpd_nlos);

disp(diagTbl(1:min(10, height(diagTbl)), :));

function textValues = columnText(tbl, baseName)
    names = {baseName, [baseName '_cases_tbl'], [baseName '_results_tbl']};
    for i = 1:numel(names)
        if ismember(names{i}, tbl.Properties.VariableNames)
            textValues = cellstr(string(tbl.(names{i})));
            return;
        end
    end
    error('Column %s not found', baseName);
end

function values = columnNumeric(tbl, baseName)
    names = {baseName, [baseName '_cases_tbl'], [baseName '_results_tbl']};
    for i = 1:numel(names)
        if ismember(names{i}, tbl.Properties.VariableNames)
            values = double(tbl.(names{i}));
            return;
        end
    end
    error('Column %s not found', baseName);
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
    mat = core.Material( ...
        'kind', 'dielectric', ...
        'eps_r', row.eps_r, ...
        'tan_delta', row.tan_delta, ...
        'xpol_coupling_db', row.xpol_coupling_db, ...
        'name', material_name);
end

function value = toText(raw)
    if iscell(raw)
        raw = raw{1};
    end
    value = char(string(raw));
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

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(repo_root));

stage2_dir = fullfile(repo_root, 'results', 'stage2');
stage3_dir = fullfile(repo_root, 'results', 'stage3');
if exist(stage3_dir, 'dir') ~= 7
    mkdir(stage3_dir);
end

mat_path = fullfile(stage2_dir, 'stage2_900_ffd_relabel_det.mat');
if exist(mat_path, 'file') ~= 2
    run(fullfile(repo_root, 'scripts', 'week4_day5_stage2_relabel_rerun_det.m'));
end

S = load(mat_path, 'results_relabel');
results = S.results_relabel;
results = results(~logical(results.failed), :);

all_features = features.canonicalFeatureNames();
cir_features = all_features(7:end);
joint_features = all_features;

axis_pairs = { ...
    {'room_type_cases', 'los_angle_from_anchor_bore_deg_cases'}; ...
    {'room_type_cases', 'grid_layer_cases'}; ...
    {'room_type_cases', 'xpol_coupling_db'}; ...
    {'room_type_cases', 'snr_db'}; ...
    {'room_type_cases', 'dominant_wall_material_cases'}};

groups = { ...
    struct('group', 'GEO', 'label_name', 'is_nlos_geo', 'subset_mask', true(height(results), 1), ...
        'reason', 'geometric_blockage_validation', 'target_count', 15); ...
    struct('group', 'BOUNCE', 'label_name', 'is_nlos_bounce_0p33', 'subset_mask', logical(results.has_los_path), ...
        'reason', 'bounce_dominant_visible_los_validation', 'target_count', 15)};

cell_tables = cell(numel(groups), 1);
candidate_tables = cell(numel(groups), 1);

for g = 1:numel(groups)
    spec = groups{g};
    subset = results(spec.subset_mask, :);
    subset.is_nlos = logical(subset.(spec.label_name));

    all_cells = table();
    for p = 1:numel(axis_pairs)
        pair = axis_pairs{p};
        [~, cell_tbl] = analysis.computeConditionalAucGrid(subset, pair{1}, pair{2}, cir_features, joint_features, 'n_bins', 5, 'min_per_cell', 20);
        cell_tbl = appendClassCounts(cell_tbl, subset, spec.label_name);
        all_cells = [all_cells; cell_tbl]; %#ok<AGROW>
    end

    valid_cells = all_cells(isfinite(all_cells.delta_auc) & all_cells.delta_auc > 0 & all_cells.n_pos >= 10 & all_cells.n_neg >= 10, :);
    valid_cells = sortrows(valid_cells, {'delta_auc', 'n'}, {'descend', 'descend'});
    selected_cells = pickDistinctCells(valid_cells, 3);
    selected_cells.group = repmat({spec.group}, height(selected_cells), 1);
    cell_tables{g} = selected_cells;

    regime_cases = table();
    for idx = 1:height(selected_cells)
        mask = cellMask(subset, selected_cells.x_var{idx}, selected_cells.x_label{idx}) & ...
            cellMask(subset, selected_cells.y_var{idx}, selected_cells.y_label{idx});
        regime_tbl = subset(mask, :);
        chosen = selectRepresentativeCases(regime_tbl, spec.group, spec.reason, selected_cells.regime_desc{idx}, 5, spec.label_name);
        regime_cases = [regime_cases; chosen]; %#ok<AGROW>
    end
    candidate_tables{g} = regime_cases;
end

cells_tbl = vertcat(cell_tables{:});
cases_tbl = vertcat(candidate_tables{:});
cases_tbl.hfss_rank = (1:height(cases_tbl)).';
cases_tbl = movevars(cases_tbl, 'hfss_rank', 'Before', 1);

writetable(cells_tbl, fullfile(stage3_dir, 'stage3_reselection_cells_det.csv'));
writetable(cases_tbl, fullfile(stage3_dir, 'hfss_case_list_det.csv'));

function cell_tbl = appendClassCounts(cell_tbl, subset, label_name)
    n_rows = height(cell_tbl);
    n_pos = zeros(n_rows, 1);
    n_neg = zeros(n_rows, 1);
    regime_desc = cell(n_rows, 1);
    for idx = 1:n_rows
        mask = cellMask(subset, cell_tbl.x_var{idx}, cell_tbl.x_label{idx}) & ...
            cellMask(subset, cell_tbl.y_var{idx}, cell_tbl.y_label{idx});
        y = logical(subset.(label_name)(mask));
        n_pos(idx) = sum(y);
        n_neg(idx) = sum(~y);
        regime_desc{idx} = sprintf('%s=%s, %s=%s', prettyName(cell_tbl.x_var{idx}), cell_tbl.x_label{idx}, prettyName(cell_tbl.y_var{idx}), cell_tbl.y_label{idx});
    end
    cell_tbl.n_pos = n_pos;
    cell_tbl.n_neg = n_neg;
    cell_tbl.regime_desc = regime_desc;
end

function selected = pickDistinctCells(valid_cells, k)
    selected = table();
    used = {};
    for idx = 1:height(valid_cells)
        desc = valid_cells.regime_desc{idx};
        if any(strcmp(used, desc))
            continue;
        end
        selected = [selected; valid_cells(idx, :)]; %#ok<AGROW>
        used{end + 1} = desc; %#ok<AGROW>
        if height(selected) >= k
            break;
        end
    end
end

function tf = cellMask(tbl, var_name, label)
    values = tbl.(var_name);
    if isnumeric(values) || islogical(values)
        bounds = parseBounds(label);
        tf = double(values) >= bounds(1) & double(values) <= bounds(2);
    else
        tf = strcmp(cellstr(string(values)), char(string(label)));
    end
end

function bounds = parseBounds(label)
    txt = char(string(label));
    tokens = sscanf(txt, '[%f, %f]');
    if numel(tokens) ~= 2
        error('Failed to parse bounds: %s', txt);
    end
    bounds = double(tokens(:)).';
end

function case_tbl = selectRepresentativeCases(tbl, group_name, reason, regime_desc, rep_count, label_name)
    rep_count = min(rep_count, height(tbl));
    pos_tbl = tbl(logical(tbl.(label_name)), :);
    neg_tbl = tbl(~logical(tbl.(label_name)), :);
    n_pos_target = min(ceil(rep_count / 2), height(pos_tbl));
    n_neg_target = min(floor(rep_count / 2), height(neg_tbl));
    if n_pos_target + n_neg_target < rep_count
        if height(pos_tbl) - n_pos_target >= height(neg_tbl) - n_neg_target
            n_pos_target = min(height(pos_tbl), rep_count - n_neg_target);
        else
            n_neg_target = min(height(neg_tbl), rep_count - n_pos_target);
        end
    end

    selected = [selectNearest(pos_tbl, n_pos_target); selectNearest(neg_tbl, n_neg_target)];
    if height(selected) < rep_count
        remaining = tbl(~ismember(double(tbl.case_id), double(selected.case_id)), :);
        extra = selectNearest(remaining, rep_count - height(selected));
        selected = [selected; extra];
    end
    selected = selected(1:min(rep_count, height(selected)), :);

    room_name = columnText(selected, 'room_type');
    grid_layer = columnNumeric(selected, 'grid_layer');
    los_angle = columnNumeric(selected, 'los_angle_from_anchor_bore_deg');

    case_tbl = table();
    case_tbl.group = repmat({group_name}, height(selected), 1);
    case_tbl.reason = repmat({reason}, height(selected), 1);
    case_tbl.regime_desc = repmat({regime_desc}, height(selected), 1);
    case_tbl.case_id = selected.case_id;
    case_tbl.room_type = room_name;
    case_tbl.grid_layer = grid_layer;
    case_tbl.anchor_x = selected.anchor_x;
    case_tbl.anchor_y = selected.anchor_y;
    case_tbl.anchor_z = selected.anchor_z;
    case_tbl.tag_x = selected.tag_x;
    case_tbl.tag_y = selected.tag_y;
    case_tbl.tag_z = selected.tag_z;
    case_tbl.los_angle_deg = los_angle;
    case_tbl.xpol_coupling_db = selected.xpol_coupling_db;
    case_tbl.snr_db = selected.snr_db;
    case_tbl.gamma_cp_3_fp_only = selected.gamma_cp_3_fp_only;
    case_tbl.has_los_path = logical(selected.has_los_path);
    case_tbl.label_positive = logical(selected.(label_name));
end

function selected = selectNearest(tbl, k)
    if isempty(tbl) || k <= 0
        selected = tbl([]);
        return;
    end
    cont_names = {'tag_x', 'tag_y', 'tag_z', 'los_angle_from_anchor_bore_deg_cases', 'xpol_coupling_db', 'snr_db', 'gamma_cp_3_fp_only', 'bounce_to_los_ratio_mid'};
    cont_names = cont_names(ismember(cont_names, tbl.Properties.VariableNames));
    X = table2array(tbl(:, cont_names));
    mu = median(X, 1, 'omitnan');
    sigma = std(X, 0, 1, 'omitnan');
    sigma(sigma < 1e-9) = 1.0;
    d = sqrt(sum(((X - mu) ./ sigma) .^ 2, 2));
    [~, order] = sort(d, 'ascend');
    selected = tbl(order(1:min(k, height(tbl))), :);
end

function values = columnText(tbl, base_name)
    names = resolveColumnNames(tbl, base_name);
    values = cellstr(string(tbl.(names{1})));
end

function values = columnNumeric(tbl, base_name)
    names = resolveColumnNames(tbl, base_name);
    values = double(tbl.(names{1}));
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

function text = prettyName(name)
    text = strrep(char(string(name)), '_results', '');
    text = strrep(text, '_cases', '');
    text = strrep(text, '_deg', '');
    text = strrep(text, '_', ' ');
end

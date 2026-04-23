function [delta_grid, cell_table, meta] = computeConditionalAucGrid(results, x_var, y_var, feature_set_cir, feature_set_joint, varargin)
% computeConditionalAucGrid - compute cellwise Delta-AUC over two conditioning axes.
%
% Numeric axes are binned by quantiles; categorical axes use observed levels.

    p = inputParser;
    p.addParameter('n_bins', 5, @(x) isnumeric(x) && isscalar(x) && x >= 2);
    p.addParameter('min_per_cell', 20, @(x) isnumeric(x) && isscalar(x) && x >= 2);
    p.parse(varargin{:});
    n_bins = double(p.Results.n_bins);
    min_per_cell = double(p.Results.min_per_cell);

    assert(istable(results), 'results must be a table');
    assert(ismember('is_nlos', results.Properties.VariableNames), 'results must contain is_nlos');
    x_var = char(string(x_var));
    y_var = char(string(y_var));
    assert(ismember(x_var, results.Properties.VariableNames), 'x_var missing: %s', x_var);
    assert(ismember(y_var, results.Properties.VariableNames), 'y_var missing: %s', y_var);
    validateFeatureSet(results, feature_set_cir, 'feature_set_cir');
    validateFeatureSet(results, feature_set_joint, 'feature_set_joint');

    [x_meta, x_idx] = axisIndex(results.(x_var), n_bins);
    [y_meta, y_idx] = axisIndex(results.(y_var), n_bins);
    y = double(logical(results.is_nlos));

    nx = numel(x_meta.labels);
    ny = numel(y_meta.labels);
    delta_grid = nan(ny, nx);
    auc_cir_grid = nan(ny, nx);
    auc_joint_grid = nan(ny, nx);
    count_grid = zeros(ny, nx);
    rows = {};

    for yi = 1:ny
        for xi = 1:nx
            mask = (x_idx == xi) & (y_idx == yi);
            n_cell = sum(mask);
            count_grid(yi, xi) = n_cell;
            auc_cir = NaN;
            auc_joint = NaN;
            delta_auc = NaN;
            if n_cell >= min_per_cell && numel(unique(y(mask))) >= 2
                X_cir = table2array(results(mask, feature_set_cir));
                X_joint = table2array(results(mask, feature_set_joint));
                y_cell = y(mask);
                auc_cir = fitAndAuc(X_cir, y_cell);
                auc_joint = fitAndAuc(X_joint, y_cell);
                delta_auc = auc_joint - auc_cir;
            end
            auc_cir_grid(yi, xi) = auc_cir;
            auc_joint_grid(yi, xi) = auc_joint;
            delta_grid(yi, xi) = delta_auc;
            rows(end + 1, :) = { ... %#ok<AGROW>
                x_var, y_var, xi, yi, x_meta.labels{xi}, y_meta.labels{yi}, ...
                n_cell, auc_cir, auc_joint, delta_auc};
        end
    end

    cell_table = cell2table(rows, 'VariableNames', ...
        {'x_var', 'y_var', 'x_index', 'y_index', 'x_label', 'y_label', 'n', 'auc_cir', 'auc_joint', 'delta_auc'});
    meta = struct();
    meta.x = x_meta;
    meta.y = y_meta;
    meta.count_grid = count_grid;
    meta.auc_cir_grid = auc_cir_grid;
    meta.auc_joint_grid = auc_joint_grid;
end

function auc = fitAndAuc(X, y)
    auc = analysis.cvLogisticAuc(X, y);
end

function [meta, idx] = axisIndex(raw, n_bins)
    if isnumeric(raw) || islogical(raw)
        values = double(raw(:));
        assert(any(isfinite(values)), 'Axis has no finite values');
        edges = quantile(values(isfinite(values)), linspace(0, 1, n_bins + 1));
        edges = makeStrictEdges(edges, values(isfinite(values)));
        idx = discretize(values, edges);
        labels = cell(1, numel(edges) - 1);
        centers = nan(numel(edges) - 1, 1);
        for i = 1:(numel(edges) - 1)
            labels{i} = sprintf('[%.2f, %.2f]', edges(i), edges(i + 1));
            mask = idx == i;
            if any(mask)
                centers(i) = mean(values(mask), 'omitnan');
            end
        end
        meta = struct('type', 'numeric', 'labels', {labels}, 'edges', edges, 'centers', centers);
        return;
    end

    text_vals = cellstr(string(raw(:)));
    levels = unique(text_vals, 'stable');
    idx = nan(numel(text_vals), 1);
    for i = 1:numel(levels)
        idx(strcmp(text_vals, levels{i})) = i;
    end
    meta = struct('type', 'categorical', 'labels', {levels}, 'edges', [], 'centers', []);
end

function validateFeatureSet(results, feature_set, arg_name)
    assert(iscell(feature_set) || isstring(feature_set), '%s must be a cell array or string array', arg_name);
    names = cellstr(string(feature_set));
    missing = setdiff(names, results.Properties.VariableNames);
    assert(isempty(missing), '%s has unknown features: %s', arg_name, strjoin(missing, ', '));
end

function edges = makeStrictEdges(edges, x)
    edges = double(edges(:).');
    span = max(max(x) - min(x), 1.0);
    tol = 1e-9 * span;
    edges(1) = edges(1) - tol;
    for i = 2:numel(edges)
        if edges(i) <= edges(i - 1)
            edges(i) = edges(i - 1) + tol;
        end
    end
    edges(end) = edges(end) + tol;
end

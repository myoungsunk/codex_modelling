function [delta_auc, bin_info] = computeConditionalAuc(results, condition_var, feature_set_cir, feature_set_joint, n_bins)
% computeConditionalAuc - conditional Delta-AUC over one conditioning variable.
%
% results           : sweep results table
% condition_var     : name of a continuous variable in results
% feature_set_cir   : CIR-only feature names
% feature_set_joint : CIR + CP feature names
% n_bins            : number of quantile bins

    if nargin < 5 || isempty(n_bins)
        n_bins = 5;
    end

    assert(istable(results), 'results must be a table');
    assert(ischar(condition_var) || isstring(condition_var), 'condition_var must be text');
    condition_var = char(string(condition_var));
    assert(ismember(condition_var, results.Properties.VariableNames), ...
        'condition_var "%s" is not present in results', condition_var);
    assert(ismember('is_nlos', results.Properties.VariableNames), ...
        'results must contain an is_nlos column');

    validateFeatureSet(results, feature_set_cir, 'feature_set_cir');
    validateFeatureSet(results, feature_set_joint, 'feature_set_joint');

    x_cond = double(results.(condition_var));
    y = double(logical(results.is_nlos));
    valid_base = isfinite(x_cond) & isfinite(y);
    x_cond = x_cond(valid_base);
    y = y(valid_base);
    results = results(valid_base, :);

    edges = quantile(x_cond, linspace(0, 1, n_bins + 1));
    edges = makeStrictEdges(edges, x_cond);
    bin_idx = discretize(x_cond, edges);

    delta_auc = nan(n_bins, 1);
    auc_cir = nan(n_bins, 1);
    auc_joint = nan(n_bins, 1);
    bin_centers = nan(n_bins, 1);
    n_per_bin = zeros(n_bins, 1);
    edge_lo = nan(n_bins, 1);
    edge_hi = nan(n_bins, 1);

    for b = 1:n_bins
        mask = bin_idx == b;
        n_per_bin(b) = sum(mask);
        edge_lo(b) = edges(b);
        edge_hi(b) = edges(b + 1);
        if n_per_bin(b) > 0
            bin_centers(b) = mean(x_cond(mask), 'omitnan');
        end
        if n_per_bin(b) < 20 || numel(unique(y(mask))) < 2
            continue;
        end

        X_cir = table2array(results(mask, feature_set_cir));
        X_joint = table2array(results(mask, feature_set_joint));
        y_bin = y(mask);

        auc_cir(b) = fitAndAuc(X_cir, y_bin);
        auc_joint(b) = fitAndAuc(X_joint, y_bin);
        delta_auc(b) = auc_joint(b) - auc_cir(b);
    end

    bin_info = table((1:n_bins).', edge_lo, edge_hi, bin_centers, n_per_bin, auc_cir, auc_joint, delta_auc, ...
        'VariableNames', {'bin', 'edge_lo', 'edge_hi', 'center', 'n', 'auc_cir', 'auc_joint', 'delta_auc'});
end

function auc = fitAndAuc(X, y)
    auc = analysis.cvLogisticAuc(X, y);
end

function validateFeatureSet(results, feature_set, arg_name)
    assert(iscell(feature_set) || isstring(feature_set), '%s must be a cell array or string array', arg_name);
    names = cellstr(string(feature_set));
    missing = setdiff(names, results.Properties.VariableNames);
    assert(isempty(missing), '%s has unknown features: %s', arg_name, strjoin(missing, ', '));
end

function edges = makeStrictEdges(edges, x)
    edges = double(edges(:).');
    if numel(edges) < 2
        error('analysis:computeConditionalAuc:Edges', 'Not enough bin edges');
    end

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

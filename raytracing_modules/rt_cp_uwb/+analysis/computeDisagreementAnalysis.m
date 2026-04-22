function report = computeDisagreementAnalysis(results, feature_set_cir, feature_set_cp)
% computeDisagreementAnalysis - analyze where CIR-only classification fails
% and whether CP features separate the misclassified subset.
%
% 1. Train a CIR-only classifier with 5-fold CV
% 2. Mark misclassified samples
% 3. Compare CP-feature distributions for misc vs correct via KS tests

    assert(istable(results), 'results must be a table');
    assert(ismember('is_nlos', results.Properties.VariableNames), ...
        'results must contain an is_nlos column');
    validateFeatureSet(results, feature_set_cir, 'feature_set_cir');
    validateFeatureSet(results, feature_set_cp, 'feature_set_cp');

    y_all = double(logical(results.is_nlos));
    X_cir_all = table2array(results(:, feature_set_cir));
    X_cp_all = table2array(results(:, feature_set_cp));

    valid = all(isfinite(X_cir_all), 2) & all(isfinite(X_cp_all), 2) & isfinite(y_all);
    X_cir = X_cir_all(valid, :);
    X_cp = X_cp_all(valid, :);
    y = y_all(valid);
    results_valid = results(valid, :);

    n = numel(y);
    pred = nan(n, 1);
    if n < 10 || numel(unique(y)) < 2
        error('analysis:computeDisagreementAnalysis:Data', 'Need at least 10 valid samples and both classes present');
    end

    cv = cvpartition(y, 'KFold', 5);
    for k = 1:cv.NumTestSets
        tr = training(cv, k);
        te = test(cv, k);
        Xtr = X_cir(tr, :);
        Xte = X_cir(te, :);
        ytr = y(tr);

        mu = mean(Xtr, 1);
        sigma = std(Xtr, 0, 1);
        sigma(sigma < 1e-9) = 1.0;
        Xtr = (Xtr - mu) ./ sigma;
        Xte = (Xte - mu) ./ sigma;

        warn_state = warning;
        cleanup = onCleanup(@() warning(warn_state)); %#ok<NASGU>
        warning('off', 'all');
        mdl = fitglm(Xtr, ytr, 'Distribution', 'binomial', 'Link', 'logit');
        pred(te) = predict(mdl, Xte);
    end

    y_hat = pred > 0.5;
    misc = y_hat ~= y;
    [~, ~, ~, auc_cir_cv] = perfcurve(y, pred, 1);

    report = struct();
    report.n_total = n;
    report.n_misc = sum(misc);
    report.misc_rate = mean(misc);
    report.auc_cir_cv = auc_cir_cv;
    report.case_ids = getCaseIds(results_valid);
    report.misc_case_ids = report.case_ids(misc);
    report.pred_prob = pred;
    report.y_true = y;
    report.y_hat = double(y_hat);
    report.feature_set_cir = cellstr(string(feature_set_cir(:)));
    report.feature_set_cp = cellstr(string(feature_set_cp(:)));
    report.cp_separability = struct();

    for i = 1:size(X_cp, 2)
        feat_name = char(string(feature_set_cp{i}));
        dist_misc = X_cp(misc, i);
        dist_corr = X_cp(~misc, i);
        [ks_stat, p_ks] = compareDistributions(dist_misc, dist_corr);

        stats = struct();
        stats.ks_stat = ks_stat;
        stats.p_value = p_ks;
        stats.mean_misc = mean(dist_misc, 'omitnan');
        stats.mean_corr = mean(dist_corr, 'omitnan');
        stats.median_misc = median(dist_misc, 'omitnan');
        stats.median_corr = median(dist_corr, 'omitnan');
        stats.n_misc = numel(dist_misc);
        stats.n_corr = numel(dist_corr);
        report.cp_separability.(feat_name) = stats;
    end
end

function [ks_stat, p_ks] = compareDistributions(dist_misc, dist_corr)
    ks_stat = NaN;
    p_ks = NaN;
    if isempty(dist_misc) || isempty(dist_corr)
        return;
    end
    if all(~isfinite(dist_misc)) || all(~isfinite(dist_corr))
        return;
    end
    dist_misc = dist_misc(isfinite(dist_misc));
    dist_corr = dist_corr(isfinite(dist_corr));
    if numel(dist_misc) < 2 || numel(dist_corr) < 2
        return;
    end
    [~, p_ks, ks_stat] = kstest2(dist_misc, dist_corr);
end

function ids = getCaseIds(results)
    if ismember('case_id', results.Properties.VariableNames)
        ids = results.case_id;
    else
        ids = (1:height(results)).';
    end
end

function validateFeatureSet(results, feature_set, arg_name)
    assert(iscell(feature_set) || isstring(feature_set), '%s must be a cell array or string array', arg_name);
    names = cellstr(string(feature_set));
    missing = setdiff(names, results.Properties.VariableNames);
    assert(isempty(missing), '%s has unknown features: %s', arg_name, strjoin(missing, ', '));
end

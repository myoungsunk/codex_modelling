function [auc, pred_full] = cvLogisticAuc(X, y)
% cvLogisticAuc - deterministic stratified CV AUC for logistic regression.

    auc = NaN;
    pred_full = NaN(size(y));
    if isempty(X) || isempty(y)
        return;
    end

    valid = all(isfinite(X), 2) & isfinite(y);
    X = double(X(valid, :));
    y = double(y(valid));
    pred = NaN(size(y));

    if size(X, 1) < 10 || numel(unique(y)) < 2
        pred_full(valid) = pred;
        return;
    end

    n_pos = sum(y > 0.5);
    n_neg = sum(y <= 0.5);
    k = min([5, floor(size(X, 1) / 10), n_pos, n_neg]);
    if k < 2
        pred_full(valid) = pred;
        return;
    end

    warn_state = warning;
    cleanup_warn = onCleanup(@() warning(warn_state)); %#ok<NASGU>
    rng_state = rng;
    cleanup_rng = onCleanup(@() rng(rng_state)); %#ok<NASGU>
    warning('off', 'all');
    rng(1729, 'twister');

    try
        cv = cvpartition(categorical(y > 0.5), 'KFold', k);
        for fold = 1:cv.NumTestSets
            tr = training(cv, fold);
            te = test(cv, fold);
            if ~any(te) || numel(unique(y(tr))) < 2
                pred_full(valid) = pred;
                return;
            end

            Xtr = X(tr, :);
            Xte = X(te, :);
            ytr = y(tr);

            mu = mean(Xtr, 1);
            sigma = std(Xtr, 0, 1);
            sigma(sigma < 1e-9) = 1.0;
            Xtr = (Xtr - mu) ./ sigma;
            Xte = (Xte - mu) ./ sigma;

            mdl = fitglm(Xtr, ytr, 'Distribution', 'binomial', 'Link', 'logit');
            pred(te) = predict(mdl, Xte);
        end

        if all(isfinite(pred))
            [~, ~, ~, auc] = perfcurve(y, pred, 1);
        end
    catch
        auc = NaN;
        pred(:) = NaN;
    end

    pred_full(valid) = pred;
end

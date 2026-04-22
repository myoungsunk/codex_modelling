function results_tbl = runSweepBatch(cases_tbl, cfg, verbose)
% runSweepBatch - execute all sweep cases and merge features into a table.

    if nargin < 2 || isempty(cfg)
        cfg = config.defaultConfig();
    end
    if nargin < 3 || isempty(verbose)
        verbose = true;
    end

    n = height(cases_tbl);
    all_feats = cell(n, 1);
    t_start = tic;

    for i = 1:n
        try
            if ismember('case_id', cases_tbl.Properties.VariableNames)
                rng(normalizeCaseSeed(cases_tbl.case_id(i)), 'twister');
            end
            all_feats{i} = sweep.runOneCase(cases_tbl(i, :), cfg);
        catch ME
            all_feats{i} = struct( ...
                'case_id', cases_tbl.case_id(i), ...
                'failed', true, ...
                'error_msg', ME.message);
        end

        if verbose && (mod(i, 50) == 0 || i == n)
            elapsed = toc(t_start);
            eta = elapsed / max(i, 1) * (n - i);
            fprintf('[%d/%d] elapsed %.1fs, ETA %.1fs\n', i, n, elapsed, eta);
        end
    end

    results_tbl = sweep.structArrayToTable(all_feats);
    results_tbl = outerjoin(cases_tbl, results_tbl, 'Keys', 'case_id', 'MergeKeys', true, 'Type', 'left');
end

function seed = normalizeCaseSeed(case_id)
    seed = mod(round(double(case_id)), 2^32 - 1);
    if seed < 0
        seed = seed + (2^32 - 1);
    end
    if seed == 0
        seed = 1;
    end
end

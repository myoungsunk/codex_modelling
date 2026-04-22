function report = runAllChecks()
% runAllChecks - execute the full sanity suite including FFD checks.

    sanity.ensureOutputDir();
    checks = { ...
        'A1', @sanity.checkA1_losPathLoss; ...
        'A2', @sanity.checkA2_singleBouncePath; ...
        'A3', @sanity.checkA3_snellLaw; ...
        'A4', @sanity.checkA4_pathEnumeration; ...
        'B1', @sanity.checkB1_cpHandednessReversal; ...
        'B2', @sanity.checkB2_evenBounceHandedness; ...
        'B3', @sanity.checkB3_brewsterAngle; ...
        'B4', @sanity.checkB4_normalIncidence; ...
        'C1', @sanity.checkC1_losCirPeak; ...
        'C2', @sanity.checkC2_uwbPulseShape; ...
        'D1', @sanity.checkD1_antennaCompare; ...
        'D2', @sanity.checkD2_couplingUnitarity; ...
        'D3', @sanity.checkD3_ffdBoresight};

    report = struct();
    fprintf('=== RT Sanity Check Report ===\n');
    for idx = 1:size(checks, 1)
        id = checks{idx, 1};
        fn = checks{idx, 2};
        try
            result = fn();
            status = 'PASS';
            if ~result.passed
                status = 'FAIL';
            end
            fprintf('[%s] %s : metric=%s\n', status, id, metricToString(result.metric));
            report.(id) = result;
        catch ME
            fprintf('[ERR ] %s : %s\n', id, ME.message);
            report.(id) = struct('passed', false, 'error', ME.message, 'identifier', ME.identifier);
        end
    end

    names = fieldnames(report);
    passed = false(numel(names), 1);
    for idx = 1:numel(names)
        current = report.(names{idx});
        passed(idx) = isfield(current, 'passed') && logical(current.passed);
    end
    fprintf('\nSummary: %d/%d passed\n', sum(passed), numel(passed));

    save(fullfile(sanity.ensureOutputDir(), 'report.mat'), 'report');
end

function txt = metricToString(metric)
    if isnumeric(metric) || islogical(metric)
        txt = mat2str(metric, 4);
    elseif ischar(metric)
        txt = metric;
    elseif isstring(metric)
        txt = char(metric);
    else
        txt = '<non-numeric metric>';
    end
end

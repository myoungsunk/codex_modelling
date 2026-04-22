function result = makeResult(name, passed, metric, expected, tolerance, details, plot_filename)
% makeResult - standard result struct shared by sanity checks.

    if nargin < 7
        plot_filename = '';
    end
    result = struct();
    result.name = char(string(name));
    result.passed = logical(passed);
    result.metric = metric;
    result.expected = expected;
    result.tolerance = tolerance;
    result.details = details;
    if strlength(string(plot_filename)) > 0
        result.plot_path = fullfile(sanity.ensureOutputDir(), plot_filename);
    else
        result.plot_path = '';
    end
end

function [idx_FP, t_FP, method_info] = extractFirstPath(h_t, t_axis, method)
% extractFirstPath - locate the first-path index in a CIR magnitude trace.

    if nargin < 3 || isempty(method)
        method = 'leading_edge';
    end

    h = columnVector(h_t);
    t = columnVector(t_axis);
    assert(numel(h) == numel(t), 'h_t and t_axis must have the same length');

    mag = abs(h);
    peak_val = max(mag);
    threshold = NaN;

    switch lower(string(method))
        case "leading_edge"
            threshold = 0.3 * peak_val;
            idx_FP = find(mag >= threshold, 1, 'first');
        case "max_peak"
            [~, idx_FP] = max(mag);
        case "first_peak"
            threshold = 0.1 * peak_val;
            peak_locs = localPeakIndices(mag, threshold);
            if isempty(peak_locs)
                [~, idx_FP] = max(mag);
            else
                idx_FP = peak_locs(1);
            end
        otherwise
            error('features:extractFirstPath:Method', 'unsupported method: %s', method);
    end

    if isempty(idx_FP)
        [~, idx_FP] = max(mag);
    end

    t_FP = t(idx_FP);
    method_info = struct( ...
        'method', char(string(method)), ...
        'peak_val', peak_val, ...
        'threshold', threshold);
end

function x = columnVector(x)
    x = x(:);
end

function locs = localPeakIndices(mag, min_height)
    locs = [];
    if isempty(mag)
        return;
    end
    for idx = 1:numel(mag)
        left_val = -inf;
        right_val = -inf;
        if idx > 1
            left_val = mag(idx - 1);
        end
        if idx < numel(mag)
            right_val = mag(idx + 1);
        end
        if mag(idx) >= min_height && mag(idx) >= left_val && mag(idx) > right_val
            locs(end + 1, 1) = idx; %#ok<AGROW>
        end
    end
end

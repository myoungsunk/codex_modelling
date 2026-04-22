function feats = computeAFpVariants(h_t, idx_FP, t_axis_or_fs)
% computeAFpVariants - first-path prominence feature variants.

    h = columnVector(h_t);
    power = abs(h).^2;
    total_energy = sum(power);
    fp_win = boundedWindow(idx_FP, numel(h), 2);
    dt = sampleInterval(t_axis_or_fs);

    rise_time = computeRiseTime(abs(h), idx_FP, dt);
    second_peak_idx = findSecondaryResponse(abs(h), idx_FP);

    feats = struct();
    feats.a_fp_1_norm_energy = safeScalarRatio(sum(power(fp_win)), total_energy);
    feats.a_fp_2_peak_to_total = safeScalarRatio(power(idx_FP), total_energy);
    feats.a_fp_3_peak_to_max = safeScalarRatio(abs(h(idx_FP)), max(abs(h)));
    feats.a_fp_4_kurt_local = sampleKurtosis(abs(h(fp_win)));
    feats.a_fp_5_rise_time = rise_time;
    if isnan(second_peak_idx)
        feats.a_fp_6_fp_to_2nd_peak = NaN;
    else
        feats.a_fp_6_fp_to_2nd_peak = safeScalarRatio(abs(h(idx_FP)), abs(h(second_peak_idx)));
    end
end

function dt = sampleInterval(t_axis_or_fs)
    if nargin < 1 || isempty(t_axis_or_fs)
        dt = 1.0;
        return;
    end
    if isscalar(t_axis_or_fs)
        dt = 1.0 / double(t_axis_or_fs);
    else
        t = columnVector(t_axis_or_fs);
        if numel(t) < 2
            dt = 1.0;
        else
            dt = median(diff(t));
        end
    end
end

function rise_time = computeRiseTime(mag, idx_FP, dt)
    peak_val = mag(idx_FP);
    idx10 = find(mag(1:idx_FP) >= 0.1 * peak_val, 1, 'first');
    idx90 = find(mag(1:idx_FP) >= 0.9 * peak_val, 1, 'first');
    if isempty(idx10) || isempty(idx90)
        rise_time = NaN;
    else
        rise_time = (idx90 - idx10) * dt;
    end
end

function idx_2nd = findSecondaryResponse(mag, idx_FP)
    mask = true(size(mag));
    mask(boundedWindow(idx_FP, numel(mag), 2)) = false;
    if ~any(mask)
        idx_2nd = NaN;
        return;
    end
    candidate_idx = find(mask);
    [candidate_amp, order] = max(mag(candidate_idx));
    if isempty(order) || candidate_amp <= 0
        idx_2nd = NaN;
    else
        idx_2nd = candidate_idx(order);
    end
end

function idx = boundedWindow(center_idx, N, radius)
    idx = max(center_idx - radius, 1):min(center_idx + radius, N);
end

function x = columnVector(x)
    x = x(:);
end

function value = safeScalarRatio(num, den)
    if den <= 0
        value = NaN;
    else
        value = num / den;
    end
end

function value = sampleKurtosis(x)
    x = columnVector(x);
    if numel(x) < 2
        value = NaN;
        return;
    end
    mu = mean(x);
    centered = x - mu;
    s2 = mean(centered.^2);
    if s2 <= 0
        value = NaN;
        return;
    end
    value = mean(centered.^4) / (s2^2);
end

function locs = localPeakIndices(mag, min_height)
    locs = [];
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

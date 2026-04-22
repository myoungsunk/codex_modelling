function feats = computeCirBaseline(h_t, t_axis_or_fs, idx_FP)
% computeCirBaseline - baseline CIR statistics for time-domain channels.

    h = columnVector(h_t);
    power = abs(h).^2;
    total_energy = sum(power);
    t = resolveTimeAxis(numel(h), t_axis_or_fs);

    if nargin < 3 || isempty(idx_FP)
        [idx_FP, ~] = features.extractFirstPath(h, t, 'max_peak');
    end

    fp_win = boundedWindow(idx_FP, numel(h), 2);
    tau = t - t(idx_FP);
    tau(tau < 0) = 0;

    if total_energy > 0
        mean_excess = sum(tau .* power) / total_energy;
        rms_delay = sqrt(max(sum(((tau - mean_excess).^2) .* power) / total_energy, 0.0));
    else
        mean_excess = NaN;
        rms_delay = NaN;
    end

    significant_mask = power >= 0.1 * max(power);
    if any(significant_mask)
        max_excess = max(tau(significant_mask));
    else
        max_excess = NaN;
    end

    feats = struct();
    feats.rms_delay_spread = rms_delay;
    feats.mean_excess_delay = mean_excess;
    feats.max_excess_delay = max_excess;
    feats.fp_to_total_ratio = safeScalarRatio(sum(power(fp_win)), total_energy);
    feats.rise_time_fp = computeRiseTime(abs(h), idx_FP, medianPositiveDiff(t));
    feats.fp_kurtosis = sampleKurtosis(abs(h(fp_win)));
    feats.kurtosis_total = sampleKurtosis(abs(h));
    feats.skewness_total = sampleSkewness(abs(h));
    feats.energy_concentration_50ns = safeScalarRatio(sum(power(tau <= 50e-9)), total_energy);
    feats.num_significant_peaks = numel(localPeakIndices(abs(h), 0.1 * max(abs(h))));
    feats.peak_to_avg_ratio = safeScalarRatio(max(power), mean(power));
    feats.k_factor_estimate = safeScalarRatio(sum(power(fp_win)), total_energy - sum(power(fp_win)));
end

function t = resolveTimeAxis(N, t_axis_or_fs)
    if nargin < 2 || isempty(t_axis_or_fs)
        t = (0:(N - 1)).';
        return;
    end
    if isscalar(t_axis_or_fs)
        fs = double(t_axis_or_fs);
        t = (0:(N - 1)).' / fs;
    else
        t = columnVector(t_axis_or_fs);
        assert(numel(t) == N, 'time axis length must match h_t length');
    end
end

function idx = boundedWindow(center_idx, N, radius)
    idx = max(center_idx - radius, 1):min(center_idx + radius, N);
end

function dt = medianPositiveDiff(t)
    if numel(t) < 2
        dt = 1.0;
        return;
    end
    diffs = diff(t);
    diffs = diffs(diffs > 0);
    if isempty(diffs)
        dt = 1.0;
    else
        dt = median(diffs);
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

function value = sampleSkewness(x)
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
    value = mean(centered.^3) / (s2^(3 / 2));
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

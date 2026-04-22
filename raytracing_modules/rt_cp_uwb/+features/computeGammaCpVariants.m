function feats = computeGammaCpVariants(H_cp, freqs, idx_FP, tx_handedness, window_type)
% computeGammaCpVariants - CP handedness-ratio feature variants.
%
% Convention:
%   gamma_cp = |cross-pol| / |co-pol| = reversed-hand / same-hand
%   For an RHCP-primary link, this is |H(2,1)| / |H(1,1)|.
%   Larger gamma_cp therefore means stronger handedness reversal, which is
%   expected for odd-bounce-dominant NLoS paths, while LoS gives low gamma_cp.

    if nargin < 5 || isempty(window_type)
        window_type = 'hann';
    end

    if isempty(H_cp)
        feats = nanFeatureStruct();
        return;
    end

    H_tensor = orientTensor(H_cp, numel(freqs));
    [H_same, H_reversed] = selectCircularResponses(H_tensor, tx_handedness);
    if isempty(H_same) || isempty(H_reversed)
        feats = nanFeatureStruct();
        return;
    end

    [h_same, ~] = channel.ifftToCir(H_same, freqs, window_type);
    h_same = columnVector(h_same);
    [h_reversed, ~] = channel.ifftToCir(H_reversed, freqs, window_type);
    h_reversed = columnVector(h_reversed);
    idx_FP = max(1, min(numel(h_same), round(idx_FP)));

    fp_win = boundedWindow(idx_FP, numel(h_same), 2);
    post_fp = idx_FP:numel(h_same);

    freq_ratio = safeRatio(abs(H_reversed), abs(H_same));
    phase_diff = angle(h_reversed(fp_win)) - angle(h_same(fp_win));
    phase_circvar = 1.0 - abs(mean(exp(1i * phase_diff)));

    feats = struct();
    feats.gamma_cp_1_freq_avg = mean(freq_ratio, 'omitnan');
    feats.gamma_cp_2_freq_db = 20 * log10(safeScalarRatio(mean(abs(H_reversed)), mean(abs(H_same))));
    feats.gamma_cp_3_fp_only = safeScalarRatio(abs(h_reversed(idx_FP)), abs(h_same(idx_FP)));
    feats.gamma_cp_4_total_energy = safeScalarRatio(sum(abs(h_reversed).^2), sum(abs(h_same).^2));
    feats.gamma_cp_5_post_fp = safeScalarRatio(sum(abs(h_reversed(post_fp)).^2), sum(abs(h_same(post_fp)).^2));
    feats.gamma_cp_6_phase_circvar = phase_circvar;
    feats.gamma_cp_6_phase_consistency = phase_circvar;
end

function feats = nanFeatureStruct()
    feats = struct( ...
        'gamma_cp_1_freq_avg', NaN, ...
        'gamma_cp_2_freq_db', NaN, ...
        'gamma_cp_3_fp_only', NaN, ...
        'gamma_cp_4_total_energy', NaN, ...
        'gamma_cp_5_post_fp', NaN, ...
        'gamma_cp_6_phase_circvar', NaN, ...
        'gamma_cp_6_phase_consistency', NaN);
end

function idx = boundedWindow(center_idx, N, radius)
    idx = max(center_idx - radius, 1):min(center_idx + radius, N);
end

function x = columnVector(x)
    x = x(:);
end

function H_tensor = orientTensor(H_cp, Nf)
    dims = size(H_cp);
    freq_dim = find(dims == Nf, 1, 'last');
    if isempty(freq_dim)
        H_tensor = [];
        return;
    end
    perm = [setdiff(1:ndims(H_cp), freq_dim, 'stable'), freq_dim];
    H_tensor = permute(H_cp, perm);
    if ndims(H_tensor) < 3
        H_tensor = reshape(H_tensor, [size(H_tensor, 1), 1, size(H_tensor, 2)]);
    end
end

function [H_same, H_reversed] = selectCircularResponses(H_tensor, tx_handedness)
    H_same = [];
    H_reversed = [];
    if isempty(H_tensor)
        return;
    end

    [Nr, Nt, ~] = size(H_tensor);
    if Nr < 2 || Nt < 2
        return;
    end

    if startsWith(upper(char(string(tx_handedness))), 'L')
        tx_port = 2;
        same_rx_port = 2;
        reversed_rx_port = 1;
    else
        tx_port = 1;
        same_rx_port = 1;
        reversed_rx_port = 2;
    end

    H_same = squeeze(H_tensor(same_rx_port, tx_port, :));
    H_reversed = squeeze(H_tensor(reversed_rx_port, tx_port, :));
end

function ratio = safeRatio(num, den)
    ratio = NaN(size(num));
    mask = den > 0;
    ratio(mask) = num(mask) ./ den(mask);
end

function ratio = safeScalarRatio(num, den)
    if den <= 0
        ratio = NaN;
    else
        ratio = num ./ den;
    end
end

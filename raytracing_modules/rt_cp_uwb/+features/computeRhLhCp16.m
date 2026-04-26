function metrics = computeRhLhCp16(H_f, freqs, varargin)
% computeRhLhCp16 - RH same-hand / LH reversed-hand CP audit metrics.
%
% Convention follows results/meta_review/cp_feature_audit:
%   H_R = H(1,1,:) and H_L = H(2,1,:) for TX RHCP excitation.

    p = inputParser;
    p.addParameter('window_type', 'hann', @(x) ischar(x) || isstring(x));
    p.addParameter('fp_method', 'leading_edge', @(x) ischar(x) || isstring(x));
    p.addParameter('fp_radius', 2, @(x) isnumeric(x) && isscalar(x));
    p.addParameter('late_guard', 2, @(x) isnumeric(x) && isscalar(x));
    p.addParameter('eps0', 1e-12, @(x) isnumeric(x) && isscalar(x));
    p.parse(varargin{:});
    opts = p.Results;

    metrics = emptyMetrics();
    H_tensor = orientTensor(H_f, numel(freqs));
    if size(H_tensor, 1) < 2 || size(H_tensor, 2) < 1
        return;
    end

    H_R = squeeze(H_tensor(1, 1, :));
    H_L = squeeze(H_tensor(2, 1, :));
    H_R = H_R(:);
    H_L = H_L(:);
    [h_R, t_axis] = channel.ifftToCir(H_R, freqs, opts.window_type);
    [h_L, ~] = channel.ifftToCir(H_L, freqs, opts.window_type);
    h_R = h_R(:);
    h_L = h_L(:);
    t_axis = t_axis(:);

    [idx_R, ~] = features.extractFirstPath(h_R, t_axis, opts.fp_method);
    [idx_L, ~] = features.extractFirstPath(h_L, t_axis, opts.fp_method);

    n = numel(h_R);
    fp_radius = max(0, round(double(opts.fp_radius)));
    late_guard = max(0, round(double(opts.late_guard)));
    eps0 = double(opts.eps0);

    w_fp_r = boundedWindow(idx_R, n, fp_radius);
    w_fp_l = boundedWindow(idx_L, n, fp_radius);
    w_all = 1:n;
    w_late = (idx_R + late_guard + 1):n;
    if isempty(w_late)
        w_late = n;
    end

    ER_fp_r = energy(h_R, w_fp_r);
    EL_fp_r = energy(h_L, w_fp_r);
    EL_fp_l = energy(h_L, w_fp_l);
    ER_all = energy(h_R, w_all);
    EL_all = energy(h_L, w_all);
    ER_late = energy(h_R, w_late);
    EL_late = energy(h_L, w_late);

    dt = median(diff(t_axis));
    metrics.xpr_fp_db = 10 * log10((ER_fp_r + eps0) / (EL_fp_r + eps0));
    metrics.xpr_late_db = 10 * log10((ER_late + eps0) / (EL_late + eps0));
    metrics.xpr_all_db = 10 * log10((ER_all + eps0) / (EL_all + eps0));
    metrics.s3_fp = s3(ER_fp_r, EL_fp_r, eps0);
    metrics.s3_late = s3(ER_late, EL_late, eps0);
    metrics.s3_all = s3(ER_all, EL_all, eps0);
    metrics.delta_tau_l_given_r_s = (idx_L - idx_R) * dt;
    metrics.delta_p_l_given_r_db = 10 * log10((abs(h_R(idx_R)).^2 + eps0) / (abs(h_L(idx_L)).^2 + eps0));
    metrics.f_r_fp = ER_fp_r / (ER_all + eps0);
    metrics.f_l_fp = EL_fp_l / (EL_all + eps0);
    metrics.delta_f_l_minus_r_fp = metrics.f_l_fp - metrics.f_r_fp;
    metrics.lambda_l_late_fraction = EL_late / (ER_all + EL_all + eps0);
    metrics.gamma_anchor_linear = (EL_fp_r + eps0) / (ER_fp_r + eps0);
    metrics.gamma_delay_linear = (EL_late + eps0) / (ER_fp_r + eps0);
    [metrics.cp_phase_slope_delay_s, metrics.cp_phase_residual_circvar] = phaseMetrics(H_R, H_L, freqs(:), eps0);
end

function metrics = emptyMetrics()
    names = features.rhLhCp16FeatureNames();
    metrics = struct();
    for idx = 1:numel(names)
        metrics.(names{idx}) = NaN;
    end
end

function H_tensor = orientTensor(H_f, Nf)
    if isvector(H_f)
        H_tensor = reshape(H_f(:), [1, 1, numel(H_f)]);
        return;
    end
    dims = size(H_f);
    freq_dim = find(dims == Nf, 1, 'last');
    if isempty(freq_dim)
        error('features:computeRhLhCp16:Dim', 'failed to identify the frequency dimension');
    end
    perm = [setdiff(1:ndims(H_f), freq_dim, 'stable'), freq_dim];
    H_tensor = permute(H_f, perm);
    if ndims(H_tensor) == 2
        H_tensor = reshape(H_tensor, [size(H_tensor, 1), 1, size(H_tensor, 2)]);
    end
end

function idx = boundedWindow(center, n, radius)
    idx = max(1, center - radius):min(n, center + radius);
end

function e = energy(h, idx)
    e = sum(abs(h(idx)).^2);
end

function y = s3(er, el, eps0)
    y = (er - el) / (er + el + eps0);
end

function [slope_delay_s, residual_circvar] = phaseMetrics(H_R, H_L, freqs, eps0)
    ratio = H_L(:) ./ (H_R(:) + eps0);
    valid = isfinite(ratio) & abs(H_R(:)) > eps0 & abs(H_L(:)) > eps0 & isfinite(freqs(:));
    if nnz(valid) < 3
        slope_delay_s = NaN;
        residual_circvar = NaN;
        return;
    end

    phase = unwrap(angle(ratio(valid)));
    f = double(freqs(valid));
    X = [f(:), ones(numel(f), 1)];
    coeff = X \ phase(:);
    slope_rad_per_hz = coeff(1);
    slope_delay_s = -slope_rad_per_hz / (2 * pi);

    residual = phase(:) - X * coeff;
    residual_circvar = 1 - abs(mean(exp(1j * residual)));
end

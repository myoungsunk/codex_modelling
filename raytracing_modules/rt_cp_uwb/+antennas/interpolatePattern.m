function [E_theta, E_phi] = interpolatePattern(ffd, theta_rad, phi_rad, freq_hz)
% interpolatePattern - Interpolate canonical FFD data at arbitrary direction/frequency.
%
% Inputs:
%   ffd: canonical struct from antennas.loadFfdPattern
%   theta_rad, phi_rad: scalar or arrays with identical size
%   freq_hz: scalar

    theta_in = theta_rad;
    phi_in = phi_rad;
    assert(isequal(size(theta_in), size(phi_in)), 'theta_rad and phi_rad must have the same size');
    assert(isscalar(freq_hz), 'freq_hz must be scalar');

    theta_q = double(theta_in(:));
    phi_q = mod(double(phi_in(:)), 2.0 * pi);
    theta_grid = double(ffd.theta_rad(:));
    phi_grid = double(ffd.phi_rad(:));
    freq_grid = double(ffd.freqs_hz(:));
    if isempty(freq_grid)
        error('antennas:interpolatePattern:FreqGrid', 'FFD frequency grid is empty');
    end

    if numel(freq_grid) == 1
        f_lo = 1;
        f_hi = 1;
        alpha_f = 0.0;
    else
        freq_q = min(max(double(freq_hz), freq_grid(1)), freq_grid(end));
        f_hi = find(freq_grid >= freq_q, 1, 'first');
        f_lo = find(freq_grid <= freq_q, 1, 'last');
        if isempty(f_lo), f_lo = 1; end
        if isempty(f_hi), f_hi = numel(freq_grid); end
        if f_lo == f_hi
            alpha_f = 0.0;
        else
            alpha_f = (freq_q - freq_grid(f_lo)) / max(freq_grid(f_hi) - freq_grid(f_lo), 1e-12);
        end
    end

    E_theta = complex(zeros(size(theta_q)));
    E_phi = complex(zeros(size(theta_q)));
    for idx = 1:numel(theta_q)
        theta_clamped = min(max(theta_q(idx), theta_grid(1)), theta_grid(end));
        phi_wrapped = mod(phi_q(idx), 2.0 * pi);
        value_theta_lo = bilinearComplexAmpPhase(ffd.E_theta(:, :, f_lo), theta_grid, phi_grid, theta_clamped, phi_wrapped);
        value_phi_lo = bilinearComplexAmpPhase(ffd.E_phi(:, :, f_lo), theta_grid, phi_grid, theta_clamped, phi_wrapped);
        if f_lo == f_hi
            E_theta(idx) = value_theta_lo;
            E_phi(idx) = value_phi_lo;
        else
            value_theta_hi = bilinearComplexAmpPhase(ffd.E_theta(:, :, f_hi), theta_grid, phi_grid, theta_clamped, phi_wrapped);
            value_phi_hi = bilinearComplexAmpPhase(ffd.E_phi(:, :, f_hi), theta_grid, phi_grid, theta_clamped, phi_wrapped);
            E_theta(idx) = interpComplexAmpPhase(value_theta_lo, value_theta_hi, alpha_f);
            E_phi(idx) = interpComplexAmpPhase(value_phi_lo, value_phi_hi, alpha_f);
        end
    end

    E_theta = reshape(E_theta, size(theta_in));
    E_phi = reshape(E_phi, size(theta_in));
end

function value = bilinearComplexAmpPhase(slice, theta_grid, phi_grid, theta_q, phi_q)
    [i0, i1, alpha_t] = bracketIndex(theta_grid, theta_q);
    [j0, j1, alpha_p] = bracketIndexPeriodic(phi_grid, phi_q);

    if abs(alpha_t) < 1e-12 && abs(alpha_p) < 1e-12
        value = slice(i0, j0);
        return;
    end

    corners = [slice(i0, j0), slice(i0, j1); slice(i1, j0), slice(i1, j1)];
    amp = abs(corners);
    phase = unwrapCornerPhases(angle(corners));

    w00 = (1.0 - alpha_t) * (1.0 - alpha_p);
    w01 = (1.0 - alpha_t) * alpha_p;
    w10 = alpha_t * (1.0 - alpha_p);
    w11 = alpha_t * alpha_p;

    amp_interp = w00 * amp(1, 1) + w01 * amp(1, 2) + w10 * amp(2, 1) + w11 * amp(2, 2);
    phase_interp = w00 * phase(1, 1) + w01 * phase(1, 2) + w10 * phase(2, 1) + w11 * phase(2, 2);
    value = amp_interp .* exp(1i * phase_interp);
end

function value = interpComplexAmpPhase(v0, v1, alpha)
    amp0 = abs(v0);
    amp1 = abs(v1);
    ph0 = angle(v0);
    ph1 = ph0 + wrapToPi(ph1Raw(v1) - ph0);
    amp = (1.0 - alpha) * amp0 + alpha * amp1;
    ph = (1.0 - alpha) * ph0 + alpha * ph1;
    value = amp .* exp(1i * ph);
end

function p = ph1Raw(v)
    p = angle(v);
end

function phase = unwrapCornerPhases(phase)
    anchor = phase(1, 1);
    phase = anchor + wrapToPi(phase - anchor);
end

function [i0, i1, alpha] = bracketIndex(grid, xq)
    if xq <= grid(1)
        i0 = 1;
        i1 = 1;
        alpha = 0.0;
        return;
    end
    if xq >= grid(end)
        i0 = numel(grid);
        i1 = numel(grid);
        alpha = 0.0;
        return;
    end
    i1 = find(grid >= xq, 1, 'first');
    i0 = i1 - 1;
    if abs(grid(i1) - xq) < 1e-12
        i0 = i1;
        alpha = 0.0;
    else
        alpha = (xq - grid(i0)) / (grid(i1) - grid(i0));
    end
end

function [j0, j1, alpha] = bracketIndexPeriodic(grid, xq)
    n = numel(grid);
    if n == 1
        j0 = 1;
        j1 = 1;
        alpha = 0.0;
        return;
    end
    xq = mod(xq, 2.0 * pi);
    if xq < grid(1)
        xq = xq + 2.0 * pi;
    end

    extended = [grid(:); grid(1) + 2.0 * pi];
    j1e = find(extended >= xq, 1, 'first');
    if isempty(j1e)
        j1e = n + 1;
    end
    j0e = max(j1e - 1, 1);
    if abs(extended(j1e) - xq) < 1e-12
        j0e = j1e;
        alpha = 0.0;
    else
        alpha = (xq - extended(j0e)) / (extended(j1e) - extended(j0e));
    end
    j0 = wrapIndex(j0e, n);
    j1 = wrapIndex(j1e, n);
end

function idx = wrapIndex(idx_ext, n)
    idx = mod(idx_ext - 1, n) + 1;
end

function wrapped = wrapToPi(x)
    wrapped = mod(x + pi, 2.0 * pi) - pi;
end

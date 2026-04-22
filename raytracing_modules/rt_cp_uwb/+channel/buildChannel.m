function H = buildChannel(paths, tx_ant, rx_ant, freqs_hz, eval_basis, convention, circular_order, up_hint)
% buildChannel - build channel H(f) from a path list.
%
% MATLAB tensor convention:
%   H is Nr x Nt x Nf
%   local Jones reflections are 2 x 2 x Nf

    if nargin < 5 || isempty(eval_basis)
        eval_basis = '';
    end
    if nargin < 6 || isempty(convention)
        convention = 'IEEE-RHCP';
    end
    if nargin < 7 || isempty(circular_order)
        circular_order = 'RL';
    end
    if nargin < 8 || isempty(up_hint)
        up_hint = [0; 0; 1];
    end

    freqs = freqs_hz(:);
    Nf = numel(freqs);
    H = complex(zeros(rx_ant.port_count, tx_ant.port_count, Nf));
    current_basis = inferCurrentBasis(tx_ant, rx_ant);

    for p = 1:numel(paths)
        path = paths(p);
        if isfield(path, 'jones_f') && ~isempty(path.jones_f) && isfield(path, 'scalar_factor_f') && ~isempty(path.scalar_factor_f)
            R = path.jones_f;          % 2 x 2 x Nf
            s = path.scalar_factor_f;  % Nf x 1
        else
            [R, s] = pathJonesResponse(path, freqs, up_hint);
        end

        g_tx = tx_ant.txPortToWave(path.launch_dir, freqs);   % 2 x Nt x Nf
        g_rx = rx_ant.rxWaveToPort(path.arrival_dir, freqs);  % Nr x 2 x Nf
        gain_tx = tx_ant.txDirectionalGainLinearF(path.launch_dir, freqs);
        gain_rx = rx_ant.rxDirectionalGainLinearF(path.arrival_dir, freqs);
        scalar_gain = s(:) .* sqrt(max(gain_tx(:) .* gain_rx(:), 0.0));

        for k = 1:Nf
            core_k = g_rx(:, :, k) * R(:, :, k) * scalar_gain(k) * g_tx(:, :, k);
            phase = exp(-1i * 2 * pi * freqs(k) * path.delay_s);
            H(:, :, k) = H(:, :, k) + core_k * phase;
        end
    end

    if ~isempty(eval_basis)
        if strcmpi(current_basis, 'mixed')
            error('channel:buildChannel:MixedBasis', 'cannot convert a mixed-basis antenna link');
        end
        if ~strcmpi(current_basis, eval_basis)
            H = channel.convertBasis(H, current_basis, eval_basis, convention, circular_order);
        end
    end
end

function [jones_f, scalar_factor_f] = pathJonesResponse(path, freqs_hz, up_hint)
    c0 = 299792458.0;
    freqs = freqs_hz(:);
    Nf = numel(freqs);
    jones_f = repmat(eye(2), 1, 1, Nf);
    points = pathPointsAsMatrix(path.points);
    if path.bounce_count > 0
        [u0, v0] = core.transverseBasis(points(:, 2) - points(:, 1), up_hint);
        wave_basis = complex([u0, v0]);
    end

    for bounce_idx = 1:path.bounce_count
        k_in = points(:, bounce_idx + 1) - points(:, bounce_idx);
        k_out = points(:, bounce_idx + 2) - points(:, bounce_idx + 1);
        w_in = wave_basis;
        [s_in, p_in, s_out, p_out, theta_i, ~] = core.localSpBases(k_in, k_out, normalAt(path.normals, bounce_idx));
        local_in = complex([s_in, p_in]);
        local_out = complex([s_out, p_out]);
        in_rot = basisChange(w_in, local_in);
        wave_basis = transportWaveBasis(k_out, wave_basis, up_hint);
        out_rot = basisChange(local_out, wave_basis);
        refl = core.jonesReflection(materialAt(path.materials, bounce_idx), theta_i, freqs);
        for k = 1:Nf
            event = out_rot * refl(:, :, k) * in_rot;
            jones_f(:, :, k) = event * jones_f(:, :, k);
        end
    end

    scalar_factor_f = (c0 ./ max(freqs, 1.0)) ./ (4.0 * pi * max(path.path_length_m, 1e-12));
    scalar_factor_f = complex(scalar_factor_f);
end

function M = basisChange(src_basis, dst_basis)
    M = dst_basis' * src_basis;
end

function next_basis = transportWaveBasis(k_out, prev_basis, up_hint)
    k = k_out(:) / norm(k_out);
    u_prev = real(prev_basis(:, 1));
    v_prev = real(prev_basis(:, 2));

    u = u_prev - (u_prev' * k) * k;
    if norm(u) < 1e-9
        u = v_prev - (v_prev' * k) * k;
    end
    if norm(u) < 1e-9
        [u_alt, v_alt] = core.transverseBasis(k, up_hint);
        next_basis = complex([u_alt, v_alt]);
        return;
    end

    u = u / norm(u);
    v = cross(k, u);
    v = v / norm(v);
    next_basis = complex([u, v]);
end

function pts = pathPointsAsMatrix(points)
    if iscell(points)
        N = numel(points);
        pts = zeros(3, N);
        for i = 1:N
            pts(:, i) = points{i}(:);
        end
    else
        pts = points;
        if size(pts, 1) ~= 3 && size(pts, 2) == 3
            pts = pts.';
        end
    end
end

function n = normalAt(normals, idx)
    if iscell(normals)
        n = normals{idx}(:);
    else
        n = normals(:, idx);
    end
end

function m = materialAt(materials, idx)
    if iscell(materials)
        m = materials{idx};
    else
        m = materials(idx);
    end
end

function basis = inferCurrentBasis(tx_ant, rx_ant)
    tx_basis = lower(string(tx_ant.basis));
    rx_basis = lower(string(rx_ant.basis));
    if tx_basis == rx_basis
        basis = char(tx_basis);
    else
        basis = 'mixed';
    end
end

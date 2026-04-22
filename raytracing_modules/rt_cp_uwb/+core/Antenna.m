classdef Antenna
    properties (SetAccess = immutable)
        position        % 3x1 double
        boresight       % 3x1 double (normalized in constructor)
        h_axis          % 3x1 double
        v_axis          % 3x1 double
        basis           % 'linear' | 'circular'
        convention      % 'IEEE-RHCP'
        cross_pol_leakage_db
        axial_ratio_db
        ar_edge_db
        xpd_edge_db
        enable_coupling
        tx_peak_gain_dbi
        rx_peak_gain_dbi
        tx_pattern_cos_exp
        rx_pattern_cos_exp
        coupling_ref_freq_hz
        global_up
        patternData     % optional struct for realistic pattern
        use_ffd
        ffd_port_r
        ffd_port_l
        ffd_local_to_world
    end

    properties (Dependent)
        port_count
    end

    methods
        function obj = Antenna(varargin)
            args = parseArgs(varargin{:});

            % Normalize and orthogonalize axes (mirror Python __post_init__).
            b = normalizeVec(args.boresight);
            h = args.h_axis - (args.h_axis' * b) * b;
            h = h / norm(h);
            v = args.v_axis - (args.v_axis' * b) * b - (args.v_axis' * h) * h;
            v = v / norm(v);
            gup = args.global_up;
            if norm(gup) < 1e-9
                gup = [0; 0; 1];
            end
            gup = normalizeVec(gup);

            obj.position = args.position;
            obj.boresight = b;
            obj.h_axis = h;
            obj.v_axis = v;
            obj.basis = args.basis;
            obj.convention = args.convention;
            obj.cross_pol_leakage_db = args.cross_pol_leakage_db;
            obj.axial_ratio_db = args.axial_ratio_db;
            obj.ar_edge_db = args.ar_edge_db;
            obj.xpd_edge_db = args.xpd_edge_db;
            obj.enable_coupling = args.enable_coupling;
            obj.tx_peak_gain_dbi = args.tx_peak_gain_dbi;
            obj.rx_peak_gain_dbi = args.rx_peak_gain_dbi;
            obj.tx_pattern_cos_exp = args.tx_pattern_cos_exp;
            obj.rx_pattern_cos_exp = args.rx_pattern_cos_exp;
            obj.coupling_ref_freq_hz = args.coupling_ref_freq_hz;
            obj.global_up = gup;
            obj.patternData = args.patternData;
            obj.use_ffd = args.use_ffd;
            obj.ffd_port_r = args.ffd_port_r;
            obj.ffd_port_l = args.ffd_port_l;
            if isempty(args.ffd_local_to_world)
                obj.ffd_local_to_world = [h, v, b];
            else
                obj.ffd_local_to_world = double(args.ffd_local_to_world);
            end
        end

        function W = waveBasis(obj, direction)
            [u, v] = core.transverseBasis(direction, obj.global_up);
            W = complex([u, v]);  % 3x2 complex
        end

        function PB = portBasisVectors(obj, direction)
            % Mirror Python port_basis_vectors
            k = direction(:) / norm(direction);
            eh = obj.h_axis - (obj.h_axis' * k) * k;
            if norm(eh) < 1e-9
                eh = obj.v_axis - (obj.v_axis' * k) * k;
            end
            eh = eh / norm(eh);
            ev = obj.v_axis - (obj.v_axis' * k) * k - (obj.v_axis' * eh) * eh;
            if norm(ev) < 1e-9
                ev = cross(k, eh);
            end
            ev = ev / norm(ev);

            if strcmpi(obj.basis, 'linear')
                PB = complex([eh, ev]);  % 3x2 complex
                return;
            end

            % Circular: IEEE-RHCP convention
            r = (eh - 1i * ev) / sqrt(2);
            l = (eh + 1i * ev) / sqrt(2);
            PB = [r, l];
        end

        function n = get.port_count(obj)
            if ~isempty(obj.patternData) && isstruct(obj.patternData) && isfield(obj.patternData, 'port_count')
                n = obj.patternData.port_count;
                return;
            end
            n = 2;
        end

        function G = txEmitMatrix(obj, direction)
            wb = obj.waveBasis(direction);
            pb = obj.portBasisVectors(direction);
            G = wb' * pb;  % 2 x port_count
        end

        function G = rxReceiveMatrix(obj, direction)
            wb = obj.waveBasis(direction);
            pb = obj.portBasisVectors(-direction(:));
            G = pb' * wb;  % port_count x 2
        end

        function C = couplingMatrix(obj, freqs_hz)
            C = obj.couplingMatrixAtDirection(freqs_hz, obj.boresight);
        end

        function C = couplingMatrixAtDirection(obj, freqs_hz, direction)
            freqs = freqs_hz(:);
            Nf = numel(freqs);
            if obj.port_count ~= 2
                error('core:Antenna:PortCount', 'couplingMatrix currently supports exactly 2 ports');
            end
            if ~obj.enable_coupling
                C = repmat(eye(2), 1, 1, Nf);
                return;
            end
            if nargin < 3 || isempty(direction)
                direction = obj.boresight;
            end
            [ar_db, xpd_db] = antennas.couplingFromAngle(obj, direction);
            C = obj.couplingMatrixCore(freqs, ar_db, xpd_db);
        end

        function C = couplingMatrixCore(obj, freqs_hz, ar_db, xpd_db)
            freqs = freqs_hz(:);
            Nf = numel(freqs);
            ar_db = expandParam(ar_db, Nf);
            xpd_db = expandParam(xpd_db, Nf);

            leak = 10 .^ (-xpd_db / 20.0);
            ar = 10 .^ (ar_db / 20.0);
            ar_leak = abs((ar - 1.0) ./ max(ar + 1.0, 1e-12));
            eps_mag = min(leak + ar_leak, 0.49);
            off = complex(eps_mag);
            den = sqrt(max(1.0 + abs(off).^2, 1e-18));

            C = complex(zeros(2, 2, Nf));
            C(1, 1, :) = reshape(1.0 ./ den, 1, 1, []);
            C(2, 2, :) = reshape(1.0 ./ den, 1, 1, []);
            C(1, 2, :) = reshape(off ./ den, 1, 1, []);
            C(2, 1, :) = reshape(-conj(off) ./ den, 1, 1, []);
        end

        function field_world = portVectorWorld(obj, port_id, freqs_hz, direction_world)
            port_id = round(double(port_id));
            freqs = freqs_hz(:);
            direction_world = normalizeVec(direction_world);
            if obj.use_ffd
                field_world = ffdPortVectorWorld(obj, port_id, freqs, direction_world);
                return;
            end
            wb = obj.waveBasis(direction_world);
            G = obj.txPortToWave(direction_world, freqs);
            field_world = complex(zeros(3, numel(freqs)));
            for k = 1:numel(freqs)
                field_world(:, k) = wb * G(:, port_id, k);
            end
        end

        function field_world = port_vector_world(obj, port_id, freqs_hz, direction_world)
            field_world = obj.portVectorWorld(port_id, freqs_hz, direction_world);
        end

        function G = txPortToWave(obj, direction, freqs_hz)
            if obj.use_ffd
                wb = obj.waveBasis(direction);
                freqs = freqs_hz(:);
                Nf = numel(freqs);
                G = complex(zeros(2, obj.port_count, Nf));
                for port_id = 1:obj.port_count
                    field_world = obj.portVectorWorld(port_id, freqs, direction);
                    for k = 1:Nf
                        G(:, port_id, k) = wb' * field_world(:, k);
                    end
                end
                return;
            end
            proj = obj.txEmitMatrix(direction);  % 2 x Nt
            C = obj.couplingMatrixAtDirection(freqs_hz, direction);
            Nf = numel(freqs_hz);
            G = complex(zeros(2, obj.port_count, Nf));
            for k = 1:Nf
                G(:, :, k) = proj * C(:, :, k);
            end
        end

        function G = rxWaveToPort(obj, direction, freqs_hz)
            if obj.use_ffd
                freqs = freqs_hz(:);
                look_dir = -direction(:);
                wb = obj.waveBasis(direction);
                Nf = numel(freqs);
                G = complex(zeros(obj.port_count, 2, Nf));
                for port_id = 1:obj.port_count
                    field_world = obj.portVectorWorld(port_id, freqs, look_dir);
                    for k = 1:Nf
                        G(port_id, :, k) = conj(field_world(:, k)).' * wb;
                    end
                end
                return;
            end
            proj = obj.rxReceiveMatrix(direction);  % Nr x 2
            C = obj.couplingMatrixAtDirection(freqs_hz, -direction(:));
            Nf = numel(freqs_hz);
            G = complex(zeros(obj.port_count, 2, Nf));
            for k = 1:Nf
                G(:, :, k) = C(:, :, k)' * proj;
            end
        end

        function gain = txDirectionalGainLinearF(obj, direction, freqs_hz)
            freqs = freqs_hz(:);
            if obj.use_ffd || hasDirectPattern(obj)
                gain = ones(size(freqs));
                return;
            end
            gain = zeros(size(freqs));
            for k = 1:numel(freqs)
                gain(k) = dirGainLin(direction, obj.boresight, obj.tx_peak_gain_dbi, obj.tx_pattern_cos_exp);
            end
        end

        function gain = rxDirectionalGainLinearF(obj, direction, freqs_hz)
            freqs = freqs_hz(:);
            look_dir = -direction(:);
            if obj.use_ffd || hasDirectPattern(obj)
                gain = ones(size(freqs));
                return;
            end
            gain = zeros(size(freqs));
            for k = 1:numel(freqs)
                gain(k) = dirGainLin(look_dir, obj.boresight, obj.rx_peak_gain_dbi, obj.rx_pattern_cos_exp);
            end
        end
    end
end

function args = parseArgs(varargin)
    defaults = struct( ...
        'position', [0; 0; 0], ...
        'boresight', [1; 0; 0], ...
        'h_axis', [0; 1; 0], ...
        'v_axis', [0; 0; 1], ...
        'basis', 'linear', ...
        'convention', 'IEEE-RHCP', ...
        'cross_pol_leakage_db', 35.0, ...
        'axial_ratio_db', 0.0, ...
        'ar_edge_db', 10.0, ...
        'xpd_edge_db', 8.0, ...
        'enable_coupling', true, ...
        'tx_peak_gain_dbi', 0.0, ...
        'rx_peak_gain_dbi', 0.0, ...
        'tx_pattern_cos_exp', 0.0, ...
        'rx_pattern_cos_exp', 0.0, ...
        'coupling_ref_freq_hz', 8.0e9, ...
        'global_up', [0; 0; 1], ...
        'patternData', [], ...
        'use_ffd', false, ...
        'ffd_port_r', [], ...
        'ffd_port_l', [], ...
        'ffd_local_to_world', []);

    if nargin == 1 && isstruct(varargin{1})
        in = varargin{1};
        args = defaults;
        names = fieldnames(in);
        for idx = 1:numel(names)
            args.(names{idx}) = in.(names{idx});
        end
    else
        p = inputParser;
        p.addParameter('position', defaults.position, @isnumeric);
        p.addParameter('boresight', defaults.boresight, @isnumeric);
        p.addParameter('h_axis', defaults.h_axis, @isnumeric);
        p.addParameter('v_axis', defaults.v_axis, @isnumeric);
        p.addParameter('basis', defaults.basis, @(x) ischar(x) || isstring(x));
        p.addParameter('convention', defaults.convention, @(x) ischar(x) || isstring(x));
        p.addParameter('cross_pol_leakage_db', defaults.cross_pol_leakage_db, @isnumeric);
        p.addParameter('axial_ratio_db', defaults.axial_ratio_db, @isnumeric);
        p.addParameter('ar_edge_db', defaults.ar_edge_db, @isnumeric);
        p.addParameter('xpd_edge_db', defaults.xpd_edge_db, @isnumeric);
        p.addParameter('enable_coupling', defaults.enable_coupling, @(x) islogical(x) || isnumeric(x));
        p.addParameter('tx_peak_gain_dbi', defaults.tx_peak_gain_dbi, @isnumeric);
        p.addParameter('rx_peak_gain_dbi', defaults.rx_peak_gain_dbi, @isnumeric);
        p.addParameter('tx_pattern_cos_exp', defaults.tx_pattern_cos_exp, @isnumeric);
        p.addParameter('rx_pattern_cos_exp', defaults.rx_pattern_cos_exp, @isnumeric);
        p.addParameter('coupling_ref_freq_hz', defaults.coupling_ref_freq_hz, @isnumeric);
        p.addParameter('global_up', defaults.global_up, @isnumeric);
        p.addParameter('patternData', defaults.patternData);
        p.addParameter('use_ffd', defaults.use_ffd, @(x) islogical(x) || isnumeric(x));
        p.addParameter('ffd_port_r', defaults.ffd_port_r);
        p.addParameter('ffd_port_l', defaults.ffd_port_l);
        p.addParameter('ffd_local_to_world', defaults.ffd_local_to_world, @(x) isempty(x) || isnumeric(x));
        p.parse(varargin{:});
        args = p.Results;
    end

    args.position = reshape(double(args.position), [3, 1]);
    args.boresight = reshape(double(args.boresight), [3, 1]);
    args.h_axis = reshape(double(args.h_axis), [3, 1]);
    args.v_axis = reshape(double(args.v_axis), [3, 1]);
    args.global_up = reshape(double(args.global_up), [3, 1]);
    args.basis = char(string(args.basis));
    args.convention = char(string(args.convention));
    args.enable_coupling = logical(args.enable_coupling);
    args.use_ffd = logical(args.use_ffd);
end

function v = normalizeVec(x)
    v = x(:);
    n = norm(v);
    if n == 0.0
        error('core:Antenna:ZeroVector', 'zero-length vector');
    end
    v = v / n;
end

function gain = dirGainLin(direction, boresight, peak_gain_dbi, cos_exp)
    d = normalizeVec(direction);
    b = normalizeVec(boresight);
    cos_psi = max(dot(d, b), 0.0);
    expo = max(cos_exp, 0.0);
    if expo > 0.0
        shape = cos_psi ^ expo;
    else
        shape = 1.0;
    end
    peak = 10 ^ (peak_gain_dbi / 10.0);
    gain = max(peak * shape, 0.0);
end

function values = expandParam(value, N)
    value = double(value);
    if isscalar(value)
        values = repmat(value, N, 1);
    else
        values = value(:);
        assert(numel(values) == N, 'parameter length must match frequency vector length');
    end
end

function tf = hasDirectPattern(obj)
    tf = isstruct(obj.patternData) && isfield(obj.patternData, 'port_patterns') && ~isempty(obj.patternData.port_patterns);
end

function field_world = ffdPortVectorWorld(obj, port_id, freqs_hz, direction_world)
    freqs = freqs_hz(:);
    direction_world = normalizeVec(direction_world);
    R = obj.ffd_local_to_world;
    direction_local = R' * direction_world;
    theta_rad = acos(clampScalar(direction_local(3), -1.0, 1.0));
    phi_rad = mod(atan2(direction_local(2), direction_local(1)), 2.0 * pi);
    [theta_hat_local, phi_hat_local] = localSphericalBasisRad(theta_rad, phi_rad);

    ffd = selectFfdPort(obj, port_id);
    field_world = complex(zeros(3, numel(freqs)));
    for k = 1:numel(freqs)
        [E_theta, E_phi] = antennas.interpolatePattern(ffd, theta_rad, phi_rad, freqs(k));
        field_local = theta_hat_local * E_theta + phi_hat_local * E_phi;
        field_world(:, k) = R * field_local;
    end
end

function ffd = selectFfdPort(obj, port_id)
    switch port_id
        case 1
            ffd = obj.ffd_port_r;
        case 2
            ffd = obj.ffd_port_l;
        otherwise
            error('core:Antenna:PortId', 'Unsupported port_id %d', port_id);
    end
    if isempty(ffd)
        error('core:Antenna:FfdPort', 'FFD port %d is not configured', port_id);
    end
end

function G = patternTxPortToWave(obj, direction, freqs_hz)
    G = [];
    F_world = patternPortWorldField(obj, direction, freqs_hz);
    if isempty(F_world)
        return;
    end
    wb = obj.waveBasis(direction);
    Nf = numel(freqs_hz);
    G = complex(zeros(2, obj.port_count, Nf));
    for k = 1:Nf
        G(:, :, k) = wb' * F_world(:, :, k);
    end
end

function G = patternRxWaveToPort(obj, direction, freqs_hz)
    G = [];
    look_dir = -direction(:);
    F_world = patternPortWorldField(obj, look_dir, freqs_hz);
    if isempty(F_world)
        return;
    end
    wb = obj.waveBasis(direction);
    Nf = numel(freqs_hz);
    G = complex(zeros(obj.port_count, 2, Nf));
    for k = 1:Nf
        G(:, :, k) = conj(F_world(:, :, k)).' * wb;
    end
end

function F_world = patternPortWorldField(obj, look_direction, freqs_hz)
    F_world = [];
    if ~hasDirectPattern(obj)
        return;
    end
    d_world = normalizeVec(look_direction);
    R = [obj.h_axis, obj.v_axis, obj.boresight];
    d_local = R' * d_world;
    theta_deg = acosd(clampScalar(d_local(3), -1.0, 1.0));
    phi_deg = atan2d(d_local(2), d_local(1));
    [theta_hat_local, phi_hat_local] = localSphericalBasis(theta_deg, phi_deg);
    theta_hat_world = R * theta_hat_local;
    phi_hat_world = R * phi_hat_local;

    freqs = freqs_hz(:);
    P = obj.port_count;
    Nf = numel(freqs);
    F_world = complex(zeros(3, P, Nf));
    for p = 1:P
        port_pattern = obj.patternData.port_patterns{p};
        [Etheta, Ephi] = interpolatePatternField(port_pattern, theta_deg, phi_deg, freqs);
        for k = 1:Nf
            F_world(:, p, k) = theta_hat_world * Etheta(k) + phi_hat_world * Ephi(k);
        end
    end
end

function [theta_hat_local, phi_hat_local] = localSphericalBasis(theta_deg, phi_deg)
    theta = deg2rad(theta_deg);
    phi = deg2rad(phi_deg);
    theta_hat_local = [cos(theta) * cos(phi); cos(theta) * sin(phi); -sin(theta)];
    phi_hat_local = [-sin(phi); cos(phi); 0.0];
end

function [theta_hat_local, phi_hat_local] = localSphericalBasisRad(theta_rad, phi_rad)
    theta_hat_local = [cos(theta_rad) * cos(phi_rad); cos(theta_rad) * sin(phi_rad); -sin(theta_rad)];
    phi_hat_local = [-sin(phi_rad); cos(phi_rad); 0.0];
end

function [Etheta, Ephi] = interpolatePatternField(port_pattern, theta_deg, phi_deg, freqs_hz)
    theta_grid = double(port_pattern.theta_deg(:));
    phi_grid = double(port_pattern.phi_deg(:));
    freq_grid = double(port_pattern.frequencies_hz(:));
    if isempty(freq_grid) || all(freq_grid == 0)
        freq_grid = 0.0;
    end

    theta_q = clampScalar(theta_deg, theta_grid(1), theta_grid(end));
    phi_q = wrapAngleDeg(phi_deg, phi_grid(1), phi_grid(end));
    freq_q = freqs_hz(:);
    if numel(freq_grid) == 1
        freq_q(:) = freq_grid(1);
    else
        freq_q = min(max(freq_q, freq_grid(1)), freq_grid(end));
    end

    theta_query = repmat(theta_q, size(freq_q));
    phi_query = repmat(phi_q, size(freq_q));
    Etheta = interpolateComplexGrid(theta_grid, phi_grid, freq_grid, port_pattern.Etheta, theta_query, phi_query, freq_q);
    Ephi = interpolateComplexGrid(theta_grid, phi_grid, freq_grid, port_pattern.Ephi, theta_query, phi_query, freq_q);
end

function values = interpolateComplexGrid(theta_grid, phi_grid, freq_grid, data, theta_q, phi_q, freq_q)
    real_part = interpn(theta_grid, phi_grid, freq_grid, real(data), theta_q, phi_q, freq_q, 'linear');
    imag_part = interpn(theta_grid, phi_grid, freq_grid, imag(data), theta_q, phi_q, freq_q, 'linear');
    values = complex(real_part, imag_part);
end

function value = wrapAngleDeg(angle_deg, phi_min, phi_max)
    span = phi_max - phi_min;
    if span <= 0
        value = angle_deg;
        return;
    end
    value = mod(angle_deg - phi_min, span) + phi_min;
    if abs(value - phi_min) < 1e-12 && angle_deg > phi_max
        value = phi_max;
    end
end

function value = clampScalar(x, lo, hi)
    value = min(max(x, lo), hi);
end

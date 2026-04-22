function ffd = loadFfdPattern(filepath, varargin)
% loadFfdPattern - Parse an HFSS FFD file into the canonical MATLAB struct.
%
% Returns fields:
%   theta_rad   : [n_theta x 1] polar angle, canonical range [0, pi]
%   phi_rad     : [n_phi x 1] azimuth, canonical range [0, 2*pi)
%   freqs_hz    : [n_freq x 1]
%   E_theta     : [n_theta x n_phi x n_freq] complex
%   E_phi       : [n_theta x n_phi x n_freq] complex
%   port_id     : inferred integer port id if known, otherwise -1
%   port_label  : inferred label such as RHCP/LHCP/H/V/UNKNOWN
%   metadata    : source and parser metadata

    p = inputParser;
    p.addParameter('notes', '', @(x) ischar(x) || isstring(x));
    p.addParameter('phi_convention', 'hfss', @(x) ischar(x) || isstring(x));
    p.addParameter('flip_ephi', [], @(x) isempty(x) || islogical(x) || isnumeric(x));
    p.parse(varargin{:});
    notes = char(string(p.Results.notes));
    phi_convention = lower(char(string(p.Results.phi_convention)));
    flip_ephi = resolveFlipEphi(phi_convention, p.Results.flip_ephi);

    filepath = char(string(filepath));
    assert(exist(filepath, 'file') == 2, 'antennas:loadFfdPattern:Missing', 'FFD file not found: %s', filepath);

    raw_text = fileread(filepath);
    raw_lines = splitlines(string(raw_text));
    lines = cleanLines(raw_lines);
    if numel(lines) < 4
        error('antennas:loadFfdPattern:ShortFile', 'FFD file is too short: %s', filepath);
    end

    theta_meta = sscanf(lines{1}, '%f');
    phi_meta = sscanf(lines{2}, '%f');
    assert(numel(theta_meta) >= 3, 'antennas:loadFfdPattern:ThetaHeader', 'Malformed theta header in %s', filepath);
    assert(numel(phi_meta) >= 3, 'antennas:loadFfdPattern:PhiHeader', 'Malformed phi header in %s', filepath);

    theta_deg_raw = linspace(theta_meta(1), theta_meta(2), round(theta_meta(3)));
    phi_deg_raw = linspace(phi_meta(1), phi_meta(2), round(phi_meta(3)));
    n_theta = numel(theta_deg_raw);
    n_phi = numel(phi_deg_raw);
    n_samples_per_freq = n_theta * n_phi;

    cursor = 3;
    freq_list = [];
    if startsWith(lower(strtrim(lines{cursor})), 'frequencies')
        freq_tokens = sscanf(lines{cursor}, '%*s %f');
        if isempty(freq_tokens)
            error('antennas:loadFfdPattern:FreqHeader', 'Malformed frequency header in %s', filepath);
        end
        n_freq = round(freq_tokens(end));
        cursor = cursor + 1;
    else
        freq_list = sscanf(lines{cursor}, '%f').';
        n_freq = numel(freq_list);
        cursor = cursor + 1;
    end

    freqs_hz = zeros(n_freq, 1);
    E_theta_raw = complex(zeros(n_theta, n_phi, n_freq));
    E_phi_raw = complex(zeros(n_theta, n_phi, n_freq));

    for fidx = 1:n_freq
        if cursor <= numel(lines) && startsWith(lower(strtrim(lines{cursor})), 'frequency')
            freq_tokens = sscanf(lines{cursor}, '%*s %f');
            assert(~isempty(freq_tokens), 'antennas:loadFfdPattern:FreqLine', 'Malformed frequency line in %s', filepath);
            freqs_hz(fidx) = freq_tokens(end);
            cursor = cursor + 1;
        elseif ~isempty(freq_list)
            freqs_hz(fidx) = freq_list(fidx);
        else
            error('antennas:loadFfdPattern:FreqBlock', 'Could not determine frequency %d in %s', fidx, filepath);
        end

        if cursor + n_samples_per_freq - 1 > numel(lines)
            error('antennas:loadFfdPattern:DataLength', 'FFD data block truncated in %s', filepath);
        end

        block = lines(cursor:(cursor + n_samples_per_freq - 1));
        cursor = cursor + n_samples_per_freq;
        for flat_idx = 1:n_samples_per_freq
            values = sscanf(block{flat_idx}, '%f');
            if numel(values) < 4
                error('antennas:loadFfdPattern:Row', 'Malformed FFD sample row in %s', filepath);
            end
            theta_idx = mod(flat_idx - 1, n_theta) + 1;
            phi_idx = floor((flat_idx - 1) / n_theta) + 1;
            E_theta_raw(theta_idx, phi_idx, fidx) = complex(values(end - 3), values(end - 2));
            E_phi_raw(theta_idx, phi_idx, fidx) = complex(values(end - 1), values(end));
        end
    end

    [phi_deg, phi_keep_idx, phi_sort_idx] = canonicalizePhiGrid(phi_deg_raw);
    E_theta = E_theta_raw(:, phi_sort_idx, :);
    E_phi = E_phi_raw(:, phi_sort_idx, :);
    E_theta = E_theta(:, phi_keep_idx, :);
    E_phi = E_phi(:, phi_keep_idx, :);
    if flip_ephi
        E_phi = -E_phi;
    end

    theta_rad = deg2rad(theta_deg_raw(:));
    phi_rad = deg2rad(phi_deg(:));
    total_power_numeric = integrateTotalPower(theta_rad, phi_rad, E_theta, E_phi);

    assert(all(diff(theta_rad) > 0), 'antennas:loadFfdPattern:ThetaGrid', 'theta grid is not strictly monotonic');
    assert(all(theta_rad >= -1e-12 & theta_rad <= pi + 1e-12), 'antennas:loadFfdPattern:ThetaRange', 'theta grid is out of [0, pi]');
    assert(all(diff(phi_rad) > 0), 'antennas:loadFfdPattern:PhiGrid', 'phi grid is not strictly monotonic');
    assert(all(phi_rad >= -1e-12 & phi_rad < 2 * pi + 1e-12), 'antennas:loadFfdPattern:PhiRange', 'phi grid is out of [0, 2*pi)');
    assert(~any(isnan(E_theta(:))) && ~any(isnan(E_phi(:))), 'antennas:loadFfdPattern:NaNField', 'NaN in E field arrays');
    assert(all(isfinite(total_power_numeric)) && all(total_power_numeric > 0), 'antennas:loadFfdPattern:Power', 'Unreasonable total power in %s', filepath);

    [port_id, port_label] = inferPortIdentity(filepath);
    file_info = dir(filepath);

    metadata = struct();
    metadata.source_filepath = filepath;
    metadata.source_filename = file_info.name;
    metadata.file_bytes = file_info.bytes;
    metadata.file_datenum = file_info.datenum;
    metadata.load_timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    metadata.notes = notes;
    metadata.format = 'HFSS_FFD';
    metadata.phi_convention = phi_convention;
    metadata.ephi_sign_flipped = logical(flip_ephi);
    metadata.header_preview = string(raw_lines(1:min(numel(raw_lines), 10)));
    metadata.n_theta_raw = n_theta;
    metadata.n_phi_raw = n_phi;
    metadata.n_phi_canonical = numel(phi_rad);
    metadata.n_freq = n_freq;
    metadata.theta_header_deg = theta_meta(:).';
    metadata.phi_header_deg = phi_meta(:).';
    metadata.theta_step_deg = median(diff(theta_deg_raw));
    metadata.phi_step_deg = median(diff(phi_deg));
    metadata.total_radiated_power_numeric = total_power_numeric(:);

    ffd = struct();
    ffd.theta_rad = theta_rad(:);
    ffd.phi_rad = phi_rad(:);
    ffd.freqs_hz = freqs_hz(:);
    ffd.E_theta = E_theta;
    ffd.E_phi = E_phi;
    ffd.port_id = port_id;
    ffd.port_label = port_label;
    ffd.metadata = metadata;
end

function flip_ephi = resolveFlipEphi(phi_convention, explicit_value)
    if ~isempty(explicit_value)
        flip_ephi = logical(explicit_value);
        return;
    end

    switch phi_convention
        case {'hfss', 'cst', 'simulator_negative_phi', 'negative_phi'}
            flip_ephi = true;
        case {'ieee', 'mathematical', 'positive_phi'}
            flip_ephi = false;
        otherwise
            error('antennas:loadFfdPattern:PhiConvention', ...
                'Unsupported phi_convention: %s', phi_convention);
    end
end

function lines = cleanLines(raw_lines)
    lines = {};
    for idx = 1:numel(raw_lines)
        line = strtrim(raw_lines(idx));
        if line == "" || startsWith(line, ["#", "!", "//"])
            continue;
        end
        lines{end + 1} = char(line); %#ok<AGROW>
    end
end

function [phi_deg, keep_idx, sort_idx] = canonicalizePhiGrid(phi_deg_raw)
    phi_wrapped = mod(phi_deg_raw(:), 360.0);
    phi_wrapped(abs(phi_wrapped - 360.0) < 1e-9) = 0.0;
    [phi_sorted, sort_idx] = sort(phi_wrapped, 'ascend');
    keep_mask = true(size(phi_sorted));
    for idx = 2:numel(phi_sorted)
        if abs(phi_sorted(idx) - phi_sorted(idx - 1)) < 1e-9
            keep_mask(idx) = false;
        end
    end
    keep_idx = find(keep_mask);
    phi_deg = phi_sorted(keep_mask);
end

function total_power = integrateTotalPower(theta_rad, phi_rad, E_theta, E_phi)
    n_freq = size(E_theta, 3);
    total_power = zeros(n_freq, 1);
    sin_theta = sin(theta_rad(:));
    weight = sin_theta * ones(1, numel(phi_rad));
    for fidx = 1:n_freq
        power_density = abs(E_theta(:, :, fidx)).^2 + abs(E_phi(:, :, fidx)).^2;
        integrand = power_density .* weight;
        theta_int = trapz(theta_rad, integrand, 1);
        total_power(fidx) = trapz(phi_rad, theta_int, 2);
    end
end

function [port_id, port_label] = inferPortIdentity(filepath)
    name = upper(string(filepath));
    if contains(name, "RHCP") || contains(name, "RH")
        port_id = 1;
        port_label = 'RHCP';
    elseif contains(name, "LHCP") || contains(name, "LH")
        port_id = 2;
        port_label = 'LHCP';
    elseif contains(name, "VERT") || contains(name, "_V") || endsWith(name, "V")
        port_id = 1;
        port_label = 'V';
    elseif contains(name, "HOR") || contains(name, "_H") || endsWith(name, "H")
        port_id = 2;
        port_label = 'H';
    else
        port_id = -1;
        port_label = 'UNKNOWN';
    end
end

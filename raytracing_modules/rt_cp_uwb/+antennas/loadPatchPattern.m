function pattern = loadPatchPattern(patternFile)
% loadPatchPattern - load representative CP patch metrics from .mat or HFSS FFD data.

    sources = resolveSources(patternFile);
    if isempty(sources)
        error('antennas:loadPatchPattern:Source', 'no usable pattern source found');
    end

    if numel(sources) == 1 && strcmpi(sources{1}.kind, 'mat')
        pattern = normalizeLoadedPattern(load(sources{1}.path), sources{1}.path);
        return;
    end

    port_patterns = cell(1, numel(sources));
    for idx = 1:numel(sources)
        if strcmpi(sources{idx}.kind, 'mat')
            port_patterns{idx} = normalizeLoadedPattern(load(sources{idx}.path), sources{idx}.path);
        else
            port_patterns{idx} = parseFfdFile(sources{idx}.path);
        end
    end
    pattern = summarizePortPatterns(port_patterns, sources);
end

function sources = resolveSources(patternFile)
    sources = {};
    if iscell(patternFile)
        for idx = 1:numel(patternFile)
            sources{end + 1} = classifySource(patternFile{idx}); %#ok<AGROW>
        end
        return;
    end
    if isstring(patternFile) && numel(patternFile) > 1
        for idx = 1:numel(patternFile)
            sources{end + 1} = classifySource(patternFile(idx)); %#ok<AGROW>
        end
        return;
    end

    pattern_path = char(string(patternFile));
    if isfolder(pattern_path)
        listing = dir(fullfile(pattern_path, '*.ffd'));
        names = {listing.name};
        cp_mask = contains(lower(names), 'rhcp') | contains(lower(names), 'lhcp');
        if any(cp_mask)
            listing = listing(cp_mask);
        end
        listing = sortByName(listing);
        for idx = 1:min(numel(listing), 2)
            sources{end + 1} = classifySource(fullfile(listing(idx).folder, listing(idx).name)); %#ok<AGROW>
        end
        return;
    end

    [~, ~, ext] = fileparts(pattern_path);
    if ismember(lower(ext), {'.txt', '.lst', '.map'})
        entries = parseArrayMap(pattern_path);
        for idx = 1:min(numel(entries), 2)
            sources{end + 1} = classifySource(entries{idx}); %#ok<AGROW>
        end
        return;
    end

    sources = {classifySource(pattern_path)};
end

function listing = sortByName(listing)
    if isempty(listing)
        return;
    end
    [~, order] = sort(lower({listing.name}));
    listing = listing(order);
end

function paths = parseArrayMap(mapFile)
    lines = splitlines(string(fileread(mapFile)));
    parent = fileparts(mapFile);
    paths = {};
    for idx = 1:numel(lines)
        line = strtrim(lines(idx));
        if line == "" || startsWith(line, ["#", "!", "//"])
            continue;
        end
        tokens = split(line);
        candidate = fullfile(parent, char(tokens(end)));
        if exist(candidate, 'file')
            paths{end + 1} = candidate; %#ok<AGROW>
        end
    end
end

function source = classifySource(pathValue)
    source = struct();
    source.path = char(string(pathValue));
    [~, ~, ext] = fileparts(source.path);
    if strcmpi(ext, '.mat')
        source.kind = 'mat';
    else
        source.kind = 'ffd';
    end
end

function pattern = normalizeLoadedPattern(loaded, source_path)
    vars = fieldnames(loaded);
    if numel(vars) == 1 && isstruct(loaded.(vars{1}))
        pattern = loaded.(vars{1});
    else
        pattern = loaded;
    end

    if isfield(pattern, 'ar_db_boresight') && isfield(pattern, 'xpd_db_boresight')
        pattern = finalizePatternStruct(pattern, source_path);
        return;
    end

    if isfield(pattern, 'theta_rad') && isfield(pattern, 'phi_rad') && isfield(pattern, 'E_theta') && isfield(pattern, 'E_phi')
        parsed = canonicalToLegacyPattern(pattern, source_path);
        pattern = summarizePortPatterns({parsed}, {struct('path', source_path, 'kind', 'mat')});
        return;
    end

    if isfield(pattern, 'Etheta') && isfield(pattern, 'Ephi')
        parsed = struct();
        parsed.source_file = source_path;
        parsed.theta_deg = fieldOrDefault(pattern, 'theta_deg', fieldOrDefault(pattern, 'thetaGridDeg', 0));
        parsed.phi_deg = fieldOrDefault(pattern, 'phi_deg', fieldOrDefault(pattern, 'phiGridDeg', 0));
        parsed.frequencies_hz = fieldOrDefault(pattern, 'frequencies_hz', fieldOrDefault(pattern, 'freqGridHz', 0));
        parsed.Etheta = pattern.Etheta;
        parsed.Ephi = pattern.Ephi;
        pattern = summarizePortPatterns({parsed}, {struct('path', source_path, 'kind', 'mat')});
        return;
    end

    error('antennas:loadPatchPattern:Mat', 'MAT file %s does not expose recognizable pattern fields', source_path);
end

function value = fieldOrDefault(s, field_name, default_value)
    if isfield(s, field_name)
        value = s.(field_name);
    else
        value = default_value;
    end
end

function pattern = parseFfdFile(pathValue)
    pattern = canonicalToLegacyPattern(antennas.loadFfdPattern(pathValue), pathValue);
end

function pattern = summarizePortPatterns(port_patterns, sources)
    n_ports = numel(port_patterns);
    [port_patterns, sources, port_handedness, file_label_handedness] = reorderPortsByHandedness(port_patterns, sources);
    ref = port_patterns{1};
    theta_deg = ref.theta_deg(:);
    phi_deg = ref.phi_deg(:);
    freqs = ref.frequencies_hz(:);

    theta_idx0 = nearestIndex(theta_deg, 0.0);
    phi_idx0 = nearestIndex(phi_deg, 0.0);
    freq_idx0 = nearestIndex(freqs, representativeFrequency(freqs));

    boresight_power = zeros(n_ports, 1);
    boresight_field = complex(zeros(n_ports, 2));
    peak_power = zeros(n_ports, 1);
    fitted_cos = zeros(n_ports, 1);

    for idx = 1:n_ports
        current = port_patterns{idx};
        sample_theta = current.Etheta(theta_idx0, phi_idx0, freq_idx0);
        sample_phi = current.Ephi(theta_idx0, phi_idx0, freq_idx0);
        boresight_field(idx, :) = [sample_theta, sample_phi];
        boresight_power(idx) = abs(sample_theta).^2 + abs(sample_phi).^2;
        power_cube = abs(current.Etheta).^2 + abs(current.Ephi).^2;
        peak_power(idx) = max(power_cube(:));
        fitted_cos(idx) = fitCosExponent(power_cube(:, :, freq_idx0), theta_deg);
    end

    primary_idx = choosePrimaryPort(port_handedness, boresight_power);
    primary_field = boresight_field(primary_idx, :);
    ar_db = polarizationAxialRatioDb(primary_field(1), primary_field(2));

    if n_ports >= 2
        sorted_power = sort(boresight_power, 'descend');
        xpd_db = 20 * log10(sqrt(sorted_power(1)) / max(sqrt(sorted_power(2)), 1e-12));
    else
        [co_mag, cross_mag] = circularComponents(primary_field(1), primary_field(2));
        xpd_db = 20 * log10(co_mag / max(cross_mag, 1e-12));
    end

    primary_handedness = port_handedness{primary_idx};
    if isempty(primary_handedness)
        primary_name = upper(string(sources{primary_idx}.path));
        if contains(primary_name, "LH")
            primary_handedness = 'L';
        else
            primary_handedness = 'R';
        end
    end

    pattern = struct();
    pattern.source_files = string(cellfun(@(s) s.path, sources, 'UniformOutput', false));
    pattern.port_count = n_ports;
    pattern.primary_port = primary_idx;
    pattern.primary_handedness = primary_handedness;
    pattern.port_handedness = string(port_handedness);
    pattern.file_label_handedness = string(file_label_handedness);
    pattern.theta_deg = theta_deg;
    pattern.phi_deg = phi_deg;
    pattern.frequencies_hz = freqs;
    pattern.boresight_field = boresight_field;
    pattern.boresight_power = boresight_power;
    pattern.ar_db_boresight = ar_db;
    pattern.xpd_db_boresight = xpd_db;
    pattern.peak_gain_dbi = 10 * log10(max(max(peak_power), 1e-12));
    pattern.fitted_cos_exp = mean(fitted_cos(isfinite(fitted_cos)));
    if isnan(pattern.fitted_cos_exp)
        pattern.fitted_cos_exp = 0.0;
    end
    pattern.port_patterns = port_patterns;
    pattern.factory = 'realistic_patch';
end

function [port_patterns, sources, port_handedness, file_label_handedness] = reorderPortsByHandedness(port_patterns, sources)
    n = numel(sources);
    port_handedness = cell(1, n);
    file_label_handedness = cell(1, n);
    order_score = zeros(1, n);
    for idx = 1:n
        file_hand = sourceHandedness(sources{idx}.path);
        actual_hand = dominantHandFromPattern(port_patterns{idx});
        if isempty(actual_hand)
            hand = file_hand;
        else
            hand = actual_hand;
        end
        port_handedness{idx} = hand;
        file_label_handedness{idx} = file_hand;
        if strcmp(hand, 'R')
            order_score(idx) = 0;
        elseif strcmp(hand, 'L')
            order_score(idx) = 1;
        else
            order_score(idx) = 2 + idx / 1000;
        end
    end
    [~, order] = sort(order_score);
    port_patterns = port_patterns(order);
    sources = sources(order);
    port_handedness = port_handedness(order);
    file_label_handedness = file_label_handedness(order);
end

function idx = choosePrimaryPort(port_handedness, boresight_power)
    idx = find(strcmp(port_handedness, 'R'), 1, 'first');
    if isempty(idx)
        [~, idx] = max(boresight_power);
    end
end

function hand = sourceHandedness(pathValue)
    name = upper(string(pathValue));
    if contains(name, "RH")
        hand = 'R';
    elseif contains(name, "LH")
        hand = 'L';
    else
        hand = '';
    end
end

function hand = dominantHandFromPattern(port_pattern)
    hand = '';
    if ~isfield(port_pattern, 'theta_deg') || ~isfield(port_pattern, 'phi_deg') || ...
            ~isfield(port_pattern, 'frequencies_hz') || ~isfield(port_pattern, 'Etheta') || ~isfield(port_pattern, 'Ephi')
        return;
    end
    theta_idx0 = nearestIndex(port_pattern.theta_deg(:), 0.0);
    phi_idx0 = nearestIndex(port_pattern.phi_deg(:), 0.0);
    freq_idx0 = nearestIndex(port_pattern.frequencies_hz(:), representativeFrequency(port_pattern.frequencies_hz(:)));
    sample_theta = port_pattern.Etheta(theta_idx0, phi_idx0, freq_idx0);
    sample_phi = port_pattern.Ephi(theta_idx0, phi_idx0, freq_idx0);
    Er = abs((sample_theta - 1i * sample_phi) / sqrt(2.0));
    El = abs((sample_theta + 1i * sample_phi) / sqrt(2.0));
    if Er >= El
        hand = 'R';
    else
        hand = 'L';
    end
end

function freq0 = representativeFrequency(freqs)
    if isempty(freqs)
        freq0 = 0.0;
    elseif all(freqs == 0)
        freq0 = 0.0;
    else
        freq0 = mean(freqs);
    end
end

function idx = nearestIndex(values, target)
    [~, idx] = min(abs(values - target));
end

function n = fitCosExponent(power_grid, theta_deg)
    theta = deg2rad(theta_deg(:));
    p = mean(power_grid, 2);
    p = p(:);
    p = p / max(max(p), 1e-12);
    mask = theta > 0 & theta < deg2rad(75.0) & p > 0 & cos(theta) > 0;
    if nnz(mask) < 2
        n = 0.0;
        return;
    end
    x = log(cos(theta(mask)));
    y = log(p(mask));
    n = max(0.0, x \ y);
end

function ar_db = polarizationAxialRatioDb(Etheta, Ephi)
    Ex = Etheta;
    Ey = Ephi;
    s0 = abs(Ex)^2 + abs(Ey)^2;
    if s0 <= 0
        ar_db = NaN;
        return;
    end
    s3 = -2 * imag(Ex * conj(Ey));
    chi = 0.5 * asin(max(min(s3 / s0, 1.0), -1.0));
    tan_chi = abs(tan(chi));
    if tan_chi < 1e-12
        ar_db = 120.0;
    else
        ar = max(1.0 / tan_chi, 1.0);
        ar_db = 20 * log10(ar);
    end
end

function [co_mag, cross_mag] = circularComponents(Etheta, Ephi)
    Er = abs((Etheta - 1i * Ephi) / sqrt(2.0));
    El = abs((Etheta + 1i * Ephi) / sqrt(2.0));
    co_mag = max(Er, El);
    cross_mag = min(Er, El);
end

function pattern = finalizePatternStruct(pattern, source_path)
    if ~isfield(pattern, 'port_count')
        pattern.port_count = 2;
    end
    if ~isfield(pattern, 'primary_handedness')
        pattern.primary_handedness = 'R';
    end
    if ~isfield(pattern, 'peak_gain_dbi')
        pattern.peak_gain_dbi = 0.0;
    end
    if ~isfield(pattern, 'fitted_cos_exp')
        pattern.fitted_cos_exp = 0.0;
    end
    pattern.source_files = string(source_path);
end

function parsed = canonicalToLegacyPattern(ffd, source_path)
    parsed = struct();
    parsed.source_file = source_path;
    parsed.theta_deg = rad2deg(ffd.theta_rad(:));
    parsed.phi_deg = rad2deg(ffd.phi_rad(:));
    parsed.frequencies_hz = ffd.freqs_hz(:);
    parsed.Etheta = ffd.E_theta;
    parsed.Ephi = ffd.E_phi;
end

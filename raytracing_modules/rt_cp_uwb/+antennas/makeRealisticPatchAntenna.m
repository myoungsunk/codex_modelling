function ant = makeRealisticPatchAntenna(patternFile, position, boresight, h_axis, v_axis)
% makeRealisticPatchAntenna - CP antenna factory backed by measured/simulated patterns.

    if isFfdPatternSpec(patternFile)
        [rhcp_path, lhcp_path] = resolveFfdPair(patternFile);
        ant = antennas.makeRealisticPatchAntennaFFD(rhcp_path, lhcp_path, position, boresight, h_axis, v_axis);
        return;
    elseif isstruct(patternFile)
        pattern = patternFile;
    elseif (ischar(patternFile) || isstring(patternFile)) && strcmpi(string(patternFile), "synthetic")
        pattern = antennas.loadPatchPatternSynthetic();
    else
        pattern = antennas.loadPatchPattern(patternFile);
    end

    if isstruct(pattern) && isfield(pattern, 'port_patterns') && isfield(pattern, 'port_handedness')
        [ffd_r, ffd_l] = summaryPatternToFfdPorts(pattern);
        ant = core.Antenna( ...
            'position', position(:), ...
            'boresight', boresight(:), ...
            'h_axis', h_axis(:), ...
            'v_axis', v_axis(:), ...
            'basis', 'circular', ...
            'convention', 'IEEE-RHCP', ...
            'cross_pol_leakage_db', pattern.xpd_db_boresight, ...
            'axial_ratio_db', pattern.ar_db_boresight, ...
            'ar_edge_db', getFieldOrDefault(pattern, 'ar_edge_db', 10.0), ...
            'xpd_edge_db', getFieldOrDefault(pattern, 'xpd_edge_db', 8.0), ...
            'enable_coupling', false, ...
            'tx_peak_gain_dbi', pattern.peak_gain_dbi, ...
            'rx_peak_gain_dbi', pattern.peak_gain_dbi, ...
            'tx_pattern_cos_exp', 0.0, ...
            'rx_pattern_cos_exp', 0.0, ...
            'patternData', pattern, ...
            'use_ffd', true, ...
            'ffd_port_r', ffd_r, ...
            'ffd_port_l', ffd_l, ...
            'ffd_local_to_world', [h_axis(:), v_axis(:), boresight(:)]);
        return;
    end

    ant = core.Antenna( ...
        'position', position(:), ...
        'boresight', boresight(:), ...
        'h_axis', h_axis(:), ...
        'v_axis', v_axis(:), ...
        'basis', 'circular', ...
        'convention', 'IEEE-RHCP', ...
        'cross_pol_leakage_db', pattern.xpd_db_boresight, ...
        'axial_ratio_db', pattern.ar_db_boresight, ...
        'ar_edge_db', getFieldOrDefault(pattern, 'ar_edge_db', 10.0), ...
        'xpd_edge_db', getFieldOrDefault(pattern, 'xpd_edge_db', 8.0), ...
        'enable_coupling', true, ...
        'tx_peak_gain_dbi', pattern.peak_gain_dbi, ...
        'rx_peak_gain_dbi', pattern.peak_gain_dbi, ...
        'tx_pattern_cos_exp', pattern.fitted_cos_exp, ...
        'rx_pattern_cos_exp', pattern.fitted_cos_exp, ...
        'patternData', pattern);
end

function tf = isFfdPatternSpec(patternFile)
    tf = false;
    if iscell(patternFile)
        tf = numel(patternFile) >= 2 && all(cellfun(@isFfdPath, patternFile(1:2)));
        return;
    end
    if isstring(patternFile) && numel(patternFile) > 1
        tf = numel(patternFile) >= 2 && all(arrayfun(@isFfdPath, patternFile(1:2)));
        return;
    end
    if ischar(patternFile) || isstring(patternFile)
        pathValue = char(string(patternFile));
        tf = isfolder(pathValue) || isFfdPath(pathValue);
    end
end

function [rhcp_path, lhcp_path] = resolveFfdPair(patternFile)
    if iscell(patternFile)
        rhcp_path = char(string(patternFile{1}));
        lhcp_path = char(string(patternFile{2}));
        return;
    end
    if isstring(patternFile) && numel(patternFile) > 1
        rhcp_path = char(patternFile(1));
        lhcp_path = char(patternFile(2));
        return;
    end

    pathValue = char(string(patternFile));
    if isfolder(pathValue)
        listing = dir(fullfile(pathValue, '*.ffd'));
        names = {listing.name};
        rh_idx = find(contains(lower(names), 'rh'), 1, 'first');
        lh_idx = find(contains(lower(names), 'lh'), 1, 'first');
        assert(~isempty(rh_idx) && ~isempty(lh_idx), 'Could not infer RH/LH FFD pair from folder %s', pathValue);
        rhcp_path = fullfile(listing(rh_idx).folder, listing(rh_idx).name);
        lhcp_path = fullfile(listing(lh_idx).folder, listing(lh_idx).name);
        return;
    end
    error('antennas:makeRealisticPatchAntenna:ResolveFfdPair', 'Could not resolve FFD pair from input');
end

function tf = isFfdPath(pathValue)
    [~, ~, ext] = fileparts(char(string(pathValue)));
    tf = strcmpi(ext, '.ffd');
end

function [ffd_r, ffd_l] = summaryPatternToFfdPorts(pattern)
    handed = string(pattern.port_handedness);
    ffd_r = [];
    ffd_l = [];
    for idx = 1:numel(pattern.port_patterns)
        current = pattern.port_patterns{idx};
        ffd = struct();
        ffd.theta_rad = deg2rad(current.theta_deg(:));
        ffd.phi_rad = deg2rad(current.phi_deg(:));
        ffd.freqs_hz = current.frequencies_hz(:);
        ffd.E_theta = current.Etheta;
        ffd.E_phi = current.Ephi;
        ffd.port_id = idx;
        if idx <= numel(handed)
            ffd.port_label = char(handed(idx));
        else
            ffd.port_label = 'UNKNOWN';
        end
        ffd.metadata = struct('source', 'summaryPatternToFfdPorts');
        if idx <= numel(handed) && handed(idx) == "R"
            ffd_r = ffd;
        elseif idx <= numel(handed) && handed(idx) == "L"
            ffd_l = ffd;
        end
    end
    assert(~isempty(ffd_r) && ~isempty(ffd_l), 'Could not recover RH/LH FFD ports from summary struct');
end

function value = getFieldOrDefault(s, field_name, default_value)
    if isstruct(s) && isfield(s, field_name)
        value = s.(field_name);
    else
        value = default_value;
    end
end

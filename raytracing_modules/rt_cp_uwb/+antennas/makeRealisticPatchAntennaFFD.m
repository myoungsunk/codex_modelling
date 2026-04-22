function ant = makeRealisticPatchAntennaFFD(ffd_rhcp_path, ffd_lhcp_path, position, boresight, h_axis, v_axis)
% makeRealisticPatchAntennaFFD - Create an antenna from HFSS FFD port files.

    [ffd_r, ffd_l, pattern_summary] = loadCachedPair(ffd_rhcp_path, ffd_lhcp_path);
    local_to_world = [normalizeVec(h_axis(:)), normalizeVec(v_axis(:)), normalizeVec(boresight(:))];

    ant = core.Antenna( ...
        'position', position(:), ...
        'boresight', boresight(:), ...
        'h_axis', h_axis(:), ...
        'v_axis', v_axis(:), ...
        'basis', 'circular', ...
        'convention', 'IEEE-RHCP', ...
        'cross_pol_leakage_db', pattern_summary.xpd_db_boresight, ...
        'axial_ratio_db', pattern_summary.ar_db_boresight, ...
        'ar_edge_db', 10.0, ...
        'xpd_edge_db', 8.0, ...
        'enable_coupling', false, ...
        'tx_peak_gain_dbi', pattern_summary.peak_gain_dbi, ...
        'rx_peak_gain_dbi', pattern_summary.peak_gain_dbi, ...
        'tx_pattern_cos_exp', 0.0, ...
        'rx_pattern_cos_exp', 0.0, ...
        'patternData', pattern_summary, ...
        'use_ffd', true, ...
        'ffd_port_r', ffd_r, ...
        'ffd_port_l', ffd_l, ...
        'ffd_local_to_world', local_to_world);
end

function [ffd_r, ffd_l, pattern_summary] = loadCachedPair(ffd_rhcp_path, ffd_lhcp_path)
    persistent cache
    key = string(ffd_rhcp_path) + "|" + string(ffd_lhcp_path);
    if ~isempty(cache) && isfield(cache, 'key') && strcmp(cache.key, key)
        ffd_r = cache.ffd_r;
        ffd_l = cache.ffd_l;
        pattern_summary = cache.pattern_summary;
        return;
    end

    ffd_a = antennas.loadFfdPattern(ffd_rhcp_path, 'notes', 'factory_input');
    ffd_b = antennas.loadFfdPattern(ffd_lhcp_path, 'notes', 'factory_input');
    validateMatchingFfd(ffd_a, ffd_b);
    [ffd_r, ffd_l] = assignFfdHands(ffd_a, ffd_b);
    pattern_summary = antennas.loadPatchPattern({ffd_rhcp_path, ffd_lhcp_path});

    cache = struct( ...
        'key', char(key), ...
        'ffd_r', ffd_r, ...
        'ffd_l', ffd_l, ...
        'pattern_summary', pattern_summary);
end

function validateMatchingFfd(ffd_a, ffd_b)
    assert(isequal(size(ffd_a.E_theta), size(ffd_b.E_theta)), 'E field size mismatch between FFD ports');
    assert(isequal(ffd_a.theta_rad, ffd_b.theta_rad), 'Theta grid mismatch between FFD ports');
    assert(isequal(ffd_a.phi_rad, ffd_b.phi_rad), 'Phi grid mismatch between FFD ports');
    assert(isequal(ffd_a.freqs_hz, ffd_b.freqs_hz), 'Frequency grid mismatch between FFD ports');
end

function [ffd_r, ffd_l] = assignFfdHands(ffd_a, ffd_b)
    hand_a = dominantHand(ffd_a);
    hand_b = dominantHand(ffd_b);
    if strcmp(hand_a, 'R') && strcmp(hand_b, 'L')
        ffd_r = ffd_a;
        ffd_l = ffd_b;
        return;
    end
    if strcmp(hand_a, 'L') && strcmp(hand_b, 'R')
        ffd_r = ffd_b;
        ffd_l = ffd_a;
        return;
    end
    error('antennas:makeRealisticPatchAntennaFFD:Handedness', 'Could not uniquely assign RHCP/LHCP FFD ports');
end

function hand = dominantHand(ffd)
    metricsR = antennas.computeFfdMetrics(ffd, 'handedness', 'RHCP');
    metricsL = antennas.computeFfdMetrics(ffd, 'handedness', 'LHCP');
    [~, fmid] = min(abs(ffd.freqs_hz - mean(ffd.freqs_hz)));
    if metricsR.boresight_xpd_db(fmid) >= metricsL.boresight_xpd_db(fmid)
        hand = 'R';
    else
        hand = 'L';
    end
end

function v = normalizeVec(x)
    v = double(x(:));
    n = norm(v);
    if n <= 1e-12
        error('antennas:makeRealisticPatchAntennaFFD:ZeroVector', 'zero-length axis vector');
    end
    v = v / n;
end

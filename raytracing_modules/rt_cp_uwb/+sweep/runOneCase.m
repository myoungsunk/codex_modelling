function features_out = runOneCase(case_row, cfg)
% runOneCase - execute the single-case RT pipeline for one sweep sample.

    if nargin < 2 || isempty(cfg)
        cfg = config.defaultConfig();
    end
    cfg = ensureCfg(cfg);
    row = normalizeCaseRow(case_row);
    seedCaseRng(row);

    mat = buildCaseMaterial(row);
    if isStage2RoomRow(row)
        geom = buildStage2Geometry(row);
        tx_pos = geom.tx_pos;
        rx_pos = geom.rx_pos;
        scene = geom.scene;
        [tx_ant, rx_ant] = buildStage2Antennas(row, geom);
    elseif isSymmetricStage1Row(row)
        geom = scenes.generateSymmetricBoresightGeometry(row, mat, row.slab_size_m);
        tx_pos = geom.tx_pos;
        rx_pos = geom.rx_pos;
        scene = geom.scene;
        [tx_ant, rx_ant] = buildSymmetricAntennas(row, geom);
    else
        [tx_pos, rx_pos, slab_center] = buildGeometry(row);
        scene = buildScene(row, mat, tx_pos, rx_pos, slab_center);
        [tx_ant, rx_ant] = buildAntennas(row, tx_pos, rx_pos, slab_center);
    end

    paths = trace.enumeratePaths(scene, tx_pos, rx_pos, 2);
    if isempty(paths)
        features_out = failureStruct(row.case_id, 'no_paths');
        return;
    end

    H = channel.buildChannel(paths, tx_ant, rx_ant, cfg.freqs);
    if ~any(isfinite(H(:))) || max(abs(H(:))) <= 1e-14
        features_out = failureStruct(row.case_id, 'zero_channel');
        return;
    end
    H_noisy = sweep.injectSnr(H, row.snr_db);
    feats = features.extractAllFeatures(H_noisy, cfg.freqs, ...
        'window_type', cfg.window_type, ...
        'feature_schema', 'canonical18', ...
        'tx_ant', tx_ant, ...
        'rx_ant', rx_ant, ...
        'tx_handedness', 'R');
    if hasNonFiniteCanonical(feats)
        features_out = failureStruct(row.case_id, 'nonfinite_features');
        return;
    end

    bounce_counts = [paths.bounce_count];
    [effective_nlos, has_los_path, los_strength_mid, max_bounce_strength_mid, dominant_bounce_count, bounce_to_los_ratio_mid] = ...
        classifyEffectiveNlos(paths, tx_ant, rx_ant, cfg);
    features_out = feats;
    features_out.case_id = row.case_id;
    features_out.failed = false;
    features_out.error_msg = '';
    features_out.num_paths = numel(paths);
    features_out.bounce_count_min = min(bounce_counts);
    features_out.bounce_count_max = max(bounce_counts);
    features_out.has_los_path = has_los_path;
    features_out.is_los = ~effective_nlos;
    features_out.is_nlos = effective_nlos;
    features_out.tx_rx_dist_m = norm(rx_pos - tx_pos);
    features_out.material_kind = mat.kind;
    features_out.los_strength_mid = los_strength_mid;
    features_out.max_bounce_strength_mid = max_bounce_strength_mid;
    features_out.bounce_to_los_ratio_mid = bounce_to_los_ratio_mid;
    features_out.dominant_bounce_count = dominant_bounce_count;
    if isStage2RoomRow(row)
        features_out.room_type = row.room_type;
        features_out.grid_layer = row.grid_layer;
        features_out.los_angle_from_anchor_bore_deg = geom.los_angle_from_anchor_bore_deg;
        features_out.los_angle_from_tag_bore_deg = geom.los_angle_from_tag_bore_deg;
        features_out.dominant_wall_material = row.dominant_wall_material;
        features_out.eps_r_multiplier = row.eps_r_multiplier;
    elseif isSymmetricStage1Row(row)
        features_out.los_angle_from_anchor_bore_deg = geom.los_angle_from_anchor_bore_deg;
        features_out.los_angle_from_tag_bore_deg = geom.los_angle_from_tag_bore_deg;
        features_out.slab_placement = row.slab_placement;
    end
end

function seedCaseRng(row)
    if isfield(row, 'case_id') && ~isempty(row.case_id) && isfinite(double(row.case_id))
        rng(normalizeCaseSeed(row.case_id), 'twister');
    end
end

function seed = normalizeCaseSeed(case_id)
    seed = mod(round(double(case_id)), 2^32 - 1);
    if seed < 0
        seed = seed + (2^32 - 1);
    end
    if seed == 0
        seed = 1;
    end
end

function cfg = ensureCfg(cfg)
    if ~isfield(cfg, 'freqs') || isempty(cfg.freqs)
        cfg.freqs = linspace(cfg.f_center - cfg.bw / 2.0, cfg.f_center + cfg.bw / 2.0, cfg.n_freq).';
    else
        cfg.freqs = cfg.freqs(:);
    end
    if ~isfield(cfg, 'window_type') || isempty(cfg.window_type)
        cfg.window_type = 'hann';
    end
end

function row = normalizeCaseRow(case_row)
    if istable(case_row)
        assert(height(case_row) == 1, 'case_row table input must contain exactly one row');
        row = table2struct(case_row);
    elseif isstruct(case_row)
        row = case_row;
    else
        error('sweep:runOneCase:CaseRow', 'case_row must be a table row or struct');
    end

    row.case_id = asDoubleField(row, 'case_id', NaN);
    row.tx_height = asDoubleField(row, 'tx_height', 1.0);
    row.rx_height = asDoubleField(row, 'rx_height', 1.0);
    row.incidence_deg = asDoubleField(row, 'incidence_deg', 30.0);
    row.eps_r = asDoubleField(row, 'eps_r', 4.0);
    row.tan_delta = asDoubleField(row, 'tan_delta', 0.01);
    row.xpol_coupling_db = asDoubleField(row, 'xpol_coupling_db', 30.0);
    row.anchor_x = asDoubleField(row, 'anchor_x', NaN);
    row.anchor_y = asDoubleField(row, 'anchor_y', NaN);
    row.anchor_z = asDoubleField(row, 'anchor_z', NaN);
    row.anchor_height = asDoubleField(row, 'anchor_height', NaN);
    row.tag_height = asDoubleField(row, 'tag_height', NaN);
    row.tag_x = asDoubleField(row, 'tag_x', NaN);
    row.tag_y = asDoubleField(row, 'tag_y', NaN);
    row.tag_z = asDoubleField(row, 'tag_z', NaN);
    row.ar_edge_db = asDoubleField(row, 'ar_edge_db', 10.0);
    row.eps_r_multiplier = asDoubleField(row, 'eps_r_multiplier', 1.0);
    row.grid_layer = asDoubleField(row, 'grid_layer', NaN);
    row.snr_db = asDoubleField(row, 'snr_db', 20.0);
    row.slab_size_m = asDoubleField(row, 'slab_size_m', 5.0);
    row.slab_tilt_deg = asDoubleField(row, 'slab_tilt_deg', 0.0);
    row.material_name = asTextField(row, 'material_name', 'wood');
    row.antenna_type = asTextField(row, 'antenna_type', 'ideal');
    row.room_type = asTextField(row, 'room_type', '');
    row.dominant_wall_material = asTextField(row, 'dominant_wall_material', row.material_name);
    row.slab_placement = asTextField(row, 'slab_placement', 'floor');
    row.los_blockage = asLogicalField(row, 'los_blockage', false);
    row.num_slabs = asDoubleField(row, 'num_slabs', 1.0);
end

function mat = buildCaseMaterial(row)
    if strcmpi(row.material_name, 'metal_pec')
        mat = core.Material( ...
            'kind', 'PEC', ...
            'pec_tm_sign', -1.0, ...
            'xpol_coupling_db', row.xpol_coupling_db, ...
            'name', 'metal_pec');
        return;
    end

    mat = core.Material( ...
        'kind', 'dielectric', ...
        'eps_r', row.eps_r, ...
        'tan_delta', row.tan_delta, ...
        'xpol_coupling_db', row.xpol_coupling_db, ...
        'name', row.material_name);
end

function [tx_pos, rx_pos, slab_center] = buildGeometry(row)
    min_incidence_deg = max(5.0, rad2deg(atan(0.3 / max(row.tx_height + row.rx_height, 1e-9))));
    incidence_deg = min(max(row.incidence_deg, min_incidence_deg), 70.0);
    th = deg2rad(incidence_deg);
    tx_rx_dist = (row.tx_height + row.rx_height) * tan(th);
    tx_pos = [0; 0; row.tx_height];
    rx_pos = [tx_rx_dist; 0; row.rx_height];
    slab_center = [tx_rx_dist / 2.0; 0; 0];
end

function geom = buildStage2Geometry(row)
    room_size = [5.0, 4.0, 3.0];
    geom = struct();
    geom.tx_pos = [row.anchor_x; row.anchor_y; row.anchor_z];
    geom.rx_pos = [row.tag_x; row.tag_y; row.tag_z];
    geom.tx_bore = [0; 0; -1];
    geom.tx_h = [1; 0; 0];
    geom.tx_v = [0; 1; 0];
    geom.rx_bore = [0; 0; 1];
    geom.rx_h = [1; 0; 0];
    geom.rx_v = [0; 1; 0];

    assert(all(geom.tx_pos >= 0) && geom.tx_pos(1) <= room_size(1) && geom.tx_pos(2) <= room_size(2) && geom.tx_pos(3) <= room_size(3), ...
        'Stage 2 TX position must lie inside the room bounds');
    assert(all(geom.rx_pos >= 0) && geom.rx_pos(1) <= room_size(1) && geom.rx_pos(2) <= room_size(2) && geom.rx_pos(3) <= room_size(3), ...
        'Stage 2 RX position must lie inside the room bounds');

    los_vec = geom.rx_pos - geom.tx_pos;
    geom.los_distance_m = norm(los_vec);
    los_dir = los_vec / max(geom.los_distance_m, 1e-12);
    geom.los_angle_from_anchor_bore_deg = acosd(clamp(dot(los_dir, geom.tx_bore)));
    geom.los_angle_from_tag_bore_deg = acosd(clamp(dot(-los_dir, geom.rx_bore)));
    geom.scene = applyStage2RoomVariants(scenes.makeRoomABCScene(row.room_type, room_size), row);
end

function scene = buildScene(row, slab_mat, tx_pos, rx_pos, slab_center)
    effective_tilt = row.slab_tilt_deg;
    effective_size = row.slab_size_m;
    if row.los_blockage && round(row.num_slabs) == 1
        effective_tilt = min(effective_tilt, 8.0);
        effective_size = max(effective_size, 0.5 * norm(rx_pos - tx_pos) + 0.75);
    end
    primary_normal = slabNormalFromTilt(effective_tilt);
    slab_size = [effective_size, effective_size];
    scene = scenes.makeSingleSlabScene(slab_mat, slab_size, slab_center, primary_normal, ...
        'surface_id', 1, 'surface_name', 'slab_main');

    if round(row.num_slabs) >= 2
        secondary_center = [slab_center(1); slab_center(2) + 0.35 * effective_size; max(0.5, 0.5 * (tx_pos(3) + rx_pos(3)))];
        secondary = core.Surface( ...
            'surface_id', 2, ...
            'name', 'slab_side', ...
            'point', secondary_center, ...
            'normal', [0; -1; 0], ...
            'u_axis', [1; 0; 0], ...
            'v_axis', [0; 0; 1], ...
            'half_u', effective_size / 2.0, ...
            'half_v', min(max(tx_pos(3), rx_pos(3)) + 0.5, effective_size), ...
            'material', slab_mat);
        scene.addSurface(secondary);
    end

    if row.los_blockage
        blocker_mat = materials.materialsLibrary('concrete');
        center = (tx_pos + rx_pos) / 2.0;
        los_dir = rx_pos - tx_pos;
        if norm(los_dir(1:2)) < 1e-9
            normal = [1; 0; 0];
            u_axis = [0; 1; 0];
        else
            normal = [los_dir(1); los_dir(2); 0];
            normal = normal / norm(normal);
            u_axis = [-normal(2); normal(1); 0];
        end
        v_axis = [0; 0; 1];
        blocker = core.Surface( ...
            'surface_id', 2, ...
            'name', 'los_blocker', ...
            'point', center, ...
            'normal', normal, ...
            'u_axis', u_axis, ...
            'v_axis', v_axis, ...
            'half_u', 0.2, ...
            'half_v', max(0.10, 0.30 * min(row.tx_height, row.rx_height)), ...
            'material', blocker_mat);
        scene.addSurface(blocker);
    end
end

function [tx_ant, rx_ant] = buildAntennas(row, tx_pos, rx_pos, slab_center)
    tx_bore = [1; 0; 0];
    tx_h = [0; 1; 0];
    tx_v = [0; 0; 1];
    rx_bore = [0; 0; 1];
    rx_h = [1; 0; 0];
    rx_v = [0; 1; 0];

    switch lower(row.antenna_type)
        case 'ideal'
            tx_ant = antennas.makeIdealCpAntenna('right', tx_pos, tx_bore, tx_h, tx_v);
            rx_ant = antennas.makeIdealCpAntenna('right', rx_pos, rx_bore, rx_h, rx_v);
        case {'patch', 'patch_ffd'}
            [ffd_rhcp_path, ffd_lhcp_path] = stage1PatchFfdPaths();
            tx_ant = antennas.makeRealisticPatchAntennaFFD(ffd_rhcp_path, ffd_lhcp_path, tx_pos, tx_bore, tx_h, tx_v);
            rx_ant = antennas.makeRealisticPatchAntennaFFD(ffd_rhcp_path, ffd_lhcp_path, rx_pos, rx_bore, rx_h, rx_v);
        case 'patch_synthetic'
            patch = stage1SyntheticPatchPattern(row);
            tx_ant = antennas.makeRealisticPatchAntenna(patch, tx_pos, tx_bore, tx_h, tx_v);
            rx_ant = antennas.makeRealisticPatchAntenna(patch, rx_pos, rx_bore, rx_h, rx_v);
        otherwise
            error('sweep:runOneCase:AntennaType', 'Unknown antenna_type: %s', row.antenna_type);
    end
end

function [tx_ant, rx_ant] = buildSymmetricAntennas(row, geom)
    switch lower(row.antenna_type)
        case 'ideal'
            tx_ant = antennas.makeIdealCpAntenna('right', geom.tx_pos, geom.tx_bore, geom.tx_h, geom.tx_v);
            rx_ant = antennas.makeIdealCpAntenna('right', geom.rx_pos, geom.rx_bore, geom.rx_h, geom.rx_v);
        case {'patch', 'patch_ffd'}
            [ffd_rhcp_path, ffd_lhcp_path] = stage1PatchFfdPaths();
            tx_ant = antennas.makeRealisticPatchAntennaFFD(ffd_rhcp_path, ffd_lhcp_path, geom.tx_pos, geom.tx_bore, geom.tx_h, geom.tx_v);
            rx_ant = antennas.makeRealisticPatchAntennaFFD(ffd_rhcp_path, ffd_lhcp_path, geom.rx_pos, geom.rx_bore, geom.rx_h, geom.rx_v);
        case 'patch_synthetic'
            patch = stage1SyntheticPatchPattern(row);
            tx_ant = antennas.makeRealisticPatchAntenna(patch, geom.tx_pos, geom.tx_bore, geom.tx_h, geom.tx_v);
            rx_ant = antennas.makeRealisticPatchAntenna(patch, geom.rx_pos, geom.rx_bore, geom.rx_h, geom.rx_v);
        otherwise
            error('sweep:runOneCase:AntennaType', 'Unknown antenna_type: %s', row.antenna_type);
    end
end

function [tx_ant, rx_ant] = buildStage2Antennas(row, geom)
    switch lower(row.antenna_type)
        case 'ideal'
            tx_ant = antennas.makeIdealCpAntenna('right', geom.tx_pos, geom.tx_bore, geom.tx_h, geom.tx_v);
            rx_ant = antennas.makeIdealCpAntenna('right', geom.rx_pos, geom.rx_bore, geom.rx_h, geom.rx_v);
        case {'patch', 'patch_ffd'}
            [ffd_rhcp_path, ffd_lhcp_path] = stage1PatchFfdPaths();
            tx_ant = antennas.makeRealisticPatchAntennaFFD(ffd_rhcp_path, ffd_lhcp_path, geom.tx_pos, geom.tx_bore, geom.tx_h, geom.tx_v);
            rx_ant = antennas.makeRealisticPatchAntennaFFD(ffd_rhcp_path, ffd_lhcp_path, geom.rx_pos, geom.rx_bore, geom.rx_h, geom.rx_v);
        case 'patch_synthetic'
            patch = stage1SyntheticPatchPattern(row);
            tx_ant = antennas.makeRealisticPatchAntenna(patch, geom.tx_pos, geom.tx_bore, geom.tx_h, geom.tx_v);
            rx_ant = antennas.makeRealisticPatchAntenna(patch, geom.rx_pos, geom.rx_bore, geom.rx_h, geom.rx_v);
        otherwise
            error('sweep:runOneCase:AntennaType', 'Unknown antenna_type: %s', row.antenna_type);
    end
end

function tf = isSymmetricStage1Row(row)
    tf = isfinite(row.anchor_height) && isfinite(row.tag_height) && isfinite(row.tag_x) && isfinite(row.tag_y);
end

function tf = isStage2RoomRow(row)
    tf = ~isempty(row.room_type) && ...
        isfinite(row.anchor_x) && isfinite(row.anchor_y) && isfinite(row.anchor_z) && ...
        isfinite(row.tag_x) && isfinite(row.tag_y) && isfinite(row.tag_z);
end

function [effective_nlos, has_los_path, los_strength_mid, max_bounce_strength_mid, dominant_bounce_count, bounce_to_los_ratio_mid] = classifyEffectiveNlos(paths, tx_ant, rx_ant, cfg)
    bounce_heavy_ratio_threshold = 0.20;
    tx_ref = antennas.makeIdealCpAntenna('right', tx_ant.position, tx_ant.boresight, tx_ant.h_axis, tx_ant.v_axis);
    rx_ref = antennas.makeIdealCpAntenna('right', rx_ant.position, rx_ant.boresight, rx_ant.h_axis, rx_ant.v_axis);
    strengths = zeros(numel(paths), 1);
    mid = ceil(numel(cfg.freqs) / 2);
    for i = 1:numel(paths)
        H_path = channel.buildChannel(paths(i), tx_ref, rx_ref, cfg.freqs);
        strengths(i) = norm(H_path(:, :, mid), 'fro');
    end

    bounce_counts = [paths.bounce_count].';
    has_los_path = any(bounce_counts == 0);
    dominant_idx = find(strengths == max(strengths), 1, 'first');
    dominant_bounce_count = bounce_counts(dominant_idx);

    los_strength_mid = 0.0;
    if has_los_path
        los_strength_mid = max(strengths(bounce_counts == 0));
    end

    if any(bounce_counts > 0)
        max_bounce_strength_mid = max(strengths(bounce_counts > 0));
    else
        max_bounce_strength_mid = 0.0;
    end

    bounce_to_los_ratio_mid = max_bounce_strength_mid / max(los_strength_mid, 1e-12);
    effective_nlos = ~has_los_path || (bounce_to_los_ratio_mid >= bounce_heavy_ratio_threshold);
end

function out = failureStruct(case_id, error_msg)
    out = struct();
    out.case_id = case_id;
    out.failed = true;
    out.error_msg = char(string(error_msg));
end

function tf = hasNonFiniteCanonical(feats)
    tf = false;
    names = features.canonicalFeatureNames();
    for i = 1:numel(names)
        value = feats.(names{i});
        if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value)
            tf = true;
            return;
        end
    end
end

function value = asDoubleField(s, name, default_value)
    if isfield(s, name)
        raw = s.(name);
        if iscell(raw)
            raw = raw{1};
        end
        value = double(raw);
    else
        value = default_value;
    end
end

function value = asTextField(s, name, default_value)
    if isfield(s, name)
        raw = s.(name);
        if iscell(raw)
            raw = raw{1};
        end
        value = char(string(raw));
    else
        value = default_value;
    end
end

function value = asLogicalField(s, name, default_value)
    if isfield(s, name)
        raw = s.(name);
        if iscell(raw)
            raw = raw{1};
        end
        value = logical(raw);
    else
        value = logical(default_value);
    end
end

function normal = slabNormalFromTilt(tilt_deg)
    tilt_rad = deg2rad(double(tilt_deg));
    normal = [sin(tilt_rad); 0; cos(tilt_rad)];
    normal = normal / norm(normal);
end

function patch = stage1SyntheticPatchPattern(row)
    patch = antennas.loadPatchPatternSynthetic('ar_edge_db', row.ar_edge_db);
end

function scene = applyStage2RoomVariants(scene, row)
    dominant_wall_mat = materials.materialsLibrary(row.dominant_wall_material);
    for idx = 1:numel(scene.surfaces)
        surf = scene.surfaces{idx};
        if startsWith(surf.name, 'wall_')
            surf.material = copyMaterialWithOverrides(dominant_wall_mat, row.eps_r_multiplier, row.xpol_coupling_db);
        else
            surf.material = copyMaterialWithOverrides(surf.material, row.eps_r_multiplier, row.xpol_coupling_db);
        end
        scene.surfaces{idx} = surf;
    end
end

function mat_out = copyMaterialWithOverrides(mat_in, eps_mult, xpol_db)
    if strcmpi(mat_in.kind, 'PEC')
        mat_out = core.Material( ...
            'kind', 'PEC', ...
            'pec_tm_sign', mat_in.pec_tm_sign, ...
            'xpol_coupling_db', xpol_db, ...
            'name', mat_in.name);
        return;
    end

    mat_out = core.Material( ...
        'kind', mat_in.kind, ...
        'eps_r', max(1.0, mat_in.eps_r * eps_mult), ...
        'tan_delta', mat_in.tan_delta, ...
        'xpol_coupling_db', xpol_db, ...
        'xpol_coupling_phase_deg', mat_in.xpol_coupling_phase_deg, ...
        'thickness_m', mat_in.thickness_m, ...
        'dispersion_model', mat_in.dispersion_model, ...
        'name', mat_in.name);
end

function value = clamp(x)
    value = min(max(double(x), -1.0), 1.0);
end

function [rhcp_path, lhcp_path] = stage1PatchFfdPaths()
    persistent cached_pair
    if ~isempty(cached_pair)
        rhcp_path = cached_pair{1};
        lhcp_path = cached_pair{2};
        return;
    end

    project_root = fileparts(fileparts(mfilename('fullpath')));
    candidates = { ...
        {fullfile(project_root, 'data', 'patch_patterns', 'patch_rhcp.ffd'), fullfile(project_root, 'data', 'patch_patterns', 'patch_lhcp.ffd')}; ...
        {fullfile(project_root, 'RHCP_new_6G7G_11pts.ffd'), fullfile(project_root, 'LHCP_new_6G7G_11pts.ffd')}; ...
        {'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\RHCP_new_6G7G_11pts.ffd', 'E:\0. CP Antenna\0. TRACK2_3_SIM\ANTENNA_SOURCE\LHCP_new_6G7G_11pts.ffd'}};

    for idx = 1:size(candidates, 1)
        current = candidates{idx};
        if all(cellfun(@(p) exist(p, 'file') == 2, current))
            cached_pair = current;
            rhcp_path = cached_pair{1};
            lhcp_path = cached_pair{2};
            return;
        end
    end

    error('sweep:runOneCase:PatchFfd', 'Could not find an RHCP/LHCP FFD pair for Stage 1 sweep');
end

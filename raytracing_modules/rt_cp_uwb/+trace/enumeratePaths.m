function paths = enumeratePaths(scene, tx_pos, rx_pos, max_reflections)
% enumeratePaths - enumerate deterministic specular paths using the image method.
%
% Supports 0/1/2 reflections for the Week 1 baseline.

    assert(ismember(max_reflections, [0, 1, 2]), 'Supports 0/1/2 bounces');

    tx = reshape(double(tx_pos), [3, 1]);
    rx = reshape(double(rx_pos), [3, 1]);
    all_surfaces = sceneSurfaces(scene);

    paths = repmat(core.PathRecord(), 0, 1);
    seen_keys = containers.Map('KeyType', 'char', 'ValueType', 'logical');

    for bounce_count = 0:max_reflections
        sequences = enumerateSequences(numel(all_surfaces), bounce_count);
        for seq_row = 1:size(sequences, 1)
            seq_idx = sequences(seq_row, :);
            surfaces = all_surfaces(seq_idx);
            points = constructPoints(tx, rx, surfaces);
            if isempty(points)
                continue;
            end
            [valid, blocked] = validatePath(points, surfaces, all_surfaces);
            if ~valid
                continue;
            end

            key = pathKey(points, seq_idx);
            if isKey(seen_keys, key)
                continue;
            end
            seen_keys(key) = true;
            paths(end + 1, 1) = buildPathRecord(points, surfaces, seq_idx, blocked); %#ok<AGROW>
        end
    end

    if ~isempty(paths)
        [~, order] = sort([paths.delay_s]);
        paths = paths(order);
    end
end

function sequences = enumerateSequences(num_surfaces, bounce_count)
    if bounce_count == 0
        sequences = zeros(1, 0);
        return;
    end
    if num_surfaces <= 0
        sequences = zeros(0, bounce_count);
        return;
    end
    if bounce_count == 1
        sequences = (1:num_surfaces).';
        return;
    end
    [first_idx, second_idx] = ndgrid(1:num_surfaces, 1:num_surfaces);
    mask = first_idx ~= second_idx;
    sequences = [first_idx(mask), second_idx(mask)];
end

function points = constructPoints(tx_pos, rx_pos, surfaces)
    points = constructPointsSourceImages(tx_pos, rx_pos, surfaces);
    if isempty(points)
        points = constructPointsReceiverImages(tx_pos, rx_pos, surfaces);
    end
end

function points = constructPointsSourceImages(tx_pos, rx_pos, surfaces)
    if isempty(surfaces)
        points = {tx_pos, rx_pos};
        return;
    end

    images = cell(1, numel(surfaces) + 1);
    images{1} = tx_pos;
    for idx = 1:numel(surfaces)
        images{idx + 1} = reflectPoint(images{idx}, surfaces{idx});
    end

    target = rx_pos;
    reflection_points = cell(1, numel(surfaces));
    for idx = numel(surfaces):-1:1
        hit = linePlaneIntersection(images{idx + 1}, target, surfaces{idx});
        if isempty(hit)
            points = {};
            return;
        end
        reflection_points{idx} = hit;
        target = hit;
    end
    points = [{tx_pos}, reflection_points, {rx_pos}];
end

function points = constructPointsReceiverImages(tx_pos, rx_pos, surfaces)
    if isempty(surfaces)
        points = {tx_pos, rx_pos};
        return;
    end

    rx_images = cell(1, numel(surfaces) + 1);
    rx_images{1} = rx_pos;
    for idx = numel(surfaces):-1:1
        out_idx = numel(surfaces) - idx + 2;
        rx_images{out_idx} = reflectPoint(rx_images{out_idx - 1}, surfaces{idx});
    end

    points = {tx_pos};
    current = tx_pos;
    for idx = 1:numel(surfaces)
        target = rx_images{numel(surfaces) - idx + 2};
        hit = linePlaneIntersection(current, target, surfaces{idx});
        if isempty(hit)
            points = {};
            return;
        end
        points{end + 1} = hit; %#ok<AGROW>
        current = hit;
    end
    points{end + 1} = rx_pos;
end

function [valid, blocked] = validatePath(points, surfaces, all_surfaces)
    blocked = false;
    for idx = 1:(numel(points) - 1)
        if norm(points{idx + 1} - points{idx}) <= 1e-9
            valid = false;
            return;
        end
    end

    for idx = 1:numel(surfaces)
        if ~surfaceContainsPoint(surfaces{idx}, points{idx + 1}, 1e-6)
            valid = false;
            return;
        end
    end

    for idx = 1:(numel(points) - 1)
        if segmentBlocked(points{idx}, points{idx + 1}, all_surfaces, 1e-7)
            valid = false;
            blocked = true;
            return;
        end
    end

    valid = true;
end

function key = pathKey(points, seq_idx)
    tol_m = 1e-6;
    point_keys = cell(1, max(numel(points) - 2, 0));
    for idx = 2:(numel(points) - 1)
        q = round(points{idx}(:).' / tol_m);
        point_keys{idx - 1} = sprintf('%d,%d,%d', q(1), q(2), q(3));
    end
    if isempty(seq_idx)
        seq_str = '[]';
    else
        seq_str = sprintf('%d,', seq_idx);
        seq_str(end) = [];
    end
    key = sprintf('%s|%s', seq_str, strjoin(point_keys, ';'));
end

function rec = buildPathRecord(points, surfaces, seq_idx, blocked)
    c0 = 299792458.0;
    bounce_count = numel(surfaces);
    total_length = pathLength(points);
    launch_dir = normalizeVec(points{2} - points{1});
    arrival_dir = normalizeVec(points{end} - points{end - 1});

    surface_ids = zeros(1, bounce_count);
    surface_names = cell(1, bounce_count);
    materials = cell(1, bounce_count);
    incidence_angles = zeros(1, bounce_count);
    normals = cell(1, bounce_count);

    for idx = 1:bounce_count
        kin = normalizeVec(points{idx + 1} - points{idx});
        normal = surfaceNormal(surfaces{idx});
        if dot(kin, normal) > 0.0
            normal = -normal;
        end
        normals{idx} = normal;
        cos_theta = min(max(-dot(kin, normal), 0.0), 1.0);
        incidence_angles(idx) = acos(cos_theta);
        surface_ids(idx) = surfaceId(surfaces{idx}, seq_idx(idx));
        surface_names{idx} = surfaceName(surfaces{idx}, sprintf('surface_%d', seq_idx(idx)));
        materials{idx} = surfaceMaterial(surfaces{idx});
    end

    rec = core.PathRecord( ...
        'points', points, ...
        'surface_ids', surface_ids, ...
        'surface_names', surface_names, ...
        'materials', materials, ...
        'bounce_count', bounce_count, ...
        'path_length_m', total_length, ...
        'delay_s', total_length / c0, ...
        'launch_dir', launch_dir, ...
        'arrival_dir', arrival_dir, ...
        'incidence_angles_rad', incidence_angles, ...
        'normals', normals, ...
        'blocked', blocked, ...
        'valid', true);
end

function surfaces = sceneSurfaces(scene)
    if isstruct(scene) && isfield(scene, 'surfaces')
        raw = scene.surfaces;
    elseif isobject(scene) && isprop(scene, 'surfaces')
        raw = scene.surfaces;
    else
        error('trace:enumeratePaths:Scene', 'scene must expose a ''surfaces'' field/property');
    end

    if isempty(raw)
        surfaces = {};
    elseif iscell(raw)
        surfaces = raw;
    else
        surfaces = arrayfun(@(item) item, raw, 'UniformOutput', false);
    end
end

function point = reflectPoint(input_point, surface)
    p = input_point(:);
    delta = p - surfacePoint(surface);
    point = p - 2.0 * dot(delta, surfaceNormal(surface)) * surfaceNormal(surface);
end

function point = linePlaneIntersection(a, b, surface)
    p0 = a(:);
    p1 = b(:);
    d = p1 - p0;
    normal = surfaceNormal(surface);
    denom = dot(normal, d);
    if abs(denom) <= 1e-9
        point = [];
        return;
    end
    t = dot(normal, surfacePoint(surface) - p0) / denom;
    point = p0 + t * d;
end

function blocked = segmentBlocked(a, b, surfaces, eps_val)
    blocked = false;
    for idx = 1:numel(surfaces)
        hit = segmentPlaneIntersection(a, b, surfaces{idx}, eps_val);
        if isempty(hit)
            continue;
        end
        blocked = true;
        return;
    end
end

function point = segmentPlaneIntersection(a, b, surface, eps_val)
    p0 = a(:);
    p1 = b(:);
    d = p1 - p0;
    normal = surfaceNormal(surface);
    denom = dot(normal, d);
    if abs(denom) <= eps_val
        point = [];
        return;
    end
    t = dot(normal, surfacePoint(surface) - p0) / denom;
    if t < -eps_val || t > 1.0 + eps_val
        point = [];
        return;
    end
    candidate = p0 + t * d;
    if ~surfaceContainsPoint(surface, candidate, max(eps_val, 1e-7))
        point = [];
        return;
    end
    if eps_val < t && t < 1.0 - eps_val
        point = candidate;
    else
        point = [];
    end
end

function total = pathLength(points)
    total = 0.0;
    for idx = 1:(numel(points) - 1)
        total = total + norm(points{idx + 1} - points{idx});
    end
end

function tf = surfaceContainsPoint(surface, point, eps_val)
    if isobject(surface)
        if ismethod(surface, 'contains_point')
            tf = surface.contains_point(point, eps_val);
            return;
        end
        if ismethod(surface, 'containsPoint')
            tf = surface.containsPoint(point, eps_val);
            return;
        end
    end

    origin = surfacePoint(surface);
    u_axis = surfaceAxis(surface, {'u_axis', 'uAxis'});
    v_axis = surfaceAxis(surface, {'v_axis', 'vAxis'});
    half_u = surfaceScalar(surface, {'half_u', 'halfU'});
    half_v = surfaceScalar(surface, {'half_v', 'halfV'});

    delta = point(:) - origin;
    u_coord = dot(delta, u_axis);
    v_coord = dot(delta, v_axis);
    tf = abs(u_coord) <= half_u + eps_val && abs(v_coord) <= half_v + eps_val;
end

function point = surfacePoint(surface)
    point = surfaceVector(surface, {'point', 'origin'});
end

function normal = surfaceNormal(surface)
    normal = normalizeVec(surfaceVector(surface, {'normal', 'n'}));
end

function axis_vec = surfaceAxis(surface, names)
    axis_vec = normalizeVec(surfaceVector(surface, names));
end

function value = surfaceScalar(surface, names)
    for idx = 1:numel(names)
        if isstruct(surface) && isfield(surface, names{idx})
            value = double(surface.(names{idx}));
            return;
        end
        if isobject(surface) && isprop(surface, names{idx})
            value = double(surface.(names{idx}));
            return;
        end
    end
    error('trace:enumeratePaths:SurfaceField', 'missing surface scalar field');
end

function value = surfaceVector(surface, names)
    for idx = 1:numel(names)
        if isstruct(surface) && isfield(surface, names{idx})
            value = reshape(double(surface.(names{idx})), [3, 1]);
            return;
        end
        if isobject(surface) && isprop(surface, names{idx})
            value = reshape(double(surface.(names{idx})), [3, 1]);
            return;
        end
    end
    error('trace:enumeratePaths:SurfaceField', 'missing surface vector field');
end

function value = surfaceMaterial(surface)
    if isstruct(surface) && isfield(surface, 'material')
        value = surface.material;
    elseif isobject(surface) && isprop(surface, 'material')
        value = surface.material;
    else
        value = [];
    end
end

function value = surfaceId(surface, fallback_id)
    if isstruct(surface) && isfield(surface, 'surface_id')
        value = double(surface.surface_id);
    elseif isstruct(surface) && isfield(surface, 'id')
        value = double(surface.id);
    elseif isobject(surface) && isprop(surface, 'surface_id')
        value = double(surface.surface_id);
    elseif isobject(surface) && isprop(surface, 'id')
        value = double(surface.id);
    else
        value = double(fallback_id);
    end
end

function value = surfaceName(surface, fallback_name)
    if isstruct(surface) && isfield(surface, 'name')
        value = char(string(surface.name));
    elseif isobject(surface) && isprop(surface, 'name')
        value = char(string(surface.name));
    else
        value = fallback_name;
    end
end

function v = normalizeVec(x)
    v = x(:);
    n = norm(v);
    if n == 0.0
        error('trace:enumeratePaths:ZeroVector', 'zero-length vector');
    end
    v = v / n;
end

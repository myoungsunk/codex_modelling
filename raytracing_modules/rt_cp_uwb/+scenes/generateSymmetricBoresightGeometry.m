function geom = generateSymmetricBoresightGeometry(case_row, slab_material, slab_size_m)
% generateSymmetricBoresightGeometry - fixed anchor / variable tag geometry.
%
% Anchor (TX): fixed above the scene, boresight downward.
% Tag (RX): variable floor-to-wall position, boresight upward.

    row = normalizeRow(case_row);

    geom = struct();
    geom.tx_pos = [0; 0; row.anchor_height];
    geom.tx_bore = [0; 0; -1];
    geom.tx_h = [1; 0; 0];
    geom.tx_v = [0; 1; 0];

    geom.rx_pos = [row.tag_x; row.tag_y; row.tag_height];
    geom.rx_bore = [0; 0; 1];
    geom.rx_h = [1; 0; 0];
    geom.rx_v = [0; 1; 0];

    los_vec = geom.rx_pos - geom.tx_pos;
    geom.los_distance_m = norm(los_vec);
    los_dir = normalizeVec(los_vec);
    geom.los_angle_from_anchor_bore_deg = acosd(clamp(dot(los_dir, geom.tx_bore)));
    geom.los_angle_from_tag_bore_deg = acosd(clamp(dot(-los_dir, geom.rx_bore)));

    room_extent = max([2.5, abs(row.tag_x) + 0.75, abs(row.tag_y) + 0.75]);
    slab_span = max(double(slab_size_m), 2.0 * room_extent);
    mid_z = 0.5 * (row.anchor_height + row.tag_height);

    switch lower(row.slab_placement)
        case 'floor'
            geom.slab_center = [0; 0; 0];
            geom.slab_normal = [0; 0; 1];
            slab_dims = [slab_span, slab_span];
        case 'ceiling'
            ceiling_z = max(row.anchor_height + 0.5, row.tag_height + 1.0);
            geom.slab_center = [0; 0; ceiling_z];
            geom.slab_normal = [0; 0; -1];
            slab_dims = [slab_span, slab_span];
        case 'wall_x'
            wall_x = signWithFallback(row.tag_x, 1.0) * room_extent;
            geom.slab_center = [wall_x; 0; mid_z];
            geom.slab_normal = [-signWithFallback(wall_x, 1.0); 0; 0];
            slab_dims = [slab_span, max(row.anchor_height + 0.5, row.tag_height + 0.5)];
        case 'wall_y'
            wall_y = signWithFallback(row.tag_y, 1.0) * room_extent;
            geom.slab_center = [0; wall_y; mid_z];
            geom.slab_normal = [0; -signWithFallback(wall_y, 1.0); 0];
            slab_dims = [slab_span, max(row.anchor_height + 0.5, row.tag_height + 0.5)];
        otherwise
            error('scenes:generateSymmetricBoresightGeometry:Placement', ...
                'unsupported slab_placement: %s', row.slab_placement);
    end

    geom.scene = scenes.makeSingleSlabScene(slab_material, slab_dims, geom.slab_center, geom.slab_normal);
end

function row = normalizeRow(case_row)
    if istable(case_row)
        assert(height(case_row) == 1, 'case_row must contain exactly one row');
        row = table2struct(case_row);
    else
        row = case_row;
    end

    row.anchor_height = asDoubleField(row, 'anchor_height', 2.5);
    row.tag_height = asDoubleField(row, 'tag_height', 0.5);
    row.tag_x = asDoubleField(row, 'tag_x', 0.0);
    row.tag_y = asDoubleField(row, 'tag_y', 0.0);
    row.slab_placement = asTextField(row, 'slab_placement', 'floor');
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

function v = normalizeVec(x)
    v = double(x(:));
    n = norm(v);
    if n <= 1e-12
        error('scenes:generateSymmetricBoresightGeometry:ZeroVector', 'zero-length vector');
    end
    v = v / n;
end

function value = clamp(x)
    value = min(max(x, -1.0), 1.0);
end

function s = signWithFallback(x, fallback)
    if abs(x) <= 1e-12
        s = sign(fallback);
    else
        s = sign(x);
    end
end

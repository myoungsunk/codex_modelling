function scene = makeRoomABCScene(room_type, room_size)
% makeRoomABCScene - Stage 2 indoor-room generator with simple clutter levels.
%
% room_type : 'A' clean room
%             'B' sparse clutter (glass pane + wood desk)
%             'C' dense clutter (B + metal cabinet + wood partition)
% room_size : [L, W, H] in meters, default [5, 4, 3]

    if nargin < 1 || isempty(room_type)
        room_type = 'A';
    end
    if nargin < 2 || isempty(room_size)
        room_size = [5.0, 4.0, 3.0];
    end

    room_size = double(room_size(:).');
    assert(numel(room_size) == 3, 'room_size must be [L, W, H]');
    L = room_size(1);
    W = room_size(2);
    H = room_size(3);

    scene = core.Scene();

    mat_wall = materials.materialsLibrary('drywall');
    mat_floor = materials.materialsLibrary('concrete');
    mat_ceil = materials.materialsLibrary('drywall');

    surface_id = 1;
    addRect('floor', [L / 2; W / 2; 0], [0; 0; 1], [L, W], mat_floor);
    addRect('ceiling', [L / 2; W / 2; H], [0; 0; -1], [L, W], mat_ceil);
    addRect('wall_west', [0; W / 2; H / 2], [1; 0; 0], [W, H], mat_wall);
    addRect('wall_east', [L; W / 2; H / 2], [-1; 0; 0], [W, H], mat_wall);
    addRect('wall_south', [L / 2; 0; H / 2], [0; 1; 0], [L, H], mat_wall);
    addRect('wall_north', [L / 2; W; H / 2], [0; -1; 0], [L, H], mat_wall);

    switch upper(char(string(room_type)))
        case 'A'
            % Clean room: only the enclosing six faces.

        case 'B'
            mat_glass = materials.materialsLibrary('glass');
            mat_wood = materials.materialsLibrary('wood');

            % Slightly inset pane to avoid coplanar overlap with the east wall.
            addRect('window_glass', [L - 0.02; 0.55 * W; 0.65 * H], [-1; 0; 0], [1.4, 1.0], mat_glass);
            addRect('desk_top', [0.68 * L; 0.62 * W; 0.75], [0; 0; 1], [1.2, 0.7], mat_wood);
            addRect('desk_front', [0.95 * L; 0.62 * W; 0.375], [-1; 0; 0], [0.7, 0.75], mat_wood);

        case 'C'
            mat_glass = materials.materialsLibrary('glass');
            mat_wood = materials.materialsLibrary('wood');
            mat_metal = materials.materialsLibrary('metal_pec');

            addRect('window_glass', [L - 0.02; 0.55 * W; 0.65 * H], [-1; 0; 0], [1.4, 1.0], mat_glass);
            addRect('desk_top', [0.68 * L; 0.62 * W; 0.75], [0; 0; 1], [1.2, 0.7], mat_wood);
            addRect('desk_front', [0.95 * L; 0.62 * W; 0.375], [-1; 0; 0], [0.7, 0.75], mat_wood);
            addRect('cabinet_front', [0.28 * L; W - 0.35; 0.9], [0; -1; 0], [0.8, 1.8], mat_metal);
            addRect('cabinet_side', [0.18 * L; W - 0.65; 0.9], [1; 0; 0], [0.6, 1.8], mat_metal);
            addRect('cabinet_top', [0.28 * L; W - 0.65; 1.8], [0; 0; 1], [0.8, 0.6], mat_metal);
            addRect('wood_partition', [0.45 * L; 0.28 * W; 1.0], [0; 1; 0], [1.2, 2.0], mat_wood);

        otherwise
            error('scenes:makeRoomABCScene:UnknownRoomType', 'Unknown room type: %s', char(string(room_type)));
    end

    function addRect(name, center, normal, dims, material)
        [u_axis, v_axis] = defaultPlaneAxes(normal);
        surf = core.Surface( ...
            'surface_id', surface_id, ...
            'name', name, ...
            'point', center(:), ...
            'normal', normal(:), ...
            'u_axis', u_axis, ...
            'v_axis', v_axis, ...
            'half_u', dims(1) / 2.0, ...
            'half_v', dims(2) / 2.0, ...
            'material', material);
        scene.addSurface(surf);
        surface_id = surface_id + 1;
    end
end

function [u_axis, v_axis] = defaultPlaneAxes(normal)
    n = normalizeVec(normal);
    if abs(n(3)) < 0.9
        ref = [0; 0; 1];
    else
        ref = [0; 1; 0];
    end
    u_axis = ref - dot(ref, n) * n;
    if norm(u_axis) < 1e-9
        ref = [1; 0; 0];
        u_axis = ref - dot(ref, n) * n;
    end
    u_axis = normalizeVec(u_axis);
    v_axis = normalizeVec(cross(n, u_axis));
end

function v = normalizeVec(x)
    v = double(x(:));
    n = norm(v);
    if n == 0.0
        error('scenes:makeRoomABCScene:ZeroVector', 'zero-length vector');
    end
    v = v / n;
end

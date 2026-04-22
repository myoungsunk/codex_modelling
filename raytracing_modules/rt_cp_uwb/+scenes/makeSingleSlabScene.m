function scene = makeSingleSlabScene(material_name, slab_size, slab_center, slab_normal, varargin)
% makeSingleSlabScene - Stage 1 baseline with one bounded rectangular slab.
%
% material_name : material key for materials.materialsLibrary or a material struct
% slab_size     : [width, height] in meters
% slab_center   : 3x1 center position
% slab_normal   : 3x1 unit normal

    if nargin < 1 || isempty(material_name)
        material_name = 'wood';
    end
    if nargin < 2 || isempty(slab_size)
        slab_size = [2.0, 2.0];
    end
    if nargin < 3 || isempty(slab_center)
        slab_center = [1; 0; 0];
    end
    if nargin < 4 || isempty(slab_normal)
        slab_normal = [0; 0; 1];
    end

    slab_size = double(slab_size(:).');
    assert(numel(slab_size) == 2, 'slab_size must be [width, height]');

    p = inputParser;
    p.addParameter('surface_id', 1, @isnumeric);
    p.addParameter('surface_name', 'slab_main', @(x) ischar(x) || isstring(x));
    p.parse(varargin{:});
    opts = p.Results;

    if isstruct(material_name)
        mat = material_name;
    else
        mat = materials.materialsLibrary(material_name);
    end
    [u_axis, v_axis] = defaultPlaneAxes(slab_normal);
    slab = core.Surface( ...
        'surface_id', double(opts.surface_id), ...
        'name', char(string(opts.surface_name)), ...
        'point', slab_center(:), ...
        'normal', slab_normal(:), ...
        'u_axis', u_axis, ...
        'v_axis', v_axis, ...
        'half_u', slab_size(1) / 2.0, ...
        'half_v', slab_size(2) / 2.0, ...
        'material', mat);

    scene = core.Scene();
    scene.addSurface(slab);
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
        error('scenes:makeSingleSlabScene:ZeroVector', 'zero-length vector');
    end
    v = v / n;
end

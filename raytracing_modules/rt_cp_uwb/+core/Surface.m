function surf = Surface(varargin)
% Surface - convenience factory for rectangular or infinite planes.
%
% Supported forms:
%   surf = core.Surface('plane', normal, offset, material, name)
%   surf = core.Surface('surface_id', 1, 'name', 'wall', 'point', ..., ...)

    if nargin >= 1 && (ischar(varargin{1}) || isstring(varargin{1})) && strcmpi(string(varargin{1}), "plane")
        assert(nargin >= 4, 'core.Surface(''plane'', normal, offset, material, [name]) requires at least 4 args');
        normal = normalizeVec(varargin{2});
        offset = double(varargin{3});
        material = varargin{4};
        if nargin >= 5
            name = char(string(varargin{5}));
        else
            name = 'plane';
        end
        [u_axis, v_axis] = defaultPlaneAxes(normal);
        surf = struct( ...
            'surface_id', 1, ...
            'name', name, ...
            'point', normal * offset, ...
            'normal', normal, ...
            'u_axis', u_axis, ...
            'v_axis', v_axis, ...
            'half_u', inf, ...
            'half_v', inf, ...
            'material', material);
        return;
    end

    p = inputParser;
    p.addParameter('surface_id', 1, @isnumeric);
    p.addParameter('name', 'surface', @(x) ischar(x) || isstring(x));
    p.addParameter('point', [0; 0; 0], @isnumeric);
    p.addParameter('normal', [0; 0; 1], @isnumeric);
    p.addParameter('u_axis', [1; 0; 0], @isnumeric);
    p.addParameter('v_axis', [0; 1; 0], @isnumeric);
    p.addParameter('half_u', inf, @isnumeric);
    p.addParameter('half_v', inf, @isnumeric);
    p.addParameter('material', [], @(x) true);
    p.parse(varargin{:});
    args = p.Results;

    surf = struct( ...
        'surface_id', double(args.surface_id), ...
        'name', char(string(args.name)), ...
        'point', reshape(double(args.point), [3, 1]), ...
        'normal', normalizeVec(args.normal), ...
        'u_axis', normalizeVec(args.u_axis), ...
        'v_axis', normalizeVec(args.v_axis), ...
        'half_u', double(args.half_u), ...
        'half_v', double(args.half_v), ...
        'material', args.material);
end

function [u_axis, v_axis] = defaultPlaneAxes(normal)
    n = normal(:);
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
    v = x(:);
    n = norm(v);
    if n == 0
        error('core:Surface:ZeroVector', 'zero-length vector');
    end
    v = v / n;
end

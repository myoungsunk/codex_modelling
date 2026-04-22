function rec = PathRecord(varargin)
% PathRecord - create path record struct mirroring the Python layout.

    p = inputParser;
    p.addParameter('points', {}, @(x) iscell(x) || isnumeric(x));
    p.addParameter('surface_ids', [], @isnumeric);
    p.addParameter('surface_names', {}, @(x) iscell(x) || isstring(x) || ischar(x));
    p.addParameter('materials', {}, @(x) iscell(x) || isstruct(x));
    p.addParameter('bounce_count', 0, @isnumeric);
    p.addParameter('path_length_m', NaN, @isnumeric);
    p.addParameter('delay_s', NaN, @isnumeric);
    p.addParameter('launch_dir', zeros(3, 1), @isnumeric);
    p.addParameter('arrival_dir', zeros(3, 1), @isnumeric);
    p.addParameter('incidence_angles_rad', [], @isnumeric);
    p.addParameter('normals', {}, @(x) iscell(x) || isnumeric(x));
    p.addParameter('blocked', false, @(x) islogical(x) || isnumeric(x));
    p.addParameter('valid', true, @(x) islogical(x) || isnumeric(x));
    p.addParameter('jones_f', [], @(x) isempty(x) || isnumeric(x));
    p.addParameter('scalar_factor_f', [], @(x) isempty(x) || isnumeric(x));
    p.addParameter('basis_up_hint', [], @(x) isempty(x) || isnumeric(x));
    p.parse(varargin{:});

    rec = p.Results;
end

function mat = Material(varargin)
% Material - create material struct for reflection computation
%
% Required fields mirroring the Python Material:
%   kind             : 'dielectric' | 'PEC'
%   eps_r            : relative permittivity (real)
%   tan_delta        : loss tangent
%   complex_eps_r    : optional complex epsilon (overrides eps_r, tan_delta)
%   xpol_coupling_db : empirical cross-pol coupling (dB)
%   xpol_coupling_phase_deg
%   pec_tm_sign      : -1.0 (IEEE convention, ENFORCED)
%   thickness_m      : NaN for infinite half-space
%   dispersion_model : 'const' | 'table' | 'debye'
%   name             : identifier string

    p = inputParser;
    p.addParameter('kind', 'dielectric', @(x) ischar(x) || isstring(x));
    p.addParameter('eps_r', 4.0, @isnumeric);
    p.addParameter('tan_delta', 0.01, @isnumeric);
    p.addParameter('complex_eps_r', [], @(x) isempty(x) || isnumeric(x));
    p.addParameter('xpol_coupling_db', 35.0, @isnumeric);
    p.addParameter('xpol_coupling_phase_deg', 0.0, @isnumeric);
    p.addParameter('pec_tm_sign', -1.0, @isnumeric);
    p.addParameter('thickness_m', NaN, @isnumeric);
    p.addParameter('dispersion_model', 'const', @(x) ischar(x) || isstring(x));
    p.addParameter('name', 'unnamed', @(x) ischar(x) || isstring(x));
    p.parse(varargin{:});

    mat = p.Results;
    mat.kind = char(string(mat.kind));
    mat.dispersion_model = char(string(mat.dispersion_model));
    mat.name = char(string(mat.name));

    % CRITICAL: PEC sign convention assertion (P2 from planner)
    if strcmpi(mat.kind, 'PEC')
        assert(mat.pec_tm_sign == -1.0, ...
            'PEC TM sign must be -1.0 (IEEE convention). Got %g', mat.pec_tm_sign);
    end
end

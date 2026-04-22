function patch = loadPatchPatternSynthetic(varargin)
% loadPatchPatternSynthetic - synthetic non-ideal CP patch surrogate.
%
% Defaults are intentionally conservative and more visibly non-ideal than
% the currently available normalized FFD export, which is close to ideal at
% boresight and not reliable for absolute gain/beamwidth.

    p = inputParser;
    p.addParameter('ar_db_boresight', 3.0, @isnumeric);
    p.addParameter('xpd_db_boresight', 20.0, @isnumeric);
    p.addParameter('peak_gain_dbi', 7.0, @isnumeric);
    p.addParameter('fitted_cos_exp', 2.0, @isnumeric);
    p.addParameter('ar_edge_db', 10.0, @isnumeric);
    p.addParameter('xpd_edge_db', 8.0, @isnumeric);
    p.addParameter('primary_handedness', 'R', @(x) ischar(x) || isstring(x));
    p.addParameter('reference_estimate', struct(), @(x) isstruct(x));
    p.parse(varargin{:});
    args = p.Results;

    patch = struct();
    patch.ar_db_boresight = double(args.ar_db_boresight);
    patch.xpd_db_boresight = double(args.xpd_db_boresight);
    patch.peak_gain_dbi = double(args.peak_gain_dbi);
    patch.fitted_cos_exp = double(args.fitted_cos_exp);
    patch.ar_edge_db = double(args.ar_edge_db);
    patch.xpd_edge_db = double(args.xpd_edge_db);
    patch.primary_handedness = char(string(args.primary_handedness));
    patch.primary_port = 1;
    patch.port_count = 2;
    patch.type = 'synthetic';
    patch.factory = 'synthetic_patch';
    patch.reference_estimate = args.reference_estimate;
end

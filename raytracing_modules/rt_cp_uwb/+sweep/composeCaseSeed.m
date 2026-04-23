function seed = composeCaseSeed(case_id, varargin)
% composeCaseSeed - build a deterministic uint32-compatible seed.
%
% Backward-compatible behavior:
%   If stage/base/replicate are all empty or zero, the seed reduces to the
%   legacy case_id-only rule so existing canonical outputs are unchanged.
%
% Salted behavior:
%   When any of stage_id/base_seed/replicate_id is supplied, the seed mixes
%   case_id with stage/base/component information to avoid cross-stage
%   collisions while keeping the run deterministic.

    p = inputParser;
    p.addParameter('stage_id', '', @(x) ischar(x) || isstring(x) || isnumeric(x));
    p.addParameter('base_seed', 0, @(x) isempty(x) || isnumeric(x));
    p.addParameter('replicate_id', 0, @(x) isempty(x) || isnumeric(x));
    p.addParameter('component', '', @(x) ischar(x) || isstring(x) || isnumeric(x));
    p.parse(varargin{:});
    opts = p.Results;

    stage_id = string(opts.stage_id);
    base_seed = normalizeNumeric(opts.base_seed);
    replicate_id = normalizeNumeric(opts.replicate_id);
    component = string(opts.component);
    case_seed = normalizeNumeric(case_id);

    if strlength(stage_id) == 0 && base_seed == 0 && replicate_id == 0
        seed = case_seed;
        return;
    end

    seed = uint32(2166136261);
    words = uint32([ ...
        base_seed, ...
        case_seed, ...
        replicate_id, ...
        hashToken(stage_id), ...
        hashToken(component)]);
    for idx = 1:numel(words)
        seed = mixWord(seed, words(idx));
    end
    seed = normalizeNumeric(seed);
end

function seed = mixWord(seed, word)
    seed = bitxor(uint32(seed), uint32(word));
    seed = uint32(mod(uint64(seed) * uint64(16777619), 2^32 - 1));
    if seed == 0
        seed = uint32(1);
    end
end

function value = hashToken(token)
    if isnumeric(token)
        value = normalizeNumeric(token);
        return;
    end

    txt = char(join(string(token), "|"));
    if isempty(txt)
        value = uint32(0);
        return;
    end

    value = uint32(2166136261);
    bytes = uint8(txt);
    for idx = 1:numel(bytes)
        value = bitxor(value, uint32(bytes(idx)));
        value = uint32(mod(uint64(value) * uint64(16777619), 2^32 - 1));
    end
    if value == 0
        value = uint32(1);
    end
end

function seed = normalizeNumeric(value)
    if isempty(value) || ~isfinite(double(value))
        seed = uint32(0);
        return;
    end
    seed = uint32(mod(round(double(value)), 2^32 - 1));
    if seed == 0 && round(double(value)) ~= 0
        seed = uint32(1);
    end
end

function feats = extractAllFeatures(H_f, freqs, varargin)
% extractAllFeatures - extract CP and CIR features from a channel response.

    p = inputParser;
    p.addParameter('window_type', 'hann', @(x) ischar(x) || isstring(x));
    p.addParameter('fp_method', 'leading_edge', @(x) ischar(x) || isstring(x));
    p.addParameter('tx_primary_port', [], @(x) isempty(x) || isnumeric(x));
    p.addParameter('rx_primary_port', [], @(x) isempty(x) || isnumeric(x));
    p.addParameter('tx_handedness', '', @(x) ischar(x) || isstring(x));
    p.addParameter('feature_schema', 'full', @(x) ischar(x) || isstring(x));
    p.addParameter('tx_ant', [], @(x) true);
    p.addParameter('rx_ant', [], @(x) true);
    p.parse(varargin{:});
    opts = p.Results;

    H_tensor = [];
    if ~isvector(H_f)
        H_tensor = orientTensor(H_f, numel(freqs));
    end

    [H_primary, primary_tx_port, primary_rx_port] = selectPrimaryResponse(H_f, freqs, opts, H_tensor);
    [h_primary, t_axis] = channel.ifftToCir(H_primary, freqs, opts.window_type);
    h_primary = columnVector(h_primary);
    [idx_FP, t_FP, method_info] = features.extractFirstPath(h_primary, t_axis, opts.fp_method);
    tx_handedness = resolveTxHandedness(opts.tx_handedness, opts.tx_primary_port, opts.tx_ant);
    [gamma_tx_port, gamma_same_rx_port, gamma_reversed_rx_port] = gammaPorts(tx_handedness, H_tensor);

    feats = struct();
    feats.idx_fp = idx_FP;
    feats.t_fp_s = t_FP;
    feats.fp_peak_val = method_info.peak_val;
    feats.fp_method = method_info.method;
    feats.primary_tx_port = primary_tx_port;
    feats.primary_rx_port = primary_rx_port;
    feats.tx_handedness = tx_handedness;
    feats.gamma_cp_same_tx_port = gamma_tx_port;
    feats.gamma_cp_same_rx_port = gamma_same_rx_port;
    feats.gamma_cp_reversed_rx_port = gamma_reversed_rx_port;

    feats = mergeStructs(feats, features.computeGammaCpVariants(H_tensor, freqs, idx_FP, tx_handedness, opts.window_type));
    feats = mergeStructs(feats, features.computeAFpVariants(h_primary, idx_FP, t_axis));
    feats = mergeStructs(feats, features.computeCirBaseline(h_primary, t_axis, idx_FP));

    schema = lower(char(string(opts.feature_schema)));
    switch schema
        case {'full', 'raw'}
            % Keep the legacy raw feature set for backward compatibility.
        case {'canonical', 'canonical18'}
            feats = features.canonicalizeFeatureStruct(feats);
        otherwise
            error('features:extractAllFeatures:FeatureSchema', 'unsupported feature_schema: %s', opts.feature_schema);
    end
end

function [H_primary, tx_port, rx_port] = selectPrimaryResponse(H_f, freqs, opts, H_tensor)
    if isvector(H_f)
        H_primary = columnVector(H_f);
        tx_port = 1;
        rx_port = 1;
        return;
    end

    [Nr, Nt, ~] = size(H_tensor);
    tx_port = resolvePrimaryPort(opts.tx_primary_port, opts.tx_ant, Nt);
    rx_port = resolvePrimaryPort(opts.rx_primary_port, opts.rx_ant, Nr);
    H_primary = squeeze(H_tensor(rx_port, tx_port, :));
end

function H_tensor = orientTensor(H_f, Nf)
    dims = size(H_f);
    freq_dim = find(dims == Nf, 1, 'last');
    if isempty(freq_dim)
        error('features:extractAllFeatures:Dim', 'failed to identify the frequency dimension');
    end
    perm = [setdiff(1:ndims(H_f), freq_dim, 'stable'), freq_dim];
    H_tensor = permute(H_f, perm);
    if ndims(H_tensor) == 2
        H_tensor = reshape(H_tensor, [size(H_tensor, 1), 1, size(H_tensor, 2)]);
    end
end

function port = resolvePrimaryPort(explicit_port, antenna, max_port)
    if ~isempty(explicit_port)
        port = explicit_port;
    elseif isstruct(antenna) && isfield(antenna, 'patternData') && isfield(antenna.patternData, 'primary_handedness')
        port = handednessToPort(antenna.patternData.primary_handedness);
    elseif isobject(antenna) && isprop(antenna, 'patternData') && isstruct(antenna.patternData) && isfield(antenna.patternData, 'primary_handedness')
        port = handednessToPort(antenna.patternData.primary_handedness);
    else
        port = 1;
    end
    port = max(1, min(max_port, round(port)));
end

function hand = resolveTxHandedness(explicit_handedness, explicit_port, antenna)
    if strlength(string(explicit_handedness)) > 0
        hand = normalizeHandedness(explicit_handedness);
        return;
    end

    if ~isempty(explicit_port)
        hand = portToHandedness(explicit_port);
        return;
    end

    if isstruct(antenna) && isfield(antenna, 'patternData') && isfield(antenna.patternData, 'primary_handedness')
        hand = normalizeHandedness(antenna.patternData.primary_handedness);
        return;
    end

    if isobject(antenna) && isprop(antenna, 'patternData') && isstruct(antenna.patternData) && isfield(antenna.patternData, 'primary_handedness')
        hand = normalizeHandedness(antenna.patternData.primary_handedness);
        return;
    end

    hand = 'R';
end

function port = handednessToPort(handedness)
    hand = normalizeHandedness(handedness);
    if startsWith(hand, 'L')
        port = 2;
    else
        port = 1;
    end
end

function hand = portToHandedness(port)
    if round(port) == 2
        hand = 'L';
    else
        hand = 'R';
    end
end

function hand = normalizeHandedness(handedness)
    hand = upper(char(string(handedness)));
    if startsWith(hand, 'L')
        hand = 'L';
    else
        hand = 'R';
    end
end

function [tx_port, same_rx_port, reversed_rx_port] = gammaPorts(tx_handedness, H_tensor)
    tx_port = handednessToPort(tx_handedness);
    same_rx_port = tx_port;
    reversed_rx_port = secondaryPort(same_rx_port, max(2, sizeOrZero(H_tensor, 1)));
    if isempty(H_tensor)
        same_rx_port = tx_port;
        reversed_rx_port = secondaryPort(same_rx_port, 2);
        return;
    end

    same_rx_port = min(same_rx_port, size(H_tensor, 1));
    tx_port = min(tx_port, size(H_tensor, 2));
    if isnan(reversed_rx_port)
        reversed_rx_port = NaN;
    else
        reversed_rx_port = min(reversed_rx_port, size(H_tensor, 1));
    end
end

function port = secondaryPort(primary_port, max_port)
    if max_port < 2
        port = NaN;
        return;
    end
    candidate = setdiff(1:max_port, primary_port, 'stable');
    if isempty(candidate)
        port = NaN;
    else
        port = candidate(1);
    end
end

function out = mergeStructs(varargin)
    out = struct();
    for idx = 1:nargin
        current = varargin{idx};
        fields = fieldnames(current);
        for fidx = 1:numel(fields)
            out.(fields{fidx}) = current.(fields{fidx});
        end
    end
end

function x = columnVector(x)
    x = x(:);
end

function n = sizeOrZero(x, dim)
    if isempty(x)
        n = 0;
    else
        n = size(x, dim);
    end
end

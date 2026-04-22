function H_out = convertBasis(H_f, src, dst, convention, circular_order)
% convertBasis - convert channel tensor between linear and circular bases.
%
% MATLAB convention:
%   H_f is Nr x Nt x Nf

    if nargin < 4 || isempty(convention)
        convention = 'IEEE-RHCP';
    end
    if nargin < 5 || isempty(circular_order)
        circular_order = 'RL';
    end

    src_norm = lower(string(src));
    dst_norm = lower(string(dst));
    H_out = H_f;
    if src_norm == dst_norm
        return;
    end

    if ~all(ismember([src_norm, dst_norm], ["linear", "circular"]))
        error('channel:convertBasis:Unsupported', 'unsupported basis conversion: %s -> %s', src, dst);
    end

    U = circularBasisMatrix(convention, circular_order);
    [Nr, Nt, Nf] = size(H_f);
    H_out = complex(zeros(Nr, Nt, Nf));

    for k = 1:Nf
        if src_norm == "linear" && dst_norm == "circular"
            H_out(:, :, k) = U' * H_f(:, :, k) * U;
        else
            H_out(:, :, k) = U * H_f(:, :, k) * U';
        end
    end
end

function U = circularBasisMatrix(convention, circular_order)
    if ~strcmpi(convention, 'IEEE-RHCP')
        error('channel:convertBasis:Convention', 'unsupported circular convention: %s', convention);
    end

    left = [1.0; 1i] / sqrt(2.0);
    right = [1.0; -1i] / sqrt(2.0);

    if strcmpi(circular_order, 'LR')
        U = [left, right];
    elseif strcmpi(circular_order, 'RL')
        U = [right, left];
    else
        error('channel:convertBasis:Order', 'unsupported circular order: %s', circular_order);
    end
end

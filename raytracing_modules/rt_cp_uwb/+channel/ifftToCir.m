function [h_t, t_axis] = ifftToCir(H_f, freqs, window_type)
% ifftToCir - convert uniformly sampled H(f) to a baseband CIR h(t).
%
% The transform is applied along the detected frequency dimension. The
% output keeps the same non-frequency dimensions as the input.

    if nargin < 3 || isempty(window_type)
        window_type = 'hann';
    end

    freq_vec = freqs(:);
    Nf = numel(freq_vec);
    assert(Nf >= 2, 'ifftToCir requires at least two frequency points');

    df = freq_vec(2) - freq_vec(1);
    assert(all(abs(diff(freq_vec) - df) < 1e-3), 'Frequencies must be uniform');

    freq_dim = detectFrequencyDimension(H_f, Nf);
    win = makeWindow(Nf, window_type);

    H_perm = moveFrequencyDimLast(H_f, freq_dim);
    input_size = size(H_perm);
    Npad = 4 * Nf;

    flat_count = prod(input_size(1:end-1));
    H_flat = reshape(H_perm, [flat_count, Nf]);
    H_win = H_flat .* win.';
    H_pad = complex(zeros(flat_count, Npad, 'like', H_win));
    H_pad(:, 1:Nf) = H_win;
    h_flat = ifft(H_pad, [], 2) * Nf;

    h_perm = reshape(h_flat, [input_size(1:end-1), Npad]);
    h_t = moveFrequencyDimFromLast(h_perm, freq_dim, size(H_f), Npad);

    dt = 1 / (Npad * df);
    t_axis = (0:(Npad - 1)).' * dt;
end

function freq_dim = detectFrequencyDimension(H_f, Nf)
    dims = size(H_f);
    candidates = find(dims == Nf);
    if isempty(candidates)
        error('channel:ifftToCir:Dim', 'no array dimension matches numel(freqs)');
    end
    if isvector(H_f)
        freq_dim = find(dims > 1, 1, 'first');
        if isempty(freq_dim)
            freq_dim = candidates(end);
        end
        return;
    end
    freq_dim = candidates(end);
end

function out = moveFrequencyDimLast(in, freq_dim)
    dims = ndims(in);
    perm = [setdiff(1:dims, freq_dim, 'stable'), freq_dim];
    out = permute(in, perm);
end

function out = moveFrequencyDimFromLast(in, freq_dim, original_size, Npad)
    dims = max(numel(original_size), ndims(in));
    perm = [setdiff(1:dims, freq_dim, 'stable'), freq_dim];
    out = ipermute(in, perm);

    new_size = original_size;
    new_size(freq_dim) = Npad;
    out = reshape(out, new_size);
end

function win = makeWindow(N, window_type)
    n = (0:(N - 1)).';
    switch lower(string(window_type))
        case "hann"
            if N == 1
                win = 1;
            else
                win = 0.5 - 0.5 * cos(2 * pi * n / (N - 1));
            end
        case "rect"
            win = ones(N, 1);
        case "blackman"
            if N == 1
                win = 1;
            else
                win = 0.42 - 0.5 * cos(2 * pi * n / (N - 1)) + 0.08 * cos(4 * pi * n / (N - 1));
            end
        otherwise
            error('channel:ifftToCir:Window', 'unsupported window_type: %s', window_type);
    end
end

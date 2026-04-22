function H_noisy = injectSnr(H, snr_db, noise_seed)
% injectSnr - add complex AWGN to a channel tensor at the requested SNR.
%
% If noise_seed is provided, the noise realization is generated from a
% local RandStream keyed by that seed so repeated runs of the same case are
% bit-exact and independent of sweep order.

    signal_power = mean(abs(H(:)).^2);
    snr_linear = 10 ^ (double(snr_db) / 10.0);
    if ~isfinite(signal_power) || signal_power <= 0.0 || ~isfinite(snr_linear) || snr_linear <= 0.0
        H_noisy = H;
        return;
    end

    noise_power = signal_power / snr_linear;
    sigma = sqrt(noise_power / 2.0);
    if nargin >= 3 && ~isempty(noise_seed) && isfinite(double(noise_seed))
        stream = RandStream('mt19937ar', 'Seed', normalizeSeed(noise_seed));
        noise_real = randn(stream, size(H));
        noise_imag = randn(stream, size(H));
    else
        noise_real = randn(size(H));
        noise_imag = randn(size(H));
    end
    noise = sigma * (noise_real + 1i * noise_imag);
    H_noisy = H + noise;
end

function seed = normalizeSeed(noise_seed)
    seed = mod(round(double(noise_seed)), 2^32 - 1);
    if seed < 0
        seed = seed + (2^32 - 1);
    end
    if seed == 0
        seed = 1;
    end
end

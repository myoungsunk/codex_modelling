# FFD Interpolation Strategy

Chosen method: **amplitude / phase separated bilinear interpolation on the (theta, phi) grid, followed by linear interpolation in frequency**.

Reasoning:
- Real/imag interpolation is simple but can distort phase near rapid phase transitions.
- Amplitude and unwrapped phase interpolation preserves polarization behavior better on coarse FFD grids.
- HFSS FFD data is already on a tensor product grid, so bilinear interpolation is fast and stable.

Pole handling:
- Near theta = 0 or theta = pi, use the nearest valid theta ring and avoid phi-sensitive interpolation because the spherical basis becomes singular.
- Clamp theta queries to the interior by a small epsilon before interpolation.

Phase strategy:
- Unwrap phase along theta first, then along phi for each frequency slice.
- Interpolate phase and amplitude separately, then reconstruct the complex field.
- If amplitude is below a small floor, fall back to nearest-neighbor phase to avoid unstable unwrap artifacts.

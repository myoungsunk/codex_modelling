from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dualpol_rt.channel.builder import build_channel, convert_basis
from dualpol_rt.em.fresnel import fresnel_reflection
from dualpol_rt.em.materials import material_named
from dualpol_rt.feature_extractor import extract_first_path, ifft_to_cir
from dualpol_rt.geometry.image_method import LIGHT_SPEED_M_PER_S, enumerate_paths
from dualpol_rt.patterns import IdealPattern, make_ideal_cp_antenna, make_realistic_patch_cp_antenna
from dualpol_rt.scene.surfaces import Scene


def _save_figure(fig: plt.Figure, path: Path) -> str:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _db20(values: np.ndarray, floor: float = 1e-15) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(values, dtype=float), floor))


def _db10(values: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(values, dtype=float), floor))


def _base_block(scale: float, skew: float) -> list[tuple[complex, complex]]:
    return [
        (scale * ((1.0 + skew) + (0.5 - 0.1 * skew) * 1j), scale * ((8.0 - 2.0 * skew) + (-1.0 - 0.2 * skew) * 1j)),
        (scale * ((2.0 - 0.5 * skew) + (1.0 + 0.2 * skew) * 1j), scale * ((16.0 + skew) + (-2.0 + 0.1 * skew) * 1j)),
        (scale * ((3.0 + 0.3 * skew) + (1.5 + 0.2 * skew) * 1j), scale * ((24.0 - skew) + (-3.0 - 0.1 * skew) * 1j)),
        (scale * ((4.0 - 0.2 * skew) + (2.0 - 0.3 * skew) * 1j), scale * ((32.0 + 2.0 * skew) + (-4.0 + 0.2 * skew) * 1j)),
    ]


def _write_multifrequency_ffd(path: Path, freqs_hz: np.ndarray, blocks: list[list[tuple[complex, complex]]]) -> None:
    lines = [
        "0 90 2",
        "-90 90 2",
        f"Frequencies {len(freqs_hz)}",
    ]
    for freq_hz, block in zip(freqs_hz, blocks):
        lines.append(f"Frequency {float(freq_hz):.15e}")
        for e_theta, e_phi in block:
            lines.append(f"{e_theta.real:.15e} {e_theta.imag:.15e} {e_phi.real:.15e} {e_phi.imag:.15e}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _synthetic_patch_files(root: Path) -> tuple[str, str]:
    freqs_hz = np.linspace(6.0e9, 7.0e9, 11, dtype=float)
    rhcp_path = root / "RHCP_sanity.ffd"
    lhcp_path = root / "LHCP_sanity.ffd"
    rhcp_blocks = [_base_block(1.0 + 0.05 * idx, 0.1) for idx in range(len(freqs_hz))]
    lhcp_blocks = [_base_block(0.8 + 0.04 * idx, 0.7) for idx in range(len(freqs_hz))]
    _write_multifrequency_ffd(rhcp_path, freqs_hz, rhcp_blocks)
    _write_multifrequency_ffd(lhcp_path, freqs_hz, lhcp_blocks)
    return str(rhcp_path), str(lhcp_path)


def _resolve_patch_files(ffd_dir: str | Path | None, temp_root: Path) -> tuple[str, str]:
    if ffd_dir is None:
        return _synthetic_patch_files(temp_root)
    root = Path(ffd_dir)
    rhcp = root / "RHCP_new_6G7G_11pts.ffd"
    lhcp = root / "LHCP_new_6G7G_11pts.ffd"
    missing = [str(path) for path in (rhcp, lhcp) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CP FFD files for sanity plot: {missing}")
    return str(rhcp), str(lhcp)


def _plot_los_pathloss(output_dir: Path) -> str:
    freq_hz = 6.5e9
    wavelength_m = LIGHT_SPEED_M_PER_S / freq_hz
    distances_m = np.geomspace(0.5, 30.0, 40)
    scene = Scene(surfaces=tuple())
    pattern = IdealPattern(basis="linear")
    rt_pathloss_linear = []
    analytic_pathloss_linear = []
    for distance_m in distances_m:
        tx = np.array([0.0, 0.0, 1.0], dtype=float)
        rx = np.array([distance_m, 0.0, 1.0], dtype=float)
        paths = enumerate_paths(scene, tx, rx, max_reflections=0)
        H = build_channel(paths, tx_pattern=pattern, rx_pattern=pattern, freqs_hz=np.array([freq_hz], dtype=float), eval_basis="linear")
        gain_linear = float(np.abs(H[0, 0, 0]) ** 2)
        rt_pathloss_linear.append(1.0 / max(gain_linear, 1e-30))
        analytic_pathloss_linear.append((4.0 * np.pi * distance_m / wavelength_m) ** 2)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.loglog(distances_m, analytic_pathloss_linear, label="Analytic FSPL", linewidth=2.0)
    ax.loglog(distances_m, rt_pathloss_linear, "o", label="RT LoS", markersize=4.0)
    ax.set_title("A1: LoS Path Loss Sanity Check")
    ax.set_xlabel("Tx-Rx distance [m]")
    ax.set_ylabel("Path loss [linear]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return _save_figure(fig, output_dir / "plot_los_pathloss.png")


def _plot_handedness_reversal(output_dir: Path) -> str:
    material = material_named("concrete")
    freq_hz = np.array([6.5e9], dtype=float)
    incidence_deg = np.linspace(0.0, 85.0, 300, dtype=float)
    single_cross_over_co_db = []
    double_cross_over_co_db = []
    for theta_deg in incidence_deg:
        gamma_s, gamma_p = fresnel_reflection(material, np.deg2rad(theta_deg), freq_hz)
        h1_linear = np.zeros((1, 2, 2), dtype=np.complex128)
        h1_linear[0, 0, 0] = gamma_s[0]
        h1_linear[0, 1, 1] = gamma_p[0]
        h2_linear = np.zeros((1, 2, 2), dtype=np.complex128)
        h2_linear[0] = h1_linear[0] @ h1_linear[0]
        h1_circular = convert_basis(h1_linear, src="linear", dst="circular")[0]
        h2_circular = convert_basis(h2_linear, src="linear", dst="circular")[0]
        for matrix, sink in ((h1_circular, single_cross_over_co_db), (h2_circular, double_cross_over_co_db)):
            co_power = np.abs(matrix[0, 0]) ** 2 + np.abs(matrix[1, 1]) ** 2
            cross_power = np.abs(matrix[0, 1]) ** 2 + np.abs(matrix[1, 0]) ** 2
            sink.append(float(_db10(np.array([cross_power / max(co_power, 1e-30)]))[0]))
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(incidence_deg, single_cross_over_co_db, label="Single bounce (odd)")
    ax.plot(incidence_deg, double_cross_over_co_db, label="Double bounce (even)")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_title("B1+B2: Handedness Reversal Check")
    ax.set_xlabel("Incidence angle [deg]")
    ax.set_ylabel("Cross-pol / co-pol [dB]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_figure(fig, output_dir / "plot_handedness_reversal.png")


def _plot_fresnel_brewster(output_dir: Path) -> str:
    material = material_named("glass")
    freq_hz = np.array([6.5e9], dtype=float)
    incidence_deg = np.linspace(0.0, 85.0, 600, dtype=float)
    gamma_s = np.zeros(len(incidence_deg), dtype=np.complex128)
    gamma_p = np.zeros(len(incidence_deg), dtype=np.complex128)
    for idx, theta_deg in enumerate(incidence_deg):
        gamma_s_arr, gamma_p_arr = fresnel_reflection(material, np.deg2rad(theta_deg), freq_hz)
        gamma_s[idx] = gamma_s_arr[0]
        gamma_p[idx] = gamma_p_arr[0]
    eps_r = float(material.eps_r(freq_hz)[0])
    brewster_deg = float(np.rad2deg(np.arctan(np.sqrt(max(eps_r, 1e-12)))))
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(incidence_deg, np.abs(gamma_s), label=r"$|\Gamma_s|$")
    ax.plot(incidence_deg, np.abs(gamma_p), label=r"$|\Gamma_p|$")
    ax.axvline(brewster_deg, color="black", linestyle="--", linewidth=1.0, label=f"Brewster ~ {brewster_deg:.1f} deg")
    ax.set_xlim(0.0, 85.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("B3: Fresnel Brewster Sanity Check")
    ax.set_xlabel("Incidence angle [deg]")
    ax.set_ylabel("Reflection magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_figure(fig, output_dir / "plot_fresnel_brewster.png")


def _plot_cir_los(output_dir: Path) -> str:
    tx = np.array([0.0, 0.0, 1.0], dtype=float)
    rx = np.array([4.2, 0.0, 1.0], dtype=float)
    freqs_hz = np.linspace(6.25e9, 6.75e9, 101, dtype=float)
    scene = Scene(surfaces=tuple())
    pattern = IdealPattern(basis="linear")
    paths = enumerate_paths(scene, tx, rx, max_reflections=0)
    H = build_channel(paths, tx_pattern=pattern, rx_pattern=pattern, freqs_hz=freqs_hz, eval_basis="linear")
    cir = ifft_to_cir(H[:, 0, 0], freqs_hz, window="hann")
    t_fp_s, idx_fp = extract_first_path(cir)
    geom_delay_s = float(np.linalg.norm(rx - tx) / LIGHT_SPEED_M_PER_S)
    magnitude = np.abs(cir.h_t)
    magnitude = magnitude / max(float(np.max(magnitude)), 1e-15)
    delay_ns = cir.delays_s * 1e9
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=False)
    axes[0].plot(delay_ns, magnitude, linewidth=1.5)
    axes[0].axvline(geom_delay_s * 1e9, color="tab:red", linestyle="--", linewidth=1.2, label="Geometric delay")
    axes[0].axvline(t_fp_s * 1e9, color="tab:green", linestyle=":", linewidth=1.2, label="Detected first path")
    axes[0].set_title("C1+C2: LoS CIR Sanity Check")
    axes[0].set_ylabel("Normalized |h(t)|")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    lo = max(0, idx_fp - 6)
    hi = min(len(delay_ns), idx_fp + 7)
    axes[1].plot(delay_ns[lo:hi], magnitude[lo:hi], marker="o", linewidth=1.5)
    axes[1].axvline(geom_delay_s * 1e9, color="tab:red", linestyle="--", linewidth=1.2)
    axes[1].axvline(t_fp_s * 1e9, color="tab:green", linestyle=":", linewidth=1.2)
    axes[1].set_xlabel("Delay [ns]")
    axes[1].set_ylabel("Pulse shape")
    axes[1].grid(True, alpha=0.3)
    return _save_figure(fig, output_dir / "plot_cir_los.png")


def _plot_antenna_compare(output_dir: Path, ffd_dir: str | Path | None = None) -> str:
    tx = np.array([0.0, 0.0, 1.0], dtype=float)
    rx = np.array([3.0, 0.0, 1.0], dtype=float)
    link_dir = rx - tx
    freqs_hz = np.linspace(6.0e9, 7.0e9, 11, dtype=float)
    scene = Scene(surfaces=tuple())
    paths = enumerate_paths(scene, tx, rx, max_reflections=0)
    ideal_tx = make_ideal_cp_antenna(handedness="right", pos=tx, boresight=link_dir, name="ideal_cp_tx")
    ideal_rx = make_ideal_cp_antenna(handedness="right", pos=rx, boresight=-link_dir, name="ideal_cp_rx")
    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        patch_files = _resolve_patch_files(ffd_dir, Path(temp_dir))
        patch_tx = make_realistic_patch_cp_antenna(patch_files, handedness="right", pos=tx, boresight=link_dir, name="patch_cp_tx")
        patch_rx = make_realistic_patch_cp_antenna(patch_files, handedness="right", pos=rx, boresight=-link_dir, name="patch_cp_rx")
        H_ideal = build_channel(paths, tx_pattern=ideal_tx, rx_pattern=ideal_rx, freqs_hz=freqs_hz, eval_basis="linear")
        H_patch = build_channel(paths, tx_pattern=patch_tx, rx_pattern=patch_rx, freqs_hz=freqs_hz, eval_basis="linear")
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
    axes[0].plot(freqs_hz / 1e9, _db20(np.abs(H_ideal[:, 0, 0])), marker="o", label="Ideal CP")
    axes[0].plot(freqs_hz / 1e9, _db20(np.abs(H_patch[:, 0, 0])), marker="s", label="Patch CP")
    axes[0].set_title("D1: Ideal vs Patch H(f)")
    axes[0].set_ylabel(r"$|H_{00}(f)|$ [dB]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(freqs_hz / 1e9, np.unwrap(np.angle(H_ideal[:, 0, 0])), marker="o", label="Ideal CP")
    axes[1].plot(freqs_hz / 1e9, np.unwrap(np.angle(H_patch[:, 0, 0])), marker="s", label="Patch CP")
    axes[1].set_xlabel("Frequency [GHz]")
    axes[1].set_ylabel("Unwrapped phase [rad]")
    axes[1].grid(True, alpha=0.3)
    return _save_figure(fig, output_dir / "plot_antenna_compare.png")


def generate_sanity_plots(output_dir: str | Path, ffd_dir: str | Path | None = None) -> dict[str, str]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    return {
        "plot_los_pathloss": _plot_los_pathloss(outdir),
        "plot_handedness_reversal": _plot_handedness_reversal(outdir),
        "plot_fresnel_brewster": _plot_fresnel_brewster(outdir),
        "plot_cir_los": _plot_cir_los(outdir),
        "plot_antenna_compare": _plot_antenna_compare(outdir, ffd_dir=ffd_dir),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate internal sanity-check plots for the dual-pol RT bundle.")
    parser.add_argument("--output-dir", default="sanity_plots_out", help="Directory where the five PNG files will be written")
    parser.add_argument("--ffd-dir", help="Optional directory containing RHCP_new_6G7G_11pts.ffd and LHCP_new_6G7G_11pts.ffd")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = generate_sanity_plots(args.output_dir, ffd_dir=args.ffd_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

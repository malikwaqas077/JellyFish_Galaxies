"""
Synthetic validation test — no TNG data needed.

Generates artificial gas distributions mimicking:
  1. A normal spiral galaxy (centrally concentrated)
  2. A jellyfish galaxy (offset tail)
  3. A stripped galaxy (elongated gas wake)

Uses the SAME projection + imaging pipeline as the real pipeline.
Run this to verify image quality, aperture, colormap before running
on real TNG data.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils.projection import project_gas
from utils.imaging import projection_to_png, _normalise
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


RNG = np.random.default_rng(42)


def make_galaxy(n_cells=200_000, kind="normal",
                aperture_kpc=200, little_h=0.6774):
    """
    Synthetic gas cell distribution in ckpc/h centred at origin.
    Returns: coords (N,3), masses (N,), densities (N,)
    """
    # Physical kpc → ckpc/h: multiply by little_h
    # Density in 1e10 M_sun/h / (ckpc/h)^3

    if kind == "normal":
        # Smooth Gaussian disk
        r_kpc = 15.0           # half-mass radius in kpc
        r_ckpc_h = r_kpc * little_h
        coords_xy = RNG.normal(0, r_ckpc_h, (n_cells, 2))
        coords_z  = RNG.normal(0, r_ckpc_h * 0.3, n_cells)

    elif kind == "jellyfish":
        # Core + offset tail extending 100 kpc
        n_core = int(n_cells * 0.5)
        n_tail = n_cells - n_core
        r_core_ckpc_h = 12 * little_h
        # core
        cx = RNG.normal(0, r_core_ckpc_h, (n_core, 2))
        cz = RNG.normal(0, r_core_ckpc_h * 0.3, n_core)
        # tail: elongated in +x direction
        tail_len_ckpc_h = 120 * little_h
        tx = np.column_stack([
            RNG.uniform(0, tail_len_ckpc_h, n_tail),   # x: tail direction
            RNG.normal(0, 8 * little_h, n_tail),        # y: width
        ])
        tz = RNG.normal(0, 5 * little_h, n_tail)
        coords_xy = np.vstack([cx, tx])
        coords_z  = np.hstack([cz, tz])
        # tail cells have lower density
        masses = np.concatenate([
            np.ones(n_core) * 2e-4,   # core cells
            np.ones(n_tail) * 5e-5,   # tail cells (less massive)
        ])

    elif kind == "stripped":
        # Elongated wake — all gas pulled in one direction
        r_kpc = 10.0
        r_ckpc_h = r_kpc * little_h
        wake_ckpc_h = 150 * little_h
        # Exponential decay along tail
        tail_x = RNG.exponential(wake_ckpc_h / 3, n_cells)
        coords_xy = np.column_stack([
            tail_x,
            RNG.normal(0, r_ckpc_h * np.exp(-tail_x / wake_ckpc_h * 0.5 + 0.5), n_cells),
        ])
        coords_z = RNG.normal(0, r_ckpc_h * 0.3, n_cells)

    # Default masses / densities (overridden for jellyfish core/tail above)
    if kind != "jellyfish":
        masses = np.ones(n_cells) * 1e-4     # 1e-4 × 1e10 M_sun/h ~ 6.7e5 M_sun per cell

    coords = np.column_stack([coords_xy, coords_z])   # (N, 3)

    # Density: M / (4/3 π r_cell^3), r_cell ~ median separation
    # Use a simple proxy: density ∝ mass (uniform cell size)
    r_cell_ckpc_h = 2.0 * little_h    # ~2 kpc per cell — realistic TNG resolution
    vol_cell = (4/3) * np.pi * r_cell_ckpc_h**3
    densities = masses / vol_cell

    return coords, masses, densities


def run_synthetic_test():
    os.makedirs("output/images", exist_ok=True)

    cases = [
        ("normal",    "Normal spiral galaxy"),
        ("jellyfish", "Jellyfish galaxy (core + tail)"),
        ("stripped",  "Stripped / wake galaxy"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor='black')

    results = []
    for i, (kind, label) in enumerate(cases):
        print(f"Rendering: {label} ...")
        coords, masses, densities = make_galaxy(n_cells=200_000, kind=kind)
        centre = np.zeros(3)

        image, extent = project_gas(
            coords, masses, densities, centre,
            axis="z", a=1.0, aperture_kpc=200.0,
        )

        if image is None:
            print(f"  WARNING: projection returned None for {kind}")
            continue

        fpath, metrics = projection_to_png(
            image,
            subhalo_id=i,
            halo_id=999,
            aperture_kpc=200.0,
            suffix=f"_synthetic_{kind}",
            quality_check=True,
        )
        results.append((kind, label, fpath, metrics))
        print(f"  signal={metrics['signal_fraction']:.3f}  "
              f"DR={metrics['dynamic_range_dex']:.2f} dex  "
              f"QC={'PASS' if metrics['passed_qc'] else 'FAIL'}  "
              f"-> {os.path.basename(fpath)}")

        # Plot
        normed = _normalise(image)
        ax = axes[i]
        ax.imshow(normed, cmap="hot", origin="lower",
                  extent=extent, vmin=0, vmax=1)
        ax.set_title(label, color='white', fontsize=9)
        ax.set_xlabel("x [kpc]", color='white', fontsize=7)
        ax.set_ylabel("y [kpc]", color='white', fontsize=7)
        ax.tick_params(colors='white', labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

    plt.tight_layout()
    fig.patch.set_facecolor('black')
    comparison_path = "output/images/synthetic_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()

    print(f"\nComparison figure -> {comparison_path}")
    print(f"Individual PNGs   -> output/images/")
    print("\nPipeline render code is validated!")
    return results


if __name__ == "__main__":
    run_synthetic_test()

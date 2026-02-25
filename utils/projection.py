"""
2-D gas surface-density projection for Arepo/TNG cells.

Arepo uses a moving Voronoi mesh (not SPH particles).
Each gas cell has a position, mass, and density; we estimate its
effective radius as r_eff = (3m / 4π ρ)^(1/3) and spread its
mass contribution with a Gaussian kernel of width σ = SMOOTH_FACTOR * r_eff.

Projection is along the z-axis by default.
Returns log10 surface-density map ready for colormapping.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from config import (APERTURE_KPC, N_PIXELS_GRID, SMOOTH_FACTOR,
                    LOG_SCALE, VMIN_PERCENTILE, VMAX_PERCENTILE, LITTLE_H)


def _to_physical_kpc(coords_ckpc_h, a=1.0):
    """
    Convert TNG comoving kpc/h → physical kpc.
    At z=0, a=1 and h≈0.6774, so physical_kpc = ckpc_h / h.
    """
    return coords_ckpc_h * a / LITTLE_H


def project_gas(coords_ckpc_h, masses_1e10msun_h, densities_1e10msun_h_kpc3,
                center_ckpc_h, axis="z", a=1.0, aperture_kpc=None):
    """
    Parameters
    ----------
    coords_ckpc_h        : (N,3) array — gas cell positions in ckpc/h
    masses_1e10msun_h    : (N,)   array — cell masses in 1e10 M_sun/h
    densities_1e10msun_h_kpc3 : (N,) array — densities in 1e10 M_sun/h / (ckpc/h)^3
    center_ckpc_h        : (3,)   array — galaxy center in ckpc/h
    axis                 : 'x', 'y', or 'z' — projection axis (LOS)
    a                    : scale factor (1.0 at z=0)
    aperture_kpc         : half-width of projection box in physical kpc

    Returns
    -------
    image  : (N_PIXELS_GRID, N_PIXELS_GRID) float array, log10 surface density
             or linear if LOG_SCALE=False
    extent : [xmin, xmax, ymin, ymax] in physical kpc (for labelling)
    """
    if aperture_kpc is None:
        aperture_kpc = APERTURE_KPC

    # Shift to galaxy centre and convert to physical kpc
    rel = coords_ckpc_h - center_ckpc_h          # (N, 3) comoving kpc/h
    rel_phys = _to_physical_kpc(rel, a=a)         # physical kpc

    # Choose projection plane axes
    axis_map = {"x": (1, 2, 0), "y": (0, 2, 1), "z": (0, 1, 2)}
    ax0, ax1, ax_los = axis_map[axis.lower()]

    px = rel_phys[:, ax0]
    py = rel_phys[:, ax1]

    # Spatial cut: only cells within the 2D aperture and along LOS (3× aperture)
    mask = ((np.abs(px) < aperture_kpc) &
            (np.abs(py) < aperture_kpc) &
            (np.abs(rel_phys[:, ax_los]) < 3 * aperture_kpc))

    if mask.sum() < 5:
        return None, None

    px = px[mask]
    py = py[mask]
    mass = masses_1e10msun_h[mask]
    dens = densities_1e10msun_h_kpc3[mask]

    # Effective cell radius in comoving kpc/h → physical kpc
    # r_eff = (3 m / 4π ρ)^(1/3)   [units cancel since both in same system]
    with np.errstate(divide="ignore", invalid="ignore"):
        r_eff_ckpc_h = (3.0 * mass / (4.0 * np.pi * np.maximum(dens, 1e-30))) ** (1.0 / 3.0)
    r_eff_kpc = _to_physical_kpc(r_eff_ckpc_h, a=a)

    # Build 2-D surface-density grid
    N = N_PIXELS_GRID
    grid = np.zeros((N, N), dtype=np.float64)
    cell_size = 2.0 * aperture_kpc / N               # kpc per pixel

    # Pixel indices for each cell centre
    ix = ((px + aperture_kpc) / cell_size).astype(int)
    iy = ((py + aperture_kpc) / cell_size).astype(int)
    valid = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)

    np.add.at(grid, (iy[valid], ix[valid]), mass[valid])

    # Per-cell Gaussian smoothing: spread mass with σ = SMOOTH_FACTOR * r_eff
    # For efficiency we use a single median-σ global smooth plus per-cell deposit.
    # For best accuracy we do a second pass blurring the grid by median r_eff.
    sigma_px = np.median(r_eff_kpc[valid]) * SMOOTH_FACTOR / cell_size
    sigma_px = np.clip(sigma_px, 0.5, N / 10)
    grid = gaussian_filter(grid, sigma=sigma_px)

    # Convert to surface density (M_sun/kpc²) — drop the 1e10 M_sun/h factor
    # for relative purposes; log-scale is what matters for the image
    surface_density = grid / (cell_size ** 2)    # [1e10 M_sun/h / kpc²]

    # Log scale
    with np.errstate(divide="ignore", invalid="ignore"):
        if LOG_SCALE:
            floor = surface_density[surface_density > 0].min() if (surface_density > 0).any() else 1e-10
            image = np.log10(np.maximum(surface_density, floor))
        else:
            image = surface_density

    extent = [-aperture_kpc, aperture_kpc, -aperture_kpc, aperture_kpc]
    return image, extent

"""
2-D gas surface-density projection for Arepo/TNG cells.

Arepo uses a moving Voronoi mesh (not SPH particles).
Each gas cell has a position, mass, and density; we estimate its
effective radius as r_eff = (3m / 4π ρ)^(1/3) and spread its
mass contribution with a Gaussian kernel of width σ = SMOOTH_FACTOR * r_eff.

Quality approach: per-cell grouped Gaussian smoothing.
  - Cells are grouped into log2 bins of their smoothing radius (in pixels).
  - Each bin is deposited onto a temporary grid and blurred with its own
    Gaussian σ, then summed.  This approximates true per-cell kernel
    smoothing without the O(N × pixels²) cost of a full per-cell loop.
  - Produces noticeably sharper cores and fainter, better-resolved tails
    compared to a single median-σ global blur.

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
    coords_ckpc_h             : (N,3) array — gas cell positions in ckpc/h
    masses_1e10msun_h         : (N,)  array — cell masses in 1e10 M_sun/h
    densities_1e10msun_h_kpc3 : (N,)  array — densities in 1e10 M_sun/h / (ckpc/h)^3
    center_ckpc_h             : (3,)  array — galaxy center in ckpc/h
    axis                      : 'x', 'y', or 'z' — projection axis (LOS)
    a                         : scale factor (1.0 at z=0)
    aperture_kpc              : half-width of projection box in physical kpc

    Returns
    -------
    image  : (N_PIXELS_GRID, N_PIXELS_GRID) float array, log10 surface density
             or linear if LOG_SCALE=False
    extent : [xmin, xmax, ymin, ymax] in physical kpc (for labelling)
    """
    if aperture_kpc is None:
        aperture_kpc = APERTURE_KPC

    # Shift to galaxy centre and convert to physical kpc
    rel      = coords_ckpc_h - center_ckpc_h     # (N, 3) comoving kpc/h
    rel_phys = _to_physical_kpc(rel, a=a)         # physical kpc

    # Choose projection plane axes
    axis_map = {"x": (1, 2, 0), "y": (0, 2, 1), "z": (0, 1, 2)}
    ax0, ax1, ax_los = axis_map[axis.lower()]

    px = rel_phys[:, ax0]
    py = rel_phys[:, ax1]

    # Spatial cut: only cells within the 2D aperture and within 3× aperture along LOS
    mask = ((np.abs(px) < aperture_kpc) &
            (np.abs(py) < aperture_kpc) &
            (np.abs(rel_phys[:, ax_los]) < 3 * aperture_kpc))

    if mask.sum() < 5:
        return None, None

    px   = px[mask]
    py   = py[mask]
    mass = masses_1e10msun_h[mask]
    dens = densities_1e10msun_h_kpc3[mask]

    # ── Effective cell radius ───────────────────────────────────────────────
    # Arepo Voronoi cells: r_eff = (3m / 4π ρ)^(1/3)
    # Units: [mass / dens] = (1e10 Msun/h) / (1e10 Msun/h / (ckpc/h)^3) = (ckpc/h)^3
    with np.errstate(divide="ignore", invalid="ignore"):
        r_eff_ckpc_h = (3.0 * mass / (4.0 * np.pi * np.maximum(dens, 1e-30))) ** (1.0 / 3.0)
    r_eff_kpc = _to_physical_kpc(r_eff_ckpc_h, a=a)

    # ── Grid setup ─────────────────────────────────────────────────────────
    N         = N_PIXELS_GRID
    cell_size = 2.0 * aperture_kpc / N           # physical kpc per pixel

    # Integer pixel indices for each cell centre
    ix = ((px + aperture_kpc) / cell_size).astype(int)
    iy = ((py + aperture_kpc) / cell_size).astype(int)
    in_grid = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)

    ix       = ix[in_grid]
    iy       = iy[in_grid]
    mass     = mass[in_grid]
    r_eff_px = r_eff_kpc[in_grid] * SMOOTH_FACTOR / cell_size
    r_eff_px = np.clip(r_eff_px, 0.3, N / 8.0)

    if len(ix) == 0:
        return None, None

    # ── Per-cell grouped Gaussian smoothing ────────────────────────────────
    # Group cells into log2 octave bins of their smoothing radius.
    # Each bin gets its own gaussian_filter pass (representative σ = 2^bin).
    # Summing all passes approximates per-cell kernel smoothing efficiently.
    grid    = np.zeros((N, N), dtype=np.float64)
    log2_s  = np.floor(np.log2(r_eff_px)).astype(int)

    for lh in range(log2_s.min(), log2_s.max() + 1):
        bin_mask = log2_s == lh
        if not bin_mask.any():
            continue
        tmp = np.zeros((N, N), dtype=np.float64)
        np.add.at(tmp, (iy[bin_mask], ix[bin_mask]), mass[bin_mask])
        sigma = max(2.0 ** lh, 0.3)
        grid += gaussian_filter(tmp, sigma=sigma)

    # ── Surface density and log scale ─────────────────────────────────────
    # grid is in 1e10 M_sun/h.  Divide by cell area to get surface density.
    surface_density = grid / (cell_size ** 2)    # [1e10 M_sun/h / kpc²]

    with np.errstate(divide="ignore", invalid="ignore"):
        if LOG_SCALE:
            pos_vals = surface_density[surface_density > 0]
            floor    = pos_vals.min() if pos_vals.size > 0 else 1e-10
            image    = np.log10(np.maximum(surface_density, floor))
        else:
            image = surface_density

    extent = [-aperture_kpc, aperture_kpc, -aperture_kpc, aperture_kpc]
    return image, extent

"""
2-D gas surface-density projection for Arepo/TNG cells.

Arepo uses a moving Voronoi mesh (not SPH particles).
Each gas cell has a position, mass, and density; we estimate its
effective radius as r_eff = (3m / 4π ρ)^(1/3) and spread its
mass contribution with a Gaussian kernel of width σ = SMOOTH_FACTOR * r_eff.

Multi-scale smoothing:  cells are grouped into log-spaced σ bins.  Each
bin is deposited onto its own grid and convolved with its representative σ,
then all bins are summed.  This preserves the sharp dense core *and* the
faint extended tail — both are rendered at their natural resolution.

Projection is along the z-axis by default.
Returns log10 surface-density map ready for colormapping.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from config import (APERTURE_KPC, N_PIXELS_GRID, SMOOTH_FACTOR,
                    LOG_SCALE, VMIN_PERCENTILE, VMAX_PERCENTILE, LITTLE_H,
                    MULTISCALE_SMOOTH, N_SMOOTH_BINS)


def _multiscale_deposit(px, py, mass, sigma_px_all, ix, iy, valid, N):
    """
    Multi-scale mass deposition.

    Cells are grouped into log-spaced σ bins.  Each bin is deposited onto
    its own sub-grid and convolved with the bin's median σ, then all sub-grids
    are summed.  This gives sharp, realistic rendering of both the dense core
    (small σ) and the faint ram-pressure tail (large σ).

    Parameters
    ----------
    px, py       : projected physical positions (kpc), length M (before valid mask)
    mass         : cell masses, length M
    sigma_px_all : per-cell σ in pixels, length M
    ix, iy       : integer pixel indices, length M
    valid        : boolean mask selecting in-bounds cells, length M
    N            : grid side length in pixels

    Returns
    -------
    grid : (N, N) float64 surface-density grid
    """
    grid_total = np.zeros((N, N), dtype=np.float64)

    sigma_v = sigma_px_all[valid]
    if sigma_v.size == 0:
        return grid_total

    # Log-spaced bin edges spanning the full σ range
    log_min = np.log10(max(sigma_v.min(), 0.3))
    log_max = np.log10(sigma_v.max())

    if log_min >= log_max:
        # All cells at same σ — single smooth pass
        grid_total = np.zeros((N, N), dtype=np.float64)
        np.add.at(grid_total, (iy[valid], ix[valid]), mass[valid])
        return gaussian_filter(grid_total, sigma=float(sigma_v.mean()))

    bin_edges = np.linspace(log_min, log_max, N_SMOOTH_BINS + 1)
    log_sigma_v = np.log10(sigma_v)

    for b in range(N_SMOOTH_BINS):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        # Include upper edge in last bin to avoid losing cells
        if b == N_SMOOTH_BINS - 1:
            in_bin = (log_sigma_v >= lo)
        else:
            in_bin = (log_sigma_v >= lo) & (log_sigma_v < hi)

        if not in_bin.any():
            continue

        # Indices into the original (pre-valid-mask) arrays
        valid_idx = np.where(valid)[0]
        bin_idx = valid_idx[in_bin]

        grid_b = np.zeros((N, N), dtype=np.float64)
        np.add.at(grid_b, (iy[bin_idx], ix[bin_idx]), mass[bin_idx])

        # Representative σ for this bin (geometric mean)
        sigma_rep = float(10 ** np.mean(log_sigma_v[in_bin]))
        sigma_rep = np.clip(sigma_rep, 0.3, N / 8.0)
        grid_total += gaussian_filter(grid_b, sigma=sigma_rep)

    return grid_total


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
    cell_size = 2.0 * aperture_kpc / N               # kpc per pixel

    # Pixel indices for each cell centre
    ix = ((px + aperture_kpc) / cell_size).astype(int)
    iy = ((py + aperture_kpc) / cell_size).astype(int)
    valid = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)

    # Per-cell smoothing width in pixels
    sigma_px_all = r_eff_kpc * SMOOTH_FACTOR / cell_size
    sigma_px_all = np.clip(sigma_px_all, 0.3, N / 8.0)

    if MULTISCALE_SMOOTH:
        grid = _multiscale_deposit(px, py, mass, sigma_px_all, ix, iy, valid, N)
    else:
        # Legacy: single global Gaussian with median σ
        grid = np.zeros((N, N), dtype=np.float64)
        np.add.at(grid, (iy[valid], ix[valid]), mass[valid])
        sigma_global = float(np.median(sigma_px_all[valid]))
        sigma_global = np.clip(sigma_global, 0.5, N / 10.0)
        grid = gaussian_filter(grid, sigma=sigma_global)

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

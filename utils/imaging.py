"""
Image rendering and saving utilities.
Converts a 2-D projection map → PNG at Zooniverse resolution.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import os

from config import (IMAGE_SIZE_PX, COLORMAP,
                    VMIN_PERCENTILE, VMAX_PERCENTILE, IMAGES_DIR)


def _normalise(image):
    """Clip to percentile range and scale to [0, 1]."""
    finite = image[np.isfinite(image)]
    if len(finite) == 0:
        return np.zeros_like(image)
    vmin = np.percentile(finite, VMIN_PERCENTILE)
    vmax = np.percentile(finite, VMAX_PERCENTILE)
    if vmax == vmin:
        return np.zeros_like(image)
    normed = np.clip((image - vmin) / (vmax - vmin), 0, 1)
    return normed


def projection_to_png(image, subhalo_id, halo_id,
                       aperture_kpc=None, suffix="", quality_check=False):
    """
    Render a projection map to a Zooniverse-ready PNG.

    Parameters
    ----------
    image       : (N, N) float array from projection.project_gas()
    subhalo_id  : int
    halo_id     : int
    aperture_kpc: float (for labelling only)
    suffix      : extra string appended to filename (e.g. "_x", "_y")
    quality_check: if True, also returns a dict of quality metrics

    Returns
    -------
    filepath : str path to saved PNG  (or None on failure)
    metrics  : dict (only if quality_check=True)
    """
    if image is None or not np.any(np.isfinite(image)):
        return (None, {}) if quality_check else None

    normed = _normalise(image)

    # Apply colormap → RGBA
    cmap = plt.get_cmap(COLORMAP)
    rgba = cmap(normed)                          # (N, N, 4) float [0,1]
    rgb_uint8 = (rgba[:, :, :3] * 255).astype(np.uint8)

    # Resize to Zooniverse standard
    pil_img = Image.fromarray(rgb_uint8, mode="RGB")
    pil_img = pil_img.resize((IMAGE_SIZE_PX, IMAGE_SIZE_PX), Image.LANCZOS)

    fname = f"halo{halo_id:06d}_sub{subhalo_id:07d}{suffix}.png"
    fpath = os.path.join(IMAGES_DIR, fname)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    pil_img.save(fpath, format="PNG", optimize=True)

    if quality_check:
        metrics = _quality_metrics(normed, image)
        return fpath, metrics
    return fpath


def _quality_metrics(normed, raw_image):
    """
    Return a dict of simple quality flags.
    Used by the quality-filter step to drop bad images.
    """
    finite = raw_image[np.isfinite(raw_image)]

    # Fraction of pixels above a low threshold (not pure background)
    signal_fraction = float(np.mean(normed > 0.05))

    # Dynamic range: difference between high and low percentiles in log space
    dynamic_range = float(np.percentile(finite, 99) - np.percentile(finite, 5)) \
                    if len(finite) > 0 else 0.0

    # Check image is not completely empty or flat
    is_empty   = signal_fraction < 0.01
    is_flat    = dynamic_range < 0.5     # less than 0.5 dex range → too flat

    # Asymmetry: galaxy off-centre gas → basic proxy for stripping
    # Compare mean brightness in inner 20% vs outer ring
    N = normed.shape[0]
    cx = cy = N // 2
    r = N // 5
    inner = normed[cy-r:cy+r, cx-r:cx+r].mean()
    outer = normed.mean()
    asymmetry_score = float(inner - outer)   # higher = more centrally concentrated

    return {
        "signal_fraction": round(signal_fraction, 4),
        "dynamic_range_dex": round(dynamic_range, 3),
        "asymmetry_score": round(asymmetry_score, 4),
        "is_empty": is_empty,
        "is_flat":  is_flat,
        "passed_qc": not (is_empty or is_flat),
    }

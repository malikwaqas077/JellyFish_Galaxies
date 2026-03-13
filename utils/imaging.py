"""
Image rendering and saving utilities.
Converts a 2-D projection map → PNG at target resolution.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
from PIL import Image
import os

from config import (IMAGE_SIZE_PX, COLORMAP,
                    VMIN_PERCENTILE, VMAX_PERCENTILE, IMAGES_DIR,
                    ARCSINH_STRETCH, ARCSINH_A)


def _normalise(image):
    """
    Clip to percentile range, scale to [0,1], then apply arcsinh stretch.

    arcsinh stretch:  f(x) = arcsinh(x/a) / arcsinh(1/a)
    Small a (e.g. 0.05) strongly boosts faint tails while compressing the
    bright core — same technique used by HST/JWST/SDSS imaging pipelines.
    """
    finite = image[np.isfinite(image)]
    if len(finite) == 0:
        return np.zeros_like(image)
    vmin = np.percentile(finite, VMIN_PERCENTILE)
    vmax = np.percentile(finite, VMAX_PERCENTILE)
    if vmax == vmin:
        return np.zeros_like(image)

    normed = np.clip((image - vmin) / (vmax - vmin), 0.0, 1.0)

    if ARCSINH_STRETCH:
        a     = float(ARCSINH_A)
        scale = float(np.arcsinh(1.0 / a))
        normed = np.arcsinh(normed / a) / scale
        normed = np.clip(normed, 0.0, 1.0)

    return normed


def projection_to_png(image, subhalo_id, halo_id,
                       aperture_kpc=None, suffix="", quality_check=False,
                       name_override=None):
    """
    Render a projection map to a PNG file.

    Parameters
    ----------
    image         : (N, N) float array from projection.project_gas()
    subhalo_id    : int
    halo_id       : int
    aperture_kpc  : float  (unused in rendering, kept for API compatibility)
    suffix        : str    appended to auto-generated filename
    quality_check : bool   if True, returns (filepath, metrics dict)
    name_override : str    if given, use this as the full output path
                           (bypasses the auto halo/sub filename pattern)

    Returns
    -------
    filepath : str | None
    metrics  : dict  (only when quality_check=True)
    """
    if image is None or not np.any(np.isfinite(image)):
        return (None, {}) if quality_check else None

    normed = _normalise(image)

    cmap      = plt.get_cmap(COLORMAP)
    rgba      = cmap(normed)
    rgb_uint8 = (rgba[:, :, :3] * 255).astype(np.uint8)
    pil_img   = Image.fromarray(rgb_uint8, mode="RGB")

    if pil_img.width != IMAGE_SIZE_PX or pil_img.height != IMAGE_SIZE_PX:
        pil_img = pil_img.resize((IMAGE_SIZE_PX, IMAGE_SIZE_PX), Image.LANCZOS)

    if name_override:
        fpath = name_override
    else:
        fname = f"halo{halo_id:06d}_sub{subhalo_id:07d}{suffix}.png"
        fpath = os.path.join(IMAGES_DIR, fname)

    os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
    pil_img.save(fpath, format="PNG", optimize=True)

    if quality_check:
        return fpath, _quality_metrics(normed, image)
    return fpath


def _quality_metrics(normed, raw_image):
    finite = raw_image[np.isfinite(raw_image)]

    signal_fraction = float(np.mean(normed > 0.05))
    dynamic_range   = float(np.percentile(finite, 99) - np.percentile(finite, 5)) \
                      if len(finite) > 0 else 0.0

    is_empty = signal_fraction < 0.01
    is_flat  = dynamic_range < 0.5

    N  = normed.shape[0]
    cx = cy = N // 2
    r  = N // 5
    inner           = normed[cy-r:cy+r, cx-r:cx+r].mean()
    asymmetry_score = float(inner - normed.mean())

    return {
        "signal_fraction":   round(signal_fraction, 4),
        "dynamic_range_dex": round(dynamic_range,   3),
        "asymmetry_score":   round(asymmetry_score, 4),
        "is_empty":          is_empty,
        "is_flat":           is_flat,
        "passed_qc":         not (is_empty or is_flat),
    }

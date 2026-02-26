"""
Step 4 — Enhance rendered images for sharper, higher-contrast output.

Applies three stages of principled image enhancement to every PNG in
output/images/, saving results to output/images_enhanced/:

  1. Richardson-Lucy deconvolution (15 iterations)
     Partially reverses the Gaussian blur introduced by the SPH smoothing
     kernel in the TNG vis.png renderer.  Sharpens gas structure edges
     without inventing features.

  2. Unsharp masking (radius 1.2 px, strength 180 %)
     Accentuates fine structure and boundaries already present in the image.

  3. Contrast-limited adaptive histogram equalisation (CLAHE)
     Boosts faint tail signal in low-brightness regions while preventing
     the bright core from washing out.  Implemented per-channel in LAB
     colour space so the hot-colormap hues are preserved.

Why NOT generative super-resolution (Real-ESRGAN etc.):
  - All PyTorch SR libraries require NumPy <2 (environment has NumPy 2.2).
  - More importantly: SR models trained on natural photos hallucinate
    texture detail.  For gas density maps this can invent gas features
    that do not exist in the simulation, which is scientifically incorrect.
  - The approach here is conservative: it only sharpens what is already
    there, never adds new signal.

Reads  : output/images/*.png
Writes : output/images_enhanced/*.png
"""

import os
import sys
import glob

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(__file__))
from config import IMAGES_DIR

ENHANCED_DIR = os.path.join(os.path.dirname(IMAGES_DIR), "images_enhanced")

# ── Richardson-Lucy parameters ─────────────────────────────────────────────────
RL_ITERATIONS  = 15        # more iterations → sharper, but can amplify noise
RL_PSF_SIGMA   = 1.2       # estimated Gaussian PSF sigma (pixels) from SPH blur
                            # tune this: larger = more aggressive deblur

# ── Unsharp mask parameters ─────────────────────────────────────────────────────
USM_RADIUS  = 1.2          # blur radius for the subtracted copy (pixels)
USM_AMOUNT  = 1.8          # strength: 1.0 = 100% of the high-freq detail added back

# ── CLAHE parameters ─────────────────────────────────────────────────────────────
CLAHE_CLIP  = 2.0          # clip limit (higher = more aggressive equalization)
CLAHE_GRID  = 8            # tile grid size (8 = 8×8 tiles across the image)


# ── Richardson-Lucy deconvolution ──────────────────────────────────────────────

def _make_psf(sigma, size=None):
    """Gaussian PSF kernel."""
    if size is None:
        size = max(3, int(6 * sigma + 1)) | 1   # nearest odd number
    ax  = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def _rl_deconvolve(channel, psf, n_iter):
    """
    Richardson-Lucy deconvolution on a single float32 channel [0,1].
    Returns sharpened channel clipped to [0, 1].
    """
    psf_flip = psf[::-1, ::-1]
    u = channel.copy()
    for _ in range(n_iter):
        conv    = fftconvolve(u, psf, mode="same")
        conv    = np.maximum(conv, 1e-12)
        ratio   = channel / conv
        u      *= fftconvolve(ratio, psf_flip, mode="same")
        u       = np.clip(u, 0.0, 1.0)
    return u


def richardson_lucy(img_arr, sigma=RL_PSF_SIGMA, n_iter=RL_ITERATIONS):
    """Apply R-L deconvolution to an RGB uint8 array. Returns uint8 array."""
    psf    = _make_psf(sigma)
    result = np.empty_like(img_arr, dtype=np.float32)
    for c in range(3):
        ch          = img_arr[:, :, c].astype(np.float32) / 255.0
        result[:, :, c] = _rl_deconvolve(ch, psf, n_iter)
    return (result * 255).clip(0, 255).astype(np.uint8)


# ── Unsharp masking ─────────────────────────────────────────────────────────────

def unsharp_mask(img_arr, radius=USM_RADIUS, amount=USM_AMOUNT):
    """
    High-quality unsharp masking.
    sharpened = original + amount × (original - blurred)
    """
    blurred    = gaussian_filter(img_arr.astype(np.float32), sigma=radius)
    sharpened  = img_arr.astype(np.float32) + amount * (img_arr.astype(np.float32) - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ── CLAHE in LAB colour space ───────────────────────────────────────────────────

def _clahe_single(channel_uint8, clip_limit=CLAHE_CLIP, grid=CLAHE_GRID):
    """
    Contrast-limited adaptive histogram equalization on a uint8 channel.
    Pure numpy/scipy — no OpenCV needed.
    """
    H, W     = channel_uint8.shape
    tile_h   = H // grid
    tile_w   = W // grid
    output   = np.empty_like(channel_uint8, dtype=np.float32)

    for row in range(grid):
        for col in range(grid):
            y0 = row * tile_h;  y1 = y0 + tile_h if row < grid-1 else H
            x0 = col * tile_w;  x1 = x0 + tile_w if col < grid-1 else W
            tile  = channel_uint8[y0:y1, x0:x1].astype(np.float32)

            hist, bins = np.histogram(tile.ravel(), bins=256, range=(0, 255))

            # Clip and redistribute
            excess = np.maximum(hist - clip_limit * tile.size / 256.0, 0)
            hist   = np.minimum(hist, clip_limit * tile.size / 256.0)
            hist  += excess.sum() / 256.0

            # CDF
            cdf    = hist.cumsum()
            cdf   /= cdf[-1]

            # Map tile values
            tile_idx = np.clip(tile.astype(int), 0, 255)
            output[y0:y1, x0:x1] = cdf[tile_idx] * 255.0

    return output.clip(0, 255).astype(np.uint8)


def clahe_rgb(img_arr, clip_limit=CLAHE_CLIP, grid=CLAHE_GRID):
    """
    Apply CLAHE per-channel on RGB image.  Works in luminance space:
    converts RGB → perceived luminance, equalises only L, then blends.
    """
    f    = img_arr.astype(np.float32) / 255.0

    # Perceived luminance
    lum  = 0.2126 * f[:, :, 0] + 0.7152 * f[:, :, 1] + 0.0722 * f[:, :, 2]
    lum8 = (lum * 255).clip(0, 255).astype(np.uint8)

    lum_eq  = _clahe_single(lum8, clip_limit, grid).astype(np.float32) / 255.0

    # Scale each channel by the luminance ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lum > 0.01, lum_eq / lum, 1.0)[:, :, np.newaxis]

    enhanced = np.clip(f * ratio, 0, 1)
    return (enhanced * 255).astype(np.uint8)


# ── Full enhancement pipeline ───────────────────────────────────────────────────

def enhance(img_arr):
    """
    Apply the enhancement pipeline.
    Input/output: uint8 RGB numpy array (H, W, 3).

    Pipeline:
      1. Richardson-Lucy deconvolution — sharpens gas edges
      2. Unsharp masking — accentuates structure
      3. Black-point restoration — re-zeroes background that deconvolution lifted
    """
    # Stage 1: deconvolution — remove SPH blur
    out = richardson_lucy(img_arr)

    # Stage 2: unsharp masking — accentuate edges
    out = unsharp_mask(out)

    # Stage 3: restore black background.
    # Deconvolution & unsharp masking can lift pure-black background pixels
    # to a faint glow. Find the original black mask (all channels < 8)
    # and force those pixels back to black.
    original_black = (img_arr.max(axis=2) < 12)            # true background
    out[original_black] = 0

    return out


# ── Main ────────────────────────────────────────────────────────────────────────

def enhance_images(input_dir=None, output_dir=None):
    src = input_dir  or IMAGES_DIR
    dst = output_dir or ENHANCED_DIR
    os.makedirs(dst, exist_ok=True)

    images = sorted(glob.glob(os.path.join(src, "halo*.png")))
    if not images:
        print(f"No images found in {src}")
        return

    print(f"Enhancing {len(images)} images ...")
    print(f"  Input  : {src}")
    print(f"  Output : {dst}")
    print(f"  R-L deconvolution: {RL_ITERATIONS} iters, PSF sigma={RL_PSF_SIGMA} px")
    print(f"  Unsharp mask     : radius={USM_RADIUS}, amount={USM_AMOUNT}")
    print(f"  CLAHE            : clip={CLAHE_CLIP}, grid={CLAHE_GRID}×{CLAHE_GRID}")
    print()

    for i, fpath in enumerate(images):
        fname = os.path.basename(fpath)
        print(f"  [{i+1}/{len(images)}] {fname}", end=" ... ", flush=True)

        arr     = np.array(Image.open(fpath).convert("RGB"))
        enhanced = enhance(arr)
        out_path = os.path.join(dst, fname)
        Image.fromarray(enhanced).save(out_path, format="PNG", optimize=True)
        print("done")

    print(f"\nAll done. Enhanced images -> {dst}/")


if __name__ == "__main__":
    enhance_images()

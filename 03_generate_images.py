"""
Step 3 — Download gas density images via the TNG vis.png API endpoint.
Rendering is done server-side; no HDF5 cutout downloads required.

The TNG vis.png endpoint returns a matplotlib figure using a jet-style colormap.
We:
  1. Auto-detect and crop the dark square data area (excludes axis labels, colorbar)
  2. Skip the top annotation rows (title, scale bar, z=0.0)
  3. Invert the jet HSV hue back to a scalar gas density
  4. Re-apply the "hot" colormap (black → red → orange → yellow → white)
  5. Resize to IMAGE_SIZE_PX and save

Key parameters (tune in constants below):
  VIS_SIZE_FACTOR : window half-width as a fraction of the host halo rViral
                    0.3 → ~300-600 physical kpc for cluster-mass halos
  TOP_SKIP_FRAC   : fraction of plot height to skip at top (annotations)

Reads  : output/data/galaxy_list.csv
Writes : output/images/*.png
         output/data/image_log.csv
"""

import csv
import io
import os
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from config import (SIMULATION, SNAPSHOT, DATA_DIR, IMAGES_DIR,
                    IMAGE_SIZE_PX, REQUEST_TIMEOUT, DOWNLOAD_RETRIES)
from utils.tng_api import get

GALAXY_LIST = os.path.join(DATA_DIR, "galaxy_list.csv")
IMAGE_LOG   = os.path.join(DATA_DIR, "image_log.csv")

# ── Visualisation parameters ───────────────────────────────────────────────────
VIS_SIZE_FACTOR = 0.3    # window half-width as fraction of host halo rViral

# Fraction of plot height to skip at top (covers title + z=0.0 + scale bar)
TOP_SKIP_FRAC = 0.09

# Polite rate-limiting between requests (seconds)
REQUEST_DELAY = 0.3

# Hot-colormap contrast percentiles
VMIN_PCT = 3.0
VMAX_PCT = 99.5


# ── URL builder ────────────────────────────────────────────────────────────────

def _vis_params():
    return {
        "partType":  "gas",
        "partField": "dens",
        "method":    "sphMap_subhalo",
        "size":      str(VIS_SIZE_FACTOR),
        "sizeType":  "rViral",
        "depthFac":  "1",
    }


def download_vis_png(subhalo_id):
    """Call the TNG vis.png endpoint and return a PIL Image, or None on failure."""
    url = (f"https://www.tng-project.org/api/{SIMULATION}"
           f"/snapshots/{SNAPSHOT}/subhalos/{subhalo_id}/vis.png")
    r = get(url, params=_vis_params())
    if r is None:
        return None
    try:
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


# ── Plot-area detection ────────────────────────────────────────────────────────

def _detect_plot_bounds(arr):
    """
    Auto-detect the dark square data area in a TNG vis.png figure.

    Works for any image size (1200×1200 for TNG300-1, 1400×1200 for TNG-Cluster,
    etc.) by scanning the centre row/column for the first dark pixel (brightness
    < 50 out of 255), then taking a square crop using the detected height.

    Returns (left, top, right, bottom) as a PIL-convention box (right/bottom
    are exclusive).
    """
    grey   = arr.mean(axis=2)          # (H, W)
    H, W   = grey.shape
    cx, cy = W // 2, H // 2

    top    = next(i for i, v in enumerate(grey[:, cx])    if v < 50)
    bottom = H - 1 - next(i for i, v in enumerate(grey[::-1, cx]) if v < 50)
    left   = next(i for i, v in enumerate(grey[cy, :])    if v < 50)

    # Use height as the side length to get a square (width may include a
    # colorbar that adds extra pixels to the right of the plot).
    side   = bottom - top
    right  = left + side

    return left, top, right + 1, bottom + 1   # +1 for exclusive PIL convention


# ── Jet → density → hot colormap ──────────────────────────────────────────────

def _extract_density_from_jet(arr_rgb):
    """
    Invert TNG's jet colormap back to a scalar density map [0, 1].

    Jet hue mapping (degrees):
      240° = blue  (lowest density / background)
      180° = cyan
      120° = green
      60°  = yellow
      0°   = red   (highest density)

    density = clip((240 - hue) / 240, 0, 1)
    Pixels darker than Cmax < 0.1 are treated as background (density = 0).
    """
    f = arr_rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]

    Cmax  = f.max(axis=2)
    Cmin  = f.min(axis=2)
    delta = Cmax - Cmin

    hue = np.zeros(Cmax.shape, dtype=np.float32)
    m   = delta > 0.0

    mr = m & (Cmax == r)
    mg = m & (Cmax == g)
    mb = m & (Cmax == b)

    hue[mr] = (60.0 * ((g[mr] - b[mr]) / delta[mr])) % 360.0
    hue[mg] =  60.0 * ((b[mg] - r[mg]) / delta[mg]) + 120.0
    hue[mb] =  60.0 * ((r[mb] - g[mb]) / delta[mb]) + 240.0

    density = np.clip((240.0 - hue) / 240.0, 0.0, 1.0)
    density[Cmax < 0.10] = 0.0   # true background pixels

    return density


def _apply_hot_colormap(density, vmin_pct=VMIN_PCT, vmax_pct=VMAX_PCT):
    """
    Normalise density with robust percentile clipping and apply the 'hot' colormap.
    Returns uint8 RGB array (H, W, 3).
    """
    signal = density[density > 0.01]
    if signal.size == 0:
        # blank image — return all-black
        return np.zeros((*density.shape, 3), dtype=np.uint8)

    vmin = float(np.percentile(signal, vmin_pct))
    vmax = float(np.percentile(signal, vmax_pct))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    normed = np.clip((density - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba   = plt.get_cmap("hot")(normed)          # (H, W, 4) float [0,1]
    return (rgba[:, :, :3] * 255).astype(np.uint8)


# ── Main image pipeline ────────────────────────────────────────────────────────

def crop_and_save(pil_img, subhalo_id, grnr):
    """
    1. Auto-detect the dark square plot area.
    2. Skip top annotation rows (title, scale bar, z annotation).
    3. Invert jet colormap → density.
    4. Apply 'hot' colormap for black-background, bright-core rendering.
    5. Resize to IMAGE_SIZE_PX and save.
    """
    arr = np.array(pil_img)
    box = _detect_plot_bounds(arr)           # (left, top, right, bottom)

    # Crop to the square plot area
    plot = arr[box[1]:box[3], box[0]:box[2]]   # (H, W, 3)
    H, W = plot.shape[:2]

    # Skip top annotation strip (title / scale bar / z=0.0)
    skip_rows = int(TOP_SKIP_FRAC * H)
    data_patch = plot[skip_rows:, :]

    # Invert jet hue → density → hot colormap
    density  = _extract_density_from_jet(data_patch)
    hot_rgb  = _apply_hot_colormap(density)

    final = Image.fromarray(hot_rgb).resize(
        (IMAGE_SIZE_PX, IMAGE_SIZE_PX), Image.LANCZOS
    )
    fname = f"halo{int(grnr):06d}_sub{int(subhalo_id):07d}_z.png"
    fpath = os.path.join(IMAGES_DIR, fname)
    final.save(fpath, format="PNG", optimize=True)
    return fpath, final


# ── Quality check ──────────────────────────────────────────────────────────────

def quality_metrics(pil_img):
    """
    Compute QC metrics on the final hot-colormap image.
    Returns dict with signal_fraction, dynamic_range_dex, asymmetry_score, passed_qc.
    """
    arr    = np.array(pil_img, dtype=float)       # (H, W, 3)
    grey   = arr.mean(axis=2) / 255.0             # (H, W), [0, 1]

    signal_fraction = float(np.mean(grey > 0.05))
    dynamic_range   = float(grey.max() - grey.min())

    is_empty = signal_fraction < 0.01
    is_flat  = dynamic_range  < 0.02

    N  = grey.shape[0]
    cx = cy = N // 2
    r  = N // 5
    inner           = grey[cy-r:cy+r, cx-r:cx+r].mean()
    asymmetry_score = float(inner - grey.mean())

    return {
        "signal_fraction":   round(signal_fraction, 4),
        "dynamic_range_dex": round(dynamic_range, 3),
        "asymmetry_score":   round(asymmetry_score, 4),
        "passed_qc":         not (is_empty or is_flat),
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def generate_images():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(GALAXY_LIST, "r") as f:
        galaxies = list(csv.DictReader(f))

    # Resume support: skip subhalos already logged
    already_done = set()
    if os.path.exists(IMAGE_LOG):
        with open(IMAGE_LOG, "r") as f:
            for r in csv.DictReader(f):
                already_done.add(r["subhalo_id"])

    log_fields = [
        "subhalo_id", "grnr", "axis", "image_path",
        "signal_fraction", "dynamic_range_dex", "asymmetry_score",
        "passed_qc", "error",
    ]
    log_mode = "a" if already_done else "w"
    log_fh   = open(IMAGE_LOG, log_mode, newline="")
    writer   = csv.DictWriter(log_fh, fieldnames=log_fields)
    if log_mode == "w":
        writer.writeheader()

    success = failed = skipped = 0
    print(f"Downloading API-rendered images for {len(galaxies)} galaxies ...")
    print(f"  (Already done: {len(already_done)}, will skip)")

    for gi, row in enumerate(galaxies):
        sub_id = row["subhalo_id"]
        grnr   = row["grnr"]
        prefix = f"[{gi+1}/{len(galaxies)}] sub={sub_id} grnr={grnr}"

        if sub_id in already_done:
            skipped += 1
            continue

        print(f"  {prefix}  requesting vis.png ...", end=" ", flush=True)
        try:
            pil_img = download_vis_png(sub_id)
            if pil_img is None:
                raise RuntimeError("API returned no image")

            fpath, final = crop_and_save(pil_img, sub_id, grnr)
            metrics = quality_metrics(final)

            tag = "OK" if metrics["passed_qc"] else "LOW_QUALITY"
            print(f"{tag}  sig={metrics['signal_fraction']:.3f}  "
                  f"DR={metrics['dynamic_range_dex']:.3f}  "
                  f"-> {os.path.basename(fpath)}")

            writer.writerow({
                "subhalo_id":        sub_id,
                "grnr":              grnr,
                "axis":              "z",
                "image_path":        fpath,
                "signal_fraction":   metrics["signal_fraction"],
                "dynamic_range_dex": metrics["dynamic_range_dex"],
                "asymmetry_score":   metrics["asymmetry_score"],
                "passed_qc":         metrics["passed_qc"],
                "error":             "",
            })
            if metrics["passed_qc"]:
                success += 1
            else:
                failed += 1

        except Exception:
            tb = traceback.format_exc()
            print("FAILED")
            print(f"    {tb.splitlines()[-1]}")
            writer.writerow({
                "subhalo_id":        sub_id,
                "grnr":              grnr,
                "axis":              "z",
                "image_path":        "",
                "signal_fraction":   "",
                "dynamic_range_dex": "",
                "asymmetry_score":   "",
                "passed_qc":         False,
                "error":             "download_failed",
            })
            failed += 1

        time.sleep(REQUEST_DELAY)  # polite rate-limiting

    log_fh.close()
    print(f"\n{'='*60}")
    print(f"  Done.  QC-passed: {success}  Low-quality/failed: {failed}  Skipped: {skipped}")
    print(f"  Images -> {IMAGES_DIR}/")
    print(f"  Log    -> {IMAGE_LOG}")


if __name__ == "__main__":
    generate_images()

"""
Step 3 — Download gas density images via the TNG vis.png API endpoint.

The vis.png endpoint renders ALL gas in the aperture (bound + ICM + stripped tails)
using server-side SPH smoothing — perfect for jellyfish morphology.

Key insight: only sizeType='rViral' is respected by the TNG API.
We compute a per-galaxy size_factor = APERTURE_KPC / halo_r200_kpc so that
every galaxy gets a fixed ~200 kpc physical half-width window regardless
of how massive its host cluster is.

Post-processing:
  1. Auto-detect the dark square plot area and crop it
  2. Skip the top annotation strip (title, scale bar, z=0.0)
  3. Invert TNG's jet HSV hue → scalar gas density
  4. Re-apply the 'hot' colormap (black→red→orange→yellow→white)
  5. Resize to IMAGE_SIZE_PX and save

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
                    IMAGE_SIZE_PX, LITTLE_H, REQUEST_TIMEOUT, DOWNLOAD_RETRIES)
from utils.tng_api import get

GALAXY_LIST = os.path.join(DATA_DIR, "galaxy_list.csv")
IMAGE_LOG   = os.path.join(DATA_DIR, "image_log.csv")

# Physical half-width of the projection window (kpc).
# The per-galaxy size_factor = APERTURE_KPC / halo_r200_kpc is passed as
# the 'size' parameter with sizeType='rViral', giving a fixed physical window.
APERTURE_KPC = 200.0

# Fraction of plot height to skip at top (title / scale bar / z=0.0 annotation)
TOP_SKIP_FRAC = 0.09

# Hot-colormap contrast percentiles (signal pixels only)
VMIN_PCT = 2.0
VMAX_PCT = 99.5

# Polite rate-limiting
REQUEST_DELAY = 0.3


# ── vis.png download ───────────────────────────────────────────────────────────

def _vis_params(size_factor):
    """
    Build vis.png request params.
    size_factor = APERTURE_KPC / halo_r200_kpc ensures a fixed physical window.
    """
    return {
        "partType":  "gas",
        "partField": "dens",
        "method":    "sphMap_subhalo",
        "size":      f"{size_factor:.6f}",
        "sizeType":  "rViral",
        "depthFac":  "1",
    }


def download_vis_png(subhalo_id, size_factor):
    """Call the TNG vis.png endpoint and return a PIL Image, or None on failure."""
    url = (f"https://www.tng-project.org/api/{SIMULATION}"
           f"/snapshots/{SNAPSHOT}/subhalos/{subhalo_id}/vis.png")
    r = get(url, params=_vis_params(size_factor))
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
    Scans the centre row/column for the first dark pixel (brightness < 50).
    Returns (left, top, right, bottom) — PIL-convention exclusive bounds.
    """
    grey   = arr.mean(axis=2)
    H, W   = grey.shape
    cx, cy = W // 2, H // 2

    top    = next(i for i, v in enumerate(grey[:, cx])       if v < 50)
    bottom = H - 1 - next(i for i, v in enumerate(grey[::-1, cx]) if v < 50)
    left   = next(i for i, v in enumerate(grey[cy, :])       if v < 50)

    side  = bottom - top      # use height to get a square (excludes colorbar)
    right = left + side

    return left, top, right + 1, bottom + 1


# ── Jet colormap inversion ─────────────────────────────────────────────────────

def _extract_density_from_jet(arr_rgb):
    """
    Invert TNG's jet colormap back to a scalar density map [0, 1].

    Jet hue: 240° (blue/background) → 0° (red/dense).
    density = clip((240 - hue_deg) / 240, 0, 1).
    Pixels with Cmax < 0.10 (very dark) are treated as background (density=0).
    """
    f     = arr_rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    Cmax  = f.max(axis=2)
    delta = Cmax - f.min(axis=2)

    hue = np.zeros(Cmax.shape, dtype=np.float32)
    m   = delta > 0.0
    mr  = m & (Cmax == r)
    mg  = m & (Cmax == g)
    mb  = m & (Cmax == b)

    hue[mr] = (60.0 * ((g[mr] - b[mr]) / delta[mr])) % 360.0
    hue[mg] =  60.0 * ((b[mg] - r[mg]) / delta[mg]) + 120.0
    hue[mb] =  60.0 * ((r[mb] - g[mb]) / delta[mb]) + 240.0

    density = np.clip((240.0 - hue) / 240.0, 0.0, 1.0)
    density[Cmax < 0.10] = 0.0
    return density


def _apply_hot_colormap(density):
    """Percentile-normalise density and apply the 'hot' colormap → uint8 RGB."""
    signal = density[density > 0.01]
    if signal.size == 0:
        return np.zeros((*density.shape, 3), dtype=np.uint8)
    vmin = float(np.percentile(signal, VMIN_PCT))
    vmax = float(np.percentile(signal, VMAX_PCT))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    normed = np.clip((density - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba   = plt.get_cmap("hot")(normed)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


# ── Main crop + render ─────────────────────────────────────────────────────────

def crop_and_save(pil_img, subhalo_id, grnr):
    """
    1. Auto-detect the square plot area and crop.
    2. Skip top annotation strip.
    3. Invert jet → density → hot colormap.
    4. Resize to IMAGE_SIZE_PX and save.
    """
    arr = np.array(pil_img)
    box = _detect_plot_bounds(arr)

    plot = arr[box[1]:box[3], box[0]:box[2]]
    H    = plot.shape[0]

    skip = int(TOP_SKIP_FRAC * H)
    data = plot[skip:, :].copy()

    # Erase bottom-left annotation (simulation name label) by painting
    # it to the TNG plot background colour (dark blue) before jet inversion.
    # The dark blue hue (≈240°) maps to density≈0 → black after inversion.
    bot_h = int(0.97 * data.shape[0])
    data[bot_h:, : data.shape[1] // 2] = (10, 10, 60)

    density = _extract_density_from_jet(data)
    hot_rgb = _apply_hot_colormap(density)

    final = Image.fromarray(hot_rgb).resize(
        (IMAGE_SIZE_PX, IMAGE_SIZE_PX), Image.LANCZOS
    )
    fname = f"halo{int(grnr):06d}_sub{int(subhalo_id):07d}_z.png"
    fpath = os.path.join(IMAGES_DIR, fname)
    final.save(fpath, format="PNG", optimize=True)
    return fpath, final


# ── Quality metrics ────────────────────────────────────────────────────────────

def quality_metrics(pil_img):
    arr  = np.array(pil_img, dtype=float)
    grey = arr.mean(axis=2) / 255.0

    signal_fraction = float(np.mean(grey > 0.05))
    dynamic_range   = float(grey.max() - grey.min())
    is_empty = signal_fraction < 0.01
    is_flat  = dynamic_range  < 0.02

    N  = grey.shape[0]; cx = cy = N // 2; r = N // 5
    asymmetry_score = float(grey[cy-r:cy+r, cx-r:cx+r].mean() - grey.mean())

    return {
        "signal_fraction":   round(signal_fraction, 4),
        "dynamic_range_dex": round(dynamic_range, 3),
        "asymmetry_score":   round(asymmetry_score, 4),
        "passed_qc":         not (is_empty or is_flat),
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def generate_images():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)

    with open(GALAXY_LIST, "r") as f:
        galaxies = list(csv.DictReader(f))

    already_done = set()
    if os.path.exists(IMAGE_LOG):
        with open(IMAGE_LOG, "r") as f:
            for r in csv.DictReader(f):
                already_done.add(r["subhalo_id"])

    log_fields = [
        "subhalo_id", "grnr", "axis", "image_path",
        "aperture_kpc", "size_factor",
        "signal_fraction", "dynamic_range_dex", "asymmetry_score",
        "passed_qc", "error",
    ]
    log_mode = "a" if already_done else "w"
    log_fh   = open(IMAGE_LOG, log_mode, newline="")
    writer   = csv.DictWriter(log_fh, fieldnames=log_fields)
    if log_mode == "w":
        writer.writeheader()

    success = failed = skipped = 0
    print(f"Downloading vis.png images for {len(galaxies)} galaxies ...")
    print(f"  Fixed aperture: {APERTURE_KPC} kpc half-width (dynamic rViral factor per galaxy)")
    print(f"  Already done: {len(already_done)}, will skip")

    for gi, row in enumerate(galaxies):
        sub_id = row["subhalo_id"]
        grnr   = row["grnr"]
        prefix = f"[{gi+1}/{len(galaxies)}] sub={sub_id} grnr={grnr}"

        if sub_id in already_done:
            skipped += 1
            continue

        # Compute dynamic size factor for fixed physical aperture
        try:
            r200 = float(row["halo_r200_kpc"])
            size_factor = APERTURE_KPC / r200 if r200 > 0 else 0.09
        except (KeyError, ValueError):
            size_factor = 0.09

        print(f"  {prefix}  size={size_factor:.4f} requesting vis.png ...",
              end=" ", flush=True)
        try:
            pil_img = download_vis_png(sub_id, size_factor)
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
                "aperture_kpc":      round(APERTURE_KPC, 1),
                "size_factor":       round(size_factor, 5),
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
                "aperture_kpc":      "",
                "size_factor":       "",
                "signal_fraction":   "",
                "dynamic_range_dex": "",
                "asymmetry_score":   "",
                "passed_qc":         False,
                "error":             "download_failed",
            })
            failed += 1

        time.sleep(REQUEST_DELAY)

    log_fh.close()
    print(f"\n{'='*60}")
    print(f"  Done.  QC-passed: {success}  Low-quality/failed: {failed}  Skipped: {skipped}")
    print(f"  Images -> {IMAGES_DIR}/")
    print(f"  Log    -> {IMAGE_LOG}")


if __name__ == "__main__":
    generate_images()

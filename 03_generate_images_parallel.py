"""
Step 3 — Download gas density images via the TNG vis.png API endpoint (PARALLEL VERSION).
Uses multiprocessing to download 8 images simultaneously for 8x speedup.
"""

import csv
import io
import os
import sys
import time
import traceback
from multiprocessing import Pool, Manager
from functools import partial

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

APERTURE_KPC = 200.0
TOP_SKIP_FRAC = 0.09
REQUEST_DELAY = 0.1  # Reduced delay for parallel processing

# Number of parallel workers
NUM_WORKERS = 8


def _vis_params(size_factor):
    """Build vis.png request params."""
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


def _detect_plot_bounds(arr):
    """Auto-detect the dark square data area in a TNG vis.png figure."""
    grey   = arr.mean(axis=2)
    H, W   = grey.shape
    cx, cy = W // 2, H // 2

    top    = next(i for i, v in enumerate(grey[:, cx])       if v < 50)
    bottom = H - 1 - next(i for i, v in enumerate(grey[::-1, cx]) if v < 50)
    left   = next(i for i, v in enumerate(grey[cy, :])       if v < 50)

    side  = bottom - top
    right = left + side

    return left, top, right + 1, bottom + 1


def crop_and_save(pil_img, subhalo_id, grnr):
    """Crop, clean, and save image."""
    arr = np.array(pil_img)
    box = _detect_plot_bounds(arr)

    plot = arr[box[1]:box[3], box[0]:box[2]]
    H    = plot.shape[0]

    skip = int(TOP_SKIP_FRAC * H)
    data = plot[skip:, :].copy()

    bot_h = int(0.97 * data.shape[0])
    data[bot_h:, : data.shape[1] // 2] = (0, 0, 128)

    final = Image.fromarray(data).resize(
        (IMAGE_SIZE_PX, IMAGE_SIZE_PX), Image.LANCZOS
    )
    fname = f"halo{int(grnr):06d}_sub{int(subhalo_id):07d}_z.png"
    fpath = os.path.join(IMAGES_DIR, fname)
    final.save(fpath, format="PNG", optimize=True)
    return fpath, final


def quality_metrics(pil_img):
    """Compute quality metrics for an image."""
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


def process_galaxy(row, already_done):
    """Process a single galaxy (for parallel execution)."""
    sub_id = row["subhalo_id"]
    grnr   = row["grnr"]
    
    if sub_id in already_done:
        return None
    
    try:
        r200 = float(row["halo_r200_kpc"])
        size_factor = APERTURE_KPC / r200 if r200 > 0 else 0.09
    except (KeyError, ValueError):
        size_factor = 0.09
    
    try:
        time.sleep(REQUEST_DELAY)  # Polite rate limiting
        pil_img = download_vis_png(sub_id, size_factor)
        if pil_img is None:
            raise RuntimeError("API returned no image")
        
        fpath, final = crop_and_save(pil_img, sub_id, grnr)
        metrics = quality_metrics(final)
        
        return {
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
            "status":            "OK" if metrics["passed_qc"] else "LOW_QUALITY",
        }
    except Exception as e:
        return {
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
            "status":            "FAILED",
        }


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
    print(f"  Fixed aperture: {APERTURE_KPC} kpc half-width")
    print(f"  Already done: {len(already_done)}, will skip")
    print(f"  Using {NUM_WORKERS} parallel workers for faster processing")

    # Filter out already processed galaxies
    pending = [g for g in galaxies if g["subhalo_id"] not in already_done]
    skipped = len(already_done)
    
    # Process in parallel
    with Pool(processes=NUM_WORKERS) as pool:
        process_fn = partial(process_galaxy, already_done=already_done)
        
        for i, result in enumerate(pool.imap_unordered(process_fn, pending)):
            if result is None:
                continue
            
            writer.writerow({k: result[k] for k in log_fields})
            log_fh.flush()
            
            status = result["status"]
            sub_id = result["subhalo_id"]
            grnr = result["grnr"]
            
            if status == "OK":
                success += 1
                sig = result["signal_fraction"]
                dr = result["dynamic_range_dex"]
                fname = os.path.basename(result["image_path"])
                print(f"  [{i+1}/{len(pending)}] sub={sub_id} grnr={grnr}  "
                      f"OK  sig={sig:.3f}  DR={dr:.3f}  -> {fname}")
            elif status == "LOW_QUALITY":
                failed += 1
                print(f"  [{i+1}/{len(pending)}] sub={sub_id} grnr={grnr}  LOW_QUALITY")
            else:
                failed += 1
                print(f"  [{i+1}/{len(pending)}] sub={sub_id} grnr={grnr}  FAILED")

    log_fh.close()
    print(f"\n{'='*60}")
    print(f"  Done.  QC-passed: {success}  Low-quality/failed: {failed}  Skipped: {skipped}")
    print(f"  Images -> {IMAGES_DIR}/")
    print(f"  Log    -> {IMAGE_LOG}")


if __name__ == "__main__":
    generate_images()

"""
Step 3 — Generate high-quality 1024×1024 images from HDF5 cutouts.

Reads each HDF5 cutout from output/cutouts/{subhalo_id}.hdf5, runs the
multi-scale Voronoi projection, and saves a 1024px PNG using arcsinh stretch.

Skips subhalos with fewer than MIN_GAS_CELLS cells (too sparse to render).
Skips images that already exist on disk.

Output images:  output/images/TNG-Cluster_{subhalo_id:06d}.png
Image log:      output/data/image_log.csv

Usage:
    python 03_generate_hd_images.py              # all cutouts
    python 03_generate_hd_images.py --workers 4  # override worker count
"""

import argparse
import csv
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR, CUTOUTS_DIR, IMAGES_DIR,
    LITTLE_H, APERTURE_KPC, MIN_GAS_CELLS,
    PARALLEL_WORKERS, IMAGE_NAME_CLUSTER, IMAGE_ID_PAD,
)
from utils.projection import project_gas
from utils.imaging import projection_to_png

CATALOG_CSV  = os.path.join(DATA_DIR, "subhalo_catalog.csv")
IMAGE_LOG    = os.path.join(DATA_DIR, "image_log.csv")
LOG_FIELDS   = [
    "subhalo_id", "grnr", "gas_category",
    "image_path", "aperture_kpc", "n_gas_cells",
    "signal_fraction", "dynamic_range_dex", "asymmetry_score",
    "passed_qc", "error",
]


def output_image_path(subhalo_id):
    fname = f"{IMAGE_NAME_CLUSTER}_{subhalo_id:0{IMAGE_ID_PAD}d}.png"
    return os.path.join(IMAGES_DIR, fname)


def dynamic_aperture(halfmassrad_gas_ckpc_h):
    """Scale aperture to the galaxy's gas extent, capped at 350 kpc."""
    r_half_kpc = float(halfmassrad_gas_ckpc_h) / LITTLE_H
    return float(np.clip(max(APERTURE_KPC, 3.0 * r_half_kpc), APERTURE_KPC, 350.0))


def load_cutout(subhalo_id):
    """
    Read gas cell data from HDF5 cutout.
    Returns (coords, masses, densities, n_cells) or raises on error.
    """
    path = os.path.join(CUTOUTS_DIR, f"{subhalo_id}.hdf5")
    with h5py.File(path, "r") as f:
        pt0 = f["PartType0"]
        coords  = pt0["Coordinates"][:]   # ckpc/h
        masses  = pt0["Masses"][:]        # 1e10 M_sun/h
        density = pt0["Density"][:]       # 1e10 M_sun/h / (ckpc/h)^3
    return coords, masses, density, len(masses)


def render_one(task):
    """
    Worker function: render a single subhalo to PNG.
    task = dict with catalog row fields.
    Returns a log dict.
    """
    sid     = int(task["subhalo_id"])
    grnr    = task["grnr"]
    cat     = task["gas_category"]
    r_half  = float(task.get("halfmassrad_gas_ckpc_h", 0) or 0)
    centre  = np.array([
        float(task["pos_x_ckpc_h"]),
        float(task["pos_y_ckpc_h"]),
        float(task["pos_z_ckpc_h"]),
    ])

    base_log = {"subhalo_id": sid, "grnr": grnr, "gas_category": cat,
                "image_path": "", "aperture_kpc": "", "n_gas_cells": 0,
                "signal_fraction": "", "dynamic_range_dex": "",
                "asymmetry_score": "", "passed_qc": False, "error": ""}

    # Already rendered?
    img_path = output_image_path(sid)
    if os.path.exists(img_path):
        return {**base_log, "image_path": img_path,
                "passed_qc": True, "error": "already_exists"}

    cutout = os.path.join(CUTOUTS_DIR, f"{sid}.hdf5")
    if not os.path.exists(cutout):
        return {**base_log, "error": "no_cutout"}

    try:
        coords, masses, density, n_cells = load_cutout(sid)
    except Exception as e:
        return {**base_log, "error": f"hdf5_read_error: {e}"}

    if n_cells < MIN_GAS_CELLS:
        return {**base_log, "n_gas_cells": n_cells, "error": "too_few_cells"}

    aperture = dynamic_aperture(r_half)

    try:
        image, extent = project_gas(
            coords, masses, density, centre,
            axis="z", a=1.0, aperture_kpc=aperture,
        )
    except Exception as e:
        return {**base_log, "n_gas_cells": n_cells,
                "aperture_kpc": round(aperture, 1),
                "error": f"projection_error: {e}"}

    if image is None:
        return {**base_log, "n_gas_cells": n_cells,
                "aperture_kpc": round(aperture, 1), "error": "empty_projection"}

    try:
        fpath, metrics = projection_to_png(
            image, sid, int(grnr),
            aperture_kpc=aperture,
            suffix="",
            quality_check=True,
            name_override=output_image_path(sid),
        )
    except Exception as e:
        return {**base_log, "n_gas_cells": n_cells,
                "aperture_kpc": round(aperture, 1),
                "error": f"render_error: {e}"}

    return {
        "subhalo_id":        sid,
        "grnr":              grnr,
        "gas_category":      cat,
        "image_path":        fpath or "",
        "aperture_kpc":      round(aperture, 1),
        "n_gas_cells":       n_cells,
        "signal_fraction":   metrics.get("signal_fraction", ""),
        "dynamic_range_dex": metrics.get("dynamic_range_dex", ""),
        "asymmetry_score":   metrics.get("asymmetry_score", ""),
        "passed_qc":         metrics.get("passed_qc", False),
        "error":             "",
    }


def run(n_workers):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(CATALOG_CSV):
        print(f"ERROR: {CATALOG_CSV} not found. Run 01_catalog_subhalos.py first.")
        sys.exit(1)

    with open(CATALOG_CSV) as f:
        catalog = list(csv.DictReader(f))
    print(f"Catalog: {len(catalog)} subhalos.")

    # Resume: skip already-logged
    logged_ids = set()
    if os.path.exists(IMAGE_LOG):
        with open(IMAGE_LOG) as f:
            for row in csv.DictReader(f):
                if row.get("error") not in ("hdf5_read_error", "projection_error",
                                            "render_error", "no_cutout"):
                    logged_ids.add(int(row["subhalo_id"]))
        print(f"Already processed: {len(logged_ids)}.  Resuming …")

    pending = [r for r in catalog if int(r["subhalo_id"]) not in logged_ids]
    print(f"Pending:           {len(pending)}")
    if not pending:
        print("All images already generated.")
        return

    log_mode = "a" if logged_ids else "w"
    log_fh   = open(IMAGE_LOG, log_mode, newline="")
    writer   = csv.DictWriter(log_fh, fieldnames=LOG_FIELDS)
    if log_mode == "w":
        writer.writeheader()

    ok = failed = skipped = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(render_one, row): row for row in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                result = fut.result()
            except Exception as e:
                result = {"subhalo_id": "?", "grnr": "?", "gas_category": "?",
                          "image_path": "", "aperture_kpc": "", "n_gas_cells": 0,
                          "signal_fraction": "", "dynamic_range_dex": "",
                          "asymmetry_score": "", "passed_qc": False,
                          "error": f"worker_crash: {e}"}
                failed += 1

            writer.writerow(result)
            log_fh.flush()

            err = result.get("error", "")
            if err == "already_exists":
                skipped += 1
            elif result.get("passed_qc"):
                ok += 1
            else:
                failed += 1

            print(f"  [{i}/{len(pending)}] sub={result['subhalo_id']:6} "
                  f"cat={result.get('gas_category','?'):6s} "
                  f"cells={result.get('n_gas_cells',0):5d} "
                  f"qc={'OK' if result.get('passed_qc') else 'NO':2s} "
                  f"{err or 'rendered'}", flush=True)

    log_fh.close()

    print(f"\n{'='*50}")
    print(f"Image generation complete.")
    print(f"  QC-passed: {ok}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed/low-quality: {failed}")
    print(f"  Images at: {IMAGES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=PARALLEL_WORKERS)
    args = parser.parse_args()
    run(args.workers)

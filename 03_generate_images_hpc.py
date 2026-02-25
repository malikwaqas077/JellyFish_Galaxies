"""
Step 3 (HPC / LOCAL mode) — Read TNG data directly from filesystem.

Use this script when you have access to the TNG data files locally
(e.g. on MPCDF, your university HPC cluster, or after bulk download).

Prerequisites:
    pip install illustris_python

Directory structure expected:
    TNG_DATA_BASE_PATH/
        TNG300-1/output/
            snapshots/snap_099.*
            groups_099/
                fof_subhalo_tab_099.*

Set TNG_DATA_PATH in config.py (or export TNG_DATA_PATH env variable).

Reads  : output/data/galaxy_list.csv
Writes : output/images/*.png
         output/data/image_log.csv  (same format as API version)
"""

import csv
import numpy as np
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(__file__))
from config import (DATA_DIR, IMAGES_DIR, LITTLE_H, APERTURE_KPC)
from utils.projection import project_gas
from utils.imaging import projection_to_png

GALAXY_LIST  = os.path.join(DATA_DIR, "galaxy_list.csv")
IMAGE_LOG    = os.path.join(DATA_DIR, "image_log.csv")

# ── Local data path ────────────────────────────────────────────────────────
# Set this to the base path of your TNG installation, e.g.:
# /virgo/simulations/IllustrisTNG/TNG300-1/output
TNG_DATA_PATH = os.environ.get(
    "TNG_DATA_PATH",
    "/path/to/TNG300-1/output"    # <-- edit this for your cluster
)
SNAPSHOT = 99

PROJ_AXES = ["z"]   # add "x", "y" for 3x more images

# ── How much radius to read around each galaxy (in ckpc/h) ────────────────
READ_RADIUS_CKPC_H = 200    # physical 200/h ~ 295 kpc


def read_gas_sphere(subhalo_id, centre_ckpc_h, radius_ckpc_h):
    """
    Load gas cells within a sphere of radius_ckpc_h around centre_ckpc_h.
    Returns (coords, masses, densities) or (None, None, None).

    Uses illustris_python for efficient particle reading.
    """
    try:
        import illustris_python as il

        # Load the gas subset for this subhalo (SUBFIND-bound particles)
        gas = il.snapshot.loadSubhalo(
            TNG_DATA_PATH, SNAPSHOT, subhalo_id, "gas",
            fields=["Coordinates", "Masses", "Density"],
        )
        if gas["count"] == 0:
            return None, None, None

        coords  = gas["Coordinates"]   # ckpc/h, comoving
        masses  = gas["Masses"]        # 1e10 M_sun/h
        density = gas["Density"]       # 1e10 M_sun/h / (ckpc/h)^3

        # Optional: also load gas from the surrounding halo (for tails)
        # This can double the number of cells but captures the tail gas
        try:
            halo_gas = il.snapshot.loadHalo(
                TNG_DATA_PATH, SNAPSHOT,
                gas.get("ParentID", 0),    # group number
                "gas",
                fields=["Coordinates", "Masses", "Density"],
            )
            if halo_gas["count"] > 0:
                # Filter to sphere only
                rel = halo_gas["Coordinates"] - centre_ckpc_h
                r   = np.linalg.norm(rel, axis=1)
                mask = r < radius_ckpc_h
                if mask.sum() > 0:
                    coords  = np.vstack([coords,  halo_gas["Coordinates"][mask]])
                    masses  = np.hstack([masses,  halo_gas["Masses"][mask]])
                    density = np.hstack([density, halo_gas["Density"][mask]])
        except Exception:
            pass    # halo gas is optional

        return coords, masses, density

    except ImportError:
        raise RuntimeError(
            "illustris_python not installed. Run: pip install illustris_python"
        )
    except Exception as e:
        print(f"    ERROR reading subhalo {subhalo_id}: {e}")
        return None, None, None


def dynamic_aperture_kpc(row):
    r_half_ckpc_h = float(row.get("halfmassrad_gas_ckpc_h", 0))
    r_half_kpc    = r_half_ckpc_h / LITTLE_H
    aperture = max(APERTURE_KPC, 3.0 * r_half_kpc)
    return min(aperture, 350.0)


def galaxy_centre(row):
    return np.array([
        float(row["pos_x_ckpc_h"]),
        float(row["pos_y_ckpc_h"]),
        float(row["pos_z_ckpc_h"]),
    ])


def generate_images_hpc():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    with open(GALAXY_LIST, "r") as f:
        galaxies = list(csv.DictReader(f))

    already_done = set()
    if os.path.exists(IMAGE_LOG):
        with open(IMAGE_LOG, "r") as f:
            for r in csv.DictReader(f):
                already_done.add((r["subhalo_id"], r["axis"]))

    log_fields = [
        "subhalo_id", "grnr", "axis", "image_path",
        "aperture_kpc", "signal_fraction", "dynamic_range_dex",
        "asymmetry_score", "passed_qc", "error",
    ]
    log_mode = "a" if already_done else "w"
    log_fh   = open(IMAGE_LOG, log_mode, newline="")
    writer   = csv.DictWriter(log_fh, fieldnames=log_fields)
    if log_mode == "w":
        writer.writeheader()

    success = failed = skipped = 0
    print(f"HPC mode: reading from {TNG_DATA_PATH}")
    print(f"Rendering {len(galaxies)} galaxies ...")

    for gi, row in enumerate(galaxies):
        sub_id  = int(row["subhalo_id"])
        grnr    = row["grnr"]
        centre  = galaxy_centre(row)
        prefix  = f"[{gi+1}/{len(galaxies)}] sub={sub_id}"
        print(f"  {prefix}", end=" ... ", flush=True)

        coords, masses, density = read_gas_sphere(
            sub_id, centre, READ_RADIUS_CKPC_H)

        if coords is None:
            print("no gas")
            for axis in PROJ_AXES:
                if (str(sub_id), axis) not in already_done:
                    writer.writerow({
                        "subhalo_id": sub_id, "grnr": grnr, "axis": axis,
                        "image_path": "", "aperture_kpc": "", "signal_fraction": "",
                        "dynamic_range_dex": "", "asymmetry_score": "",
                        "passed_qc": False, "error": "no_gas",
                    })
            continue
        print(f"{len(masses)} cells", end=" ")

        aperture = dynamic_aperture_kpc(row)

        for axis in PROJ_AXES:
            if (str(sub_id), axis) in already_done:
                skipped += 1
                continue
            try:
                image, extent = project_gas(
                    coords, masses, density, centre,
                    axis=axis, a=1.0, aperture_kpc=aperture)

                fpath, metrics = projection_to_png(
                    image, sub_id, int(grnr),
                    aperture_kpc=aperture, suffix=f"_{axis}",
                    quality_check=True)

                qc = metrics.get("passed_qc", False)
                print(f"  axis={axis} {'OK' if (fpath and qc) else 'LOW_Q'}")
                writer.writerow({
                    "subhalo_id":       sub_id, "grnr": grnr, "axis": axis,
                    "image_path":       fpath or "",
                    "aperture_kpc":     round(aperture, 1),
                    "signal_fraction":  metrics.get("signal_fraction", ""),
                    "dynamic_range_dex":metrics.get("dynamic_range_dex", ""),
                    "asymmetry_score":  metrics.get("asymmetry_score", ""),
                    "passed_qc":        metrics.get("passed_qc", False),
                    "error":            "",
                })
                success += 1 if (fpath and qc) else 0
                failed  += 0 if (fpath and qc) else 1
            except Exception:
                print(f"  axis={axis} RENDER ERROR")
                writer.writerow({
                    "subhalo_id": sub_id, "grnr": grnr, "axis": axis,
                    "image_path": "", "aperture_kpc": round(aperture, 1),
                    "signal_fraction": "", "dynamic_range_dex": "",
                    "asymmetry_score": "", "passed_qc": False,
                    "error": "render_error",
                })
                failed += 1

    log_fh.close()
    print(f"\nDone. QC-passed: {success}  Failed: {failed}  Skipped: {skipped}")


if __name__ == "__main__":
    if not os.path.exists(TNG_DATA_PATH):
        print(f"ERROR: TNG_DATA_PATH not found: {TNG_DATA_PATH}")
        print("Set env variable: export TNG_DATA_PATH=/path/to/TNG300-1/output")
        sys.exit(1)
    generate_images_hpc()

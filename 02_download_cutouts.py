"""
Step 2 — Download HDF5 cutouts for cataloged subhalos.

Reads subhalo_catalog.csv and downloads the gas particle data
(Coordinates, Masses, Density) for subhalos that pass the gas quality gate:
  - gas_category must NOT be in SKIP_GAS_CATEGORIES (default: skips NONE)
  - mass_gas_1e10msun_h >= MIN_DOWNLOAD_GAS_MASS (default: 0.05)

This prevents wasting bandwidth and disk space on gas-stripped satellites
that would produce blank images.

Cutout files are saved as:
  output/cutouts/{subhalo_id}.hdf5

Writes a download log:  output/data/download_log.csv
"""

import csv
import os
import sys
import time
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATA_DIR, CUTOUTS_DIR, HDF5_GAS_FIELDS,
    PARALLEL_WORKERS, MIN_DOWNLOAD_GAS_MASS, SKIP_GAS_CATEGORIES,
)
from utils.tng_api import download_file

CATALOG_CSV  = os.path.join(DATA_DIR, "subhalo_catalog.csv")
DOWNLOAD_LOG = os.path.join(DATA_DIR, "download_log.csv")
LOG_FIELDS   = ["subhalo_id", "gas_category", "status", "file_path",
                "file_size_kb", "error"]


def cutout_path(subhalo_id):
    return os.path.join(CUTOUTS_DIR, f"{subhalo_id}.hdf5")


def verify_hdf5(path):
    """Return True if file exists and is a valid non-empty HDF5."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with h5py.File(path, "r") as f:
            return "PartType0" in f
    except Exception:
        return False


def download_one(row):
    sid       = int(row["subhalo_id"])
    cat       = row["gas_category"]
    out_path  = cutout_path(sid)

    # Already downloaded and valid
    if verify_hdf5(out_path):
        return {"subhalo_id": sid, "gas_category": cat,
                "status": "skipped", "file_path": out_path,
                "file_size_kb": round(os.path.getsize(out_path) / 1024, 1),
                "error": ""}

    url = (f"https://www.tng-project.org/api/TNG-Cluster/"
           f"snapshots/99/subhalos/{sid}/cutout.hdf5"
           f"?gas={HDF5_GAS_FIELDS}")

    os.makedirs(CUTOUTS_DIR, exist_ok=True)
    ok = download_file(url, out_path)

    if not ok or not verify_hdf5(out_path):
        # Clean up partial file
        if os.path.exists(out_path):
            os.remove(out_path)
        return {"subhalo_id": sid, "gas_category": cat,
                "status": "failed", "file_path": "",
                "file_size_kb": 0, "error": "download_failed"}

    size_kb = round(os.path.getsize(out_path) / 1024, 1)
    return {"subhalo_id": sid, "gas_category": cat,
            "status": "ok", "file_path": out_path,
            "file_size_kb": size_kb, "error": ""}


def run():
    os.makedirs(CUTOUTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(CATALOG_CSV):
        print(f"ERROR: {CATALOG_CSV} not found. Run 01_catalog_subhalos.py first.")
        sys.exit(1)

    with open(CATALOG_CSV) as f:
        catalog = list(csv.DictReader(f))
    print(f"Catalog has {len(catalog)} subhalos.")

    # ── Quality gate: skip gas-empty and gas-too-sparse subhalos ─────────────
    before = len(catalog)
    catalog = [
        r for r in catalog
        if r.get("gas_category") not in SKIP_GAS_CATEGORIES
        and float(r.get("mass_gas_1e10msun_h", 0) or 0) >= MIN_DOWNLOAD_GAS_MASS
    ]
    skipped_by_filter = before - len(catalog)
    print(f"Quality gate: skipped {skipped_by_filter} subhalos "
          f"(NONE category or gas < {MIN_DOWNLOAD_GAS_MASS}×10¹⁰ M☉/h)")
    print(f"Eligible for download: {len(catalog)} subhalos")

    # Show gas category breakdown of what we will download
    cats = {}
    for r in catalog:
        cats[r.get("gas_category", "?")] = cats.get(r.get("gas_category", "?"), 0) + 1
    for cat, n in sorted(cats.items()):
        print(f"  {cat:8s}: {n}")

    # Load already-logged IDs (resume support)
    logged_ids = set()
    if os.path.exists(DOWNLOAD_LOG):
        with open(DOWNLOAD_LOG) as f:
            for row in csv.DictReader(f):
                if row["status"] in ("ok", "skipped"):
                    logged_ids.add(int(row["subhalo_id"]))
        print(f"Already downloaded: {len(logged_ids)}.  Resuming …")

    pending = [r for r in catalog if int(r["subhalo_id"]) not in logged_ids]
    print(f"Pending downloads:  {len(pending)}")

    if not pending:
        print("All cutouts already downloaded.")
        return

    log_mode = "a" if logged_ids else "w"
    log_fh   = open(DOWNLOAD_LOG, log_mode, newline="")
    writer   = csv.DictWriter(log_fh, fieldnames=LOG_FIELDS)
    if log_mode == "w":
        writer.writeheader()

    ok_count = failed = skipped = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futs = {pool.submit(download_one, row): row for row in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            result = fut.result()
            writer.writerow(result)
            log_fh.flush()

            s = result["status"]
            if s == "ok":
                ok_count += 1
            elif s == "skipped":
                skipped += 1
            else:
                failed += 1

            elapsed = time.time() - t0
            rate    = i / elapsed
            eta     = (len(pending) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(pending)}] sub={result['subhalo_id']:6d} "
                  f"cat={result['gas_category']:6s} {s:8s} "
                  f"{result['file_size_kb']:7.1f} KB  "
                  f"ETA {eta/60:.1f} min", flush=True)

    log_fh.close()

    total_size_mb = sum(
        os.path.getsize(cutout_path(int(r["subhalo_id"])))
        for r in catalog
        if verify_hdf5(cutout_path(int(r["subhalo_id"])))
    ) / (1024 ** 2)

    print(f"\n{'='*50}")
    print(f"Downloads complete.")
    print(f"  OK:      {ok_count}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")
    print(f"  Total cutouts on disk: {ok_count + skipped}")
    print(f"  Disk usage: {total_size_mb:.1f} MB")


if __name__ == "__main__":
    run()

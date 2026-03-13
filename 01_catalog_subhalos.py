"""
Step 1 — Catalog all satellite subhalos with gas/stellar properties.

Reads clusters.csv (produced by 01_find_clusters.py), paginates through
every satellite in each cluster, fetches full subhalo properties, and
records them WITHOUT downloading any HDF5 data.

Gas categories saved to catalog:
  NONE   — gas_mass = 0            (fully stripped)
  LOW    — 0 < gas_mass < 0.1      (stripped, "red & dead")
  MEDIUM — 0.1 ≤ gas_mass < 1.0   (possible ongoing stripping)
  HIGH   — gas_mass ≥ 1.0          (gas-rich, prime jellyfish)
  (masses in 1e10 M_sun/h units)

Writes: output/data/subhalo_catalog.csv
"""

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BASE_URL, DATA_DIR,
    MIN_GAS_MASS_1E10MSUN_H, MAX_GAS_MASS_1E10MSUN_H,
    MIN_STELLAR_MASS_1E10MSUN_H, MAX_STELLAR_MASS_1E10MSUN_H,
    MAX_HALFMASS_GAS_CKPC_H,
    MAX_CLUSTERS, MAX_GALAXIES_PER_CLUSTER,
)
from utils.tng_api import get_json

CLUSTERS_CSV = os.path.join(DATA_DIR, "clusters.csv")
CATALOG_CSV  = os.path.join(DATA_DIR, "subhalo_catalog.csv")
CHECKPOINT   = os.path.join(DATA_DIR, ".catalog_checkpoint")

CATALOG_FIELDS = [
    "subhalo_id", "grnr", "gas_category",
    "mass_gas_1e10msun_h", "mass_stars_1e10msun_h",
    "sfr", "halfmassrad_gas_ckpc_h",
    "pos_x_ckpc_h", "pos_y_ckpc_h", "pos_z_ckpc_h",
    "halo_r200_kpc", "bcg_pos_x", "bcg_pos_y", "bcg_pos_z",
    "cutout_url",
]

# Pagination early-exit: stop when subhalo total mass drops below this
MIN_TOTAL_MASS_LOG = 8.5    # log10(M/M_sun) — lower than before to catch stripped sats
MAX_SATS_TO_PAGE   = 5000   # safety cap per cluster


def gas_category(m_gas):
    if m_gas == 0.0:
        return "NONE"
    elif m_gas < 0.1:
        return "LOW"
    elif m_gas < 1.0:
        return "MEDIUM"
    return "HIGH"


def paginate_satellites(grnr):
    """Paginate satellite subhalos for one cluster, early-exit below mass floor."""
    url    = f"{BASE_URL}/subhalos/"
    params = {"limit": 500, "grnr": grnr, "primary_flag": 0,
              "order_by": "-mass_log_msun"}
    sats = []
    while url:
        data = get_json(url, params=params)
        if data is None:
            break
        for r in data.get("results", []):
            if r["mass_log_msun"] < MIN_TOTAL_MASS_LOG:
                return sats
            sats.append(r)
            if len(sats) >= MAX_SATS_TO_PAGE:
                return sats
        url    = data.get("next")
        params = {}
    return sats


def fetch_satellite_details(sat_summary, r200_kpc, bcg_pos):
    """Fetch full subhalo record; return catalog row or None."""
    try:
        d = get_json(sat_summary["url"])
        if d is None:
            return None
        if d.get("subhaloflag", 1) == 0:
            return None

        m_gas   = float(d.get("mass_gas",        0) or 0)
        m_star  = float(d.get("mass_stars",       0) or 0)
        r_half  = float(d.get("halfmassrad_gas",  0) or 0)

        # Stellar mass bounds
        if not (MIN_STELLAR_MASS_1E10MSUN_H <= m_star <= MAX_STELLAR_MASS_1E10MSUN_H):
            return None
        # Gas mass bounds
        if not (MIN_GAS_MASS_1E10MSUN_H <= m_gas <= MAX_GAS_MASS_1E10MSUN_H):
            return None
        # Gas radius guard (only meaningful for gas-bearing galaxies)
        if m_gas > 0.01 and r_half > MAX_HALFMASS_GAS_CKPC_H:
            return None

        return {
            "subhalo_id":            d["id"],
            "grnr":                  d["grnr"],
            "gas_category":          gas_category(m_gas),
            "mass_gas_1e10msun_h":   round(m_gas,  6),
            "mass_stars_1e10msun_h": round(m_star, 6),
            "sfr":                   round(float(d.get("sfr", 0) or 0), 6),
            "halfmassrad_gas_ckpc_h":round(r_half, 3),
            "pos_x_ckpc_h":          round(d["pos_x"], 2),
            "pos_y_ckpc_h":          round(d["pos_y"], 2),
            "pos_z_ckpc_h":          round(d["pos_z"], 2),
            "halo_r200_kpc":         round(r200_kpc, 1),
            "bcg_pos_x":             bcg_pos[0],
            "bcg_pos_y":             bcg_pos[1],
            "bcg_pos_z":             bcg_pos[2],
            "cutout_url":            d["cutouts"]["subhalo"],
        }
    except Exception:
        return None


def run():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(CLUSTERS_CSV):
        print(f"ERROR: {CLUSTERS_CSV} not found. Run 01_find_clusters.py first.")
        sys.exit(1)

    with open(CLUSTERS_CSV) as f:
        clusters = list(csv.DictReader(f))
    if MAX_CLUSTERS:
        clusters = clusters[:MAX_CLUSTERS]
    print(f"Loaded {len(clusters)} clusters from {CLUSTERS_CSV}")

    # Resume support
    start_ci = 0
    seen_ids  = set()
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            start_ci = int(f.read().strip())
        print(f"Checkpoint found — resuming from cluster {start_ci + 1}/{len(clusters)}")
    if os.path.exists(CATALOG_CSV) and start_ci > 0:
        with open(CATALOG_CSV) as f:
            for row in csv.DictReader(f):
                seen_ids.add(int(row["subhalo_id"]))
        print(f"Already cataloged: {len(seen_ids)} subhalos")

    log_mode = "a" if seen_ids else "w"
    fh = open(CATALOG_CSV, log_mode, newline="")
    writer = csv.DictWriter(fh, fieldnames=CATALOG_FIELDS)
    if log_mode == "w":
        writer.writeheader()

    total = len(seen_ids)
    cat_counts = {"NONE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0}

    for ci, cluster in enumerate(clusters):
        if ci < start_ci:
            continue

        grnr    = int(cluster["grnr"])
        r200    = float(cluster["R_approx_kpc"])
        bcg_pos = (float(cluster["bcg_pos_x"]),
                   float(cluster["bcg_pos_y"]),
                   float(cluster["bcg_pos_z"]))

        print(f"  [{ci+1}/{len(clusters)}] grnr={grnr}  M~{cluster['M_approx_msun']}",
              end="  ...", flush=True)

        sats = paginate_satellites(grnr)
        print(f"  {len(sats)} candidates", end="  ->", flush=True)

        if not sats:
            print(" 0 kept")
            continue

        if MAX_GALAXIES_PER_CLUSTER:
            sats = sats[:MAX_GALAXIES_PER_CLUSTER]

        kept = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(fetch_satellite_details, s, r200, bcg_pos): s
                    for s in sats}
            for fut in as_completed(futs):
                row = fut.result()
                if row and int(row["subhalo_id"]) not in seen_ids:
                    writer.writerow(row)
                    seen_ids.add(int(row["subhalo_id"]))
                    cat_counts[row["gas_category"]] += 1
                    kept += 1
                    total += 1

        fh.flush()
        print(f"  {kept} kept  (total {total})")

        # Save checkpoint after every cluster
        with open(CHECKPOINT, "w") as f:
            f.write(str(ci + 1))
        time.sleep(0.05)

    fh.close()

    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    print(f"\n{'='*55}")
    print(f"Catalog complete:  {total} subhalos total")
    print(f"  NONE   (fully stripped): {cat_counts['NONE']}")
    print(f"  LOW    (gas-poor):       {cat_counts['LOW']}")
    print(f"  MEDIUM (moderate gas):   {cat_counts['MEDIUM']}")
    print(f"  HIGH   (gas-rich / JF):  {cat_counts['HIGH']}")
    print(f"Saved -> {CATALOG_CSV}")


if __name__ == "__main__":
    run()

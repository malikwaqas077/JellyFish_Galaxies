"""
Step 1 — Find cluster-mass halos in TNG300-1 at z=0.

Strategy (TNG API does NOT have a /halos/ list endpoint):
  - Query primary subhalos (BCGs) sorted by total mass (descending)
  - The mass_log_msun field for BCGs equals log10(M_FoF / M_sun)
    which is a good proxy for the host cluster mass
  - Filter by mass_log_msun > 13.7  (~5 x 10^13 M_sun, conservative cluster floor)
  - Fetch individual BCG records in parallel to get: grnr, position, halfmassrad

Writes: output/data/clusters.csv
"""

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from config import (BASE_URL, DATA_DIR, LITTLE_H, MAX_CLUSTERS, REQUEST_TIMEOUT)
from utils.tng_api import get_json, paginate

OUTPUT_FILE = os.path.join(DATA_DIR, "clusters.csv")

# proxy threshold: BCG total mass > 10^13.5 M_sun ~ 3e13 M_sun
# This captures all clusters AND massive groups (where jellyfish are also found)
BCG_MASS_LOG_MIN = 13.5
MAX_BCG_FETCH    = 800      # TNG300-1 has ~700 clusters; 800 is safe upper bound


def fetch_bcg_details(bcg_summary):
    """Fetch full BCG record from its URL."""
    try:
        data = get_json(bcg_summary["url"])
        if data is None:
            return None
        return {
            "bcg_id":         data["id"],
            "grnr":           data["grnr"],
            "mass_log_msun":  bcg_summary["mass_log_msun"],
            "bcg_pos_x":      data["pos_x"],    # ckpc/h
            "bcg_pos_y":      data["pos_y"],
            "bcg_pos_z":      data["pos_z"],
            "halfmassrad_gas_ckpc_h": data.get("halfmassrad_gas", 0),
        }
    except Exception as e:
        print(f"    WARN: failed to fetch BCG {bcg_summary['id']}: {e}")
        return None


def find_clusters():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Querying TNG300-1 snapshot 99 for massive BCGs ...")
    print(f"  BCG mass proxy threshold: log10(M/M_sun) > {BCG_MASS_LOG_MIN}")

    # ── Paginate BCG list, stop when mass drops below threshold ─────────────
    url    = f"{BASE_URL}/subhalos/"
    params = {
        "limit":        500,
        "primary_flag": 1,
        "order_by":     "-mass_log_msun",
    }

    bcg_list = []
    page_count = 0
    while url and len(bcg_list) < MAX_BCG_FETCH:
        data = get_json(url, params=params)
        if data is None:
            break
        page_count += 1
        results = data.get("results", [])
        for r in results:
            if r["mass_log_msun"] < BCG_MASS_LOG_MIN:
                url = None    # stop paginating — remaining are below threshold
                break
            bcg_list.append(r)
        else:
            url = data.get("next")
        params = {}    # next URL already encodes params

    print(f"  Found {len(bcg_list)} BCGs above mass threshold "
          f"(scanned {page_count} pages)")

    if MAX_CLUSTERS:
        bcg_list = bcg_list[:MAX_CLUSTERS]
        print(f"  Limiting to MAX_CLUSTERS={MAX_CLUSTERS}")

    # ── Fetch full BCG details in parallel (grnr, position) ─────────────────
    print(f"  Fetching full BCG records (parallel) ...")
    clusters = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_bcg_details, b): b for b in bcg_list}
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result:
                clusters.append(result)
            if done % 50 == 0:
                print(f"    {done}/{len(bcg_list)} BCGs fetched ...")

    # Sort by grnr for reproducibility
    clusters.sort(key=lambda x: x["grnr"])
    print(f"  {len(clusters)} clusters with full details retrieved.")

    # ── Save ─────────────────────────────────────────────────────────────────
    fieldnames = [
        "grnr", "bcg_id", "mass_log_msun",
        "bcg_pos_x", "bcg_pos_y", "bcg_pos_z",
        "halfmassrad_gas_ckpc_h",
        "M_approx_msun",
        "R_approx_kpc",
    ]
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in clusters:
            m_approx = 10 ** c["mass_log_msun"]
            # Rough R200c estimate from M200c (Bryan & Norman 1998):
            # R200c ~ 1.6 Mpc x (M200c / 10^15 Msun)^(1/3)
            r_approx_kpc = 1600.0 * (m_approx / 1e15) ** (1.0 / 3.0)
            writer.writerow({
                **c,
                "M_approx_msun":  f"{m_approx:.3e}",
                "R_approx_kpc":   round(r_approx_kpc, 1),
            })

    print(f"Saved {len(clusters)} clusters -> {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    find_clusters()

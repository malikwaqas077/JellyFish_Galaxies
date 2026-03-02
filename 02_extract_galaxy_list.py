"""
Step 2 — For each cluster, find satellite galaxies with enough gas to image.

Strategy:
  - For each cluster (grnr), paginate satellite subhalos sorted by mass (desc)
  - Stop paginating when mass drops below MIN_TOTAL_MASS_LOG (early exit)
  - Fetch full details in parallel for mass-passing satellites
  - Keep those with mass_gas >= MIN_GAS_MASS and mass_stars >= MIN_STAR_MASS

Reads  : output/data/clusters.csv
Writes : output/data/galaxy_list.csv
"""

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from config import (BASE_URL, DATA_DIR, LITTLE_H,
                    MIN_GAS_MASS_1E10MSUN_H, MAX_GAS_MASS_1E10MSUN_H,
                    MIN_STELLAR_MASS_1E10MSUN_H, MAX_STELLAR_MASS_1E10MSUN_H,
                    MAX_HALFMASS_GAS_CKPC_H, MAX_GALAXIES_PER_CLUSTER)
from utils.tng_api import get_json

CLUSTERS_FILE = os.path.join(DATA_DIR, "clusters.csv")
OUTPUT_FILE   = os.path.join(DATA_DIR, "galaxy_list.csv")

# Total subhalo mass floor (log10 M_sun) for early-stop pagination
# A galaxy with M_gas > 10^8 M_sun + M_* > 10^8 M_sun has M_total >> 10^9 M_sun
# (DM-dominated: M_total ~ 10x baryons)
MIN_TOTAL_MASS_LOG = 9.0    # log10(SubhaloMass / M_sun) > 9.0

# Max satellites to paginate per cluster (safety cap)
MAX_SATS_TO_PAGE   = 2000


def paginate_satellites(grnr):
    """
    Return list of {id, mass_log_msun, url} for satellites in cluster grnr
    with mass_log_msun > MIN_TOTAL_MASS_LOG (early-exit pagination).
    """
    url    = f"{BASE_URL}/subhalos/"
    params = {
        "limit":        500,
        "grnr":         grnr,
        "primary_flag": 0,
        "order_by":     "-mass_log_msun",
    }
    sats   = []
    while url:
        data = get_json(url, params=params)
        if data is None:
            break
        for r in data.get("results", []):
            if r["mass_log_msun"] < MIN_TOTAL_MASS_LOG:
                return sats       # everything below is lighter → stop
            sats.append(r)
            if len(sats) >= MAX_SATS_TO_PAGE:
                return sats
        url    = data.get("next")
        params = {}
    return sats


def fetch_satellite_details(sat_summary):
    """Fetch full subhalo record and return standardised row or None."""
    try:
        d = get_json(sat_summary["url"])
        if d is None:
            return None

        # Must be a proper galaxy (not DM-only or numerical artifact)
        if d.get("subhaloflag", 1) == 0:
            return None

        m_gas    = d.get("mass_gas",          0) or 0
        m_star   = d.get("mass_stars",        0) or 0
        r_half_g = d.get("halfmassrad_gas",   0) or 0

        # Stellar mass bounds (we need visible galaxies)
        if m_star < MIN_STELLAR_MASS_1E10MSUN_H or m_star > MAX_STELLAR_MASS_1E10MSUN_H:
            return None
        
        # Gas mass bounds (NOW INCLUDES GAS-POOR GALAXIES for classifier training)
        if m_gas < MIN_GAS_MASS_1E10MSUN_H or m_gas > MAX_GAS_MASS_1E10MSUN_H:
            return None
        
        # Half-mass radius guard: only check if galaxy has significant gas
        # Gas-poor galaxies (m_gas < 0.01) can have undefined/large r_half_g
        if m_gas > 0.01 and r_half_g > MAX_HALFMASS_GAS_CKPC_H:
            return None

        return {
            "subhalo_id":              d["id"],
            "grnr":                    d["grnr"],
            "mass_gas_1e10msun_h":     round(m_gas,  6),
            "mass_stars_1e10msun_h":   round(m_star, 6),
            "pos_x_ckpc_h":            round(d["pos_x"], 2),
            "pos_y_ckpc_h":            round(d["pos_y"], 2),
            "pos_z_ckpc_h":            round(d["pos_z"], 2),
            "halfmassrad_gas_ckpc_h":  round(d.get("halfmassrad_gas", 0), 3),
            "cutout_url":              d["cutouts"]["subhalo"],
            "sfr":                     round(d.get("sfr", 0), 4),
        }
    except Exception as e:
        return None


def extract_galaxies():
    with open(CLUSTERS_FILE, "r") as f:
        clusters = list(csv.DictReader(f))
    
    # Checkpoint file for resume capability
    CHECKPOINT_FILE = os.path.join(DATA_DIR, ".extraction_checkpoint")
    start_cluster = 0
    
    # Check for existing checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            start_cluster = int(f.read().strip())
        print(f"Resuming from cluster {start_cluster+1}/{len(clusters)} (checkpoint found)")
    else:
        print(f"Processing {len(clusters)} clusters (fresh start)")

    all_galaxies = []
    fieldnames = [
        "subhalo_id", "grnr", "halo_r200_kpc",
        "mass_gas_1e10msun_h", "mass_stars_1e10msun_h",
        "pos_x_ckpc_h", "pos_y_ckpc_h", "pos_z_ckpc_h",
        "halfmassrad_gas_ckpc_h",
        "bcg_pos_x", "bcg_pos_y", "bcg_pos_z",
        "cutout_url", "sfr",
    ]
    
    # Load existing galaxies if resuming
    if start_cluster > 0 and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            all_galaxies = list(csv.DictReader(f))
        print(f"Loaded {len(all_galaxies)} galaxies from previous run")

    for ci, cluster in enumerate(clusters):
        # Skip already processed clusters
        if ci < start_cluster:
            continue
        grnr     = int(cluster["grnr"])
        r200_kpc = float(cluster["R_approx_kpc"])
        bcg_pos  = (float(cluster["bcg_pos_x"]),
                    float(cluster["bcg_pos_y"]),
                    float(cluster["bcg_pos_z"]))

        print(f"  [{ci+1}/{len(clusters)}] grnr={grnr}  "
              f"M~{cluster['M_approx_msun']}  R200~{r200_kpc:.0f} kpc",
              end="  ...", flush=True)

        # Paginate satellites (early exit below mass floor)
        sats = paginate_satellites(grnr)
        print(f"  {len(sats)} candidate satellites", end="  ->", flush=True)

        if not sats:
            print(" 0 kept")
            continue

        if MAX_GALAXIES_PER_CLUSTER:
            sats = sats[:MAX_GALAXIES_PER_CLUSTER]

        # Fetch full details in parallel (reduced workers to avoid API throttling)
        kept = []
        processed = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(fetch_satellite_details, s) for s in sats]
            for fut in as_completed(futs):
                row = fut.result()
                processed += 1
                if row:
                    row["halo_r200_kpc"] = round(r200_kpc, 1)
                    row["bcg_pos_x"]     = bcg_pos[0]
                    row["bcg_pos_y"]     = bcg_pos[1]
                    row["bcg_pos_z"]     = bcg_pos[2]
                    kept.append(row)
                # Progress indicator for large clusters
                if len(sats) > 500 and processed % 100 == 0:
                    print(f" {processed}/{len(sats)}", end="", flush=True)

        print(f" {len(kept)} kept (gas + stellar mass cuts)")
        all_galaxies.extend(kept)
        
        # Save checkpoint after each cluster (enables resume)
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(str(ci + 1))
        
        # Periodic save every 10 clusters to avoid data loss
        if (ci + 1) % 10 == 0 or ci == len(clusters) - 1:
            os.makedirs(DATA_DIR, exist_ok=True)
            # Sort by grnr then subhalo_id
            all_galaxies_sorted = sorted(all_galaxies, key=lambda r: (int(r["grnr"]), int(r["subhalo_id"])))
            with open(OUTPUT_FILE, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_galaxies_sorted)
            print(f"  → Saved {len(all_galaxies)} galaxies (checkpoint: {ci+1}/{len(clusters)})")

    # Final save
    os.makedirs(DATA_DIR, exist_ok=True)
    all_galaxies.sort(key=lambda r: (int(r["grnr"]), int(r["subhalo_id"])))
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_galaxies)
    
    # Remove checkpoint on successful completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"\n{'='*60}")
    print(f"Total galaxies selected: {len(all_galaxies)}")
    print(f"Saved -> {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    extract_galaxies()

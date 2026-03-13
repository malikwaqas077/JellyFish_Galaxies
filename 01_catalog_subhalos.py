"""
Step 1 — Catalog all satellite subhalos with gas/stellar properties.

Queries the TNG API for every cluster, then paginates through its satellites
and records key properties WITHOUT downloading any HDF5 data.  The result is
a lightweight CSV that tells us exactly which subhalos are worth downloading.

Gas categories written to catalog:
  LOW    — 0 < gas_mass < 0.1  (×10¹⁰ M_sun/h)   stripped satellites
  MEDIUM — 0.1 ≤ gas_mass < 1  (×10¹⁰ M_sun/h)   possible stripping
  HIGH   — gas_mass ≥ 1        (×10¹⁰ M_sun/h)   jellyfish candidates
  NONE   — gas_mass = 0        (fully stripped)

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
    MIN_CLUSTER_MASS_1E10MSUN_H,
    MIN_GAS_MASS_1E10MSUN_H, MAX_GAS_MASS_1E10MSUN_H,
    MIN_STELLAR_MASS_1E10MSUN_H, MAX_STELLAR_MASS_1E10MSUN_H,
    MAX_HALFMASS_GAS_CKPC_H,
    MAX_CLUSTERS, MAX_GALAXIES_PER_CLUSTER, PARALLEL_WORKERS,
)
from utils.tng_api import get_json, paginate

CATALOG_CSV = os.path.join(DATA_DIR, "subhalo_catalog.csv")
CATALOG_FIELDS = [
    "subhalo_id", "grnr", "gas_category",
    "mass_gas_1e10msun_h", "mass_stars_1e10msun_h",
    "sfr", "halfmassrad_gas_ckpc_h",
    "pos_x_ckpc_h", "pos_y_ckpc_h", "pos_z_ckpc_h",
    "halo_r200_kpc", "bcg_pos_x", "bcg_pos_y", "bcg_pos_z",
    "cutout_url",
]


def gas_category(m_gas):
    if m_gas == 0.0:
        return "NONE"
    elif m_gas < 0.1:
        return "LOW"
    elif m_gas < 1.0:
        return "MEDIUM"
    else:
        return "HIGH"


def fetch_clusters():
    """Return list of cluster dicts passing the mass threshold."""
    print("Fetching cluster list …")
    url = f"{BASE_URL}/subhalos/"
    params = {"limit": 100, "order_by": "-mass_gas"}
    all_subs = paginate(url, params)

    clusters = []
    for s in all_subs:
        bcg_mass = s.get("mass_log_msun", 0) or 0
        # Also accept via SubhaloMassType field (index 4 = stars)
        m_tot = s.get("mass", 0) or 0
        # Use primary subhalo of each group: grnr == subhalo index for BCGs
        if s.get("grnr") != s.get("id"):
            continue
        # Filter by minimum halo mass (approximate via BCG total mass)
        if m_tot < MIN_CLUSTER_MASS_1E10MSUN_H:
            continue
        clusters.append(s)
        if MAX_CLUSTERS and len(clusters) >= MAX_CLUSTERS:
            break

    print(f"  Found {len(clusters)} clusters above mass threshold.")
    return clusters


def fetch_cluster_r200(grnr):
    """Fetch R200 for a group/halo."""
    url = f"{BASE_URL}/halos/{grnr}/"
    data = get_json(url)
    if data is None:
        return None
    return data.get("Group_R_Crit200", None)   # ckpc/h


def fetch_satellites(cluster, r200_ckpc_h):
    """Return list of satellite rows for one cluster."""
    grnr      = cluster["id"]
    bcg_pos   = (cluster.get("pos_x", 0), cluster.get("pos_y", 0), cluster.get("pos_z", 0))
    # R200 in physical kpc
    r200_kpc  = (r200_ckpc_h / 0.6774) if r200_ckpc_h else 0

    url    = f"{BASE_URL}/subhalos/"
    params = {"limit": 100, "grnr": grnr, "order_by": "-mass_gas"}
    subs   = paginate(url, params)

    rows = []
    for s in subs:
        sid = s.get("id")
        if sid == grnr:
            continue   # skip BCG itself

        m_gas   = float(s.get("mass_gas",   0) or 0)
        m_stars = float(s.get("mass_stars", 0) or 0)
        r_half  = float(s.get("halfmassrad_gas", 0) or 0)   # ckpc/h

        # Mass filters
        if m_gas < MIN_GAS_MASS_1E10MSUN_H or m_gas > MAX_GAS_MASS_1E10MSUN_H:
            continue
        if m_stars < MIN_STELLAR_MASS_1E10MSUN_H or m_stars > MAX_STELLAR_MASS_1E10MSUN_H:
            continue
        # Gas radius filter only for gas-bearing galaxies
        if m_gas > 0.01 and r_half > MAX_HALFMASS_GAS_CKPC_H:
            continue

        cutout_url = (f"https://www.tng-project.org/api/TNG-Cluster/"
                      f"snapshots/99/subhalos/{sid}/cutout.hdf5")

        rows.append({
            "subhalo_id":            sid,
            "grnr":                  grnr,
            "gas_category":          gas_category(m_gas),
            "mass_gas_1e10msun_h":   round(m_gas,   6),
            "mass_stars_1e10msun_h": round(m_stars, 6),
            "sfr":                   round(float(s.get("sfr", 0) or 0), 6),
            "halfmassrad_gas_ckpc_h":round(r_half,  3),
            "pos_x_ckpc_h":          s.get("pos_x", 0),
            "pos_y_ckpc_h":          s.get("pos_y", 0),
            "pos_z_ckpc_h":          s.get("pos_z", 0),
            "halo_r200_kpc":         round(r200_kpc, 2),
            "bcg_pos_x":             bcg_pos[0],
            "bcg_pos_y":             bcg_pos[1],
            "bcg_pos_z":             bcg_pos[2],
            "cutout_url":            cutout_url,
        })

        if MAX_GALAXIES_PER_CLUSTER and len(rows) >= MAX_GALAXIES_PER_CLUSTER:
            break

    return rows


def run():
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Load already-processed subhalo IDs (resume support) ──────────────────
    done_ids = set()
    if os.path.exists(CATALOG_CSV):
        with open(CATALOG_CSV) as f:
            for row in csv.DictReader(f):
                done_ids.add(int(row["subhalo_id"]))
        print(f"Resuming — {len(done_ids)} subhalos already cataloged.")

    clusters = fetch_clusters()
    if not clusters:
        print("No clusters found. Check API_KEY and network.")
        return

    # Pre-fetch R200 for all clusters in parallel
    print(f"Fetching R200 for {len(clusters)} clusters …")
    r200_map = {}
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futs = {pool.submit(fetch_cluster_r200, c["id"]): c["id"] for c in clusters}
        for fut in as_completed(futs):
            grnr = futs[fut]
            r200_map[grnr] = fut.result() or 2000.0   # fallback 2000 ckpc/h

    # ── Stream results to CSV ─────────────────────────────────────────────────
    mode = "a" if done_ids else "w"
    fh   = open(CATALOG_CSV, mode, newline="")
    writer = csv.DictWriter(fh, fieldnames=CATALOG_FIELDS)
    if mode == "w":
        writer.writeheader()

    total = 0
    cat_counts = {"NONE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0}

    for ci, cluster in enumerate(clusters):
        grnr = cluster["id"]
        print(f"  [{ci+1}/{len(clusters)}] cluster grnr={grnr}", end=" … ", flush=True)
        rows = fetch_satellites(cluster, r200_map.get(grnr, 2000.0))
        new_rows = [r for r in rows if int(r["subhalo_id"]) not in done_ids]
        for row in new_rows:
            writer.writerow(row)
            cat_counts[row["gas_category"]] += 1
        fh.flush()
        total += len(new_rows)
        print(f"{len(new_rows)} new satellites (total {total + len(done_ids)})")
        time.sleep(0.1)

    fh.close()

    grand_total = total + len(done_ids)
    print(f"\n{'='*50}")
    print(f"Catalog complete:  {grand_total} subhalos")
    print(f"  NONE   (fully stripped): {cat_counts['NONE']}")
    print(f"  LOW    (gas-poor):       {cat_counts['LOW']}")
    print(f"  MEDIUM (moderate gas):   {cat_counts['MEDIUM']}")
    print(f"  HIGH   (gas-rich / JF):  {cat_counts['HIGH']}")
    print(f"Written to: {CATALOG_CSV}")


if __name__ == "__main__":
    run()

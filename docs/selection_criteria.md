# Selection Criteria for TNG Jellyfish Galaxy Dataset

This document describes the criteria used to select galaxy clusters and satellite galaxies for the jellyfish galaxy image dataset.

---

## Simulation

| Parameter | Value |
|-----------|-------|
| Simulation | **TNG-Cluster** |
| Snapshot | 99 (z = 0) |
| Description | Zoom-in cosmological simulation of 352 massive galaxy clusters |

TNG-Cluster is ideal for jellyfish studies because massive clusters have strong intracluster medium (ICM) that causes ram-pressure stripping of infalling galaxies.

---

## Step 1: Cluster Selection

Clusters are identified by their **Brightest Cluster Galaxy (BCG)**, which is the primary subhalo (central galaxy) of each Friends-of-Friends (FoF) halo.

### Criteria

| Parameter | Threshold | Physical Meaning |
|-----------|-----------|------------------|
| `BCG_MASS_LOG_MIN` | 13.5 | log₁₀(M_total / M☉) > 13.5 |
| Equivalent mass | > 3 × 10¹³ M☉ | Conservative cluster floor (includes massive groups) |

### Method

1. Query TNG API for primary subhalos (`primary_flag = 1`) sorted by total mass
2. Filter by `mass_log_msun > 13.5`
3. Fetch full BCG records to get: `grnr` (group number), position, half-mass radius

### Result

**13 clusters** selected from TNG-Cluster snapshot 99.

---

## Step 2: Galaxy (Satellite) Selection

For each cluster, satellite galaxies are selected based on gas and stellar mass cuts.

### Criteria

| Parameter | Min Value | Max Value | Physical Reason |
|-----------|-----------|-----------|-----------------|
| **Gas mass** | 0.001 | 100.0 | Catch partially-stripped galaxies; exclude ICM-filling subclusters |
| **Stellar mass** | 0.01 | 50.0 | Avoid tiny dwarfs; exclude BCGs and massive subclusters |
| **Gas half-mass radius** | — | 100.0 | Galaxy-scale objects only (not subclusters) |

All masses are in TNG internal units: **1 × 10¹⁰ M☉/h**

To convert to physical solar masses, multiply by `h ≈ 0.6774`:
- `0.001 × 0.6774 × 10¹⁰ = 6.8 × 10⁶ M☉`
- `100.0 × 0.6774 × 10¹⁰ = 6.8 × 10¹¹ M☉`

### Selection Logic

A satellite galaxy is included if **all** conditions are met:

```
MIN_GAS_MASS   ≤ mass_gas   ≤ MAX_GAS_MASS
MIN_STELLAR_MASS ≤ mass_stars ≤ MAX_STELLAR_MASS
halfmassrad_gas ≤ MAX_HALFMASS_GAS
```

### Rationale

| Cut | Purpose |
|-----|---------|
| Min gas mass (7×10⁶ M☉) | Include gas-poor / heavily stripped galaxies that may still show tails |
| Max gas mass (7×10¹¹ M☉) | Exclude massive gas-rich subclusters that would dominate the image |
| Min stellar mass (7×10⁷ M☉) | Exclude tiny dwarf galaxies with unreliable morphologies |
| Max stellar mass (3×10¹¹ M☉) | Exclude BCGs and massive subclusters |
| Max gas radius (150 kpc) | Ensure galaxy-scale objects, not extended ICM structures |

---

## Dataset Summary

### Cluster Distribution

| grnr | BCG ID | Mass (M☉) | R200 (kpc) | Galaxies |
|------|--------|-----------|------------|----------|
| 0 | 0 | 2.59 × 10¹⁵ | 2198 | 154 |
| 1 | 23206 | 4.61 × 10¹³ | 574 | 30 |
| 2 | 24687 | 1.49 × 10¹⁴ | 848 | 22 |
| 3 | 25757 | 5.39 × 10¹³ | 604 | 30 |
| 4 | 26830 | 1.33 × 10¹⁴ | 817 | 2 |
| 5 | 27160 | 7.59 × 10¹³ | 678 | 4 |
| 6 | 27523 | 5.09 × 10¹³ | 593 | 18 |
| 7 | 28066 | 8.58 × 10¹³ | 706 | 3 |
| 8 | 28454 | 6.95 × 10¹³ | 658 | 2 |
| 9 | 28711 | 5.13 × 10¹³ | 595 | 10 |
| 10 | 29138 | 3.94 × 10¹³ | 544 | 2 |
| 11 | 29281 | 3.53 × 10¹³ | 525 | 9 |
| 12 | 29511 | 4.07 × 10¹³ | 550 | 5 |
| **Total** | | | | **291** |

### Key Statistics

| Metric | Value |
|--------|-------|
| Total clusters | 13 |
| Total galaxies | 291 |
| Cluster mass range | 3.5 × 10¹³ – 2.6 × 10¹⁵ M☉ |
| Cluster R200 range | 525 – 2198 kpc |
| Galaxies per cluster | 2 – 154 |

---

## Image Generation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `APERTURE_KPC` | 200.0 | Half-width of projection window (kpc) |
| `IMAGE_SIZE_PX` | 424 | Output image size (pixels) |
| `COLORMAP` | jet (native TNG) | Dark blue background → red for dense gas |

The 200 kpc aperture (400 kpc total) captures most jellyfish tails while keeping the galaxy centered.

---

## Configuration File Reference

All parameters are defined in `config.py`:

```python
# Cluster selection
MIN_CLUSTER_MASS_1E10MSUN_H = 10_000   # ~7×10^13 M_sun

# Galaxy selection
MIN_GAS_MASS_1E10MSUN_H     = 0.001    # ~7×10^6 M_sun
MAX_GAS_MASS_1E10MSUN_H     = 100.0    # ~7×10^11 M_sun
MIN_STELLAR_MASS_1E10MSUN_H = 0.01     # ~7×10^7 M_sun
MAX_STELLAR_MASS_1E10MSUN_H = 50.0     # ~3×10^11 M_sun
MAX_HALFMASS_GAS_CKPC_H     = 100.0    # ~150 physical kpc
```

---

## Data Files

| File | Description |
|------|-------------|
| `output/data/clusters.csv` | 13 clusters with BCG info, mass, R200 |
| `output/data/galaxy_list.csv` | 291 galaxies with positions, masses, SFR |
| `output/data/image_log.csv` | Image metadata and quality metrics |
| `output/images/*.png` | Gas density images (424×424 px, jet colormap) |

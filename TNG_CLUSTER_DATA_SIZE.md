# TNG-Cluster Data Size Guide

## Overview

**TNG-Cluster** is a massive cosmological simulation with 352 zoom-in galaxy clusters.

**Total dataset size:** **209.7 TB** (all snapshots, all data types)

---

## What You Actually Need

### For Rendering Galaxy Images

You **don't** need the full 209.7 TB! Here's what you need:

#### Option 1: Single Snapshot (Minimal)

**What:** Snapshot 99 (z=0, present day) - just our 188 clusters  
**Size:** ~1.5 TB  
**Contains:**
- Gas particle positions, velocities, masses, temperatures
- Star particle data
- Dark matter (optional, can skip)

**Sufficient for:** Rendering all 1,863 galaxies at any resolution

#### Option 2: Cutout Server (Smart Choice)

**What:** Download only specific subhalo regions via API  
**Size:** ~10-50 GB (for 1,863 galaxies)  
**How:** Use TNG cutout service to download 200 kpc radius around each galaxy

**Example:**
```python
# Instead of downloading 1.5 TB snapshot:
# Download just the region around each galaxy
for subhalo_id in galaxy_ids:
    cutout = get(f"https://tng-project.org/api/TNG-Cluster/snapshots/99/subhalos/{subhalo_id}/cutout.hdf5?gas=Coordinates,Density,Masses,Velocities,Temperature")
    # Each cutout: ~5-50 MB
```

**Total:** 1,863 galaxies × 20 MB avg = ~37 GB

✅ **This is the smart approach!**

---

## Detailed Breakdown

### Full TNG-Cluster Dataset Structure

```
TNG-Cluster/
├── snapshots/          209.7 TB total (100 snapshots across cosmic time)
│   ├── snapshot_000/   1.5 TB   (z=127, Big Bang)
│   ├── snapshot_050/   4.4 TB   (z=1.0)
│   ├── snapshot_099/   1.5 TB   (z=0, present day) ← WE NEED THIS
│   └── ...
├── groups/             ~10 GB   (halo/galaxy catalogs)
├── trees/              ~50 GB   (merger trees)
└── other/              ~5 TB    (additional data products)
```

### Snapshot 99 Breakdown (z=0)

**Total size:** ~1.5 TB

**File structure:**
```
snapshot_099/
├── snap_099.0.hdf5     ~300 GB  (Gas particles)
├── snap_099.1.hdf5     ~100 GB  (Dark matter - can skip!)
├── snap_099.2.hdf5     ~50 GB   (Star particles)
├── snap_099.3.hdf5     ~10 GB   (Black holes)
└── snap_099.[4-7].hdf5 ~1 TB    (More dark matter - can skip!)
```

**What you actually need for gas images:**
- `snap_099.0.hdf5` (gas) = **~300 GB**
- `snap_099.2.hdf5` (stars, optional) = **~50 GB**
- **Total: ~350 GB** (skip dark matter!)

---

## Storage Requirements Comparison

| Method | Size | Download Time (100 Mbps) | Pros | Cons |
|--------|------|-------------------------|------|------|
| **API Images** | 93 MB | 10 seconds | ✅ Instant | ❌ Fixed quality |
| **Cutouts (Smart)** | 37 GB | 1 hour | ✅ Custom rendering | ⚠️ Need expertise |
| **Snapshot (Gas only)** | 350 GB | 8 hours | ✅ Full control | ⚠️ Large storage |
| **Full Snapshot** | 1.5 TB | 33 hours | ✅ Everything | ❌ Huge, includes DM |
| **All Snapshots** | 210 TB | 6 months | ✅ Time evolution | ❌ Absurd overkill |

---

## Recommended Approach: Cutout Service

### Why Cutouts?

Instead of downloading the full 1.5 TB snapshot (99.9% of which you don't need), download just the particles near each galaxy:

**Advantages:**
- **40× smaller:** 37 GB vs 1.5 TB
- **Targeted:** Only particles within 200 kpc of each galaxy
- **Efficient:** Download in parallel (1,863 small files)
- **Sufficient:** Contains all gas/stars needed for rendering

**How it works:**
1. For each subhalo_id, request a cutout via API
2. TNG servers extract just that region from snapshot
3. You download a small HDF5 file (~20 MB)
4. Render locally with full control

### Example Code

```python
import requests
import h5py

# Your TNG API key
headers = {"api-key": "YOUR_KEY"}

# Download cutout for one galaxy
subhalo_id = 25758  # High-gas example
cutout_url = (
    f"https://www.tng-project.org/api/TNG-Cluster/snapshots/99/"
    f"subhalos/{subhalo_id}/cutout.hdf5"
    f"?gas=Coordinates,Density,Masses,Velocities,Temperature"
)

response = requests.get(cutout_url, headers=headers)
with open(f"cutout_{subhalo_id}.hdf5", "wb") as f:
    f.write(response.content)

# Now render locally with custom parameters
with h5py.File(f"cutout_{subhalo_id}.hdf5", "r") as f:
    gas_coords = f['PartType0/Coordinates'][:]
    gas_density = f['PartType0/Density'][:]
    gas_mass = f['PartType0/Masses'][:]
    gas_temp = f['PartType0/Temperature'][:]
    
    # Custom rendering (temperature-weighted, high-res, etc.)
    # ... your rendering code here ...
```

---

## Cost-Benefit Analysis

### Scenario 1: API Images (Current ✓)

**Cost:**
- Storage: 93 MB
- Time: 5 hours
- Expertise: None

**Benefit:**
- 1,856 images ready immediately
- Good enough for 80-90% ML accuracy

**Verdict:** ✅ **Best starting point**

---

### Scenario 2: Cutout Service (If Needed)

**Cost:**
- Storage: 37 GB
- Time: 2-4 hours download + 20-30 hours rendering
- Expertise: Moderate (Python, astrophysics basics)

**Benefit:**
- Full rendering control (resolution, temperature, projections)
- 10-20% ML accuracy improvement
- Publication-quality images

**Verdict:** ✅ **Good upgrade path if API plateaus**

---

### Scenario 3: Full Snapshot (Overkill)

**Cost:**
- Storage: 350 GB (gas only) or 1.5 TB (full)
- Time: 8-33 hours download + 30-50 hours rendering
- Expertise: High (simulation data formats, parallel processing)

**Benefit:**
- Maximum flexibility
- Can render any galaxy, any method
- Research-grade quality

**Verdict:** ⚠️ **Only for publishing or advanced research**

---

## My Recommendation

### Phase 1: API Images (DONE ✓)
- **Cost:** 93 MB, 5 hours
- **Status:** Complete, 1,856 images ready
- **Action:** Train classifier now

### Phase 2: IF Needed - Cutout Service
- **Trigger:** Classifier accuracy <85%
- **Cost:** 37 GB, ~30 hours total
- **Approach:**
  1. Download cutouts for ~200 problematic galaxies
  2. Re-render at high quality
  3. Retrain classifier with mixed dataset

### Phase 3: IF Publishing - Full Snapshot
- **Trigger:** Preparing manuscript for ApJ/MNRAS
- **Cost:** 350 GB, ~50 hours total
- **Approach:**
  1. Download gas-only snapshot
  2. Render all 1,863 at publication quality
  3. Generate x, y, z projections (5,589 images)

---

## Practical Download Guide

### If You Decide to Download Cutouts

**Step 1: Get API Key**
```bash
# Register at https://www.tng-project.org/users/register/
# Get API key from https://www.tng-project.org/users/profile/
```

**Step 2: Download Script**
```python
import time
import requests
import csv

API_KEY = "your_key_here"
headers = {"api-key": API_KEY}

# Read your galaxy list
with open('output/data/galaxy_list.csv', 'r') as f:
    galaxies = list(csv.DictReader(f))

# Download cutouts (with rate limiting)
for i, g in enumerate(galaxies):
    subhalo_id = g['subhalo_id']
    
    # Request cutout (200 kpc radius, gas only)
    url = (f"https://www.tng-project.org/api/TNG-Cluster/snapshots/99/"
           f"subhalos/{subhalo_id}/cutout.hdf5"
           f"?gas=Coordinates,Density,Masses,Temperature")
    
    response = requests.get(url, headers=headers)
    
    if response.ok:
        with open(f"cutouts/cutout_{subhalo_id}.hdf5", "wb") as f:
            f.write(response.content)
        print(f"[{i+1}/{len(galaxies)}] Downloaded {subhalo_id}")
    else:
        print(f"[{i+1}/{len(galaxies)}] FAILED {subhalo_id}: {response.status_code}")
    
    # Be nice to TNG servers (rate limit)
    time.sleep(1)

print(f"\nTotal size: ~{len(galaxies) * 20 / 1024:.1f} GB")
```

**Step 3: Render Locally**
```python
# See API_vs_LOCAL_DATA.md for rendering examples
```

---

## Disk Space Planning

### Minimum Setup (Current)
```
API Images:           93 MB
Code + Docs:          10 MB
Total:               103 MB
```

### Cutout Setup (Upgrade)
```
API Images:           93 MB
Cutout HDF5 files:    37 GB
Rendered images:      3.7 GB (high-res)
Code + Docs:          10 MB
Total:               ~41 GB
```

### Full Snapshot Setup (Research)
```
API Images:           93 MB
Snapshot (gas):      350 GB
Cutouts (optional):   37 GB
Rendered images:      3.7 GB
Code + Docs:          10 MB
Total:              ~391 GB
```

---

## When to Upgrade?

### Stick with API if:
✅ Classifier achieves >85% accuracy  
✅ Faint galaxies not critical  
✅ Speed > quality  
✅ Proof-of-concept stage  

### Upgrade to Cutouts if:
⚠️ Classifier plateaus at 75-85% accuracy  
⚠️ Missing faint jellyfish (false negatives)  
⚠️ Need multi-view augmentation  
⚠️ Preparing for publication  

### Upgrade to Full Snapshot if:
❌ Cutouts insufficient (rarely happens)  
❌ Need time evolution (multi-snapshot study)  
❌ Doing novel physics analysis  
❌ Building public image database  

---

## Bottom Line

**TNG-Cluster full dataset:** 209.7 TB (absurdly huge)  
**What you need:** 37 GB (cutouts) or 350 GB (snapshot)  
**What you have:** 93 MB (API images)

**Recommendation:** Start with API images (done!), upgrade to cutouts only if needed.

**Storage ratio:**
- API: 1×
- Cutouts: 400×
- Snapshot: 3,800×
- Full dataset: 2,200,000× (don't even think about it!)

✅ **You made the right choice using the API!**

# Simulation Data Explained: Snapshots vs Cutouts

## Understanding Cosmological Simulations

Think of TNG-Cluster as a **massive computer simulation** that recreates the universe's evolution from the Big Bang to today.

---

## What is a Snapshot?

### Simple Explanation

A **snapshot** is like taking a photograph of the entire simulated universe at one moment in time.

**Analogy:**
Imagine you're watching a movie of the universe evolving. A snapshot is like pausing the movie and saving one frame.

### Technical Details

**TNG-Cluster has 100 snapshots:**
- Snapshot 0: Universe at z=127 (just after Big Bang)
- Snapshot 50: Universe at z=1.0 (6 billion years ago)
- **Snapshot 99: Universe at z=0 (today, present day)** ← We use this!

**What's in a snapshot?**

A snapshot contains the positions, velocities, masses, and properties of **every single particle** in the simulation:

```
Snapshot 99 (z=0, present day):
├── Gas particles:        ~2 billion particles    (300 GB)
├── Dark matter particles: ~20 billion particles   (1 TB)
├── Star particles:       ~500 million particles  (50 GB)
└── Black holes:          ~352 supermassive BHs   (10 GB)

Total: ~22 billion particles = 1.5 TB of data
```

**Each particle has properties:**
```python
Gas Particle #1,234,567:
  - Position: (x=50234.5, y=12345.6, z=98765.4) kpc
  - Velocity: (vx=200, vy=-150, vz=300) km/s
  - Mass: 1.2e6 solar masses
  - Temperature: 10^6 Kelvin
  - Density: 0.001 particles/cm³
  - Metallicity: 0.02 (2% metals)
```

---

## What is "Gas" in a Snapshot?

### Simple Explanation

When we say "snapshot gas," we mean the **gas particles only** (not dark matter, not stars).

**What is gas in galaxies?**
- Hydrogen and helium (mostly)
- Forms stars
- Creates the beautiful nebulae and clouds you see in space images
- Gets stripped away in clusters (making jellyfish!)

### Why Download Gas Only?

**Full snapshot:** 1.5 TB
- 300 GB: Gas (✅ we need this!)
- 1 TB: Dark matter (❌ invisible, don't need for images)
- 50 GB: Stars (⚠️ optional, for context)
- 10 GB: Black holes (❌ don't need)

**Smart choice:** Download only gas = **300 GB** instead of 1.5 TB

**Why we need gas:**
- Jellyfish galaxies are defined by **gas stripping**
- We're rendering **gas density** projections
- Dark matter is invisible (doesn't emit light)
- Stars are optional (we already know stellar masses from API)

---

## What is a Cutout?

### Simple Explanation

A **cutout** is like cutting out just one galaxy from the full snapshot.

**Analogy:**
- **Snapshot** = Photograph of entire city from space
- **Cutout** = Zoomed-in photo of just your house and neighborhood

Instead of downloading the entire universe (1.5 TB), you ask TNG servers:
> "Hey, give me just the particles within 200 kpc of galaxy #25758"

The server extracts that small region and sends you a tiny file (~20 MB).

### Visual Example

```
Full Snapshot (1.5 TB):
┌────────────────────────────────────────┐
│  •  •   cluster1  •  •  •  •  •  •    │
│    •  •  •  • •  • •  •   •  •   •    │
│  •   •  cluster2   • • •  •  •  •  •  │
│    •  • •   •  • • •   •  •  •    •   │
│  •  •   •  • •  •   cluster3  •  •    │
│   •   • •  •  •  •  • •  • •  •  •    │
└────────────────────────────────────────┘
        22 billion particles

Cutout for galaxy in cluster2 (20 MB):
        ┌──────┐
        │ • •  │
        │ • • •│ ← Just this 200 kpc box
        │  • • │
        └──────┘
        ~100,000 particles
```

### Technical Details

**Cutout Service API:**
```python
# Request cutout for subhalo 25758
url = (
    "https://www.tng-project.org/api/TNG-Cluster/snapshots/99/"
    "subhalos/25758/cutout.hdf5"
    "?gas=Coordinates,Density,Masses,Temperature"
)

# This tells TNG servers:
# 1. Open snapshot 99 (1.5 TB file on their disks)
# 2. Find subhalo 25758
# 3. Extract all gas particles within 200 kpc
# 4. Send me just those particles (~20 MB)
```

**What you get:**
```
cutout_25758.hdf5 (20 MB):
├── PartType0/             # Gas particles
│   ├── Coordinates        # x,y,z positions (100k × 3 = 300k floats)
│   ├── Density            # Gas density (100k floats)
│   ├── Masses             # Particle masses (100k floats)
│   └── Temperature        # Gas temperature (100k floats)
```

---

## Comparison

### What Each Contains

| Data Type | Snapshot (1.5 TB) | Cutout (20 MB) |
|-----------|-------------------|----------------|
| **Scope** | Entire universe | One galaxy + surroundings |
| **Particles** | 22 billion | ~100,000 |
| **Galaxies** | All 352 clusters | 1 galaxy |
| **Region** | 300,000 kpc box | 400 kpc box (200 kpc radius) |

### When to Use Each

**Use Snapshot if:**
- You want to render **all** galaxies in the simulation
- You need **complete context** (neighboring galaxies, ICM, large-scale structure)
- You're doing **statistical analysis** across entire clusters
- You have **350 GB free space** and **8+ hours** to download

**Use Cutouts if:**
- You have a **specific list** of galaxies (like your 1,863)
- You want **only those galaxies** at high quality
- You have **40 GB free space** and **1-2 hours** to download
- You want **parallel downloads** (faster than one 1.5 TB file)

---

## How Rendering Works

### With API (Current Method)

```
You → TNG Server: "Give me image of subhalo 25758"
            ↓
TNG Server: Opens snapshot
            Extracts particles
            Renders image with their parameters
            Sends PNG (50 KB)
            ↓
You ← PNG image (fixed quality, no control)
```

**Pros:** Fast, simple  
**Cons:** Fixed quality, no customization

---

### With Cutout (Upgrade Method)

```
You → TNG Server: "Give me cutout of subhalo 25758"
            ↓
TNG Server: Opens snapshot
            Extracts particles
            Sends particle data (20 MB HDF5)
            ↓
You ← Particle data (positions, masses, temps)
            ↓
You: Render locally with Python
     - Choose resolution (424px or 4096px?)
     - Choose colormap (hot or viridis?)
     - Temperature weighting (T < 10^5 K only?)
     - Multiple projections (x, y, z views?)
     ↓
Custom high-quality images!
```

**Pros:** Full control, high quality  
**Cons:** Requires Python/astrophysics knowledge

---

### With Snapshot (Maximum Method)

```
You → TNG Server: "Give me full snapshot 99"
            ↓
TNG Server: Sends entire 1.5 TB file
            ↓
You: Download for 30+ hours
     Load snapshot with Python (requires 32+ GB RAM!)
     Select galaxies manually
     Render with custom code
     ↓
Maximum quality images!
```

**Pros:** Complete flexibility, offline work  
**Cons:** Huge storage, long download, high complexity

---

## Real Example: Your High-Gas Galaxy

### Galaxy: subhalo_25758 (155×10¹⁰ M☉ gas - the biggest!)

**Method 1: API (what you have)**
```
Size: 50 KB PNG
Resolution: 424×424 px
Download: 5 seconds
Quality: Good, but faint tails clipped
```

**Method 2: Cutout**
```
Raw data: 20 MB HDF5 (100k gas particles)
Render to: 2048×2048 PNG (5 MB)
Download: 20 seconds
Quality: Excellent, all tails visible
Custom: Temperature-weighted, cool gas only
```

**Method 3: Snapshot**
```
Full snapshot: 1.5 TB (includes this galaxy + 21.9 billion other particles)
Your galaxy: Same 100k particles as cutout
Download: 30 hours for everything
Quality: Same as cutout (but with 99.999% extra data you don't need!)
```

---

## Storage Math

### For Your 1,863 Galaxies

**API Images:**
```
1,863 galaxies × 50 KB = 93 MB
✅ Perfect!
```

**Cutouts:**
```
1,863 cutouts × 20 MB = 37 GB
✅ Still reasonable
```

**Snapshot:**
```
1 snapshot = 1.5 TB (entire universe)
❌ 40× bigger than needed!
```

---

## Why Cutouts Are Smart

Think of it like downloading movies:

**Snapshot approach:**
- Download entire Netflix catalog (1.5 TB)
- Watch 1,863 movies from it
- Delete the other 99.9% you don't need
- **Wasteful!**

**Cutout approach:**
- Download just the 1,863 movies you want (37 GB)
- Perfect quality
- No wasted bandwidth/storage
- **Smart!**

**API approach:**
- Stream those 1,863 movies (93 MB compressed)
- Lower quality, but instant
- **Convenient!**

---

## Particle Data Format

### What's in an HDF5 Cutout File?

```python
import h5py

# Open cutout
with h5py.File('cutout_25758.hdf5', 'r') as f:
    
    # Gas particles (PartType0)
    gas_pos = f['PartType0/Coordinates'][:]    # Shape: (100000, 3)
    gas_mass = f['PartType0/Masses'][:]        # Shape: (100000,)
    gas_temp = f['PartType0/Temperature'][:]   # Shape: (100000,)
    gas_rho = f['PartType0/Density'][:]        # Shape: (100000,)
    
    # Each particle:
    # gas_pos[0] = [x, y, z] in kpc
    # gas_mass[0] = mass in 10^10 M_sun/h
    # gas_temp[0] = temperature in Kelvin
    # gas_rho[0] = density in (10^10 M_sun/h) / (ckpc/h)^3
    
    # Example particle:
    print(f"Particle 0:")
    print(f"  Position: {gas_pos[0]}")      # [12345.6, 23456.7, 34567.8] kpc
    print(f"  Mass: {gas_mass[0]}")         # 0.000012 × 10^10 M_sun
    print(f"  Temperature: {gas_temp[0]}")  # 1500000 K (hot!)
    print(f"  Density: {gas_rho[0]}")       # 0.0001
```

This is **raw particle data** - you render it yourself!

---

## When to Use What?

### Decision Tree

```
Do you need better quality than API?
    │
    ├─ No → ✅ Use API images (current, 93 MB)
    │
    └─ Yes → Do you need ALL 1,863 galaxies?
              │
              ├─ Yes → Use Cutouts (37 GB, 1-2 hours)
              │
              └─ No (just top 100-200) → Use Cutouts (2-4 GB, 15 min)
                   │
                   └─ Still not enough? → Snapshot (350 GB, 8 hours)
                        │
                        └─ STILL not enough? → You don't need more!
```

---

## Rendering Complexity

### API: No Coding
```
Done! Images already rendered.
```

### Cutouts: Medium Coding
```python
# ~100 lines of Python
# Libraries: h5py, numpy, matplotlib
# Skills needed: Basic Python, understand projections
# Time to learn: 1-2 days
```

### Snapshot: Advanced Coding
```python
# ~500 lines of Python
# Libraries: h5py, numpy, scipy, yt/arepo
# Skills needed: Python + astrophysics + parallel processing
# Time to learn: 1-2 weeks
# Hardware: 32+ GB RAM, multi-core CPU
```

---

## My Recommendation (Again!)

### For Your Use Case

**Goal:** Train jellyfish classifier

**Current status:** 1,856 API images ready

**Action plan:**
1. ✅ Train classifier on API images (done next!)
2. ✅ Evaluate accuracy
3. 🔄 IF accuracy <85%:
   - Download cutouts for ~200 problematic galaxies
   - Re-render those at high quality
   - Retrain with mixed dataset
4. ⚠️ IF still not enough (rare):
   - Download cutouts for all 1,863 galaxies
   - Full high-quality dataset

**Most likely outcome:**
API images will give you 85-90% accuracy → no upgrade needed! 🎯

---

## Summary

| Term | What It Is | Size | Use Case |
|------|-----------|------|----------|
| **Snapshot** | Full universe at one time | 1.5 TB | Research, exploration |
| **Snapshot (gas)** | Just gas particles | 350 GB | Custom rendering, all galaxies |
| **Cutout** | One galaxy + surroundings | 20 MB | Targeted high-quality rendering |
| **API Image** | Pre-rendered PNG | 50 KB | Fast results, good quality |

**For you:** API images are perfect to start! Upgrade only if needed.

---

## Additional Resources

**If you want to learn more:**
- TNG Project: https://www.tng-project.org/
- Data access guide: https://www.tng-project.org/data/
- Python example scripts: https://www.tng-project.org/data/docs/scripts/

**Need help with cutouts?**
See `API_vs_LOCAL_DATA.md` for rendering examples and code.

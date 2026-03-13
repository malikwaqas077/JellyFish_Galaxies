# Disk Space Analysis

**Date:** 2026-03-02  
**System:** /dev/sda5 (root partition)

---

## Current Status

### Overall Disk Usage

```
Total:     228 GB
Used:       61 GB (27%)
Available: 156 GB (68%)
```

**Status:** ✅ **Healthy** - Plenty of free space

---

## Home Directory Breakdown

**Total home usage:** 37 GB

### Top Space Consumers

| Directory | Size | Purpose |
|-----------|------|---------|
| Android | 6.4 GB | Android SDK/tools |
| android-studio | 2.9 GB | IDE |
| flutter | 1.5 GB | Flutter SDK |
| drivemate | 1.5 GB | Project |
| Downloads | 1.1 GB | Downloaded files |

**Other:** ~24 GB (various projects, config files, etc.)

---

## JellyFish Project

**Total:** 48 MB (0.03% of disk, 0.13% of home)

### Breakdown

```
JellyFish_Galaxies/
├── output/images/     25 MB  (2,129 PNG files)
├── output/data/      580 KB  (CSV files)
├── code/logs/         ~10 MB  (extraction/generation logs)
└── documentation/     ~12 MB  (markdown, reports)
```

**Images:** 2,129 files × ~12 KB avg = 25 MB  
(Very efficient PNG compression!)

---

## Upgrade Capacity Analysis

### Can You Fit Different Approaches?

| Upgrade Option | Size Needed | Available | Fits? | After Install |
|----------------|-------------|-----------|-------|---------------|
| **Current (API)** | 48 MB | 156 GB | ✅ Done! | 156 GB free |
| **Cutouts** | 37 GB | 156 GB | ✅ YES | 119 GB free |
| **Snapshot (gas)** | 350 GB | 156 GB | ❌ NO | -194 GB (need more!) |
| **Full Snapshot** | 1.5 TB | 156 GB | ❌ NO | Way over! |

**Conclusion:**
- ✅ You CAN upgrade to cutouts if needed
- ❌ You CANNOT fit full snapshot without external drive
- ✅ Plenty of room for ML models, experiments, etc.

---

## Recommendations

### Option 1: Stay with API (Current) ✅

**Pros:**
- Already complete
- Minimal storage (48 MB)
- Leaves 156 GB for ML training, models, experiments
- Fast iteration

**Storage remaining:** 156 GB (plenty!)

---

### Option 2: Upgrade to Cutouts (If Needed)

**Storage plan:**
```
Current usage:        61 GB
Cutouts download:    +37 GB
Rendered images:      +4 GB
Total after:         102 GB
Remaining:           126 GB (55% free) ✅
```

**Still comfortable!**

**When to do this:**
- Classifier accuracy plateaus <85%
- Need higher resolution for faint features
- Preparing for publication

---

### Option 3: External Storage (For Snapshot)

If you absolutely need the full snapshot:

**Options:**
1. **External HDD (500 GB):** ~$30
   - Download snapshot → external drive
   - Render galaxies → copy to local
   - Keep snapshot offline for reference

2. **Cloud storage (Google Drive 2 TB):** ~$10/month
   - Upload rendered images for backup
   - Keep local storage free

3. **Clean up existing data:**
   - Android (6.4 GB) - archive old projects?
   - Downloads (1.1 GB) - clean up?
   - Could free 5-10 GB, still not enough for snapshot

**Verdict:** Not worth it unless doing serious research

---

## Storage Growth Projections

### If You Continue ML Work

**ML Training additions:**
```
Training data:        ~5 GB   (augmented images)
Model checkpoints:    ~2 GB   (saved weights)
TensorBoard logs:     ~1 GB   (training metrics)
Experiments:          ~5 GB   (various trials)
Total ML additions:  ~13 GB
```

**Projected usage:**
```
Current:              61 GB
ML work:             +13 GB
Future:               74 GB
Remaining:           154 GB (67% free) ✅
```

**Still very comfortable!**

---

### If You Add Cutouts

**Full research setup:**
```
Current:              61 GB
Cutouts:             +37 GB
Rendered high-res:    +4 GB
ML work:             +13 GB
Total:               115 GB
Remaining:           113 GB (50% free) ✅
```

**Still manageable!**

---

## Space-Saving Tips

### If You Need More Space Later

**Low-hanging fruit:**
1. **Downloads folder (1.1 GB):**
   - Archive or delete old files
   - Potential: +1 GB

2. **Old Android builds (6.4 GB):**
   - If not actively developing
   - Potential: +5-6 GB

3. **Log files (project logs):**
   - Compress or delete old logs
   - Potential: +500 MB

4. **System cache:**
   ```bash
   sudo apt clean
   sudo apt autoremove
   ```
   - Potential: +1-2 GB

**Total potential:** ~8-10 GB (not much in the grand scheme)

---

## Comparison to Requirements

### Your Needs vs Available Space

| Task | Needs | Available | Margin |
|------|-------|-----------|--------|
| API images | 48 MB | 156 GB | 3,250× |
| Cutouts | 37 GB | 156 GB | 4.2× |
| ML training | 13 GB | 156 GB | 12× |
| **All combined** | 50 GB | 156 GB | **3.1×** ✅ |

**Verdict:** You have 3× the space you need for full ML pipeline with cutouts!

---

## External Storage Options

### If You Want Snapshot (350 GB)

**Option A: External HDD**
- **Seagate 500 GB:** ~$30
- **WD 1 TB:** ~$50
- **USB 3.0:** Fast enough for rendering

**Option B: Cloud (Not Recommended)**
- **Too slow** for 350 GB download/upload
- **Expensive** for long-term storage
- **Not practical** for active work

**Option C: Upgrade Internal SSD**
- **Add 500 GB SSD:** ~$40-50
- **Permanent solution**
- **Faster than external**

**Recommendation:** Only if you're doing serious research publishing

---

## Monitoring Disk Usage

### Commands to Check Space

```bash
# Overall disk usage
df -h

# Home directory breakdown
du -sh /home/waqas/*

# JellyFish project size
du -sh /home/waqas/JellyFish_Galaxies

# Find large files
find /home/waqas -type f -size +100M

# Clean package cache
sudo apt clean
sudo apt autoremove
```

---

## Conclusion

**Current status:** ✅ **Excellent**
- 156 GB free (68% of disk)
- JellyFish project: only 48 MB (0.03%)
- Room for full ML pipeline + cutouts

**Recommendations:**
1. ✅ **Continue with API images** (already done, efficient)
2. ✅ **You CAN afford cutouts** if needed (37 GB)
3. ❌ **Skip snapshot** unless absolutely necessary (350 GB)
4. ✅ **No storage concerns** for foreseeable ML work

**Bottom line:** You're in great shape! No storage issues. 🎯

# Branch: feature/2000-images-mixed-gas

## Goal
Generate **2,000+ galaxy images** with varying gas content (LOW/MEDIUM/HIGH) for training a jellyfish classifier model.

## Strategy Change

### Master Branch (Baseline)
- **Target:** High-quality jellyfish candidates only
- **Criteria:** Strict gas mass requirements (> 7,000 M_sun minimum)
- **Result:** 760 images (38% of 2,000 target)
- **Limitation:** TNG-Cluster has limited gas-rich satellites

### This Branch (2000+ Images)
- **Target:** 2,000+ satellite galaxies with MIXED gas content
- **Criteria:** Include gas-poor, gas-moderate, and gas-rich
- **Philosophy:** Cast wide net, let ML classifier sort them out
- **Expected mix:**
  - **LOW gas** (0-0.1 × 10¹⁰ M_sun): "Red & dead" stripped satellites
  - **MEDIUM gas** (0.1-1.0 × 10¹⁰ M_sun): Possible ongoing stripping
  - **HIGH gas** (>1.0 × 10¹⁰ M_sun): Prime jellyfish candidates

## Config Changes

### Cluster Selection
```python
MIN_CLUSTER_MASS: 3,000 → 1,000 (×10¹⁰ M_sun/h)
```
- Lowers threshold from ~2×10¹³ to ~7×10¹² solar masses
- Captures more clusters (expect 300-400 vs 188)

### Galaxy Selection
```python
MIN_GAS_MASS: 0.000001 → 0.0 (×10¹⁰ M_sun/h)
MIN_STELLAR_MASS: 0.00001 → 0.001 (×10¹⁰ M_sun/h)
MAX_HALFMASS_GAS: 500 → 1000 (kpc)
```

**Key change:** `MIN_GAS_MASS = 0.0`
- Includes completely gas-stripped satellites
- These are scientifically interesting (post-stripping stage)
- Provides negative examples for classifier

### Code Logic Updates
```python
# Half-mass radius check: only enforced for gas-rich galaxies
if m_gas > 0.01 and r_half_g > MAX_HALFMASS_GAS_CKPC_H:
    return None
```
- Gas-poor galaxies (<0.01 × 10¹⁰ M_sun) can have undefined/large gas radius
- Prevents rejection of legitimate stripped satellites

## Expected Outcomes

### Cluster Count
- **Before:** 188 clusters
- **After:** ~350-400 clusters (estimated)

### Satellites Per Cluster
- **Before:** ~4 avg (strict gas cuts)
- **After:** ~8-15 avg (inclusive cuts)

### Total Galaxies
- **Target:** 2,000+
- **Expected:** 2,500-4,000 (if estimates correct)

### Gas Distribution (estimated)
- **LOW:** 40-50% (~1,000-1,500 images) - stripped satellites
- **MEDIUM:** 30-40% (~800-1,200 images) - ongoing stripping
- **HIGH:** 10-20% (~400-800 images) - jellyfish candidates

## Classifier Training Benefits

1. **Balanced dataset:** Mix of positive (jellyfish) and negative (non-jellyfish) examples
2. **Gas context:** Model learns what gas content looks like at different stages
3. **Real distribution:** Reflects actual cluster satellite populations
4. **Transfer learning:** Can distinguish jellyfish features from just "has gas"

## Next Steps

1. ✅ Update config and extraction logic
2. 🔄 Run extraction: `python3 02_extract_galaxy_list.py`
3. 🔄 Generate images: `python3 03_generate_images_parallel.py`
4. 📊 Analyze gas distribution in `galaxy_list.csv`
5. 🖼️ Generate sample gallery by gas category
6. 🤖 Feed to classifier model for training

## Comparison Files

To compare with baseline:
```bash
# Baseline (master branch)
git checkout master
cat output/data/galaxy_list.csv | wc -l  # 775 lines (774 galaxies + header)

# This branch
git checkout feature/2000-images-mixed-gas
# After running: should see 2000+ lines
```

## Rollback

If this approach doesn't work:
```bash
git checkout master
# Baseline preserved with 760 high-quality images
```

---

**Branch Author:** AI Assistant  
**Date:** 2026-02-28  
**Commit:** [Will be updated after first commit]

#!/bin/bash
# MAXIMUM IMAGE GENERATION MODE - Continuous Run
# This script will keep generating images until all possible galaxies are processed

set -e

LOG_DIR="logs_maximum"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/maximum_run_$TIMESTAMP.log"

echo "==================================================================" | tee -a "$MAIN_LOG"
echo "  JELLYFISH GALAXIES - MAXIMUM IMAGE GENERATION MODE" | tee -a "$MAIN_LOG"
echo "==================================================================" | tee -a "$MAIN_LOG"
echo "Started: $(date)" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Step 1: Find ALL clusters
echo "[STEP 1/3] Finding ALL clusters from TNG-Cluster..." | tee -a "$MAIN_LOG"
python3 01_find_clusters.py 2>&1 | tee -a "$MAIN_LOG"
CLUSTERS=$(wc -l < output/data/clusters.csv)
CLUSTERS=$((CLUSTERS - 1))  # Subtract header
echo "✓ Found $CLUSTERS clusters" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Step 2: Extract ALL galaxies
echo "[STEP 2/3] Extracting galaxies from $CLUSTERS clusters..." | tee -a "$MAIN_LOG"
python3 02_extract_galaxy_list.py 2>&1 | tee -a "$MAIN_LOG"
GALAXIES=$(wc -l < output/data/galaxy_list.csv)
GALAXIES=$((GALAXIES - 1))  # Subtract header
echo "✓ Extracted $GALAXIES galaxies" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Step 3: Generate ALL images (parallel mode)
echo "[STEP 3/3] Generating images for $GALAXIES galaxies (8 parallel workers)..." | tee -a "$MAIN_LOG"
python3 03_generate_images_parallel.py 2>&1 | tee -a "$MAIN_LOG"
IMAGES=$(ls output/images/*.png 2>/dev/null | wc -l)
echo "✓ Generated $IMAGES images" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

echo "==================================================================" | tee -a "$MAIN_LOG"
echo "  MAXIMUM MODE COMPLETE!" | tee -a "$MAIN_LOG"
echo "==================================================================" | tee -a "$MAIN_LOG"
echo "Finished: $(date)" | tee -a "$MAIN_LOG"
echo "Clusters processed: $CLUSTERS" | tee -a "$MAIN_LOG"
echo "Galaxies extracted: $GALAXIES" | tee -a "$MAIN_LOG"
echo "Images generated: $IMAGES" | tee -a "$MAIN_LOG"
echo "Success rate: $(echo "scale=2; $IMAGES * 100 / $GALAXIES" | bc)%" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "Output directory: output/images/" | tee -a "$MAIN_LOG"
echo "Log file: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "==================================================================" | tee -a "$MAIN_LOG"

#!/bin/bash
# AUTO-CONTINUE PIPELINE - Waits for extraction, then starts image generation automatically
# This ensures no duplicates and continuous operation

set -e

LOG_FILE="auto_continue.log"

echo "======================================" | tee -a "$LOG_FILE"
echo "  AUTO-CONTINUE MAXIMUM MODE" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Wait for extraction to complete
echo "⏳ Waiting for galaxy extraction to complete..." | tee -a "$LOG_FILE"
while ps aux | grep -q "[p]ython3 02_extract_galaxy"; do
    CURRENT_GALAXIES=$(($(wc -l < output/data/galaxy_list.csv 2>/dev/null || echo 1) - 1))
    if [ $CURRENT_GALAXIES -gt 0 ]; then
        echo "  In progress... ($CURRENT_GALAXIES galaxies so far) - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
    fi
    sleep 60  # Check every minute
done

echo "✓ Galaxy extraction complete!" | tee -a "$LOG_FILE"
sleep 5

# Check results
NEW_GALAXIES=$(($(wc -l < output/data/galaxy_list.csv) - 1))
EXISTING_IMAGES=$(wc -l < output/data/image_log.csv)
EXISTING_IMAGES=$((EXISTING_IMAGES - 1))  # Subtract header
TO_GENERATE=$((NEW_GALAXIES - EXISTING_IMAGES))

echo "" | tee -a "$LOG_FILE"
echo "📊 Extraction Results:" | tee -a "$LOG_FILE"
echo "  Total galaxies: $NEW_GALAXIES" | tee -a "$LOG_FILE"
echo "  Already processed: $EXISTING_IMAGES" | tee -a "$LOG_FILE"
echo "  NEW images to generate: $TO_GENERATE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $TO_GENERATE -le 0 ]; then
    echo "✓ No new images needed! All galaxies already processed." | tee -a "$LOG_FILE"
    exit 0
fi

# Start image generation with deduplication
echo "🎨 Starting parallel image generation ($TO_GENERATE new images)..." | tee -a "$LOG_FILE"
echo "   8 parallel workers | Will skip $EXISTING_IMAGES existing images" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python3 03_generate_images_parallel.py 2>&1 | tee -a "$LOG_FILE"

# Final stats
FINAL_IMAGES=$(ls output/images/*.png 2>/dev/null | wc -l)
NEW_GENERATED=$((FINAL_IMAGES - EXISTING_IMAGES))

echo "" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"
echo "  ✓ MAXIMUM MODE COMPLETE!" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📊 Final Statistics:" | tee -a "$LOG_FILE"
echo "  Total galaxies extracted: $NEW_GALAXIES" | tee -a "$LOG_FILE"
echo "  Total images generated: $FINAL_IMAGES" | tee -a "$LOG_FILE"
echo "  NEW images this run: $NEW_GENERATED" | tee -a "$LOG_FILE"
echo "  Duplicates avoided: $EXISTING_IMAGES" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Success rate: $(echo "scale=2; $FINAL_IMAGES * 100 / $NEW_GALAXIES" | bc)%" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"

# Optional: Auto-commit to git
read -p "Commit new images to git? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git add output/
    git commit -m "Add $NEW_GENERATED new jellyfish galaxy images (MAXIMUM mode: $FINAL_IMAGES total)"
    echo "✓ Committed to git (not pushed - run 'git push' to upload)"
fi

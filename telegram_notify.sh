#!/bin/bash
# Telegram Progress Notifications for JellyFish Pipeline

WORK_DIR="/home/waqas/JellyFish_Galaxies"
cd "$WORK_DIR"

# Gather statistics
CLUSTERS_TOTAL=$(wc -l < output/data/clusters.csv 2>/dev/null || echo 1)
CLUSTERS_TOTAL=$((CLUSTERS_TOTAL - 1))

GALAXIES_TOTAL=$(wc -l < output/data/galaxy_list.csv 2>/dev/null || echo 1)
GALAXIES_TOTAL=$((GALAXIES_TOTAL - 1))

IMAGES_TOTAL=$(ls output/images/*.png 2>/dev/null | wc -l)

# Check what's running
EXTRACTION_RUNNING="❌ Stopped"
GENERATION_RUNNING="⏳ Not started"

if ps aux | grep -q "[p]ython3 02_extract_galaxy_list.py"; then
    EXTRACTION_PID=$(ps aux | grep "[p]ython3 02_extract_galaxy" | awk '{print $2}' | head -1)
    EXTRACTION_TIME=$(ps -p $EXTRACTION_PID -o etime= 2>/dev/null | xargs || echo "unknown")
    EXTRACTION_RUNNING="✅ Running (${EXTRACTION_TIME})"
    
    # Get current cluster from log
    CURRENT_CLUSTER=$(tail -1 extract_2000plus_restart.log 2>/dev/null | grep -oP '\[\d+/\d+\]' || echo "?/?")
fi

if ps aux | grep -q "[p]ython3 03_generate_images"; then
    GEN_PID=$(ps aux | grep "[p]ython3 03_generate_images" | awk '{print $2}' | head -1)
    GEN_TIME=$(ps -p $GEN_PID -o etime= 2>/dev/null | xargs || echo "unknown")
    GENERATION_RUNNING="✅ Running (${GEN_TIME})"
fi

# Calculate progress
if [ $GALAXIES_TOTAL -gt 0 ]; then
    PROGRESS=$(echo "scale=1; $IMAGES_TOTAL * 100 / $GALAXIES_TOTAL" | bc 2>/dev/null || echo "0.0")
    REMAINING=$((GALAXIES_TOTAL - IMAGES_TOTAL))
else
    PROGRESS="0.0"
    REMAINING="?"
fi

# Determine phase
PHASE="🔄 Phase 1: Extracting Galaxies"
if [ $GALAXIES_TOTAL -ge 1000 ]; then
    PHASE="🎨 Phase 2: Generating Images"
fi

# Build message
MESSAGE="🌌 *JellyFish Galaxies Pipeline Update*

$PHASE

📊 *Statistics:*
• Clusters found: $CLUSTERS_TOTAL
• Galaxies extracted: $GALAXIES_TOTAL
• Images generated: $IMAGES_TOTAL
• Progress: $PROGRESS%
• Remaining: $REMAINING images

🔄 *Status:*
• Extraction: $EXTRACTION_RUNNING
• Image Generation: $GENERATION_RUNNING

🎯 *Target:* 2000+ images from TNG-Cluster
🤖 *Watchdog:* Active (checks every 10 min)

_Next update in 2 hours_
⏰ $(date '+%Y-%m-%d %H:%M:%S')"

# Send to Telegram (escape special characters for markdown)
MESSAGE_ESCAPED=$(echo "$MESSAGE" | sed 's/\./\\./g; s/-/\\-/g; s/(/\\(/g; s/)/\\)/g; s/+/\\+/g')

# Log the notification
echo "[$(date)] Sending Telegram notification..." >> telegram_notify.log
echo "$MESSAGE" >> telegram_notify.log
echo "---" >> telegram_notify.log

# Use OpenClaw to send message (will be sent via Telegram channel)
# The message will be sent to the current chat context
echo "$MESSAGE"

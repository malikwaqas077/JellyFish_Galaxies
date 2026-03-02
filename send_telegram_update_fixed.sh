#!/bin/bash
# Send Telegram progress update (stores message for OpenClaw to pick up)

WORK_DIR="/home/waqas/JellyFish_Galaxies"
cd "$WORK_DIR"

# Create updates directory
mkdir -p telegram_updates

# Gather statistics
CLUSTERS=$(wc -l < output/data/clusters.csv 2>/dev/null || echo 1)
CLUSTERS=$((CLUSTERS - 1))

GALAXIES=$(wc -l < output/data/galaxy_list.csv 2>/dev/null || echo 1)
GALAXIES=$((GALAXIES - 1))

IMAGES=$(ls output/images/*.png 2>/dev/null | wc -l)

# Calculate progress
if [ $GALAXIES -gt 0 ]; then
    PROGRESS=$(echo "scale=1; $IMAGES * 100 / $GALAXIES" | bc 2>/dev/null || echo "0.0")
    REMAINING=$((GALAXIES - IMAGES))
else
    PROGRESS="0.0"
    REMAINING="?"
fi

# Check extraction status
if ps aux | grep -q "[p]ython3 02_extract_galaxy"; then
    EXTRACT_PID=$(ps aux | grep "[p]ython3 02_extract_galaxy" | awk '{print $2}' | head -1)
    EXTRACT_TIME=$(ps -p $EXTRACT_PID -o etime= 2>/dev/null | xargs || echo "?")
    CURRENT=$(tail -1 extract_2000plus_restart.log 2>/dev/null | grep -oP '\[\d+/188\]' || echo "")
    EXTRACT_STATUS="✅ Running $CURRENT ($EXTRACT_TIME)"
else
    EXTRACT_STATUS="❌ Stopped"
fi

# Check image generation status
if ps aux | grep -q "[p]ython3 03_generate_images"; then
    GEN_PID=$(ps aux | grep "[p]ython3 03_generate_images" | awk '{print $2}' | head -1)
    GEN_TIME=$(ps -p $GEN_PID -o etime= 2>/dev/null | xargs || echo "?")
    GEN_STATUS="✅ Running ($GEN_TIME)"
else
    if [ $GALAXIES -ge 1000 ]; then
        GEN_STATUS="⚠️ Should be running"
    else
        GEN_STATUS="⏳ Not started"
    fi
fi

# Determine phase
PHASE="🔄 Phase 1: Extracting Galaxies"
if [ $GALAXIES -ge 1000 ] && [ $IMAGES -lt $GALAXIES ]; then
    PHASE="🎨 Phase 2: Generating Images"
elif [ $IMAGES -ge 2000 ]; then
    PHASE="🎉 Complete - 2000+ Images Done!"
fi

# Build message
MESSAGE="🌌 *JellyFish Galaxies Update*

$PHASE

📊 *Statistics:*
• Clusters: $CLUSTERS
• Galaxies: $GALAXIES
• Images: *$IMAGES* / 2000+ target
• Progress: *$PROGRESS%*
• Remaining: $REMAINING

🔄 *Status:*
• Extraction: $EXTRACT_STATUS
• Generation: $GEN_STATUS
• Watchdog: ✅ Active

⏰ $(date '+%Y-%m-%d %H:%M')
_Next update in 2 hours_"

# Save to file for pickup
TIMESTAMP=$(date +%s)
echo "$MESSAGE" > "telegram_updates/update_${TIMESTAMP}.txt"

# Log
echo "[$(date)] Update saved: Clusters=$CLUSTERS, Galaxies=$GALAXIES, Images=$IMAGES" >> telegram_notify.log

# Output for immediate send
echo "$MESSAGE"

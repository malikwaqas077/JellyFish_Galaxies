#!/bin/bash
# JellyFish Galaxies Pipeline Watchdog
# Ensures the pipeline keeps running and auto-restarts if stopped

set -e

WORK_DIR="/home/waqas/JellyFish_Galaxies"
LOG_FILE="$WORK_DIR/watchdog.log"
LOCK_FILE="$WORK_DIR/.watchdog.lock"
MAX_LOG_SIZE=10485760  # 10MB

cd "$WORK_DIR"

# Rotate log if too large
if [ -f "$LOG_FILE" ] && [ $(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE") -gt $MAX_LOG_SIZE ]; then
    mv "$LOG_FILE" "$LOG_FILE.old"
fi

# Prevent multiple watchdog instances
if [ -f "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE")
    if ps -p "$LOCK_PID" > /dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watchdog already running (PID: $LOCK_PID)" >> "$LOG_FILE"
        exit 0
    fi
fi
echo $$ > "$LOCK_FILE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if extraction is running
check_extraction() {
    if ps aux | grep -q "[p]ython3 02_extract_galaxy_list.py"; then
        return 0  # Running
    else
        return 1  # Not running
    fi
}

# Check if image generation is running
check_image_generation() {
    if ps aux | grep -q "[p]ython3 03_generate_images"; then
        return 0  # Running
    else
        return 1  # Not running
    fi
}

# Check if extraction is complete
extraction_complete() {
    if [ ! -f output/data/galaxy_list.csv ]; then
        return 1  # Not complete
    fi
    
    # Check if we have enough clusters processed
    CLUSTERS_FOUND=$(wc -l < output/data/clusters.csv 2>/dev/null || echo 0)
    CLUSTERS_FOUND=$((CLUSTERS_FOUND - 1))  # Subtract header
    
    if [ $CLUSTERS_FOUND -eq 0 ]; then
        return 1  # No clusters yet
    fi
    
    # Check if extraction log shows completion
    if grep -q "Total galaxies selected:" extract_2000plus_restart.log 2>/dev/null || \
       grep -q "Saved ->" extract_2000plus_restart.log 2>/dev/null; then
        return 0  # Complete
    fi
    
    return 1  # Not complete
}

# Check if extraction is stuck (no progress in 30 minutes)
extraction_stuck() {
    LOG_FILES="extract_2000plus_restart.log extract_2000plus.log"
    
    for LOG_F in $LOG_FILES; do
        if [ -f "$LOG_F" ]; then
            # Get last modification time
            LAST_MOD=$(stat -c %Y "$LOG_F" 2>/dev/null || stat -f %m "$LOG_F" 2>/dev/null || echo 0)
            NOW=$(date +%s)
            DIFF=$((NOW - LAST_MOD))
            
            # If log hasn't been updated in 30 minutes and process is running
            if [ $DIFF -gt 1800 ] && check_extraction; then
                return 0  # Stuck
            fi
        fi
    done
    
    return 1  # Not stuck
}

# Start extraction
start_extraction() {
    log "Starting galaxy extraction (188 clusters, 2000+ target)"
    nohup python3 02_extract_galaxy_list.py > extract_2000plus_restart.log 2>&1 &
    log "Extraction started (PID: $!)"
}

# Start image generation
start_image_generation() {
    GALAXIES=$(wc -l < output/data/galaxy_list.csv 2>/dev/null || echo 1)
    GALAXIES=$((GALAXIES - 1))
    
    log "Starting parallel image generation ($GALAXIES galaxies)"
    nohup python3 03_generate_images_parallel.py > image_gen_auto.log 2>&1 &
    log "Image generation started (PID: $!)"
}

# Main watchdog logic
main() {
    log "=== Watchdog Check ==="
    
    # Phase 1: Check extraction
    if ! extraction_complete; then
        if check_extraction; then
            # Extraction is running
            if extraction_stuck; then
                log "⚠️  Extraction appears stuck! Restarting..."
                pkill -f "python3 02_extract_galaxy_list.py" || true
                sleep 5
                start_extraction
            else
                log "✓ Extraction running normally (cluster $(tail -1 extract_2000plus_restart.log 2>/dev/null | grep -oP '\[\d+/\d+\]' || echo 'unknown'))"
            fi
        else
            # Extraction not running but should be
            log "⚠️  Extraction stopped! Restarting..."
            start_extraction
        fi
    else
        # Phase 2: Extraction complete, check image generation
        log "✓ Extraction complete"
        
        GALAXIES=$(wc -l < output/data/galaxy_list.csv 2>/dev/null || echo 1)
        GALAXIES=$((GALAXIES - 1))
        IMAGES=$(ls output/images/*.png 2>/dev/null | wc -l)
        REMAINING=$((GALAXIES - IMAGES))
        
        # Check completion: allow up to 2% failure rate (astronomical data tolerance)
        SUCCESS_RATE=$(awk "BEGIN {print ($IMAGES / $GALAXIES) * 100}")
        
        if [ $REMAINING -gt 0 ] && awk "BEGIN {exit !($SUCCESS_RATE < 98)}"; then
            if check_image_generation; then
                log "✓ Image generation running ($IMAGES/$GALAXIES images, $REMAINING remaining)"
            else
                # Check if generation is paused
                if [ -f "$WORK_DIR/.pause_generation" ]; then
                    log "⏸️  Image generation paused (see .pause_generation file)"
                else
                    log "⚠️  Image generation not running! Starting..."
                    start_image_generation
                fi
            fi
        else
            if [ $REMAINING -eq 0 ]; then
                log "🎉 COMPLETE! All $GALAXIES images generated!"
            else
                log "🎉 COMPLETE! $IMAGES/$GALAXIES images ($SUCCESS_RATE% success)"
                log "   Remaining $REMAINING likely failed permanently (corrupt TNG data)"
            fi
            # Mission accomplished - clean up lock and exit
            rm -f "$LOCK_FILE"
            exit 0
        fi
    fi
    
    # Statistics
    CLUSTERS=$(wc -l < output/data/clusters.csv 2>/dev/null || echo 1)
    CLUSTERS=$((CLUSTERS - 1))
    log "Stats: Clusters=$CLUSTERS, Galaxies=$GALAXIES, Images=$IMAGES"
    
    # Clean up lock
    rm -f "$LOCK_FILE"
}

# Run main logic
main

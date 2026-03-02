#!/bin/bash

# JellyFish Pipeline Restart Monitor
# Only alerts when the watchdog actually restarts something

LOG_FILE="/home/waqas/JellyFish_Galaxies/watchdog.log"
STATE_FILE="/home/waqas/JellyFish_Galaxies/.monitor_state"
TELEGRAM_SCRIPT="/home/waqas/.openclaw/workspace/send_telegram_alert.sh"

# Get the last line number we checked
if [ -f "$STATE_FILE" ]; then
    LAST_LINE=$(cat "$STATE_FILE")
else
    LAST_LINE=0
fi

# Get current line count
CURRENT_LINE=$(wc -l < "$LOG_FILE")

# If there are new lines, check for restart events
if [ $CURRENT_LINE -gt $LAST_LINE ]; then
    # Extract new lines
    NEW_LINES=$(tail -n +$((LAST_LINE + 1)) "$LOG_FILE" | head -n $((CURRENT_LINE - LAST_LINE)))
    
    # Check for restart keywords
    if echo "$NEW_LINES" | grep -qE "(Starting|Restarting|started|Image generation started|Extraction started)"; then
        # Extract relevant restart info
        RESTART_INFO=$(echo "$NEW_LINES" | grep -E "(Starting|Restarting|started)" | tail -5)
        
        # Get current stats
        STATS=$(bash /home/waqas/JellyFish_Galaxies/pipeline_control.sh status 2>&1 | grep -A 10 "Progress:")
        
        # Format alert message
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        MESSAGE="🔄 **JellyFish Pipeline Auto-Restart**

⏰ $TIMESTAMP

🔧 **Action Taken:**
$RESTART_INFO

📊 **Current Status:**
$STATS"
        
        # Send to Telegram via OpenClaw
        echo "$MESSAGE"
        
        # You can add telegram notification here if needed
    fi
    
    # Update state file
    echo "$CURRENT_LINE" > "$STATE_FILE"
fi

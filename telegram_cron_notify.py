#!/usr/bin/env python3
"""
Telegram Progress Notifier - Runs from cron
Sends pipeline updates via wake event to trigger OpenClaw notification
"""

import os
import subprocess
import sys
from datetime import datetime

WORK_DIR = "/home/waqas/JellyFish_Galaxies"
os.chdir(WORK_DIR)

def count_lines(filepath):
    """Count lines in a file (excluding header)"""
    try:
        with open(filepath, 'r') as f:
            return max(0, len(f.readlines()) - 1)
    except:
        return 0

def count_images():
    """Count PNG files in images directory"""
    try:
        result = subprocess.run(
            ['ls', 'output/images/*.png'],
            capture_output=True,
            text=True,
            shell=True
        )
        return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    except:
        return 0

def check_process(pattern):
    """Check if a process is running"""
    result = subprocess.run(
        ['ps', 'aux'],
        capture_output=True,
        text=True
    )
    return pattern in result.stdout

def get_process_time(pattern):
    """Get process runtime"""
    result = subprocess.run(
        f"ps aux | grep '{pattern}' | grep -v grep | awk '{{print $10}}'",
        capture_output=True,
        text=True,
        shell=True
    )
    return result.stdout.strip() or "?"

def get_current_cluster():
    """Get current cluster being processed"""
    try:
        with open('extract_2000plus_restart.log', 'r') as f:
            last_line = f.readlines()[-1]
            # Extract [X/188] pattern
            import re
            match = re.search(r'\[(\d+)/188\]', last_line)
            if match:
                return match.group(0)
    except:
        pass
    return ""

# Gather stats
clusters = count_lines('output/data/clusters.csv')
galaxies = count_lines('output/data/galaxy_list.csv')
images = count_images()

# Calculate progress
if galaxies > 0:
    progress = round(images * 100 / galaxies, 1)
    remaining = galaxies - images
else:
    progress = 0.0
    remaining = "?"

# Check processes
extract_running = check_process("python3 02_extract_galaxy")
gen_running = check_process("python3 03_generate_images")

if extract_running:
    extract_time = get_process_time("02_extract_galaxy")
    current_cluster = get_current_cluster()
    extract_status = f"✅ Running {current_cluster} ({extract_time})"
else:
    extract_status = "❌ Stopped"

if gen_running:
    gen_time = get_process_time("03_generate_images")
    gen_status = f"✅ Running ({gen_time})"
else:
    if galaxies >= 1000:
        gen_status = "⚠️ Should be running"
    else:
        gen_status = "⏳ Not started"

# Determine phase
if images >= 2000:
    phase = "🎉 Complete - 2000+ Images Done!"
elif galaxies >= 1000 and images < galaxies:
    phase = "🎨 Phase 2: Generating Images"
else:
    phase = "🔄 Phase 1: Extracting Galaxies"

# Build message
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
message = f"""🌌 JellyFish Galaxies Update

{phase}

📊 Statistics:
• Clusters: {clusters}
• Galaxies: {galaxies}
• Images: {images} / 2000+ target
• Progress: {progress}%
• Remaining: {remaining}

🔄 Status:
• Extraction: {extract_status}
• Generation: {gen_status}
• Watchdog: ✅ Active

⏰ {timestamp}
Next update in 2 hours"""

# Write to trigger file
with open('telegram_updates/.trigger', 'w') as f:
    f.write(message)

# Log
with open('telegram_notify.log', 'a') as f:
    f.write(f"[{timestamp}] Update: Clusters={clusters}, Galaxies={galaxies}, Images={images}\n")

# Output message (for manual testing)
print(message)
print("\n✓ Update message prepared")

#!/bin/bash
# JellyFish Galaxies Pipeline Control Script

WORK_DIR="/home/waqas/JellyFish_Galaxies"
cd "$WORK_DIR"

show_status() {
    echo "========================================================"
    echo "  JELLYFISH GALAXIES PIPELINE STATUS"
    echo "========================================================"
    echo ""
    
    # Check processes
    echo "🔄 Running Processes:"
    if ps aux | grep -q "[p]ython3 02_extract_galaxy"; then
        EXTRACT_PID=$(ps aux | grep "[p]ython3 02_extract_galaxy" | awk '{print $2}' | head -1)
        EXTRACT_TIME=$(ps -p $EXTRACT_PID -o etime= 2>/dev/null || echo "unknown")
        echo "  ✓ Galaxy extraction RUNNING (PID: $EXTRACT_PID, Runtime: $EXTRACT_TIME)"
    else
        echo "  ⚠️  Galaxy extraction NOT running"
    fi
    
    if ps aux | grep -q "[p]ython3 03_generate_images"; then
        GEN_PID=$(ps aux | grep "[p]ython3 03_generate_images" | awk '{print $2}' | head -1)
        GEN_TIME=$(ps -p $GEN_PID -o etime= 2>/dev/null || echo "unknown")
        echo "  ✓ Image generation RUNNING (PID: $GEN_PID, Runtime: $GEN_TIME)"
    else
        echo "  ⚠️  Image generation NOT running"
    fi
    echo ""
    
    # Check cron
    echo "🤖 Watchdog:"
    if crontab -l 2>/dev/null | grep -q "watchdog.sh"; then
        echo "  ✓ Cron job ACTIVE (runs every 10 minutes)"
        MINUTES=$((10#$(date +%M)))
        echo "  Next check in: ~$(( 10 - MINUTES % 10 )) minutes"
    else
        echo "  ⚠️  Cron job NOT installed"
    fi
    echo ""
    
    # Statistics
    echo "📊 Progress:"
    CLUSTERS=$(wc -l < output/data/clusters.csv 2>/dev/null || echo 1)
    CLUSTERS=$((CLUSTERS - 1))
    GALAXIES=$(wc -l < output/data/galaxy_list.csv 2>/dev/null || echo 1)
    GALAXIES=$((GALAXIES - 1))
    IMAGES=$(ls output/images/*.png 2>/dev/null | wc -l)
    
    echo "  Clusters found: $CLUSTERS"
    echo "  Galaxies extracted: $GALAXIES"
    echo "  Images generated: $IMAGES"
    
    if [ $GALAXIES -gt 0 ]; then
        PERCENT=$(echo "scale=1; $IMAGES * 100 / $GALAXIES" | bc 2>/dev/null || echo "0")
        echo "  Progress: $PERCENT%"
        REMAINING=$((GALAXIES - IMAGES))
        if [ $REMAINING -gt 0 ]; then
            echo "  Remaining: $REMAINING images"
        fi
    fi
    
    echo ""
    echo "📝 Recent Activity:"
    tail -5 watchdog.log 2>/dev/null | sed 's/^/  /'
    echo ""
    echo "========================================================"
}

start_watchdog_manual() {
    echo "Starting watchdog manually..."
    ./watchdog.sh
    echo "✓ Watchdog check complete!"
}

install_cron() {
    echo "Installing cron job (runs every 10 minutes)..."
    (crontab -l 2>/dev/null | grep -v "jellyfish_watchdog"; echo "# JellyFish Galaxies Pipeline Watchdog - runs every 10 minutes") | crontab -
    (crontab -l; echo "*/10 * * * * /home/waqas/JellyFish_Galaxies/watchdog.sh") | crontab -
    echo "✓ Cron job installed!"
    crontab -l | grep -A1 "JellyFish"
}

remove_cron() {
    echo "Removing cron job..."
    crontab -l 2>/dev/null | grep -v "watchdog.sh" | grep -v "JellyFish Galaxies" | crontab -
    echo "✓ Cron job removed!"
}

stop_all() {
    echo "Stopping all pipeline processes..."
    pkill -f "python3 02_extract_galaxy_list.py" && echo "  ✓ Stopped extraction" || echo "  - Extraction not running"
    pkill -f "python3 03_generate_images" && echo "  ✓ Stopped image generation" || echo "  - Image generation not running"
    echo "✓ All processes stopped!"
}

force_start_extraction() {
    echo "Force starting extraction..."
    stop_all
    sleep 2
    nohup python3 02_extract_galaxy_list.py > extract_2000plus_restart.log 2>&1 &
    echo "✓ Extraction started (PID: $!)"
}

force_start_generation() {
    echo "Force starting image generation..."
    pkill -f "python3 03_generate_images" || true
    sleep 2
    nohup python3 03_generate_images_parallel.py > image_gen_auto.log 2>&1 &
    echo "✓ Image generation started (PID: $!)"
}

show_logs() {
    echo "======== WATCHDOG LOG (last 30 lines) ========"
    tail -30 watchdog.log 2>/dev/null || echo "No watchdog log yet"
    echo ""
    echo "======== EXTRACTION LOG (last 20 lines) ========"
    tail -20 extract_2000plus_restart.log 2>/dev/null || echo "No extraction log yet"
}

case "${1:-status}" in
    status)
        show_status
        ;;
    start)
        start_watchdog_manual
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        sleep 2
        start_watchdog_manual
        ;;
    install-cron)
        install_cron
        ;;
    remove-cron)
        remove_cron
        ;;
    force-extract)
        force_start_extraction
        ;;
    force-generate)
        force_start_generation
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "JellyFish Galaxies Pipeline Control"
        echo ""
        echo "Usage: $0 {command}"
        echo ""
        echo "Commands:"
        echo "  status          - Show current status (default)"
        echo "  start           - Run watchdog check manually"
        echo "  stop            - Stop all pipeline processes"
        echo "  restart         - Stop and restart pipeline"
        echo "  install-cron    - Install cron job (auto-check every 10 min)"
        echo "  remove-cron     - Remove cron job"
        echo "  force-extract   - Force restart galaxy extraction"
        echo "  force-generate  - Force restart image generation"
        echo "  logs            - Show recent logs"
        echo ""
        echo "Examples:"
        echo "  $0 status              # Check what's running"
        echo "  $0 install-cron        # Set up auto-restart"
        echo "  $0 logs                # View logs"
        ;;
esac

#!/bin/bash
# Real-time monitoring for MAXIMUM MODE

echo "======================================"
echo "  JELLYFISH GALAXIES - LIVE MONITOR"
echo "======================================"
echo ""

while true; do
    clear
    echo "======================================"
    echo "  MAXIMUM IMAGE GENERATION - LIVE"
    echo "======================================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check running processes
    if ps aux | grep -q "[p]ython3 02_extract_galaxy"; then
        echo "STATUS: 🔄 Extracting galaxies..."
        STAGE="EXTRACTION"
    elif ps aux | grep -q "[p]ython3 03_generate_images"; then
        echo "STATUS: 🎨 Generating images..."
        STAGE="GENERATION"
    else
        echo "STATUS: ⏸️  Waiting/Idle"
        STAGE="IDLE"
    fi
    echo ""
    
    # Progress stats
    if [ -f output/data/clusters.csv ]; then
        CLUSTERS=$(($(wc -l < output/data/clusters.csv) - 1))
        echo "📊 Clusters found: $CLUSTERS"
    fi
    
    if [ -f output/data/galaxy_list.csv ]; then
        GALAXIES=$(($(wc -l < output/data/galaxy_list.csv) - 1))
        echo "🌌 Galaxies extracted: $GALAXIES"
    fi
    
    IMAGES=$(ls output/images/*.png 2>/dev/null | wc -l)
    echo "🖼️  Images generated: $IMAGES"
    
    if [ -f output/data/galaxy_list.csv ] && [ $IMAGES -gt 0 ]; then
        PERCENT=$(echo "scale=1; $IMAGES * 100 / $GALAXIES" | bc 2>/dev/null || echo "0")
        echo "📈 Progress: $PERCENT%"
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Show recent activity
    if [ -f extract_maximum.log ]; then
        echo "Latest extraction activity:"
        tail -3 extract_maximum.log | sed 's/^/  /'
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo "Refreshing in 10 seconds..."
    
    sleep 10
done

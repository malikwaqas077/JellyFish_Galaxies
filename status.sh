#!/bin/bash
# Quick status check for MAXIMUM MODE

echo "========================================================"
echo "  JELLYFISH GALAXIES - MAXIMUM MODE STATUS"
echo "========================================================"
echo "Time: $(date)"
echo ""

# Check what's running
echo "🔄 Active Processes:"
if ps aux | grep -q "[p]ython3 02_extract_galaxy"; then
    echo "  ✓ Galaxy extraction RUNNING"
fi
if ps aux | grep -q "[p]ython3 03_generate_images"; then
    echo "  ✓ Image generation RUNNING"
fi
if ps aux | grep -q "[a]uto_continue_maximum.sh"; then
    echo "  ✓ Auto-continue monitor ACTIVE (waiting for extraction)"
fi

if ! ps aux | grep -qE "[p]ython3 0[23]_|[a]uto_continue"; then
    echo "  ⚠️  No active processes detected"
fi
echo ""

# Current stats
echo "📊 Current Statistics:"
if [ -f output/data/clusters.csv ]; then
    CLUSTERS=$(($(wc -l < output/data/clusters.csv) - 1))
    echo "  Clusters found: $CLUSTERS"
fi

if [ -f output/data/galaxy_list.csv ]; then
    GALAXIES=$(($(wc -l < output/data/galaxy_list.csv) - 1))
    echo "  Galaxies extracted: $GALAXIES"
    
    if [ -f backups/galaxy_list_528_backup.csv ]; then
        OLD_GALAXIES=528
        NEW_GALAXIES=$((GALAXIES - OLD_GALAXIES))
        if [ $NEW_GALAXIES -gt 0 ]; then
            echo "    └─ NEW galaxies: +$NEW_GALAXIES"
        fi
    fi
fi

IMAGES=$(ls output/images/*.png 2>/dev/null | wc -l)
echo "  Images generated: $IMAGES"

if [ -f output/data/image_log.csv ]; then
    LOGGED=$(($(wc -l < output/data/image_log.csv) - 1))
    if [ $LOGGED -ne $IMAGES ]; then
        echo "    └─ Logged: $LOGGED (sync check)"
    fi
fi

echo ""
echo "📈 Progress:"
if [ -f output/data/galaxy_list.csv ] && [ $IMAGES -gt 0 ]; then
    PERCENT=$(echo "scale=1; $IMAGES * 100 / $GALAXIES" | bc 2>/dev/null || echo "0")
    echo "  Image generation: $PERCENT% complete"
    
    REMAINING=$((GALAXIES - IMAGES))
    if [ $REMAINING -gt 0 ]; then
        echo "  Remaining: $REMAINING images"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Recent activity
if [ -f extract_maximum.log ]; then
    echo "📝 Latest Extraction Activity:"
    tail -3 extract_maximum.log | sed 's/^/  /'
    echo ""
fi

if [ -f auto_continue_output.log ]; then
    echo "🤖 Auto-Continue Status:"
    tail -3 auto_continue_output.log | sed 's/^/  /'
    echo ""
fi

echo "========================================================"
echo "Run './monitor_maximum.sh' for live monitoring"
echo "========================================================"

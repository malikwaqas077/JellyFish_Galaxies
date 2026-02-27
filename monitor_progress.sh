#!/bin/bash
# Progress monitor for JellyFish_Galaxies image generation

echo "=== JellyFish Galaxy Image Generation Progress ==="
echo ""
echo "Target: 437 galaxies"
echo ""

while true; do
    IMG_COUNT=$(ls output/images/*.png 2>/dev/null | wc -l)
    PERCENT=$(echo "scale=1; $IMG_COUNT / 437 * 100" | bc)
    
    echo -ne "\rImages generated: $IMG_COUNT / 437 ($PERCENT%)   "
    
    if [ "$IMG_COUNT" -ge 437 ]; then
        echo ""
        echo "✓ Complete!"
        break
    fi
    
    sleep 10
done

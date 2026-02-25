"""
Step 4 — Quality filter and summary report.

Reads  : output/data/image_log.csv
Writes : output/data/final_dataset.csv  (only QC-passed images)
         output/data/qc_report.txt
         output/data/qc_grid.png         (visual montage of sample images)
"""

import csv
import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR, IMAGES_DIR, IMAGE_SIZE_PX

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_LOG    = os.path.join(DATA_DIR, "image_log.csv")
FINAL_CSV    = os.path.join(DATA_DIR, "final_dataset.csv")
REPORT_TXT   = os.path.join(DATA_DIR, "qc_report.txt")
GRID_PNG     = os.path.join(DATA_DIR, "qc_grid.png")

# Stricter QC thresholds (can be relaxed after visual inspection)
QC_MIN_SIGNAL    = 0.02     # at least 2% pixels above noise
QC_MIN_DYNA_RANGE = 0.8     # at least 0.8 dex dynamic range


def run_qc():
    with open(IMAGE_LOG, "r") as f:
        rows = list(csv.DictReader(f))

    passed = []
    failed_qc = []
    errors = []

    for row in rows:
        if row.get("error"):
            errors.append(row)
            continue
        sig  = float(row.get("signal_fraction", 0) or 0)
        dr   = float(row.get("dynamic_range_dex", 0) or 0)
        path = row.get("image_path", "")

        ok = (
            sig  >= QC_MIN_SIGNAL and
            dr   >= QC_MIN_DYNA_RANGE and
            os.path.exists(path)
        )
        if ok:
            passed.append(row)
        else:
            failed_qc.append(row)

    # Write final dataset CSV
    with open(FINAL_CSV, "w", newline="") as f:
        if passed:
            writer = csv.DictWriter(f, fieldnames=passed[0].keys())
            writer.writeheader()
            writer.writerows(passed)

    # Report
    total = len(rows)
    report = (
        f"TNG Jellyfish Extractor — QC Report\n"
        f"{'='*45}\n"
        f"Total images attempted : {total}\n"
        f"Download/render errors : {len(errors)}\n"
        f"Failed QC thresholds   : {len(failed_qc)}\n"
        f"Passed QC              : {len(passed)}\n"
        f"Pass rate              : {100*len(passed)/max(total,1):.1f}%\n\n"
        f"QC thresholds applied:\n"
        f"  signal_fraction ≥ {QC_MIN_SIGNAL}\n"
        f"  dynamic_range   ≥ {QC_MIN_DYNA_RANGE} dex\n\n"
        f"Output: {FINAL_CSV}\n"
    )
    print(report)
    with open(REPORT_TXT, "w") as f:
        f.write(report)

    # Visual grid — random sample of passed images
    _make_grid(passed, n=min(64, len(passed)))

    return passed


def _make_grid(rows, n=64):
    """Compose an n-image preview grid (8×8 max)."""
    sample = random.sample(rows, n) if len(rows) >= n else rows
    cols   = min(8, n)
    nrows  = (len(sample) + cols - 1) // cols

    thumb  = 80   # px per thumbnail in the grid
    grid   = Image.new("RGB", (cols * thumb, nrows * thumb), color=(0, 0, 0))

    for i, row in enumerate(sample):
        path = row.get("image_path", "")
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).resize((thumb, thumb), Image.LANCZOS)
            col = i % cols
            r   = i // cols
            grid.paste(img, (col * thumb, r * thumb))
        except Exception:
            pass

    grid.save(GRID_PNG)
    print(f"QC grid saved → {GRID_PNG}")


if __name__ == "__main__":
    run_qc()

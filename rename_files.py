"""
Rename images and update CSVs to follow the naming convention:
  TNG-Cluster_{subhalo_id:06d}.png
"""
import os
import re
import csv

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "images")
DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "data")
CLUSTER   = "TNG-Cluster"
PAD       = 6  # zero-pad to 6 digits (max ID ~37693)

# ── 1. Rename images ──────────────────────────────────────────────────────────
renamed = {}   # old_name -> new_name (without directory)
skipped = []

for fname in sorted(os.listdir(IMAGE_DIR)):
    m = re.match(r"halo\d+_sub0*(\d+)_z\.png", fname)
    if not m:
        skipped.append(fname)
        continue

    subhalo_id = int(m.group(1))
    new_name = f"{CLUSTER}_{subhalo_id:0{PAD}d}.png"
    src = os.path.join(IMAGE_DIR, fname)
    dst = os.path.join(IMAGE_DIR, new_name)

    if os.path.exists(dst):
        print(f"  SKIP (already exists): {new_name}")
        renamed[fname] = new_name
        continue

    os.rename(src, dst)
    renamed[fname] = new_name

print(f"Renamed {len(renamed)} images.")
if skipped:
    print(f"Skipped (unexpected format): {skipped}")

# ── 2. Update image_log.csv and final_dataset.csv ────────────────────────────
for csv_name in ("image_log.csv", "final_dataset.csv"):
    csv_path = os.path.join(DATA_DIR, csv_name)
    if not os.path.exists(csv_path):
        print(f"Not found, skipping: {csv_name}")
        continue

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue

    fieldnames = list(rows[0].keys())
    updated = 0
    for row in rows:
        old_path = row.get("image_path", "")
        if not old_path:
            continue
        old_fname = os.path.basename(old_path)
        if old_fname in renamed:
            new_fname = renamed[old_fname]
            row["image_path"] = os.path.join(IMAGE_DIR, new_fname)
            updated += 1

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {updated} rows in {csv_name}.")

print("\nDone. Sample new filenames:")
for old, new in list(renamed.items())[:5]:
    print(f"  {old}  ->  {new}")

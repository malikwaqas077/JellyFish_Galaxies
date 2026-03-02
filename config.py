"""
TNG Jellyfish Image Extractor — Configuration
All tunable parameters live here. Edit this file only.
"""

# ── TNG API ───────────────────────────────────────────────────────────────────
API_KEY        = "faa0959886d51fa3258568782eca5f78"
SIMULATION     = "TNG-Cluster"       # zoom-in simulation of 352 massive clusters
SNAPSHOT       = 99                  # z = 0
BASE_URL       = f"https://www.tng-project.org/api/{SIMULATION}/snapshots/{SNAPSHOT}"

# ── Cluster selection ─────────────────────────────────────────────────────────
# M200c threshold in 1e10 M_sun/h (TNG internal units)
# 1e14 M_sun ≈ 1e4 × 1e10 M_sun  (h≈0.6774 → 1e14/0.6774/1e10 ≈ 14762 in code units)
MIN_CLUSTER_MASS_1E10MSUN_H = 3_000    # ~2×10^13 M_sun → more clusters! (was 10_000)

# ── Galaxy selection inside clusters ─────────────────────────────────────────
# Masses in TNG internal units: 1e10 M_sun/h  (multiply by 0.6774×1e10 to get M_sun)
# ULTRA-PERMISSIVE SETTINGS FOR 2000+ IMAGES!
MIN_GAS_MASS_1E10MSUN_H     = 0.000001 # ~7e3 M_sun — ULTRA LOW, catch everything!
MAX_GAS_MASS_1E10MSUN_H     = 1000.0   # ~7e12 M_sun — no upper limit effectively
MIN_STELLAR_MASS_1E10MSUN_H = 0.00001  # ~7e4 M_sun — extremely tiny objects
MAX_STELLAR_MASS_1E10MSUN_H = 1000.0   # ~7e12 M_sun — no upper limit effectively  
MAX_HALFMASS_GAS_CKPC_H     = 500.0    # ~740 physical kpc — allow very extended

# ── Image generation ──────────────────────────────────────────────────────────
IMAGE_SIZE_PX  = 424                   # Zooniverse Galaxy Zoo standard (px × px)
APERTURE_KPC   = 200.0                 # physical projection window half-width in kpc
                                       # → 400 kpc total → captures most JF tails
N_PIXELS_GRID  = 1024                  # internal render grid (downsampled to 424)
SMOOTH_FACTOR  = 0.6                   # Gaussian smooth σ = SMOOTH_FACTOR × cell_radius
LOG_SCALE      = True                  # log10 surface density
COLORMAP       = "hot"                 # black bg → white/yellow core; shows tails well
VMIN_PERCENTILE = 3                    # lower clip percentile — keep faint tail signal
VMAX_PERCENTILE = 99.9                 # upper clip percentile

# ── Cosmology (TNG uses Planck 2015) ─────────────────────────────────────────
H0   = 67.74                           # km/s/Mpc
OMEGA_M = 0.3089
LITTLE_H = H0 / 100.0                  # ≈ 0.6774

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR     = "output"
IMAGES_DIR     = "output/images"
DATA_DIR       = "output/data"
LOGS_DIR       = "output/logs"

# ── Pipeline limits (set to None to process all) ──────────────────────────────
MAX_CLUSTERS   = None                  # MAXIMUM MODE: Process ALL clusters!
MAX_GALAXIES_PER_CLUSTER = None        # None = all qualifying galaxies per cluster!
REQUEST_TIMEOUT = 60                   # seconds per HTTP request
DOWNLOAD_RETRIES = 3

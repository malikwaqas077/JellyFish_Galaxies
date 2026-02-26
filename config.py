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
MIN_CLUSTER_MASS_1E10MSUN_H = 10_000   # ~7×10^13 M_sun → conservative cluster floor

# ── Galaxy selection inside clusters ─────────────────────────────────────────
# Masses in TNG internal units: 1e10 M_sun/h  (multiply by 0.6774×1e10 to get M_sun)
MIN_GAS_MASS_1E10MSUN_H     = 0.001    # ~7e6 M_sun — catch partially-stripped galaxies
MAX_GAS_MASS_1E10MSUN_H     = 100.0    # ~7e11 M_sun — exclude ICM-filling subclusters
MIN_STELLAR_MASS_1E10MSUN_H = 0.01     # ~7e7 M_sun — avoid tiny dwarfs
MAX_STELLAR_MASS_1E10MSUN_H = 50.0     # ~3e11 M_sun — exclude BCGs / massive subclusters
MAX_HALFMASS_GAS_CKPC_H     = 100.0    # ~150 physical kpc — galaxies, not subclusters

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
MAX_CLUSTERS   = None                  # None = all clusters
MAX_GALAXIES_PER_CLUSTER = None        # None = all members
REQUEST_TIMEOUT = 60                   # seconds per HTTP request
DOWNLOAD_RETRIES = 3

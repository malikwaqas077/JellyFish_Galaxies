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
MIN_CLUSTER_MASS_1E10MSUN_H = 1_000    # ~6.8×10^12 M_sun → MAXIMUM clusters! (lowered from 3000)

# ── Galaxy selection inside clusters ─────────────────────────────────────────
# Masses in TNG internal units: 1e10 M_sun/h  (multiply by 0.6774×1e10 to get M_sun)
# MAXIMUM PERMISSIVE MODE: 2000+ images with LOW/MEDIUM/HIGH gas mix
# Classifier model will filter jellyfish vs non-jellyfish
#
# Gas content categories for classifier training:
#   LOW:    0 - 0.1 × 1e10 M_sun/h    (gas-poor, stripped "red & dead")
#   MEDIUM: 0.1 - 1.0 × 1e10 M_sun/h  (moderate gas, possible stripping)
#   HIGH:   > 1.0 × 1e10 M_sun/h      (gas-rich, prime jellyfish candidates)
#
MIN_GAS_MASS_1E10MSUN_H     = 0.0      # NO MINIMUM! Include gas-poor satellites
MAX_GAS_MASS_1E10MSUN_H     = 1000.0   # ~7e12 M_sun — no upper limit
MIN_STELLAR_MASS_1E10MSUN_H = 0.001    # ~6.8e6 M_sun — visible galaxies only (lowered from 0.00001)
MAX_STELLAR_MASS_1E10MSUN_H = 1000.0   # ~7e12 M_sun — no upper limit
MAX_HALFMASS_GAS_CKPC_H     = 1000.0   # ~1.5 Mpc — allow extended structures (doubled)

# ── Image generation ──────────────────────────────────────────────────────────
IMAGE_SIZE_PX  = 1024                  # High-resolution output (px × px)
APERTURE_KPC   = 200.0                 # physical projection window half-width in kpc
                                       # → 400 kpc total → captures most JF tails
N_PIXELS_GRID  = 1024                  # internal render grid (matches output; no downsampling)
SMOOTH_FACTOR  = 0.3                   # Gaussian smooth σ = SMOOTH_FACTOR × cell_radius
MULTISCALE_SMOOTH = True               # per-cell σ grouped into bins (True = realistic rendering)
N_SMOOTH_BINS  = 8                     # number of σ-bins for multi-scale deposition
LOG_SCALE      = True                  # log10 surface density
COLORMAP       = "inferno"             # perceptually-uniform; better for faint tail detection
VMIN_PERCENTILE = 2                    # lower clip percentile — shows faint tails
VMAX_PERCENTILE = 99.9                 # upper clip percentile
ARCSINH_STRETCH = True                 # apply arcsinh stretch after log-scale normalisation
ARCSINH_A      = 0.05                  # arcsinh softening: smaller → more contrast in faint regions

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

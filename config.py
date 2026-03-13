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
MIN_CLUSTER_MASS_1E10MSUN_H = 1_000    # ~6.8×10^12 M_sun — captures all massive groups

# ── Galaxy catalog filters (step 1: catalog subhalos) ────────────────────────
# Gas content categories for classifier training:
#   LOW:    0 – 0.1 × 1e10 M_sun/h    (gas-poor / stripped "red & dead")
#   MEDIUM: 0.1 – 1.0 × 1e10 M_sun/h  (moderate gas, possible stripping)
#   HIGH:   > 1.0 × 1e10 M_sun/h      (gas-rich, prime jellyfish candidates)
MIN_GAS_MASS_1E10MSUN_H     = 0.0      # include fully gas-stripped satellites
MAX_GAS_MASS_1E10MSUN_H     = 1000.0
MIN_STELLAR_MASS_1E10MSUN_H = 0.001    # ~6.8e6 M_sun — visible galaxies only
MAX_STELLAR_MASS_1E10MSUN_H = 1000.0
MAX_HALFMASS_GAS_CKPC_H     = 1000.0   # ~1.5 Mpc comoving

# Minimum gas cells in HDF5 cutout before we bother rendering
MIN_GAS_CELLS = 20

# ── HDF5 cutout fields to download ───────────────────────────────────────────
HDF5_GAS_FIELDS = "Coordinates,Masses,Density"

# ── Image generation ──────────────────────────────────────────────────────────
IMAGE_SIZE_PX   = 1024                 # output PNG resolution
APERTURE_KPC    = 200.0                # projection window half-width in physical kpc
N_PIXELS_GRID   = 1024                 # internal render grid (= output size, no resize)
SMOOTH_FACTOR   = 0.3                  # Gaussian σ = SMOOTH_FACTOR × cell_radius
MULTISCALE_SMOOTH = True               # per-cell σ binned (preserves core + faint tail)
N_SMOOTH_BINS   = 8                    # σ-bins for multi-scale deposition
LOG_SCALE       = True                 # log10 surface density
COLORMAP        = "inferno"            # perceptually uniform; great for faint tails
VMIN_PERCENTILE = 2
VMAX_PERCENTILE = 99.9
ARCSINH_STRETCH = True                 # JWST-style: boosts faint signal, compresses core
ARCSINH_A       = 0.05

# Image filename convention:  TNG-Cluster_{subhalo_id:06d}.png
IMAGE_NAME_CLUSTER = "TNG-Cluster"
IMAGE_ID_PAD       = 6

# ── Cosmology (TNG uses Planck 2015) ─────────────────────────────────────────
H0       = 67.74
OMEGA_M  = 0.3089
LITTLE_H = H0 / 100.0                  # ≈ 0.6774

# ── Output directories ────────────────────────────────────────────────────────
OUTPUT_DIR   = "output"
IMAGES_DIR   = "output/images"
DATA_DIR     = "output/data"
CUTOUTS_DIR  = "output/cutouts"        # raw HDF5 files live here
LOGS_DIR     = "output/logs"

# ── Pipeline limits (None = process all) ─────────────────────────────────────
MAX_CLUSTERS              = None
MAX_GALAXIES_PER_CLUSTER  = None
REQUEST_TIMEOUT           = 60
DOWNLOAD_RETRIES          = 3
PARALLEL_WORKERS          = 8

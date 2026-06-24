# Paths
DATA_INTERIM   = "data/interim"
DATA_RAW_CRIME = "data/raw/crime"
OUTPUT_DIR     = "results"

# Time
YEAR_PREDICT   = 2025
YEAR_BASE      = 2024
START_YEAR     = 2015
OBSERVED_YEAR  = 2025

# Model
N_REGIMES      = 6
N_QUANTILES    = 5
N_STEPS        = 1
MIN_MUNI       = 30
BINS           = 5
MA_WINDOW      = 9
STD_THRESHOLD  = 2.0
AUX_MODEL      = 'ma'

# CRI
CRI_ALPHA      = 0.5
CRI_BETA       = 0.3
CRI_GAMMA      = 0.2
USE_SPATIAL_LAG = True
SPATIAL_LAMBDA = 0.2
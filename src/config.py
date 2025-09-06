from pathlib import Path

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA = BASE_DIR / "data" / "raw" / "online_retail_ii.xlsx"
PROC_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

# ---- Data prep ----
MIN_PRICE = 0.01  # drop rows with unit price <= this
MIN_QTY = 1  # drop rows with quantity < this
TOP_COUNTRIES = 5  # keep only top N countries separate
MIN_OBS_PER_SKU = 60  # require at least this many daily observations per SKU

# ---- Optimization ----
COST_RATIO = 0.60  # proxy: unit_cost = median_price * COST_RATIO
GRID_STEPS = 25  # number of candidate prices to simulate per SKU
PRICE_LOWER_MULT = 0.80  # lower bound factor relative to historical min price
PRICE_UPPER_MULT = 1.20  # upper bound factor relative to historical max price

# ---- Backtest ----
HOLDOUT_WEEKS = 8  # number of weeks reserved for holdout evaluation
SEED = 42  # random seed for reproducibility

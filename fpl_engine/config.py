"""Configuration constants for the FPL engine."""

from pathlib import Path

# ── API ──────────────────────────────────────────────────────────────────────
FPL_BASE_URL = "https://fantasy.premierleague.com/api"
REQUEST_DELAY = 0.35  # seconds between API calls (rate limiting)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Model ────────────────────────────────────────────────────────────────────
POSITIONS = ["GKP", "DEF", "MID", "FWD"]
POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# Rolling window sizes for feature engineering
ROLLING_WINDOWS = [3, 5, 10]

# Budget and squad constraints
TOTAL_BUDGET = 1000  # FPL uses 10× (e.g. 100.0m = 1000)
SQUAD_SIZE = 15
STARTING_XI = 11
MAX_PER_TEAM = 3

POSITION_CONSTRAINTS = {
    "GKP": (2, 2),   # (squad_min, squad_max)
    "DEF": (5, 5),
    "MID": (5, 5),
    "FWD": (3, 3),
}

FORMATION_CONSTRAINTS = {
    "GKP": (1, 1),   # (xi_min, xi_max)
    "DEF": (3, 5),
    "MID": (2, 5),
    "FWD": (1, 3),
}

# ── Optimizer ────────────────────────────────────────────────────────────────
TRANSFER_HIT_COST = 4  # points penalty per extra transfer
PLANNING_HORIZON = 3   # gameweeks to plan ahead

# Ownership-aware weighting
DIFFERENTIAL_LAMBDA = 0.15   # bonus weight for low-ownership picks
RISK_PENALTY_MU = 0.05       # downside penalty weight

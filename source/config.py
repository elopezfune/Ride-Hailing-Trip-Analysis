from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent

# Data Path
# =========
PATH_TO_DATA = PROJECT / "data/intervals_challenge.json"

# Seed for reproducibility
# ========================
SEED = 42


# Numerical variables
# ===================
NUM_VARS = ['duration','distance']


# Road distance tag
# =================
ROAD = list(map(str, range(9))) #['9','a','b','c','d','e','f']


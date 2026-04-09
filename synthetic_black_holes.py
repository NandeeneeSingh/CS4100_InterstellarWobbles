import numpy as np
import pandas as pd
from pathlib import Path

N_CLONES    = 100
RANDOM_SEED = 42
BH_MASS_MIN = 10.0
BH_MASS_MAX = 20.0
FP_ITER     = 80
MIN_SIN_I   = 1e-3

K1 = 10**3 
P = 86400  
G = 6.674e-11 
M_SUN = 1.989e30 

CONVERSION_CONSTANT = (K1**3 * P) / (2 * np.pi * G * M_SUN) # ≈ 1.0385e-7

INPUT_CSV  = Path("data/03_physics_anchored/with_m1_sb1_cleaned.csv")
OUTPUT_CSV = Path("data/03_physics_anchored/synthetic_bh_mc_expanded.csv")

CORE_FEATURES = [
    "source_id", "period", "eccentricity", "m1_solar_m",
]

# Physics helpers (from mc_expansion.py)

def solve_m2_min(f_mass, m1, sin_i=1.0, n_iter=FP_ITER):
    """
    Solve for minimum companion mass M2 [M_sun] via fixed-point iteration.
    """
    f  = np.asarray(f_mass, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    # Initial guess
    m2 = (f * m1**2) ** (1.0 / 3.0) / sin_i
    for _ in range(n_iter):
        m2 = (f * (m1 + m2)**2) ** (1.0 / 3.0) / sin_i
    return m2

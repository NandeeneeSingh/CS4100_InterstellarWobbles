import numpy as np
import pandas as pd
from pathlib import Path

# Parameters for Monte Carlo expansion
N_CLONES = 100      # Clones / actual system
RANDOM_SEED = 42    
FP_ITER = 80        # fixed point iterations
MIN_SIN_I = 1e-3    # face-on singulary prevention (i < ~0.06°)

# Use a consistent conversion constant for mass function calculations (K1 in km/s, P in days → f in solar masses)
# Necessary since the data we are using is not in SI units, but the physics equations are. 
K1 = 10**3 # convert km/s to m/s
P = 86400  # convert days to seconds
G = 6.674e-11 # gravitational constant - m^3/kg/s^2
M_SUN = 1.989e30 # solar mass in kg

CONVERSION_CONSTANT = (K1**3 * P) / (2 * np.pi * G * M_SUN) # ≈ 1.0385e-7

INPUT_CSV  = Path("data/03_physics_anchored/with_m1_sb1_cleaned.csv")
OUTPUT_CSV = Path("data/03_physics_anchored/sb1_mc_expanded.csv")

CORE_FEATURES = [
    "source_id",
    "period",
    "eccentricity",
    "semi_amplitude_primary",   # K1 (km/s) — radial velocity amplitude
    "m1_solar_m",       # primary mass estimate (solar masses)
]

# Physics equation helpers
def compute_mass_function(K1_kms, P_days, e):
    """
    Computes SB1 spectroscopoic binary mass function f(M) in solar masses
    f(M) = CONVERSION_CONSTANT · K1^3 · P · (1 - e^2)^1.5
    """
    return CONVERSION_CONSTANT * K1_kms**3 * P_days * (1.0 - e**2)**1.5

def solve_m2(f_mass, m1, sin_i, iters=FP_ITER):
    """
    Solve for m2 via fixed-point iteration of the mass function cubic:
    M2^(k+1) = cbrt( f · (M1 + M2^(k))^2 ) / sin(i)

    If inclination is smaller, companion mass must be larger to produce same obsevered K1, so each sampled
    inclination gives different M2. This is how we get a distribution of possible M2 values for each real system.
    """
    f = np.asarray(f_mass, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    sin_i = np.asarray(sin_i, dtype=float)

    # Initial guess (assuming m2 << m1)
    m2 = (f * m1**2) ** (1.0 / 3.0) / sin_i

    for _ in range(iters):
        m2 = (f * (m1 + m2)**2) ** (1.0 / 3.0) / sin_i

    return m2


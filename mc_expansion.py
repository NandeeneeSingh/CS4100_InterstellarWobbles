import numpy as np
import pandas as pd
from pathlib import Path

# Parameters for Monte Carlo expansion
N_CLONES = 100     
RANDOM_SEED = 42    
FP_ITER = 80        # fixed point iterations
MIN_SIN_I = 1e-3    # to avoid division by zero in face-on situation

# Conversion data for constant equation
K1 = 10**3          # km/s to m/s
P = 86400           # days to seconds
G = 6.674e-11       # gravitational constant - m^3/kg/s^2
M_SUN = 1.989e30    # solar mass (kg)

# Conversion constant to convert mass function to solar masses. 
CONVERSION_CONSTANT = (K1**3 * P) / (2 * np.pi * G * M_SUN) # ≈ 1.0385e-7

INPUT_CSV  = Path("data/03_physics_anchored/with_m1_sb1_cleaned.csv")
OUTPUT_CSV = Path("data/03_physics_anchored/sb1_mc_expanded.csv")

CORE_FEATURES = [
    "source_id",
    "period",
    "eccentricity",
    "semi_amplitude_primary",  
    "m1_solar_m",
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

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # Load SB1 Data
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}, run calculate_m1_sb1.py first.")
    
    df = pd.read_csv(INPUT_CSV, usecols=lambda col: col in CORE_FEATURES)
    print(f"Loaded {len(df)} SB1 systems from {INPUT_CSV}")

    df = df[CORE_FEATURES].dropna().copy() 
    print(f"{len(df)} systems after dropping rows with missing core features.")

    # Compute binary mass function for each system
    df["f_mass"] = compute_mass_function(
        df["semi_amplitude_primary"].values(),
        df["period"].values(),
        df["eccentricity"].values(),
    )

    # Minimum companion mass (conservative lower bound sin_i = 1)
    df["m2_min"] = solve_m2(
        df["f_mass"].values(),
        df["m1_solar_m"].values(),
        sin_i=1.0,
    )

    n_systems = len(df)
    print(f"\nMass function f(M) - descriptive stats:")
    print(df["f_mass"].describe().to_string())
    print(f"\nMinimum companion mass m2_min_solar — descriptive stats:")
    print(df["m2_min_solar"].describe().to_string())

    # Monte Carlo expansion
    clone_df = df.loc[df.index.repeat(N_CLONES)].reset_index(drop=True)
    clone_df["clone_id"] = np.title(np.arrange(N_CLONES), n_systems)

    # Sample inclinations from the geometric distribution.
    # cos(i) ~ Uniform(0, 1) -> p(i) = sin(i)
    cos_i_samples = rng.uniform(0.0, 1.0, size=len(clone_df))
    sin_i_samples = np.sqrt(1.0 - cos_i_samples**2)
    clone_df["i_sample_deg"] = np.degrees(np.arccos(sin_i_samples))

    # Solve for true companion mass at each sampled inclination
    clone_df["m2_solar"] = solve_m2(
        clone_df["f_mass"].values(),
        clone_df["m1_solar_m"].values(),
        sin_i_samples,
    )

    # Save expanded dataset
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    clone_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nMone Carlo expansion complete. Expanded dataset saved to {OUTPUT_CSV}")
    print(f"Real systems   : {n_systems:,}")
    print(f"Clones/system  : {N_CLONES}")
    print(f"Total rows     : {len(clone_df):,}")
    print(f"\nSampled companion mass m2_solar — descriptive stats:")
    print(clone_df["m2_solar"].describe().to_string())

    # m2_solar >= m2_min_solar for all clones
    below_min = (clone_df["m2_solar"] < clone_df["m2_min_solar"] * 0.99).sum()
    if below_min > 0:
        print(f"\n  WARNING: {below_min} clones have m2_solar < m2_min_solar "
              f"(may indicate edge-case convergence issues)")
    else:
        print(f"\n  Sanity check passed: all sampled M2 ≥ minimum M2 (as expected)")

if __name__ == "__main__":
    main()
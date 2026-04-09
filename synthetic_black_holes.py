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

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
    
    # Load the authenitc SB1 orbital data
    df = pd.read_csv(INPUT_CSV, usecols=lambda c: c in set(CORE_FEATURES))
    df = df[CORE_FEATURES].dropna().copy()
    n_systems = len(df)

    print(f"Loaded {n_systems} real SB1 systems (P, e, M1)")

    # Synethesize black holes
    df = df.copy() # seperate object
    df["source_id"] = df["source_id"].astype(str)
    df.loc[:, "m2_true"] = rng.uniform(BH_MASS_MIN, BH_MASS_MAX, size=n_systems)
    df.loc[:, "source_id"] = "SYNTH_BH_" + df["source_id"]

    # Monte carlo expansion
    clone_df = df.loc[df.index.repeat(N_CLONES)].reset_index(drop=True)
    clone_df["clone_id"] = np.tile(np.arange(N_CLONES), n_systems)

    # Sampling inclination
    cos_i_samples = rng.uniform(0.0, 1.0, size=len(clone_df))
    sin_i_samples = np.sqrt(1.0 - cos_i_samples**2)
    clone_df["i_sample_deg"] = np.degrees(np.arccos(cos_i_samples))

    # Forward physics: Compute synthetic observable K1
    m1 = clone_df["m1_solar_m"].values
    m2 = clone_df["m2_true"].values
    sin_i = sin_i_samples

    f_true = (m2 * sin_i)**3 / (m1 + m2)**2

    P = clone_df["period"].values
    e = clone_df["eccentricity"].clip(lower=0, upper=0.999).values # make sure e != 1

    denominator = CONVERSION_CONSTANT * P * (1.0 - e**2)**1.5
    clone_df = clone_df.copy() 
    clone_df.loc[:, "semi_amplitude_primary"] = np.cbrt(f_true / denominator)

    # Backward physics: Anchor targets m2_min and f_mass
    clone_df.loc[:, "f_mass"] = f_true
    clone_df.loc[:, "m2_min_solar"] = solve_m2_min(f_true, m1)
    clone_df.loc[:, "m2_solar"] = clone_df["m2_true"] # Actual injected mass

    # Save 
    keep_cols = [
        "source_id", "period", "eccentricity", "semi_amplitude_primary",
        "m1_solar_m", "f_mass", "m2_min_solar", "clone_id", "i_sample_deg", "m2_solar"
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    clone_df[keep_cols].to_csv(OUTPUT_CSV, index=False)

    print(f"Synthetic Data Generation complete, dataset saved to {OUTPUT_CSV}")
    print(f"Real anchors   : {n_systems:,}")
    print(f"Clones/system  : {N_CLONES}")
    print(f"Total rows     : {len(clone_df):,}")
    print(f"\nSynthetic K_1 (km/s) generated stats:")
    print(clone_df["semi_amplitude_primary"].describe().to_string())

if __name__ == "__main__":
    main()
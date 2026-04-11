import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from random_forest import RandomForest as CustomRandomForest

N_CLONES = 100
RANDOM_SEED = 42

CLASS_LABELS = {0: "(0) Min_Mass_<_1.4", 1: "(1) Min_Mass_1.4_to_3.0", 2: "(2) Min_Mass_>_3.0"}
OUTPUT_DIR = Path("data/06_validation")

# Prints accuracy of each class in a formatted table
def print_class_accuracy(system_df, true_col, label):
    print(f"  {'Class':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}") # Table Headings 
    print(f"  {'─'*55}")
    for cls, cls_label in CLASS_LABELS.items(): # Loop through classes 0 - 2
        mask = system_df[true_col] == cls # Filter data at once using vectorized boolean mask
        total = mask.sum()
        if total == 0: # No samples for this class, skip to avoid division by zero
            continue
        correct = (system_df.loc[mask, "predicted_class"] == cls).sum() # Only look at rows where the true class matched the predicted class
        print(f"  {cls_label:<25} {correct:>8} {total:>8} {correct/total*100:>9.1f}%")
    print()

# Rearranges Binary Mass Function equation to solve for M2
def solve_m2(f_mass, m1, sin_i=1.0, n_iter=80):
    f = np.asarray(f_mass, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    sin_i = np.clip(np.asarray(sin_i, dtype=float), 1e-3, 1.0) # np.clip useed to make sure sin i is never less than 0.001
    m2 = (f * m1**2) ** (1.0 / 3.0) / sin_i # Initial guess for M2 based on the assumption that M2 << M1
    for _ in range(n_iter): # Refine M2 value until convergence
        m2 = (f * (m1 + m2)**2) ** (1.0 / 3.0) / sin_i
    return m2

# m_solar < 1.4  --> Class 0 (Chandrasekhar Mass Limit for white dwarfs) -- Soup
# m_solar < 3.0 --> Class 1 (Tolman–Oppenheimer–Volkoff Limit for neutron stars) -- Intermediate
# m_solar >= 3.0 --> Class 2 (High-Mass Stellar Remnants, likely black holes) -- HMD
def map_target(m_solar):
    return np.where(m_solar < 1.4, 0, np.where(m_solar < 3.0, 1, 2))

def validate_sb2(rf: CustomRandomForest):
    input_csv = Path("data/03_physics_anchored/with_m1_sb2_forward_model_results.csv")
    # Safety check
    if not input_csv.exists():
        print(f"Skipping SB2 validation, missing {input_csv}")
        return

    df = pd.read_csv(input_csv)
    # Drop NaN values in columns needed for calculations
    df = df.dropna(subset=["period", "eccentricity", "semi_amplitude_primary", "m1_solar_m", "m2_sin3i"]).copy()

    n_systems = len(df)
    clone_df = df.loc[df.index.repeat(N_CLONES)].reset_index(drop=True) # Dataframe of clones for each SB2 system
    
    random_tilt = np.random.default_rng(RANDOM_SEED + 1) # Random generated tilt for each SB2 system clone

    cos_i = random_tilt.uniform(0.0, 1.0, size=len(clone_df)) # Orbital tilt requires 3D geometry -- uniform distribution between 0 and 1
    clone_df.loc[:, "i_sample_deg"] = np.degrees(np.arccos(cos_i))
    sin_i = np.sin(np.radians(clone_df["i_sample_deg"])) # Orbital tilt requires 3D geometry -- convert to radians for sine calculation
    clone_df.loc[:, "m2_true_sample"] = clone_df["m2_sin3i"] / (sin_i**3)

    # .loc changes original dateframe
    # Calculate and classify possible M2 based on random sample of tilt, cos_i, and m2_sin3i value
    clone_df.loc[:, "true_class"] = map_target(clone_df["m2_true_sample"])

    # Feed features into Random Forest + get probabilities for each class
    features = ["period", "eccentricity", "semi_amplitude_primary", "m1_solar_m", "i_sample_deg"]
    subset_db = clone_df[features].values # Create subset of data with only the features needed for prediction
    
    decimal_prob_confidence = rf.predict_proba(subset_db)
    clone_df.loc[:, "prob_soup"] = decimal_prob_confidence[:, 0]
    clone_df.loc[:, "prob_inter"] = decimal_prob_confidence[:, 1]
    clone_df.loc[:, "prob_hmdr"] = decimal_prob_confidence[:, 2]
    
    # Average AI predictions across clones for each SB2 system --> single predicted class
    # Use mode (lambda) to determine most statistically probably physical class fo reach system
    system_df = clone_df.groupby("source_id").agg({
        "prob_soup": "mean", "prob_inter": "mean", "prob_hmdr": "mean",
        "true_class": lambda x: x.mode()[0], "m2_sin3i": "first"
    }).reset_index()

    # ArgMax : choose highest predicted probability for final class prediction for each system
    system_df["predicted_class"] = np.argmax(system_df[["prob_soup", "prob_inter", "prob_hmdr"]].values, axis=1)
    system_df = system_df.assign(
        predicted_label=system_df["predicted_class"].map(CLASS_LABELS),
        true_label=system_df["true_class"].map(CLASS_LABELS),
        correct=system_df["predicted_class"] == system_df["true_class"]
    )

    decimal_acc = system_df["correct"].mean() # Decimal accuracy across all systems
    print(f"  SB2 — Overall Accuracy: {decimal_acc*100:.2f}%")
    print_class_accuracy(system_df, "true_class", "SB2")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_cols = ["source_id", "m2_sin3i", "true_label", "predicted_label", "correct", "prob_soup", "prob_inter", "prob_hmdr"]
    system_df[out_cols].to_csv(OUTPUT_DIR / "validation_sb2.csv", index=False)
    print(f"  Artifact saved: {OUTPUT_DIR / 'validation_sb2.csv'}")

def main():

    model_path = Path("data/03_physics_anchored/rf_model.pkl")
    # Error handling to ensure model exists before validation
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train and save model before validation.")
        
    print("Loading Custom Random Forest...\n")
    with open(model_path, "rb") as f:
        rf = pickle.load(f)

    print("Double-Lined Spectroscopic Binaries (Minimum Mass Ground Truth)")
    print(f"{'─'*65}")
    validate_sb2(rf)
    print(f"{'─'*65}\n")

if __name__ == "__main__":
    main()

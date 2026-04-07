import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from step3_custom_ml import CustomRandomForest, DecisionTree, Node

N_CLONES = 100
RANDOM_SEED = 42

CLASS_LABELS = {0: "Min_Mass_<_1.4", 1: "Min_Mass_1.4_to_3.0", 2: "Min_Mass_>_3.0"}
OUTPUT_DIR = Path("data/06_validation")

def print_per_class_accuracy(system_df, true_col, label):
    print(f"  {'Class':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'─'*55}")
    for cls, cls_label in CLASS_LABELS.items():
        mask = system_df[true_col] == cls
        total = mask.sum()
        if total == 0:
            continue
        correct = (system_df.loc[mask, "predicted_class"] == cls).sum()
        print(f"  {cls_label:<25} {correct:>8} {total:>8} {correct/total*100:>9.1f}%")
    print()

def solve_m2(f_mass, m1, sin_i=1.0, n_iter=80):
    f = np.asarray(f_mass, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    sin_i = np.clip(np.asarray(sin_i, dtype=float), 1e-3, 1.0)
    m2 = (f * m1**2) ** (1.0 / 3.0) / sin_i
    for _ in range(n_iter):
        m2 = (f * (m1 + m2)**2) ** (1.0 / 3.0) / sin_i
    return m2

def map_target(m_solar):
    return np.where(m_solar < 1.4, 0, np.where(m_solar < 3.0, 1, 2))

def validate_orbital(rf: CustomRandomForest):
    input_csv = Path("data/03_physics_anchored/with_m1_orbital_cleaned.csv")
    if not input_csv.exists():
        print(f"Skipping Orbital validation, missing {input_csv}")
        return

    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["period", "wobble_amplitude", "parallax", "m1_solar_m"]).copy()
    df.loc[:, "eccentricity"] = 0.0 # Orbital solutions without provided eccentricity assumed circular

    df.loc[:, "a1_au"] = df["wobble_amplitude"] / df["parallax"]
    P_yr = df["period"] / 365.25
    df.loc[:, "f_ast"] = (df["a1_au"]**3) / (P_yr**2)
    df.loc[:, "m2_true"] = solve_m2(df["f_ast"], df["m1_solar_m"], sin_i=1.0)
    df.loc[:, "true_class"] = map_target(df["m2_true"])

    n_systems = len(df)
    clone_df = df.loc[df.index.repeat(N_CLONES)].reset_index(drop=True)
    
    rng = np.random.default_rng(RANDOM_SEED)
    cos_i = rng.uniform(0.0, 1.0, size=len(clone_df))
    sin_i = np.sqrt(1.0 - cos_i**2)
    clone_df.loc[:, "i_sample_deg"] = np.degrees(np.arccos(cos_i))
    clone_df.loc[:, "semi_amplitude_primary"] = (2 * np.pi * clone_df["a1_au"] * sin_i) / (clone_df["period"] * np.sqrt(1 - clone_df["eccentricity"]**2)) * 1731.4568

    features = ["period", "eccentricity", "semi_amplitude_primary", "m1_solar_m", "i_sample_deg"]
    X = clone_df[features].values
    
    probas = rf.predict_proba(X)
    clone_df.loc[:, "prob_soup"] = probas[:, 0]
    clone_df.loc[:, "prob_inter"] = probas[:, 1]
    clone_df.loc[:, "prob_hmdr"] = probas[:, 2]
    
    system_df = clone_df.groupby("source_id").agg({
        "prob_soup": "mean", "prob_inter": "mean", "prob_hmdr": "mean",
        "true_class": "first", "m2_true": "first"
    }).reset_index()

    system_df["predicted_class"] = np.argmax(system_df[["prob_soup", "prob_inter", "prob_hmdr"]].values, axis=1)
    system_df = system_df.assign(
        predicted_label=system_df["predicted_class"].map(CLASS_LABELS),
        true_label=system_df["true_class"].map(CLASS_LABELS),
        correct=system_df["predicted_class"] == system_df["true_class"]
    )

    acc = system_df["correct"].mean()
    print(f"  Gold Test (Orbital) — Overall Accuracy: {acc*100:.2f}%")
    print_per_class_accuracy(system_df, "true_class", "Orbital")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_cols = ["source_id", "m2_true", "true_label", "predicted_label", "correct", "prob_soup", "prob_inter", "prob_hmdr"]
    system_df[out_cols].to_csv(OUTPUT_DIR / "validation_orbital.csv", index=False)
    print(f"  Artifact saved: {OUTPUT_DIR / 'validation_orbital.csv'}")

def validate_sb2(rf: CustomRandomForest):
    input_csv = Path("data/03_physics_anchored/with_m1_sb2_forward_model_results.csv")
    if not input_csv.exists():
        print(f"Skipping SB2 validation, missing {input_csv}")
        return

    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["period", "eccentricity", "semi_amplitude_primary", "m1_solar_m", "m2_sin3i"]).copy()

    n_systems = len(df)
    clone_df = df.loc[df.index.repeat(N_CLONES)].reset_index(drop=True)
    
    rng = np.random.default_rng(RANDOM_SEED + 1)
    cos_i = rng.uniform(0.0, 1.0, size=len(clone_df))
    clone_df.loc[:, "i_sample_deg"] = np.degrees(np.arccos(cos_i))

    sin_i = np.sin(np.radians(clone_df["i_sample_deg"]))
    clone_df.loc[:, "m2_true_sample"] = clone_df["m2_sin3i"] / (sin_i**3)
    clone_df.loc[:, "true_class"] = map_target(clone_df["m2_true_sample"])

    features = ["period", "eccentricity", "semi_amplitude_primary", "m1_solar_m", "i_sample_deg"]
    X = clone_df[features].values
    
    probas = rf.predict_proba(X)
    clone_df.loc[:, "prob_soup"] = probas[:, 0]
    clone_df.loc[:, "prob_inter"] = probas[:, 1]
    clone_df.loc[:, "prob_hmdr"] = probas[:, 2]
    
    system_df = clone_df.groupby("source_id").agg({
        "prob_soup": "mean", "prob_inter": "mean", "prob_hmdr": "mean",
        "true_class": lambda x: x.mode()[0], "m2_sin3i": "first"
    }).reset_index()

    system_df["predicted_class"] = np.argmax(system_df[["prob_soup", "prob_inter", "prob_hmdr"]].values, axis=1)
    system_df = system_df.assign(
        predicted_label=system_df["predicted_class"].map(CLASS_LABELS),
        true_label=system_df["true_class"].map(CLASS_LABELS),
        correct=system_df["predicted_class"] == system_df["true_class"]
    )

    acc = system_df["correct"].mean()
    print(f"  Silver Test (SB2)   — Overall Accuracy: {acc*100:.2f}%")
    print_per_class_accuracy(system_df, "true_class", "SB2")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_cols = ["source_id", "m2_sin3i", "true_label", "predicted_label", "correct", "prob_soup", "prob_inter", "prob_hmdr"]
    system_df[out_cols].to_csv(OUTPUT_DIR / "validation_sb2.csv", index=False)
    print(f"  Artifact saved: {OUTPUT_DIR / 'validation_sb2.csv'}")

def main():
    print(f"\n{'─'*65}")
    print(f"Step 6: Validation via Independent Datasets")
    print(f"{'─'*65}")
    
    model_path = Path("data/03_physics_anchored/custom_rf_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run Step 3 first.")
        
    print("Loading Custom Random Forest...\n")
    with open(model_path, "rb") as f:
        rf = pickle.load(f)

    print("Gold Test: Orbital Astrometric Systems (True Mass Ground Truth)")
    print(f"{'─'*65}")
    validate_orbital(rf)
    print("Silver Test: Double-Lined Spectroscopic Binaries (Minimum Mass Ground Truth)")
    print(f"{'─'*65}")
    validate_sb2(rf)
    print(f"{'─'*65}\n")

if __name__ == "__main__":
    main()

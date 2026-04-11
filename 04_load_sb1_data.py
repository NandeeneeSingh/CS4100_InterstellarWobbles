import numpy as np
import pandas as pd
import pickle

from random_forest import RandomForest, DecisionTree, Node


clones = pd.read_csv("data/03_physics_anchored/sb1_mc_expanded.csv")

with open("data/03_physics_anchored/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

cols = [
    "period",
    "eccentricity",
    "semi_amplitude_primary",
    "m1_solar_m",
    "i_sample_deg",
]
X = clones[cols].values

print(f"Running inference on {len(X):,} clones...")
probas = model.predict_proba(X)

clones["low_mass_prob"] = probas[:, 0]
clones["mid_mass_prob"] = probas[:, 1]
clones["high_mass_prob"] = probas[:, 2]

# average the 100 clone predictions back into one row per real system
systems = (
    clones.groupby("source_id")
    .agg(
        low_mass_prob=("low_mass_prob", "mean"),
        mid_mass_prob=("mid_mass_prob", "mean"),
        high_mass_prob=("high_mass_prob", "mean"),
        orbital_period=("period", "first"),
        eccentricity=("eccentricity", "first"),
        k1=("semi_amplitude_primary", "first"),
        host_mass=("m1_solar_m", "first"),
        min_companion_mass=("m2_min_solar", "first"),
    )
    .reset_index()
)

meta = pd.read_csv(
    "data/03_physics_anchored/with_m1_sb1_cleaned.csv",
    usecols=["source_id", "goodness_of_fit", "parallax", "parallax_error"],
)
systems = systems.merge(meta, on="source_id", how="left")

snr = systems["parallax"] / systems["parallax_error"]
bad = (systems["goodness_of_fit"] > 20) | (snr < 5) | (systems["eccentricity"] > 0.99)

bucket_names = ["Min_Mass_<_1.4", "Min_Mass_1.4_to_3.0", "Min_Mass_>_3.0"]
best = systems[["low_mass_prob", "mid_mass_prob", "high_mass_prob"]].values.argmax(
    axis=1
)
systems["predicted_class"] = np.array(bucket_names)[best]
systems["confidence"] = systems[
    ["low_mass_prob", "mid_mass_prob", "high_mass_prob"]
].max(axis=1)

systems.loc[bad, "predicted_class"] = "Systemic_Artifact"
systems.loc[bad, "confidence"] = 1.0

results = systems[bad | (systems["confidence"] >= 0.85)].copy()
results = results.sort_values("high_mass_prob", ascending=False)

results.to_csv("data/05_results/final_master_catalogue.csv", index=False)

print(f"\n{len(results):,} systems written")
print(results["predicted_class"].value_counts().to_string())

hmdr = results[results["predicted_class"] == "Min_Mass_>_3.0"]
print("\nTop black hole candidates:")
print(
    hmdr[["source_id", "high_mass_prob", "min_companion_mass", "orbital_period"]]
    .head(5)
    .to_string(index=False)
)

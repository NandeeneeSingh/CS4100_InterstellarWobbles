import pandas as pd
from pathlib import Path

RAW_CSV = Path("data/raw/gaia_sb1_raw.csv")
CLEAN_DIR = Path("data/cleaned")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = CLEAN_DIR / "gaia_sb1_cleaned.csv"

def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {RAW_CSV}\n"
            "Run query_gaia_sb1.py first."
        )
    
    df = pd.read_csv(RAW_CSV)
    print(f"Initial rows: {len(df)}")

    # Keep only core fields that matter for modeling and quality checks
    keep_cols = [
        "source_id",
        "phot_g_mean_mag",
        "solution_id",
        "nss_solution_type",
        "ra",
        "dec",
        "parallax",
        "period",
        "period_error",
        "t_periastron",
        "t_periastron_error",
        "eccentricity",
        "eccentricity_error",
        "arg_periastron",
        "arg_periastron_error",
        "center_of_mass_velocity",
        "center_of_mass_velocity_error",
        "semi_amplitude_primary",
        "semi_amplitude_primary_error",
        "inclination",
        "inclination_error",
        "rv_n_obs_primary",
        "rv_n_good_obs_primary",
        "efficiency",
        "significance",
        "goodness_of_fit",
        "conf_spectro_period",
        "flags",
    ]
    df = df[keep_cols].copy()
    print(f"After keeping columns: {len(df)}")

    # Drop rows missing core orbital parameters
    df = df.dropna(
        subset=[
            "period",
            "t_periastron",
            "eccentricity",
            "arg_periastron",
            "center_of_mass_velocity",
            "semi_amplitude_primary",
        ]
    )
    print(f"After dropping rows with missing core parameters: {len(df)}")

    # Physical sanity cuts
    df = df[
        (df["nss_solution_type"] == "SB1")
        & (df["period"] > 0)
        & (df["semi_amplitude_primary"] > 0)
        & (df["eccentricity"] >= 0)
        & (df["eccentricity"] < 1)
    ].copy()

    # Quality cuts
    # We can loosened/tightened later if needed.
    if "rv_n_good_obs_primary" in df.columns:
        df = df[df["rv_n_good_obs_primary"] >= 20]
        print(f"After rv_n_good_obs_primary >= 20: {len(df)}")

    if "efficiency" in df.columns:
        df = df[df["efficiency"] >= 0.2]
        [print(f"After efficiency >= 0.2: {len(df)}")]

    if "significance" in df.columns:
        df = df[df["significance"] >= 10]
        [print(f"After significance >= 10: {len(df)}")]

    if "period" in df.columns:
        df = df[df["period"].between(2, 1000)]
        print(f"After period between 2 and 1000 days: {len(df)}")

    # One row per solution
    #if "solution_id" in df.columns:
    #    df = df.drop_duplicates(subset=["solution_id"])

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Cleaned rows: {len(df)}")
    print(f"Saved cleaned data to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

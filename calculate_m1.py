import pandas as pd
import numpy as np
import os

# 1. Define filenames of cleaned gaia data 
FILES_TO_PROCESS = [
    'SB1-ForwardModel/data/cleaned/gaia_sb1_cleaned.csv',
    'data/sb2-model/sb2_forward_model_results.csv',
    'data/cleaned_nss_orbital_results.csv'
]

# Master folder for all outputs to be in one place 
OUTPUT_DIR = 'data/categorization_ready'

def run_stellar_mass_pipeline():
    # Safety check: make sure output path exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    for path in FILES_TO_PROCESS:
        if not os.path.exists(path):
            print(f"Skipping: {path} (File not found)")
            continue

        print(f"Processing: {path}")
        df = pd.read_csv(path)

        # 2. Remove invalid parallax values (<= 0)
        df = df[df['parallax'] > 0].copy()

        # 3. Calculate distance in parsecs
        df['distance_pc'] = 1000 / df['parallax']

        # 4. Calculate absolute G-band magnitude(M_G)
        df['M_G'] = df['phot_g_mean_mag'] + 5 - 5 * np.log10(df['distance_pc'])

        # 5. Estimate mass of Primary Star (M1) in Solar Masses
        #    Power law approximation: M1 = 10^((4.83 - M_G) / 10)
        df['m1_solar_m'] = 10 ** ((4.7 - df['M_G']) / 10)

        # Save to master folder 
        base_name = os.path.basename(path)
        output_path = os.path.join(OUTPUT_DIR, f"with_m1_{base_name}")

        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    run_stellar_mass_pipeline()
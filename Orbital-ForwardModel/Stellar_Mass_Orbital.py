import pandas as pd
import numpy as np
import os 

# Define path to the cleaned data
INPUT_FILE: 'data/cleaned_nss_orbital_results.csv'
OUTPUT_FILE: 'data/orbital_with_m1.csv'

def run_stellar_mass_calculation(input_file, output_file):
    # Load the cleaned data
    df = pd.read_csv(input_file)

    # 1. Calculate the distance in parsecs
    df['distance_pc'] = 1000 / df['parallax']

    # 2. Calculate the absolute G-band magnitude
    df['M_G'] = df['phot_g_mean_mag'] + 5 - 5 * np.log10(df['distance_pc'])

    # 3. Calculate the mass of the primary star (M1) in solar masses
    df['m1_solar_m'] = 10 ** ((4.83 - df['M_G']) / 10)

    # Save the updated DataFrame to the output file
    df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    run_stellar_mass_calculation(INPUT_FILE, OUTPUT_FILE)
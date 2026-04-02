import pandas as pd
import numpy as np
import os

INPUT_FILE = os.path.join('data/03_physics_anchored/with_m1_orbital_cleaned.csv')
OUTPUT_DIRECTORY = os.path.join('data/04_categorized')
OUTPUT_FILE = os.path.join(OUTPUT_DIRECTORY, 'orbital_categorized.csv')

def run_orbital_monteCarlo(iterations=10000):
    # Safety Check: Ensure output directory exists
    if not os.path.exists(INPUT_FILE):
        print(f"Could not find {INPUT_FILE}")
        return 
    
    df = pd.read_csv(INPUT_FILE)
    most_likely_category = []
    confidence_levels = []

    print(f"Running Monte Carlo simulation for {len(df)} entries with {iterations} iterations each...")

    for i, row in df.iterrows():

        # 1. Setup distributions based on Gaia data
        #    Sample M1, parallax, and semi-major axis proxy
        m1_val = row['m1_solar_m']
        m1_samples = np.random.normal(m1_val, m1_val * 0.1, iterations) # Primary mass

        # 2. Parallax, assuming 2% error
        p_val = row['parallax']
        p_err = row['parallax_error'] if 'parallax_error' in row else p_val * 0.02
        parallax_samples = np.random.normal(p_val, p_err, iterations)
        parallax_samples[parallax_samples <= 0] = 1e-9  # Prevents division by 0


        # 3. Angular wobble from a_thiele_innes constant 
        a_val = row['a_thiele_innes']
        a_err = row['a_thiele_innes_error'] if 'a_thiele_innes_error' in row else abs(a_val) * 0.05
        a_samples = np.random.normal(a_val, a_err, iterations)

        # 4. Calculate M2 using the Mass Function 
        #    m2 ~ (a / p) * m1^(2/3) * P^(2/3)
        period_year = row['period'] / 365.25
        m2_samples = (a_samples / parallax_samples) * (m1_samples ** (2/3)) * (period_year ** (2/3))
        
        # 5. Categorize the results based on M2
        is_remnant = m2_samples > 3.0  # Threshold for compact object
        is_degenerate = (m2_samples <= 3.0) & (m2_samples >= 0.08)  # Between NS and WD
        is_substellar = (m2_samples < 0.08) & (m2_samples > 0) # Below hydrogen burning limit
        is_artifact = (m2_samples <= 0) | (np.isnan(m2_samples))  # Non-physical results

        # 4. Count categories and determine final classification
        counts = {
            "High-Mass Dark Remnant": np.sum(is_remnant), 
            "Degenerate Compact Object": np.sum(is_degenerate), 
            "Substrellar Perturbator": np.sum(is_substellar), 
            "Systemic Artifact": np.sum(is_artifact)
        }

        # 5. Determine category with the highest counts 
        highest_count = max(counts, key=counts.get)
        confidence_level = counts[highest_count] / iterations

        # If the confidence lebvel is too low and there is no clear highest_count, then it's an artifact
        if confidence_level < 0.50: 
            highest_count = "Systemic Artifact"
        
        most_likely_category.append(highest_count)
        confidence_levels.append(confidence_level)

    # Save results 
    df['most_likely_category'] = most_likely_category
    df['category_confidence'] = confidence_levels

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Finished! Results saved to {OUTPUT_FILE}")
    print(f"Summary of Analysis:")
    print(df['most_likely_category'].value_counts())

if __name__ == "__main__":
    run_orbital_monteCarlo()
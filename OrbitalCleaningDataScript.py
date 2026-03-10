import pandas as pd
import numpy as np

# Load Gaia Orbital Solutions data
df = pd.read_csv('nss_orbital_results_2.csv')

# 1. Remove rows of missing or unrealistic parallax values
df = df[df['parallax'] > 0]

# 2. Filter for high-confidence wobbles
# Scientists use > 5 for binaries, but > 10 for planets. Let's use > 10 to account for any weirdly shaped stars or other sources of noise that could be mistaken for a wobble.
df = df[df['significance'] > 10]

# 3. Filter by orbital period 
# Gaia DR3's baseline is around 34 months, so we can only reliably detect planets with orbital periods shorter than that. Let's set a cutoff at 1000 days to be safe.
df = df[df['period'] < 1000]

# 4. A RUWE filter will flag unresolved binaries, which can produce false positives. Let's set a cutoff at 1.4, which is commonly used in the Gaia community.
# We want stars that wobble, but aren't so weird that we can't trust the data.
df = df[(df['ruwe'] > 1.4) & (df['ruwe'] < 4.0)]



# --- Forward modeling to estimate planet masses ---

# Calculate Wobble Amplitude (alpha) in arcseconds
# Alpha = sqrt(a^2 + b^2) where a and b are the Thiele-Innes constants from Gaia's orbital solutions
df['wobble_amplitude'] = np.sqrt(df['a_thiele_innes']**2 + df['b_thiele_innes']**2)

# Save cleaned data to a new CSV file
df.to_csv('cleaned_nss_orbital_results.csv', index=False)
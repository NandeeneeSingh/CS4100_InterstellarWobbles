import pandas as pd
import numpy as np
from astroquery.gaia import Gaia

# Define ADQL query 
query = """
SELECT
    source_id, ra, dec, parallax, significance, period, 
    ruwe, a_thiele_innes, b_thiele_innes, phot_g_mean_mag, bp_rp
FROM gaiadr3.nss_acceleration_astrometric
WHERE parallax >0
AND period < 1000
"""

# Execute the query and retrieve the data
job = Gaia.launch_job(query)
results = job.get_results()

# Convert results to a pandas DataFrame
df = results.to_pandas()

# Forward Model to estimate the mass of the companion
df['wobble_amplitude'] = np.sqrt(df['a_thiele_innes']**2 + df['b_thiele_innes']**2)

# Save result 
df.to_csv('data/cleaned_gaia_data.csv', index=False)

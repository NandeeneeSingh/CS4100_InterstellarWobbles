import pandas as pd
import numpy as np
from astroquery.gaia import Gaia

# Define ADQL query 
query = """
SELECT
    nss.source_id, 
    gs.ra, 
    gs.dec, 
    gs.parallax, 
    nss.significance, 
    nss.period, 
    gs.ruwe, 
    nss.a_thiele_innes, 
    nss.b_thiele_innes, 
    gs.phot_g_mean_mag, 
    gs.bp_rp
FROM gaiadr3.nss_two_body_orbit AS nss
JOIN gaiadr3.gaia_source AS gs ON nss.source_id = gs.source_id
WHERE gs.parallax > 0
  AND nss.period < 1000
  AND nss.nss_solution_type = 'Orbital'
"""

# Execute the query and retrieve the data
job = Gaia.launch_job(query)
results = job.get_results()

# Convert results to a pandas DataFrame
df = results.to_pandas()

# Forward Model to estimate the mass of the companion
df['wobble_amplitude'] = np.sqrt(df['a_thiele_innes']**2 + df['b_thiele_innes']**2)

# Save result 
df.to_csv('../data/02_cleaned/orbital_cleaned.csv', index=False)

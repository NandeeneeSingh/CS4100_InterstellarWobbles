import pandas as pd
import numpy as np

# ran this on Gaia Archive to get the initial SB2 dataset (saved as data/sb2.csv):
# SELECT
#     nss.source_id, nss.nss_solution_type, nss.period, nss.eccentricity,
#     nss.goodness_of_fit, nss.semi_amplitude_primary, nss.semi_amplitude_secondary,
#     m.m1, m.m2, m.fluxratio,
#     gs.ra, gs.dec, gs.parallax, gs.parallax_error,
#     gs.parallax_over_error, gs.ruwe, gs.phot_g_mean_mag, gs.bp_rp
# FROM gaiadr3.nss_two_body_orbit AS nss
# JOIN gaiadr3.binary_masses AS m ON nss.source_id = m.source_id
# JOIN gaiadr3.gaia_source AS gs ON nss.source_id = gs.source_id
# WHERE nss.nss_solution_type IN ('SB2', 'SB2C')
# AND m.m2 > 0

df = pd.read_csv('data/sb2.csv')

# Remove unrealistic parallax values
df = df[df['parallax'] > 0]

# Drop rows missing columns needed for the forward model
df = df.dropna(subset=[
    'semi_amplitude_primary', 'semi_amplitude_secondary',
    'period', 'eccentricity',
    'parallax', 'parallax_error',
    'phot_g_mean_mag', 'bp_rp'
])

# Only keep systems with reliable distance measurements
df = df[df['parallax_over_error'] > 10]

# Both semi-amplitudes must be positive (SB2 = both stars detected)
df = df[df['semi_amplitude_primary'] > 0]
df = df[df['semi_amplitude_secondary'] > 0]

df.to_csv('data/sb2_cleaned.csv', index=False)
print(f"Cleaned SB2 data: {len(df)} systems")
import pandas as pd
import numpy as np
from scipy import constants as const

# Load cleaned SB2 data
df = pd.read_csv('data/sb2_cleaned.csv')

# G in units for: K in km/s, P in days, M in solar masses
PREFACTOR_CONST = 1.036e-7

# Mass ratio (exact — no unknowns for SB2)
# q = M1/M2 = K2/K1
df['mass_ratio_q'] = df['semi_amplitude_secondary'] / df['semi_amplitude_primary']

# Mass function
# M2*sin^3(i) = P*(1-e^2)^(3/2) / (2*pi*G) * (K1+K2)^2 * K1
# M1*sin^3(i) = P*(1-e^2)^(3/2) / (2*pi*G) * (K1+K2)^2 * K2

# Shared prefactor across both equations
prefactor = PREFACTOR_CONST * df['period'] * (1 - df['eccentricity']**2)**1.5

K_total = df['semi_amplitude_primary'] + df['semi_amplitude_secondary']

df['m2_sin3i'] = prefactor * K_total**2 * df['semi_amplitude_primary']
df['m1_sin3i'] = prefactor * K_total**2 * df['semi_amplitude_secondary']

df.to_csv('data/sb2_forward_model_results.csv', index=False)

# Sanity check: our m_sin3i should always be <= Gaia's reported mass (since sin^3(i) <= 1)
print(f"M2*sin3i <= Gaia M2: {(df['m2_sin3i'] <= df['m2']).mean()*100:.1f}%")
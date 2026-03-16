import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv('data/cleaned/sb2_cleaned.csv')

PREFACTOR_CONST = 1.036e-7

def solve_kepler(M, e, tol=1e-10, max_iter=1000):
    """Solve Kepler's equation M = E - e*sin(E) via Newton's method."""
    M = np.asarray(M)
    E = M.copy()
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = f / fp
        E -= dE
        if np.all(np.abs(dE) < tol):
            break
    return E

def sb2_radial_velocity(t, P, T0, e, omega_deg, gamma, K1, K2):
    """
    Forward model for SB2 radial velocity.
    Returns predicted RV for both stars:
        vr1(t) = gamma + K1 [cos(nu + omega) + e*cos(omega)]
        vr2(t) = gamma - K2 [cos(nu + omega) + e*cos(omega)]
    """
    omega = np.radians(omega_deg)

    # Mean anomaly
    n = 2 * np.pi / P
    M = np.mod(n * (np.asarray(t) - T0), 2 * np.pi)

    # Eccentric anomaly
    E = solve_kepler(M, e)

    # True anomaly
    nu = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )

    # RV for both components
    rv_term = np.cos(nu + omega) + e * np.cos(omega)
    vr1 = gamma + K1 * rv_term
    vr2 = gamma - K2 * rv_term

    return vr1, vr2

# --- Plot example system ---
row = df[df['period'] > 50].iloc[0]
t = np.linspace(0, row['period'], 500)

vr1, vr2 = sb2_radial_velocity(
    t, P=row['period'], T0=row['t_periastron'], e=row['eccentricity'],
    omega_deg=row['arg_periastron'], gamma=row['center_of_mass_velocity'],
    K1=row['semi_amplitude_primary'], K2=row['semi_amplitude_secondary']
)

plt.figure(figsize=(8, 5))
plt.plot(t, vr1, label='Primary (K1)')
plt.plot(t, vr2, label='Secondary (K2)')
plt.xlabel('Time (days)')
plt.ylabel('Radial Velocity (km/s)')
plt.title(f'SB2 Forward Model RV Curve (source_id={row["source_id"]})')
plt.legend()
plt.tight_layout()
plt.savefig('data/sb2-model/sb2_example_rv_curve.png', dpi=200)
print(f"Saved example RV curve to data/sb2-model/sb2_example_rv_curve.png")

# --- Mass equations derived from the forward model ---

# Mass ratio (exact for SB2)
df['mass_ratio_q'] = df['semi_amplitude_secondary'] / df['semi_amplitude_primary']

# M*sin^3(i) for both components
prefactor = PREFACTOR_CONST * df['period'] * (1 - df['eccentricity']**2)**1.5
K_total = df['semi_amplitude_primary'] + df['semi_amplitude_secondary']

df['m2_sin3i'] = prefactor * K_total**2 * df['semi_amplitude_primary']
df['m1_sin3i'] = prefactor * K_total**2 * df['semi_amplitude_secondary']

df.to_csv('data/sb2-model/sb2_forward_model_results.csv', index=False)

# Sanity check: m_sin3i should be <= Gaia's reported mass since sin^3(i) <= 1
print(f"M2*sin3i <= Gaia M2: {(df['m2_sin3i'] <= df['m2']).mean()*100:.1f}%")
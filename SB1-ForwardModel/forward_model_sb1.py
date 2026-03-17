import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CLEAN_CSV = Path("data/sb1-model/cleaned/gaia_sb1_cleaned.csv")
OUTPUT_PNG = Path("data/sb1-model/cleaned/example_rv_curve.png")

def solve_keplr(M, e, tol=1e-10, max_iter=1000):
    """
    Solve Kepler's equation:
        M = E - e * sin(E)
    using Newton's method.
    """
    M = np.asarray(M)
    E = M.copy()                      # Initial guess: E = M

    for _ in range(max_iter):         # iteratively improve our guess (E) until convergence
        f = E - e * np.sin(E) - M     # Kepler's equation
        fp = 1 - e * np.cos(E)        # Derivative
        dE = f / fp                   # Newton's step
        E -= dE                       # Update guess

        if np.all(np.abs(dE) < tol):  # Stop when solution stabilizes
            break
    
    return E                          # Eccentric anomaly

def sb1_radial_velocity(t, P, T0, e, omega_deg, gamma, K):
    """
    Forward model for SB1 radial velocity:
        v_r(t) = gamma + K [cos(nu(t) + omega) + e cos(omega)]
    where nu(t) is the true anomaly (at time t).
    """
    omega = np.radians(omega_deg)

    # Mean anomaly
    n = 2 * np.pi / P
    M = n * (np.asarray(t) - T0)
    M = np.mod(M, 2 * np.pi)  # Wrap to [0, 2pi]

    # Eccentric anomaly
    E = solve_keplr(M, e)

    # True anomaly
    nu = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )

    # Radial velocity
    vr = gamma + K * (np.cos(nu + omega) + e * np.cos(omega))
    return vr

def main():
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"Clean file not found: {CLEAN_CSV}\n"
            "Run clean_sb1_data.py first."
        )
    
    df = pd.read_csv(CLEAN_CSV)

    if len(df) == 0:
        raise ValueError("Cleaned dataframe is empty.")
    
    # Pick the first row as an example system
    row = df.iloc[0]

    P = row["period"]
    T0 = row["t_periastron"]
    e = row["eccentricity"]
    omega = row["arg_periastron"]
    gamma = row["center_of_mass_velocity"]
    K = row["semi_amplitude_primary"]

    # Create time grid across one orbital period
    t = np.linspace(T0, T0 + P, 500)
    vr = sb1_radial_velocity(t, P, T0, e, omega, gamma, K)

    plt.figure(figsize=(8, 5))
    plt.plot(t, vr)
    plt.xlabel("Time (days)")
    plt.ylabel("Radial Velocity (km/s)")
    plt.title(f"Example SB1 RV Curve (source_id={row['source_id']})")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)

    print(f"Saved example RV curve to: {OUTPUT_PNG}")
    print(f"\nExample system parameterss used:")
    print(f"source_id = {row['source_id']}")
    print(f"P = {P}")
    print(f"T0 = {T0}")
    print(f"e = {e}")
    print(f"omega = {omega}")
    print(f"gamma = {gamma}")
    print(f"K = {K}")

if __name__ == "__main__":
    main()

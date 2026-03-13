import numpy as np

def forward_model(P, e, i, omega, Omega, a_pl, parallax, M_star, M_plan):
    """
    Predicts Thiele-Innes constants based on orbital parameters.
    P: Period (days)
    e: Eccentricity
    i, omega, Omega: Angles (radians)
    a_pl: Planet semi-major axis (AU)
    parallax: mas
    """
    # 1. Calculate the star's wobble amplitude (alpha) in mas
    # Using the mass ratio: alpha = (M_plan / M_star) * (a_pl * parallax)
    alpha = (M_plan / M_star) * a_pl * parallax
    
    # 2. Calculate Thiele-Innes Constants (The "Wobble Fingerprint")
    A = alpha * (np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.sin(Omega) * np.cos(i))
    B = alpha * (np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(Omega) * np.cos(i))
    F = alpha * (-np.sin(omega) * np.cos(Omega) - np.cos(omega) * np.sin(Omega) * np.cos(i))
    G = alpha * (-np.sin(omega) * np.sin(Omega) + np.cos(omega) * np.cos(Omega) * np.cos(i))
    
    return {'A': A, 'B': B, 'F': F, 'G': G}

# Example: A Jupiter-mass planet around a Sun-like star at 10 parsecs
params = forward_model(P=4332, e=0.05, i=np.radians(45), omega=0, Omega=0, 
                       a_pl=5.2, parallax=100.0, M_star=1.0, M_plan=0.00095)
print(params)
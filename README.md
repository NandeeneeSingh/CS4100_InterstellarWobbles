# CS4100 Interstellar Wobbles 

## Step 1: Data Ingestion & Physical Anchoring

**The Data:** We query the Gaia DR3 archive for three specific system types: SB1 (classic radial velocity), SB2 (validation control group), and Orbital (3D astrometry).

**The Physics:** We calculate the mass of the visible host star (M1) using its luminosity and distance. This serves as the gravitational anchor for the rest of the pipeline.

## Step 2: The Two-Stage Bayesian Optimizer

Because telescope data inherently contains noise, and radial velocity data is missing the orbital inclination (i), we cannot use deterministic math. Instead, we use a stochastic probability engine:

The Physics Engine: We map the data to theoretical models—using Kepler's Third Law for 3D Astrometric wobble, and a highly optimized Newton-Raphson solver to resolve the Mass Function for 1D Spectroscopic binaries.

The Stochastic Simulation: For every single star system, the pipeline runs a 10,000-iteration Monte Carlo simulation. It randomly samples Gaia's specific error bars (for parallax, period, and semi-amplitude) and simulates thousands of random orbital tilts. This outputs a true probability distribution of the dark companion's mass (M2​).

## Step 3: Unsupervised Anomaly Detection

Gaia telescope data is notoriously noisy. Instead of using hardcoded human rules (e.g., deleting rows based on arbitrary error thresholds), we deploy AI to clean the data.

**The ML Filter:** The pipeline feeds the physical results into an Isolation Forest (an Unsupervised Machine Learning model).

**The Action:** The AI autonomously clusters mathematically stable "real physics" systems and isolates erratic, noisy data points, officially flagging them as "Systemic Artifacts."

## Step 4: Probabilistic Classification & Visualization

**Confidence Matrices:** Because the pipeline is probabilistic, it does not output rigid, deterministic labels. Instead, it outputs confidence levels (e.g., "This system has a 92% probability of being a High-Mass Dark Remnant, and an 8% probability of being a Degenerate Compact Object").

**Visual Validation:** The pipeline generates a final population distribution histogram, as well as an HR Diagram to visually validate the foundational M1 physics.

**Step 5**: Literature Cross-Matching (Validation)

To prove the efficacy of the pipeline, the final AI-cleaned catalog is cross-referenced against external databases of known astronomical objects.


`Python version 3.14.0`

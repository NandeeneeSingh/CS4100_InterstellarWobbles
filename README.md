# CS4100 Interstellar Wobbles 

## Step 1: Data Ingestion & Physical Anchoring

**The Data:** We query the Gaia DR3 archive for three specific system types: SB1 (classic radial velocity), SB2 (validation control group), and Orbital (3D astrometry).

**The Physics:** We calculate the mass of the visible host star (M1) using its luminosity and distance. This serves as the gravitational anchor for the rest of the pipeline.

## Step 2: The Two-Stage Bayesian Optimizer

We use a two-stage approach to efficiently solve the inverse physics problem:

**Phase A (Informed Prior Generation):** A standard Monte Carlo stochastic search throws 10,000 random mass and tilt guesses at our forward physics models. This broadly maps the parameter space and finds the mathematical "foothills" of the true answer.

**Phase B (Bayesian Parameter Estimation):** The best guesses from Phase A are injected into a Markov Chain Monte Carlo (MCMC) algorithm. The MCMC dynamically learns the topology of the data, stepping through the parameter space to find the exact posterior probability distribution of the dark companion's mass (M2) and orbital inclination (i).

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

# CS4100 Interstellar Wobbles 

## Step 1: Feature Engineering & The Monte Carlo Expansion

- **The Data:** We clean the Gaia DR3 SB1 dataset, isolating the available physical features (orbital period, eccentricity, radial velocity amplitude, and primary mass).

- **The Physics:** To solve the missing inclination angle (i), we implement a Monte Carlo simulation. For every real SB1 system, we generate 100 computational clones, assigning each a random orbital tilt drawn from a true geometric distribution. This statistical spread completely bypasses the physics "degeneracy wall."

## Step 2: Synthetic Data Generation (The "Black Hole" Injection)

- **The Process:** We use forward physics models to simulate the wobbles of a primary star being pulled by a high-mass anomaly (10-20 solar masses).

- **The Goal:** We apply the exact same Monte Carlo cloning process to these synthetic anomalies, simulating them across all possible viewing angles (from edge-on to face-on). This forces the AI to learn the full physical boundary of a black hole wobble, creating a perfectly balanced training set.

## Step 3: Minimum Mass Target & Standard Forest Architecture

- **The ML Model:** We train a standard Random Forest classifier, utilizing a manual implementation of scikit's predict_proba() function to output class probabilities rather than hard predictions.

- **The Target Shift:** Instead of predicting absolute classes (like High-Mass Dark Remnant), we train the model to predict Minimum Mass thresholds (e.g., Min_Mass_>_3_Solar). This perfectly aligns the machine learning goal with the strict mathematical limits of the Binary Mass Function.

## Step 4: Probabilistic Inference on SB1 Data

- **The Action:** We feed the unlabeled, Monte-Carlo-expanded SB1 data into the trained Random Forest.

- **The Output:** The model evaluates the statistical clones and aggregates the class ratios from every tree, outputting a mathematically sound final Probability Distribution for each system's minimum mass.


## Step 5: Anomaly Classification & Artifact Filtering

- **The Filter:** We apply strict thresholding to the probability outputs. If a system's highest probability is a "Systemic Artifact" (bad Gaia data or instrumental noise), or if it fails to reach a high confidence threshold, it is autonomously flagged as noise and discarded.

- **The Result:** The pipeline produces a pristine, highly-curated catalog of high-confidence black hole and anomaly candidates for publication and telescope follow-up.

`Python version 3.14.0`

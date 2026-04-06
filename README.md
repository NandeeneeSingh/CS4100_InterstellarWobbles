# CS4100 Interstellar Wobbles 

## Core AI and ML Concepts 

- **Ensemble Learning (Bagging):** By building a Random Forest entirely from scratch in pure Python, we implement "Bootstrap Aggregating" (Bagging). Each tree sees a different random subset of the data, which reduces the overall variance and makes the model more robust to noisy Gaia data.
- **Supervised Transfer Learning (Domain Adaptation):** We train on domains where ground truth or physical boundaries are known (e.g., synthetic physics injections and SB2 extrapolations) and transfer that knowledge to SB1 systems (the target domain where companion labels are unseen). 
- **Probabilistic Classification (Soft Voting):** Instead of a "hard" vote (e.g., "This is a Black Hole"), our custom leaf nodes calculate the percentage of classes bounded in that leaf to return a probability distribution. This elevates the model from a simple classifier to an uncertainty-aware system.
- **Physics-Informed Data Augmentation:** Because the dataset lacks enough confirmed real black holes, we use the binary mass function and orbital equations to generate synthetic training samples (Monte Carlo inclination clones). This "teaches" the AI the mathematical signature of a black hole so it can recognize one in real, unlabeled data.

---

## High-Level Project Overview & Steps

### Step 1: Feature Alignment, Cleaning & The Monte Carlo Expansion
- **The Data:** We clean the Gaia DR3 SB1 dataset, identifying and isolating the physical features available (orbital period, eccentricity, radial velocity amplitude, and primary star mass).
- **The Physics:** To solve the missing inclination angle ($i$) degeneracy, we implement a Monte Carlo simulation. For every real system, we generate 100 computational clones, assigning each a random orbital tilt drawn from a true geometric distribution. This statistical spread forces the model to learn the full physical boundaries.

### Step 2: Synthetic Data Generation (The "Black Hole" Injection)
- **The Process:** We use forward physics equations to simulate the "wobbles" of a primary star being pulled by a high-mass anomaly (10-20 solar masses).
- **The Goal:** We apply the exact same Monte Carlo cloning process to these synthetic anomalies across all possible viewing angles. This creates a perfectly balanced training set so the AI doesn't just learn to ignore rare anomalies like black holes.

### Step 3: Minimum Mass Target & Custom Random Forest Architecture
- **The ML Model:** We build a Random Forest classifier entirely from scratch using pure Python/NumPy. At the leaf node level, instead of returning a single label, it calculates the percentage of each physical class present in that leaf for pure soft voting.
- **The Target Shift:** Rather than predicting absolute string classes naturally, we train the model to predict Minimum Mass thresholds (e.g., `Min_Mass_>_3.0`). This perfectly aligns the machine learning goal with the strict mathematical limits of the physical Binary Mass Function.

### Step 4: Probabilistic Inference on SB1 Data
- **The Action:** We feed the unlabeled, Monte-Carlo-expanded SB1 data into our trained custom forest.
- **The Output:** The model evaluates the statistical clones and aggregates the class ratios from every tree, outputting a mathematically sound final Probability Distribution for each specific anomaly.

### Step 5: Anomaly Classification & Artifact Filtering
- **The Filter:** We apply strict thresholding to the probability outputs. If a system fails to reach a high confidence threshold ($>85\%$), it is discarded. If the astrometric metadata suggests bad Gaia data or instrumental noise, the system's class is overridden and explicitly flagged as a `Systemic_Artifact`. 
- **The Result:** The pipeline produces a pristine, highly-curated catalog of high-confidence black hole candidates as well as identified noise artifacts for further scientific study and telescope follow-up.

### Step 6: Validation via Independent Datasets
- **Orbital Data (The Gold Test):** Used strictly as a final hold-out set to validate the model's accuracy. For these fully-resolved astrometric systems, we implemented complete physical orbital equations to calculate the exact *True Mass* for verification.
- **SB2 Data (The Silver Test):** Used as a secondary test to evaluate the model's performance on double-lined systems. Because both stars are visible in the spectra, we used physics equations to mathematically derive the *Minimum Mass* to serve as our ground truth.
- **The Result:** This multi-tiered evaluation proves that the probabilistic logic learned from synthetic injections successfully scales out to completely novel, fully-resolved physical systems.

---
`Python version 3.14.0`
Note: It's recommended that you install the `DataWrangler` VS Code extension to open large CSVs.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import os


INPUT_DIR = os.path.join('data/04_categorized')
OUTPUT_DIR = os.path.join('data/05_results')

def merge_and_run_ml():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("1. Loading and Harmonizing Datasets...")
    
    # Load all three categorized datasets
    try:
        df_orb = pd.read_csv(os.path.join(INPUT_DIR, 'orbital_categorized.csv'))
        df_sb1 = pd.read_csv(os.path.join(INPUT_DIR, 'sb1_categorized.csv'))
        df_sb2 = pd.read_csv(os.path.join(INPUT_DIR, 'sb2_categorized.csv'))
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the CSVs. {e}")
        return

    
    # Merge into one master dataset
    df = pd.concat([df_orb, df_sb1, df_sb2], ignore_index=True)
    print(f"Merged Catalog Size: {len(df)} systems.")

    print("2. Prepping Data for the AI Isolation Forest...")
    # Define the quality metrics we want the AI to analyze
    ml_features = ['ruwe', 'significance', 'goodness_of_fit', 'efficiency']
    
    # Filter only to features that actually exist in the merged dataframe
    valid_features = [col for col in ml_features if col in df.columns]
    
    ml_data = df[valid_features].copy()
    ml_data = ml_data.fillna(ml_data.median())

    print("3. Running Anomaly Detection...")
    # contamination=0.1 means we expect roughly 10% of the data to be noisy junk
    iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    
    # Predict (-1 = Anomaly, 1 = Normal)
    df['anomaly_score'] = iso_forest.fit_predict(ml_data)

    # OVERRIDE the physical category if the AI flagged it as an artifact
    df.loc[df['anomaly_score'] == -1, 'most_likely_category'] = 'Systemic Artifact'

    print("4. Generating Final Visualizations...")
    sns.set_theme(style="whitegrid")
    
    # PLOT 1: Population Distribution Histogram

    plt.figure(figsize=(10, 6))
    category_colors = {
        "High-Mass Dark Remnant": "black",
        "Degenerate Compact Object": "purple",
        "Substrellar Perturbator": "orange", 
        "Systemic Artifact": "red"
    }

    sns.countplot(
        data=df, 
        y='most_likely_category', 
        order=category_colors.keys(),
        palette=category_colors
    )
    plt.title('Distribution of Dark Companions (AI-Cleaned)', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Companion Category', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'population_histogram.png'), dpi=300)
    plt.close()

    # PLOT 2: Observational HR Diagram

    if 'bp_rp' in df.columns and 'M_G' in df.columns:
        plt.figure(figsize=(8, 10))
        
        # Plot real physics systems
        real_systems = df[df['most_likely_category'] != 'Systemic Artifact']
        plt.scatter(real_systems['bp_rp'], real_systems['M_G'], 
                    c='blue', s=4, alpha=0.6, label='Valid Systems')
        
        # Plot artifacts in red
        artifacts = df[df['most_likely_category'] == 'Systemic Artifact']
        plt.scatter(artifacts['bp_rp'], artifacts['M_G'], 
                    c='red', s=4, alpha=0.8, marker='x', label='AI Flagged Artifacts')

        # Invert Y-axis (Brighter stars have lower/negative magnitudes)
        plt.gca().invert_yaxis()
        
        plt.title('Observational Hertzsprung-Russell Diagram', fontsize=16)
        plt.xlabel('Color (BP - RP) [Temperature Proxy]', fontsize=12)
        plt.ylabel('Absolute Magnitude (M_G) [Luminosity Proxy]', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'hr_diagram.png'), dpi=300)
        plt.close()

    # Save
    final_output_path = os.path.join(OUTPUT_DIR, 'final_master_catalog.csv')
    df.to_csv(final_output_path, index=False)
    print(f"Pipeline Complete! Master Catalog and Visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    merge_and_run_ml()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import ScalarFormatter
import os

sns.set_theme(style="whitegrid", palette="muted") 
validation_file_path = "data/06_validation/validation_sb2.csv"

# Safety
if not os.path.exists(validation_file_path):
    print("Error: Missing validation data. Please run 06A_validation.py first.")
    exit()

df = pd.read_csv(validation_file_path)

# Mappings to translate class labels for better readability in plots
class_mapping = {
    "(0) Min_Mass_<_1.4": 0,
    "(1) Min_Mass_1.4_to_3.0": 1,
    "(2) Min_Mass_>_3.0": 2
}

predicted_name_mapping = {
    "(0) Min_Mass_<_1.4": "White Dwarf",
    "(1) Min_Mass_1.4_to_3.0": "Neutron Star",
    "(2) Min_Mass_>_3.0": "Black Hole"
}

# Apply mappings to create new columns for plotting
df["true_class_num"] = df["true_label"].map(class_mapping)
df["predicted_class_num"] = df["predicted_label"].map(class_mapping)
df["predicted_class_name"] = df["predicted_label"].map(predicted_name_mapping)

# Calculate model's confidence in its predictions by taking the max predicted probability across classes
df["max_confidence"] = df[["prob_soup", "prob_inter", "prob_hmdr"]].max(axis=1)


'''
Plot 1: True class vs. Predicted class (Heatmap)
'''
confusion_matrix_data = confusion_matrix(df["true_class_num"], df["predicted_class_num"], labels=[0, 1, 2])
labels = ["White Dwarf\n(<1.4 M☉)", "Neutron Star\n(1.4-3.0 M☉)", "Black Hole\n(>3.0 M☉)"]

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_data, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})

# Calculate overall accuracy for the title 
title_accuracy = np.trace(confusion_matrix_data) / np.sum(confusion_matrix_data)

# Plotting 
plt.title(f"SB2 Validation Confusion Matrix ({title_accuracy: .1%} Accuracy)", fontsize=14, pad=15, fontweight="bold")
plt.ylabel("True Class", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Class", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("sb2_confusion_matrix.png", dpi=300)
plt.close()
print("Saved confusion matrix plot as sb2_confusion_matrix.png")


'''
Plot 2: Distribution of max_confidence scores for final candidates (KDE Plot)
'''
plt.figure(figsize=(10, 6))

sns.kdeplot(df["max_confidence"], fill=True, color="#6f42c1", alpha=0.6, linewidth=2)
plt.axvline(0.85, color="#dc3545", linestyle="--", linewidth=2.5, label="85% Confidence Threshold")

# Plotting
plt.title("Distribution of Model Confidence Scores for SB2 Systems", fontsize=14, pad=15, fontweight="bold")
plt.xlabel("Max Confidence", fontsize=12, fontweight="bold")
plt.ylabel("Density of Star Systems", fontsize=12, fontweight="bold")
plt.legend(fontsize=11)
plt.xlim(0.3,1.05)
plt.tight_layout()
plt.savefig("sb2_confidence_distribution.png", dpi=300)
plt.close()
print("Saved confidence distribution plot as sb2_confidence_distribution.png")


'''
Plot 3: True Minimum Mass vs. Predicted Probability of being a High-Mass Dark Remnant (Scatter Plot)
'''
plt.figure(figsize=(10, 6))

sns.scatterplot(data=df, x="m2_sin3i", y="prob_hmdr",
                hue="predicted_class_name",
                palette={'White Dwarf': '#0dcaf0', 'Neutron Star': '#fd7e14', 'Black Hole': '#212529'},
                alpha=0.7, s=40, edgecolor='w', linewidth=0.5)

plt.axvline(3.0, color='#dc3545', linestyle='--', linewidth=2, label='TOV Mass Limit (3.0 M☉)')

# Plotting 
plt.title("True Minimum Mass vs. Predicted Probability of High-Mass Dark Remnant", fontsize=14, pad=15, fontweight="bold")
plt.xlabel("Minimum Mass (M☉)", fontsize=12, fontweight="bold")
plt.ylabel("Predicted Probability of HMDR", fontsize=12, fontweight="bold")
plt.legend(title="Predicted Class", fontsize=10, title_fontsize=11)

# Capping x-axis to prevent outliers from dominating the plot
max_x = min(df["m2_sin3i"].max() + 1, 20)
plt.xlim(0, max_x)

plt.tight_layout()
plt.savefig("sb2_mass_vs_hmdr_probability.png", dpi=300)
plt.close()
print("Saved mass vs. HMDR probability plot as sb2_mass_vs_hmdr_probability.png")


'''
Plot 3 (LOGARITHMIC SCALE): True Minimum Mass vs. Predicted Probability of being a High-Mass Dark Remnant (Scatter Plot)
'''
plt.figure(figsize=(10, 6))

sns.scatterplot(data=df, x="m2_sin3i", y="prob_hmdr",
                hue="predicted_class_name",
                palette={'White Dwarf': '#0dcaf0', 'Neutron Star': '#fd7e14', 'Black Hole': '#212529'},
                alpha=0.5, s=25, edgecolor='w', linewidth=0.3)

plt.xscale("log")

plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.xticks([0.5, 1, 2, 3, 5, 10, 20])

plt.axvline(3.0, color='#dc3545', linestyle='--', linewidth=2, label='TOV Mass Limit (3.0 M☉)')

# Plotting 
plt.title("True Minimum Mass vs. Predicted Probability of High-Mass Dark Remnant (Log Scale View)", fontsize=14, pad=15, fontweight="bold")
plt.xlabel("Minimum Mass (M☉) [Log Scale]", fontsize=12, fontweight="bold")
plt.ylabel("Predicted Probability of HMDR", fontsize=12, fontweight="bold")
plt.legend(title="Predicted Class", fontsize=10, title_fontsize=11)

plt.xlim(0.4, 25)

plt.tight_layout()
plt.savefig("sb2_mass_vs_hmdr_probability_log.png", dpi=300)
plt.close()
print("Saved mass vs. HMDR probability plot as sb2_mass_vs_hmdr_probability_log.png")

print("\nAll visualizations generated and saved successfully!")
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_csv("data/06_validation/validation_sb2.csv")
y_true = df["true_label"]
y_pred = df["predicted_label"]

print("=== Original 3-Class Evaluation ===")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\n" + "="*50)
print("=== Grouped Binary Evaluation (0 & 1 vs 2) ===")
print("="*50)

def group_label(label):
    if "(0)" in label or "(1)" in label:
        return "(0+1) Min_Mass_<=_3.0"
    return "(2) Min_Mass_>_3.0"

y_true_grouped = y_true.apply(group_label)
y_pred_grouped = y_pred.apply(group_label)

print("Accuracy:", accuracy_score(y_true_grouped, y_pred_grouped))
print("\nClassification Report:")
print(classification_report(y_true_grouped, y_pred_grouped))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true_grouped, y_pred_grouped))

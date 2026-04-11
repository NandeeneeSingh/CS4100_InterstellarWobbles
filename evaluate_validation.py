import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_csv("data/06_validation/validation_sb2.csv")
y_true = df["true_label"]
y_pred = df["predicted_label"]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

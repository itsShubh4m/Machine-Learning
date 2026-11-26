
# PRACTICAL 2: KNN FOR BREAST CANCER MEDICAL DIAGNOSIS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

# Step 1: Load Dataset
cancer = load_breast_cancer()

X = cancer.data
y = 1 - cancer.target  # 0=Benign, 1=Malignant

print("Dataset Loaded Successfully!")
print("Total Records:", len(X))
print("Total Features:", X.shape[1])

print("\nClass Distribution:")
print(pd.Series(y).value_counts())

# GRAPH 1: CLASS DISTRIBUTION
plt.figure(figsize=(6, 4))
plt.bar(["Benign", "Malignant"],
        [(y == 0).sum(), (y == 1).sum()])
plt.title("Tumor Class Distribution")
plt.xlabel("Tumor Type")
plt.ylabel("Number of Samples")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# GRAPH 2: FEATURE COMPARISON
plt.figure(figsize=(7, 5))

plt.scatter(X[y == 0, 0], X[y == 0, 1],
            label="Benign", alpha=0.6)

plt.scatter(X[y == 1, 0], X[y == 1, 1],
            label="Malignant", alpha=0.6)

plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.title("Benign vs Malignant Feature Comparison")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Step 2: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Step 3: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Apply KNN with Different Distance Metrics
metrics = ["euclidean", "manhattan", "minkowski", "chebyshev"]

results = []
models = {}

for metric in metrics:

    model = KNeighborsClassifier(
        n_neighbors=5,
        metric=metric
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    results.append([acc*100, pre*100, rec*100, f1*100])
    models[metric] = model

    print("\nDistance Metric:", metric)
    print("Accuracy:", round(acc*100, 2), "%")

# Step 5: Display Comparison Table
df = pd.DataFrame(
    results,
    index=metrics,
    columns=["Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\nPerformance Comparison:")
print(df.round(2))

# GRAPH 3: ACCURACY COMPARISON
plt.figure(figsize=(7, 4))
plt.bar(df.index, df["Accuracy"])
plt.title("KNN Accuracy Comparison")
plt.xlabel("Distance Metrics")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# GRAPH 4: PERFORMANCE METRICS
df.plot(kind="bar", figsize=(9, 5))
plt.title("KNN Performance Metrics")
plt.xlabel("Distance Metrics")
plt.ylabel("Score (%)")
plt.ylim(0, 105)
plt.xticks(rotation=15)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# Step 6: Select Best Model
best_metric = df["Accuracy"].idxmax()
best_model = models[best_metric]

pred = best_model.predict(X_test)

print("\nBest Distance Metric:", best_metric)
print("Best Accuracy:", round(df.loc[best_metric, "Accuracy"], 2), "%")

# Step 7: Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, pred,
    target_names=["Benign", "Malignant"]
))

# GRAPH 5: CONFUSION MATRIX
ConfusionMatrixDisplay.from_predictions(
    y_test, pred,
    display_labels=["Benign", "Malignant"],
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# GRAPH 6: ROC CURVE
plt.figure(figsize=(8, 5))

for metric, model in models.items():

    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)

    auc_score = roc_auc_score(y_test, prob)

    plt.plot(
        fpr, tpr,
        label=metric + " (AUC=" + str(round(auc_score, 3)) + ")"
    )

plt.plot([0, 1], [0, 1], "--", label="Random Classifier")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


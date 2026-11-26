
# PRACTICAL 1: KNN FOR BOTANICAL SPECIES IDENTIFICATION
# Dataset: Wheat Seeds Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


# STEP 1: LOAD DATASET

columns = [
    "Area", "Perimeter", "Compactness",
    "KernelLength", "KernelWidth",
    "Asymmetry", "GrooveLength", "Species"
]

data = pd.read_csv(
    "seeds.csv",
    sep=r"\s+",
    header=None,
    names=columns
)

species_names = {
    1: "Kama",
    2: "Rosa",
    3: "Canadian"
}

print("========== KNN BOTANICAL SPECIES IDENTIFICATION ==========")

print("\nDataset Loaded Successfully!")
print("Total Records:", len(data))
print("\nFirst 5 Records:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nSpecies Distribution:")
print(data["Species"].value_counts().sort_index())


# GRAPH 1: DATASET DISTRIBUTION

plt.figure(figsize=(7, 5))

counts = data["Species"].value_counts().sort_index()

plt.bar(
    [species_names[i] for i in counts.index],
    counts.values
)

plt.title("Wheat Seed Species Distribution")
plt.xlabel("Wheat Species")
plt.ylabel("Number of Samples")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()


# STEP 2: SEPARATE FEATURES AND TARGET

X = data.drop("Species", axis=1).values
y = data["Species"].values


# GRAPH 2: FEATURE COMPARISON

plt.figure(figsize=(8, 5))

for species in [1, 2, 3]:

    subset = data[data["Species"] == species]

    plt.scatter(
        subset["Area"],
        subset["KernelLength"],
        label=species_names[species],
        alpha=0.7
    )

plt.title("Wheat Species Feature Comparison")
plt.xlabel("Seed Area")
plt.ylabel("Kernel Length")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()


# STEP 3: SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# STEP 4: FEATURE SCALING

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# STEP 5: APPLY KNN FOR K = 3, 5, 7

k_values = [3, 5, 7]

accuracies = []
models = {}

print("\n========== KNN ACCURACY RESULTS ==========")

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    accuracies.append(accuracy * 100)

    models[k] = model

    print("\nK =", k)
    print("Accuracy:", round(accuracy * 100, 2), "%")


# GRAPH 3: KNN ACCURACY COMPARISON

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    accuracies,
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=8
)

plt.title("KNN Accuracy Comparison")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy (%)")
plt.xticks(k_values)
plt.grid(True, alpha=0.3)

for i in range(len(k_values)):

    plt.annotate(
        str(round(accuracies[i], 2)) + "%",
        (k_values[i], accuracies[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center"
    )

plt.tight_layout()


# STEP 6: MANUAL EUCLIDEAN DISTANCE

def euclidean_distance(a, b):

    distance = 0

    for i in range(len(a)):

        distance += (a[i] - b[i]) ** 2

    return math.sqrt(distance)


# STEP 7: MANUAL DISTANCE FOR FIRST 5 TEST SAMPLES

print("\n========== MANUAL EUCLIDEAN DISTANCE ==========")

for i in range(5):

    test_sample = X_test[i]

    distances = []

    for j in range(len(X_train)):

        d = euclidean_distance(
            test_sample,
            X_train[j]
        )

        distances.append((d, y_train[j]))

    distances.sort(key=lambda x: x[0])

    print("\nTest Sample:", i + 1)

    print(
        "Actual Species:",
        species_names[y_test[i]]
    )

    for k in k_values:

        nearest = distances[:k]

        labels = [item[1] for item in nearest]

        prediction = min(
            set(labels),
            key=lambda label: (-labels.count(label), label)
        )

        print("\nK =", k)
        print("Nearest Distances:")

        for distance, label in nearest:

            print(
                round(distance, 4),
                "-",
                species_names[label]
            )

        print(
            "Predicted Species:",
            species_names[prediction]
        )


# STEP 8: SELECT BEST K VALUE

best_index = accuracies.index(max(accuracies))
best_k = k_values[best_index]

best_model = models[best_k]

best_predictions = best_model.predict(X_test)

print("\n========== BEST MODEL ==========")

print("Best K Value:", best_k)

print(
    "Best Accuracy:",
    round(accuracies[best_index], 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        best_predictions,
        labels=[1, 2, 3],
        target_names=["Kama", "Rosa", "Canadian"],
        zero_division=0
    )
)


# GRAPH 4: CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[1, 2, 3]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Kama", "Rosa", "Canadian"]
)

fig, ax = plt.subplots(figsize=(7, 5))

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title("KNN Confusion Matrix (K = " + str(best_k) + ")")
plt.tight_layout()


# STEP 9: DISPLAY ALL GRAPHS

plt.show()



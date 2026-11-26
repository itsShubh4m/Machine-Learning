
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# 1. Load Handwritten Digits Dataset
digits = load_digits()

X = digits.data
y = digits.target

print("Total Images:", len(X))
print("Number of Features:", X.shape[1])
print("Number of Classes:", len(np.unique(y)))

# 2. Display First 10 Digit Images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"Digit: {y[i]}")
    ax.axis("off")

plt.suptitle("Sample Handwritten Digit Images")
plt.tight_layout()
plt.show()

# 3. Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Create and Train KNN Model
model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
model.fit(X_train, y_train)

# 5. Predict Test Data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nKNN Model Accuracy:", accuracy * 100, "%")

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# 7. Visualize 3 Test Images and Their 5 Nearest Neighbors
distances, indices = model.kneighbors(X_test[:3])

fig, axes = plt.subplots(3, 6, figsize=(13, 7))

for i in range(3):

    # Test Image
    axes[i, 0].imshow(X_test[i].reshape(8, 8), cmap="gray")
    axes[i, 0].set_title(
        f"Test\nActual: {y_test[i]}\nPredicted: {y_pred[i]}"
    )
    axes[i, 0].axis("off")

    # Five Nearest Neighbors
    for j in range(5):
        neighbor_index = indices[i][j]

        axes[i, j + 1].imshow(
            X_train[neighbor_index].reshape(8, 8),
            cmap="gray"
        )

        axes[i, j + 1].set_title(
            f"Neighbor {j+1}\nDigit: {y_train[neighbor_index]}\n"
            f"Dist: {distances[i][j]:.2f}"
        )
        axes[i, j + 1].axis("off")

plt.suptitle("3 Test Images and Their 5 Nearest Neighbors")
plt.tight_layout()
plt.show()

# 8. Digit Class Distribution
plt.figure(figsize=(8, 5))
plt.hist(y, bins=np.arange(11)-0.5, edgecolor="black")
plt.xticks(range(10))
plt.title("Distribution of Handwritten Digits")
plt.xlabel("Digit")
plt.ylabel("Number of Images")
plt.grid(axis="y")
plt.show()

# 9. Average Pixel Intensity Heatmap
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for digit, ax in enumerate(axes.flat):
    avg_image = digits.images[y == digit].mean(axis=0)
    sns.heatmap(avg_image, cmap="viridis", cbar=False, ax=ax)
    ax.set_title(f"Digit {digit}")
    ax.axis("off")

plt.suptitle("Average Pixel Intensity of Each Digit")
plt.tight_layout()
plt.show()

# 10. Correct vs Incorrect Predictions
correct = np.sum(y_test == y_pred)
incorrect = np.sum(y_test != y_pred)

plt.figure(figsize=(6, 4))
plt.bar(["Correct", "Incorrect"], [correct, incorrect])
plt.title("Correct vs Incorrect Predictions")
plt.ylabel("Number of Predictions")
plt.show()

# 11. Precision, Recall and F1-Score
report = classification_report(
    y_test, y_pred, output_dict=True, zero_division=0
)

precision = [report[str(i)]["precision"] for i in range(10)]
recall = [report[str(i)]["recall"] for i in range(10)]
f1 = [report[str(i)]["f1-score"] for i in range(10)]

x = np.arange(10)
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar(x - width, precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar(x + width, f1, width, label="F1-Score")

plt.xticks(x, range(10))
plt.xlabel("Digit")
plt.ylabel("Score")
plt.title("Precision, Recall and F1-Score")
plt.legend()
plt.show()

# 12. Accuracy for Each Digit
digit_accuracy = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(8, 5))
plt.bar(range(10), digit_accuracy)
plt.xticks(range(10))
plt.ylim(0, 1.05)
plt.xlabel("Digit")
plt.ylabel("Accuracy")
plt.title("Accuracy for Each Digit")
plt.show()

# 13. Misclassified Images
wrong_indices = np.where(y_test != y_pred)[0][:10]

if len(wrong_indices) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for ax in axes.flat:
        ax.axis("off")

    for i, index in enumerate(wrong_indices):
        axes.flat[i].imshow(
            X_test[index].reshape(8, 8),
            cmap="gray"
        )
        axes.flat[i].set_title(
            f"Actual: {y_test[index]}\nPredicted: {y_pred[index]}"
        )
        axes.flat[i].axis("off")

    plt.suptitle("Misclassified Handwritten Digits")
    plt.tight_layout()
    plt.show()
else:
    print("No misclassified images found.")

# 14. Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nPractical Completed Successfully!")

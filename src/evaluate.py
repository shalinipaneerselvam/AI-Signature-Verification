import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import matplotlib.pyplot as plt
import os

# =========================
# Configuration
# =========================
MODEL_PATH = "models/signature_model.keras"
TEST_DIR = "data/test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# =========================
# Load Model
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# =========================
# Load Test Dataset
# =========================
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Class mapping:", test_data.class_indices)

# =========================
# Predictions
# =========================
probabilities = model.predict(test_data)

predictions = (probabilities >= 0.5).astype(int).flatten()
true_labels = test_data.classes

# =========================
# Accuracy
# =========================
accuracy = accuracy_score(true_labels, predictions)

print("\n==============================")
print("Evaluation Results")
print("==============================")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy (%): {accuracy * 100:.2f}%")

# =========================
# Classification Report
# =========================
print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predictions,
        target_names=["Forged", "Genuine"]
    )
)

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(true_labels, predictions)

print("\nConfusion Matrix:")
print(cm)

# =========================
# Save Confusion Matrix
# =========================
os.makedirs("results/figures", exist_ok=True)

plt.figure(figsize=(6, 5))
plt.imshow(cm)

plt.title("Signature Verification - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks([0, 1], ["Forged", "Genuine"])
plt.yticks([0, 1], ["Forged", "Genuine"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()

output_path = "results/figures/confusion_matrix.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("\nConfusion matrix saved to:")
print(output_path)
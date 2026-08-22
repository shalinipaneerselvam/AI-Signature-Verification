import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# =========================
# Configuration
# =========================
MODEL_PATH = "models/signature_model.keras"
IMG_SIZE = (128, 128)

# =========================
# Check Image Path
# =========================
if len(sys.argv) < 2:
    print("Usage:")
    print("python src/predict.py <image_path>")
    sys.exit()

image_path = sys.argv[1]

# =========================
# Load Model
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# Load Image
# =========================
image = Image.open(image_path)

# Convert to grayscale
image = image.convert("L")

# Resize
image = image.resize(IMG_SIZE)

# Convert to numpy
image_array = np.array(image, dtype=np.float32)

# Normalize
image_array = image_array / 255.0

# Add channel dimension
image_array = np.expand_dims(image_array, axis=-1)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# =========================
# Prediction
# =========================
prediction = model.predict(image_array, verbose=0)[0][0]

# =========================
# Result
# =========================
if prediction >= 0.5:
    result = "GENUINE"
    confidence = prediction * 100
else:
    result = "FORGED"
    confidence = (1 - prediction) * 100

print("\n==============================")
print("Signature Verification Result")
print("==============================")
print("Image:", image_path)
print("Prediction:", result)
print(f"Confidence: {confidence:.2f}%")
print("==============================")
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# Configuration
# =========================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
MODEL_PATH = "models/signature_model.keras"

# =========================
# Data Loading
# =========================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Class mapping:", train_data.class_indices)

# =========================
# CNN Model
# =========================
model = models.Sequential([
    layers.Input(shape=(128, 128, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

# =========================
# Compile
# =========================
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# Training
# =========================
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

# =========================
# Evaluation
# =========================
loss, accuracy = model.evaluate(test_data)

print("\n==============================")
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
print("==============================")

# =========================
# Save
# =========================
model.save(MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)
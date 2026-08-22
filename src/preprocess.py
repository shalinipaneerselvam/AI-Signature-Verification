import os
from PIL import Image

# Source and destination folders
DATA_DIR = "data"

folders = [
    "train/genuine",
    "train/forged",
    "test/genuine",
    "test/forged"
]

TARGET_SIZE = (128, 128)

for folder in folders:
    folder_path = os.path.join(DATA_DIR, folder)

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".png"):
            continue

        file_path = os.path.join(folder_path, filename)

        try:
            # Open image
            image = Image.open(file_path)

            # Convert everything to grayscale
            image = image.convert("L")

            # Resize
            image = image.resize(TARGET_SIZE)

            # Save
            image.save(file_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

print("================================")
print("Preprocessing completed!")
print("All images converted to:")
print("- Grayscale")
print("- 128 x 128 pixels")
print("================================")
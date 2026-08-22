# AI Signature Verification

An AI-based signature verification system that uses deep learning to classify handwritten signatures as **Genuine** or **Forged**.

## 📌 Project Overview

Signature verification is an important application in banking, finance, legal documents, and identity verification.

This project uses a **Convolutional Neural Network (CNN)** to analyze signature images and classify them into two categories:

* Genuine Signature
* Forged Signature

The model was trained using the **CEDAR Signature Dataset** and achieved approximately **97.35% test accuracy**.

## 🎯 Objectives

* Detect whether a signature is genuine or forged.
* Preprocess signature images for deep learning.
* Train a CNN-based image classification model.
* Evaluate model performance using accuracy and confusion matrix.
* Provide a prediction script for testing individual signature images.

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Pillow
* Git & GitHub

## 📂 Project Structure

```text
AI-Signature-Verification/
│
├── app.py
│
├── models/
│   └── signature_model.keras
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── results/
│   └── figures/
│       └── confusion_matrix.png
│
├── .gitignore
└── README.md
```

## 🔄 Workflow

```text
Signature Image
       ↓
Image Preprocessing
       ↓
Resize to 128 × 128
       ↓
Grayscale Conversion
       ↓
CNN Model
       ↓
Prediction
       ↓
Genuine / Forged
```

## 🧹 Data Preprocessing

The signature images are preprocessed before training:

1. Convert images to grayscale.
2. Resize images to **128 × 128 pixels**.
3. Organize images into Genuine and Forged classes.
4. Split the dataset into training and testing sets.

## 🧠 Model

A Convolutional Neural Network (CNN) is used for image classification.

The model learns visual characteristics of handwritten signatures and predicts whether a given signature belongs to the genuine or forged class.

## 📊 Results

| Metric        |        Result |
| ------------- | ------------: |
| Test Accuracy |    **97.35%** |
| Test Loss     |    **0.1397** |
| Image Size    | **128 × 128** |
| Classes       |         **2** |

### Confusion Matrix

The confusion matrix generated during evaluation is available at:

`results/figures/confusion_matrix.png`

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/shalinipaneerselvam/AI-Signature-Verification.git
cd AI-Signature-Verification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows PowerShell:**

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn pillow
```

### 5. Run preprocessing

```bash
python src/preprocess.py
```

### 6. Train the model

```bash
python src/train.py
```

### 7. Evaluate the model

```bash
python src/evaluate.py
```

### 8. Predict a signature

```bash
python src/predict.py
```

## 📈 Model Performance

The trained CNN achieved **97.35% accuracy on the test dataset**, demonstrating strong classification performance on the prepared signature data.

## 🔮 Future Improvements

* Add a web-based user interface for uploading signatures.
* Improve robustness using data augmentation.
* Experiment with transfer learning models such as MobileNet or EfficientNet.
* Add confidence scores to predictions.
* Test the model on signatures from different datasets.
* Deploy the application using Streamlit or Flask.

## 👩‍💻 Author

**Shalini Panneerselvam**

B.Tech Information Technology Student

GitHub:
https://github.com/shalinipaneerselvam

## ⭐ Project

If you find this project useful, consider giving it a ⭐ on GitHub.

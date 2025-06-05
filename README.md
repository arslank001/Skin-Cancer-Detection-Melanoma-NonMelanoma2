# 🧠 Skin Cancer Detection Web App using ResNet50

A deep learning-powered web application for detecting **Melanoma** and **Non-Melanoma** skin cancers using **ResNet50** architecture. This application leverages pre-trained models and fine-tuning to provide accurate, real-time classification of dermoscopic skin images through a clean, user-friendly Streamlit interface.

---

## 📌 Features

- 🔍 **Upload a dermoscopic image**
- 🩺 **Predicts Melanoma or Non-Melanoma**
- 📊 **Displays confidence score**
- ⚙️ **Trained using ResNet50 with transfer learning**
- 🧪 **Optimized threshold from ROC curve**
- ☁️ **Automatically downloads `.h5` model from Google Drive**
- 📱 **Responsive Streamlit UI**

---

## 🧰 Tech Stack

| Component       | Technology                |
|----------------|----------------------------|
| Model          | ResNet50 (Keras / TensorFlow) |
| Frontend       | Streamlit                  |
| Libraries      | NumPy, Pillow, scikit-learn, matplotlib |
| Deployment     | Streamlit Cloud            |
| Model Format   | `.h5` (HDF5 format)        |

---

## 🧬 Dataset Summary

- **Training Set**:
  - Melanoma: 4312 images
  - Non-Melanoma: 4312 images
- **Testing Set**:
  - Melanoma: 1079 images
  - Non-Melanoma: 1079 images

The dataset is balanced and underwent preprocessing, including:
- Image resizing (224x224)
- Normalization
- Data augmentation
- Class weighting for imbalance handling

---

## 📦 Installation

```bash
git clone https://github.com/arslank001/skin-cancer-detection-web-app-using-resnet50.git
cd skin-cancer-detection-web-app-using-resnet50
pip install -r requirements.txt

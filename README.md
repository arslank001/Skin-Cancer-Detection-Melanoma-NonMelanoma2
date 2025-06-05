# 🧠 Skin Cancer Detection Web App using ResNet50

A deep learning-powered web application for detecting **Melanoma** and **Non-Melanoma** skin cancers using **ResNet50** architecture. This application leverages pre-trained models and fine-tuning to provide accurate, real-time classification of dermoscopic skin images through a clean, user-friendly Streamlit interface.

---

## 📌 Features

- 🔍 **Upload a dermoscopic image**
- 🩺 **Predicts Melanoma or Non-Melanoma**
- 📊 **Displays confidence score**
- ⚙️ **Trained using ResNet50 with transfer learning**
- 🧪 **Optimized threshold from ROC curve**
- ☁️ **Automatically downloads `.h5` model from Google Drive/second option is run locally**
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

git clone https://github.com/arslank001/skin-cancer-detection-web-app-using-resnet50.git
cd skin-cancer-detection-web-app-using-resnet50
pip install -r requirements.txt

---

## 🚀 Run the App
For running of web app on localhosts, run the below given two commands in anaconda prompt:
  - cd "C:\Users\LENOVO\Desktop\testing skin cancer app\Web App and Connected Model"     (Path where your app.py file and melanoma_model.h5 file must exist)
  - python -m streamlit run app.py

---

## 🧾 Requirements
- Python 3.10 or 3.11
- TensorFlow 2.13.0
- Streamlit 1.45.1
- Additional dependencies in requirements.txt

---

## 📸 Sample Screenshot
**🔹 Before Prediction**
![Before Prediction](web app images/screenshot_before.png)

**🔹 After Prediction**
![After Prediction](web app images/screenshot_after.png)

---

## 🙋 About Melanoma and Non-Melanoma
- Melanoma is a dangerous type of skin cancer that develops from pigment-containing cells. Early detection is vital.
- Non-Melanoma skin cancers are more common and less aggressive, but still require medical attention.

---

## ✍️ Author
Arsalan Tahir
🔗 LinkedIn
🐙 GitHub

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
import os

# === Download model from Google Drive if not exists ===
MODEL_FILENAME = "melanoma_model.h5"

# Only download if model is not already present
if not os.path.exists(MODEL_FILENAME):
    st.info("Downloading model. Please wait a moment...")
    url = "https://drive.google.com/uc?id=1P66ypqXbrMnt0IuziCe59RI2irB7JWyY"
    response = requests.get(url)
    with open(MODEL_FILENAME, "wb") as f:
        f.write(response.content)
    st.success("Model downloaded successfully!")

# === Load the model ===
model = tf.keras.models.load_model(MODEL_FILENAME)

# === Optimal Threshold (set from training output) ===
OPTIMAL_THRESHOLD = 0.4820

# === App Title ===
st.set_page_config(page_title="Skin Cancer Detection", layout="centered")
st.title("🔬 Skin Cancer Detection Web App")
st.subheader("Predicts if a dermoscopic image is **Melanoma** or **Non-Melanoma**")

# === Informative Description ===
with st.expander("🧬 What is Melanoma vs Non-Melanoma?"):
    st.markdown("""
    **Melanoma** is a dangerous type of skin cancer that can spread to other parts of the body if not detected early.  
    **Non-Melanoma** refers to other common, less aggressive skin cancers like basal cell carcinoma or squamous cell carcinoma.  
    Upload a **dermoscopic image** to get a prediction.
    """)

# === Image Upload ===
uploaded_file = st.file_uploader("📤 Upload a skin image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

    # === Preprocess ===
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # === Prediction ===
    prediction_prob = model.predict(img_array)[0][0]
    prediction_class = "Melanoma" if prediction_prob > OPTIMAL_THRESHOLD else "Non-Melanoma"

    # === Show Result ===
    st.markdown("---")
    st.markdown(f"<h3 style='text-align: center;'>🩺 Prediction: <span style='color:#ff4b4b'>{prediction_class}</span></h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'>Confidence Score: <strong>{prediction_prob:.4f}</strong></p>", unsafe_allow_html=True)

# === Footer ===
st.markdown("---")
st.caption("Developed by ARSALAN TAHIR • Powered by TensorFlow + Streamlit")

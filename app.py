import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# === Load Trained Model ===
MODEL_PATH = "melanoma_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# === Optimal Threshold from training ===
OPTIMAL_THRESHOLD = 0.4820  # Update if needed

# === App Title ===
st.set_page_config(page_title="Melanoma Detector", layout="centered")
st.title("🔬 Skin Cancer Detection Web App")
st.subheader("Predicts if a dermoscopic image is **Melanoma** or **Non-Melanoma**")

# === Informative Description ===
with st.expander("🧬 What is Melanoma vs Non-Melanoma?"):
    st.markdown("""
**Melanoma** is a serious form of skin cancer that begins in cells known as melanocytes.  
It can spread to other organs if not detected early.

**Non-Melanoma** includes more common, less aggressive types like basal cell carcinoma (BCC) or squamous cell carcinoma (SCC).  
These are usually easier to treat and less likely to spread.

This tool helps screen skin lesion images, but it's **not a replacement for medical advice**. Please consult a dermatologist for clinical evaluation.
""")

# === Image Upload ===
uploaded_file = st.file_uploader("📤 Upload a dermoscopic image...", type=["jpg", "jpeg", "png"])

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

    # === Result Display ===
    st.markdown("### 🩺 Prediction Result")
    st.success(f"🧾 **Diagnosis:** `{prediction_class}`")
    st.info(f"📊 **Confidence Score:** `{prediction_prob:.4f}`")

# === Footer ===
st.markdown("---")
st.caption("Developed by ARSALAN TAHIR • Powered by TensorFlow + Streamlit")

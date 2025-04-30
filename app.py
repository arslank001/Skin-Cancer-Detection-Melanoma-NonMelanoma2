import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
MODEL_PATH = "melanoma_detector_weights.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# App UI
st.title("Skin Cancer Detection Web App")
st.subheader("Upload a dermoscopic image to detect Melanoma or Non-Melanoma")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    result = "Melanoma" if prediction > 0.5 else "Non-Melanoma"

    st.success(f"Prediction: {result}")

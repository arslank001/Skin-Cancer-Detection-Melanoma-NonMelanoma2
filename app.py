{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c79a0332-56a2-4106-a26e-db13bff200fd",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'streamlit'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mstreamlit\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mst\u001b[39;00m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mnumpy\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mnp\u001b[39;00m\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mtensorflow\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mtf\u001b[39;00m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'streamlit'"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.preprocessing import image\n",
    "from PIL import Image\n",
    "import os\n",
    "\n",
    "# Load the trained model from your full path\n",
   
     "MODEL_PATH = "melanoma_detector_weights.h5\"\n",
    "model = tf.keras.models.load_model(MODEL_PATH)\n",
    "\n",
    "# App title\n",
    "st.title(\"Skin Cancer Detection Web App\")\n",
    "st.subheader(\"Upload a dermoscopic image to detect Melanoma or Non-Melanoma\")\n",
    "\n",
    "# File uploader\n",
    "uploaded_file = st.file_uploader(\"Choose an image...\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    # Load and display image\n",
    "    img = Image.open(uploaded_file)\n",
    "    st.image(img, caption=\"Uploaded Image\", use_column_width=True)\n",
    "\n",
    "    # Preprocessing\n",
    "    img = img.resize((224, 224))\n",
    "    img_array = image.img_to_array(img) / 255.0\n",
    "    img_array = np.expand_dims(img_array, axis=0)\n",
    "\n",
    "    # Predict\n",
    "    prediction = model.predict(img_array)\n",
    "    result = \"Melanoma\" if prediction > 0.5 else \"Non-Melanoma\"\n",
    "\n",
    "    # Display result\n",
    "    st.success(f\"Prediction: {result}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "0f46d0c5-a5db-418d-bdd8-eeb1ab5a0daa",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (tfenv)",
   "language": "python",
   "name": "tfenv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

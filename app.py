import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import pickle

# Load models
leaf_model = load_model("mobilenetv2_base_model.h5")
with open("best_model.pkl", "rb") as f:
    pickle_model = pickle.load(f)

# Class names and recommendations
class_names = ["Healthy", "Rust", "Phoma", "Cercospora"]
recommendations = {
    "Rust": "Apply copper-based fungicide weekly.",
    "Phoma": "Prune affected branches and improve drainage.",
    "Cercospora": "Improve shade, reduce overhead irrigation.",
    "Healthy": "Your plant is healthy. No action needed."
}

# Prediction function
def predict_disease(image, model):
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_idx, confidence

# Streamlit app
st.set_page_config(page_title="Coffee Disease Detector", layout="centered")
st.title("☕ Coffee Leaf & Berry Disease Detector")

st.write("Upload a photo of a coffee **leaf** or **berry** to detect any disease.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
disease_type = st.selectbox("Select type of image", ["Leaf", "Berry"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        model = leaf_model if disease_type == "Leaf" else berry_model
        class_idx, confidence = predict_disease(image, model)

        label = class_names[class_idx]
        recommendation = recommendations[label]

        st.success(f"Prediction: **{label}** ({confidence * 100:.2f}%)")
        st.info(f"Recommendation: {recommendation}")

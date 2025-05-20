import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import numpy as np

# Load models
@st.cache_resource
def load_models():
    berry_model = load_model("models/mobilenetv2_base_model.h5")
    with open("models/best_model.pkl", "rb") as f:
        leaf_model = pickle.load(f)
    return berry_model, leaf_model

berry_model, leaf_model = load_models()

# Class dictionaries
berry_classes = ['Coffee__Berry_borer', 'Coffee__Damaged_bean', 'Coffee__Healthy_bean']
leaf_classes = ['miner', 'rust', 'phoma', 'no disease']

recommendations = {
    'Coffee__Berry_borer': "Apply insecticide and practice proper pruning.",
    'Coffee__Damaged_bean': "Check for fungal infections and improve drying techniques.",
    'Coffee__Healthy_bean': "No issue detected.",
    'miner': "Apply recommended pesticide and remove affected leaves.",
    'rust': "Use resistant varieties and copper-based fungicides.",
    'phoma': "Improve drainage and use fungicide treatment.",
    'no disease': "Healthy leaf. No action needed."
}

# Image preprocessor
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.title("☕ Coffee Disease Detector")

uploaded_image = st.file_uploader("Upload an image of a coffee leaf or berry", type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    disease_type = st.radio("Select image type:", ["Leaf", "Berry"])

    if st.button("Predict"):
        st.write("Analyzing...")
        processed = preprocess_image(image)

        if disease_type == "Berry":
            prediction = berry_model.predict(processed)
            predicted_class = berry_classes[np.argmax(prediction)]
        else:
            prediction = leaf_model.predict(processed)
            predicted_class = leaf_classes[np.argmax(prediction)]

        st.success(f"Prediction: **{predicted_class}**")
        st.info(f"Recommendation: {recommendations[predicted_class]}")

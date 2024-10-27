import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import joblib
import os

# Load the EfficientNetB3 model for feature extraction
base_model = EfficientNetB3(weights="imagenet",include_top=False,input_shape=(256,256,3))
base_model.trainable=False

# Load the pre-trained Random Forest model
rf_classifier = joblib.load('random_forest_model.pkl')  # Make sure you have saved your RandomForestClassifier

# Function to preprocess and extract features from an image
def extract_features_from_image(image):
    image = image.resize((256, 256))
    image = np.array(image)
    
    # Ensure the image has 3 channels
    if image.shape[-1] == 1 or len(image.shape) == 2:  # Handle grayscale images
        image = np.stack((image,) * 3, axis=-1)
    
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    features = base_model(image)
    features = tf.reshape(features, (features.shape[0], -1))
    return features.numpy()

# Streamlit interface
st.title("Pneumonia Detection Interface")
st.write("Upload an image and get the classification result.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Extract features from the image
    features = extract_features_from_image(image)

    # Predict using the Random Forest model
    prediction = rf_classifier.predict(features)
    class_names = {0:'Normal', 1:'Pneumonia'}

    # Display the result
    st.write(f"Predicted class: {class_names[prediction[0]]}")

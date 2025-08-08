import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Define class names in the same order as your model was trained
# class_names = [
#     "Bacterial Spot",
#     "Healthy",
#     "Late Blight",
#     "Mosaic Virus",
#     "Yellow Leaf Curl Virus",
#     "Leaf Mold"
# ]
# class_names = [
#     "Healthy",
#     "Yellow Leaf Curl Virus",
#     "Late Blight",
#     "Leaf Mold",
#     "Mosaic Virus",
#     "Bacterial Spot"
# ]
class_names = [
    "Leaf Mold",
    "Bacterial Spot",
    "Late Blight",
    "Yellow Leaf Curl Virus",
    "Healthy",
    "Mosaic Virus"
    
]


# Load the trained CNN model
model = tf.keras.models.load_model("tomato_disease_mobilenetv2_model.h5")

def predict_image(img):
    img = img.resize((150, 150))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return class_names[predicted_class], confidence

# Streamlit UI
st.title("Tomato Leaf Disease Classifier")
st.write("Upload an image of a tomato leaf to classify its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)
    
    predicted_class, confidence = predict_image(image_data)
    st.write(f"### Prediction: {predicted_class} ({confidence*100:.2f}% confidence)")

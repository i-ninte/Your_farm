import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the SavedModel
model_path = "C:/Users/Alif Osman Otoo/Desktop/your_farm/converted_savedmodel/model.savedmodel/saved_model.pb"

model = tf.saved_model.load(model_path)

# Function to perform inference
infer = model.signatures["serving_default"]

# Load the labels
class_names = open("C:/Users/Alif Osman Otoo/Desktop/your_farm/converted_savedmodel/labels.txt", "r").readlines()


st.title("Crop Disease Prediction")

uploaded_file = st.file_uploader("Upload an image of the crop", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image = np.array(image)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Convert the image to a tensor
    tensor = tf.convert_to_tensor(image)

    # Perform inference
    prediction = infer(tensor)['sequential_3']  # Using the correct output layer name
    prediction = prediction.numpy()
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    st.write(f"Class: {class_name}")
    st.write(f"Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%")

import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_option = st.sidebar.radio(
    "Choose a model",
    ("CIFAR-10 CNN", "MobileNetV2")
)

# CIFAR-10 Class Names
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Load CIFAR-10 model function
def cifar10_model():
    # Load and preprocess CIFAR-10 dataset
    (X_train, y_train), _ = datasets.cifar10.load_data()
    X_train = X_train / 255.0
    y_train = y_train

    # Build the CIFAR-10 CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    # Load pre-trained weights if available
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    return model

# Load MobileNetV2 model function
def mobilenetv2_model():
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    # Add custom layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Function to preprocess image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array / 255.0

# Main logic
if model_option == "CIFAR-10 CNN":
    st.title("CIFAR-10 Image Classification")
    model = cifar10_model()
    uploaded_file = st.file_uploader("Upload an image (32x32 resolution):", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        preprocessed_image = preprocess_image(image, (32, 32))
        prediction = model.predict(preprocessed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence_score = np.max(prediction)
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence Score: **{confidence_score * 100:.2f}%**")

elif model_option == "MobileNetV2":
    st.title("MobileNetV2 Image Classification")
    model = mobilenetv2_model()
    uploaded_file = st.file_uploader("Upload an image (224x224 resolution):", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        preprocessed_image = preprocess_image(image, (224, 224))
        prediction = model.predict(preprocessed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence_score = np.max(prediction)
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence Score: **{confidence_score * 100:.2f}%**")

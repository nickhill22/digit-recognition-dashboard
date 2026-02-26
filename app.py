import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np

st.title("Handwritten Digit Recognition")

@st.cache_resource
def load_model():
    (x_train, y_train), _ = keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)

    model = keras.Sequential([
        keras.Input(shape=(28,28,1)),
        keras.layers.Conv2D(32, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=1, verbose=0)
    return model

model = load_model()

st.success("Model Loaded Successfully ✅")

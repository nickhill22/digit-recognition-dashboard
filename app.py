import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import os

st.set_page_config(page_title="Digit Recognition", layout="centered")
st.title("Handwritten Digit Recognition")

MODEL_PATH = "mnist_cnn.keras"

@st.cache_resource
def get_model():

    # If model does not exist, train it
    if not os.path.exists(MODEL_PATH):

        st.info("Training model for first time... Please wait ⏳")

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train = x_train / 255.0
        x_test = x_test / 255.0

        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

        model.save(MODEL_PATH)

        st.success("Model trained and saved successfully ✅")

    return tf.keras.models.load_model(MODEL_PATH)


model = get_model()
st.success("Model Ready ✅")

st.write("Draw a digit below and click Predict")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):

    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")
    else:
        img = canvas_result.image_data
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)
        digit = np.argmax(prediction)

        st.subheader(f"Predicted Digit: {digit}")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("mnist_cnn.keras")

st.title("🧠 Handwritten Digit Recognition Dashboard")
st.write("Draw a digit below:")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
)

def predict(img):
    img = img.resize((28, 28))
    img = img.convert("L")
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array)
    return prediction

if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data[:, :, 0].astype("uint8"))
    prediction = predict(img)
    digit = np.argmax(prediction)

    st.subheader(f"Prediction: {digit}")

    fig, ax = plt.subplots()
    ax.bar(range(10), prediction[0])

    st.pyplot(fig)

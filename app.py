import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

st.title("Handwritten Digit Recognizer")
st.write("Upload a digit image (28x28 pixels, black on white)")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    image = ImageOps.invert(image).resize((28,28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

      st.image(image, caption="Processed Image",width=150)
      st.write(f"### Predicted Digit: {predicted_digit}")

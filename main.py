from neural_network import *
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2


nw = Network([28 * 28, 10, 10], [Tanh(), Tanh()], "./network")


SIZE = 28 * 10
canvas_result = st_canvas(
    fill_color="#ffffff",
    stroke_width=20,
    stroke_color="#000000",
    background_color="#ffffff",
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    img = cv2.resize(canvas_result.image_data.astype("uint8"), (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype("float32")
    img = 255 - img
    img /= 255
    img = img.reshape(784, 1)
    output = nw.forward(img)

    st.write(f"# {np.argmax(output)}")

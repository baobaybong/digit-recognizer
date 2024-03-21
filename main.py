import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from utils import predict, format, CNN, ResNet

from PIL import Image
import numpy as np
import os

def main():
    st.set_page_config(
        page_title="Digit Recognizer",
        page_icon="🎨",
    )

    st.title("Digit recognizer🎨🖌️")
    st.subheader("Draw a digit:")
    col1, col2 = st.columns(2)

    with col1:
        canvas_result = st_canvas(
            background_color="#FFFFFF",
            height=300,
            width=300,
        )
    
    with col2:
        num_models = 2 if st.checkbox("Compare two models") else 1

        cols = st.columns(num_models)

        for id, col in enumerate(cols):
            with col:
                models = os.listdir("models")
                model_selected = st.selectbox(
                    "Choose model",
                    models,
                    format_func=lambda x: x.split(".")[0],  
                    key=id
                )
                
                if canvas_result.image_data is not None:
                    img = np.array(format(canvas_result.image_data))[0] * 255
                    st.image(Image.fromarray(img).convert('L'))

                    if np.all(img == 0):
                        st.write("Draw a digit to predict!")
                    else:
                        st.write("Predicted...")

                        st.markdown("""
                        <style>
                        .prediction {
                            font-size:50px !important;
                            font-weight: bold;
                        }
                        </style>
                        """, unsafe_allow_html=True)

                        st.markdown(f'<p class="prediction">{predict(canvas_result.image_data, model_selected)} !!</p>', unsafe_allow_html=True)
                else:
                    st.write("Image data empty")
    
    with st.expander("See model descriptions"):
        with open("README.md", "r") as file:
            st.markdown(file.read().split("## Models:")[1])

    with st.expander("About me"):
        st.write("Made by Bao Bay Bong")
        st.write("Check out my GitHub here:", "https://github.com/baobaybong")

if __name__ == "__main__":
    main()